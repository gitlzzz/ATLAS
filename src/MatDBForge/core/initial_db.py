"""
This script generates a pandas dataframe containing a set of
base (unperturbed) structures and a certain number of structures
with an applied perturbation with respect to the temperature.
"""

import itertools as it
import json as js
import os
import pathlib
import re
import warnings
from io import BytesIO, TextIOWrapper
from multiprocessing import Pool
from typing import Union

import ase.io as aseio
import catkit.gen.surface as cts
import time
import emmet
import numpy as np
import pandas as pd
import pymatgen.io.vasp as vasp
import rich.progress as riprg
import rich.console as ricns
import rich.align as rialg
import rich.live as riliv
from aiida import orm
from dscribe.descriptors import SOAP
from mp_api.client import MPRester
from pymatgen.core.periodic_table import Element, Species
from pymatgen.core.structure import Lattice, Structure
from pymatgen.core.surface import Slab
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from slugify import slugify

import MatDBForge.core.structure as mdf_struct
from MatDBForge.core import exceptions as mdbex
from MatDBForge.core import utils as ut

# Filtering certain warnings
warnings.filterwarnings("ignore", category=vasp.outputs.UnconvergedVASPWarning)


class BinaryPhaseDiagram:
    """
    Class representing a binary phase diagram of materials.

    Parameters:
        material (str): The name of the material.
        *phases (Phase): Variable number of Phase objects representing the
                         phases in the diagram.

    Attributes:
        phases (list): List of Phase objects representing the phases in the diagram.
        material (str): The name of the material.

    """

    def __init__(self, material: str, *phases: "Phase"):
        self.phases = []
        self.phase_names = []
        self.material = material
        self.phase_dict = {}

        for phase in phases:
            self.add_phase(phase)

        for phase in self.phases:
            self.phase_names.append(phase.name)

    def add_phase(self, phase):
        """Add a phase to the phase diagram.

        Parameters:
            phase (Phase): The Phase object to add.

        """
        self.phases.append(phase)
        phase.phasediagram = self.__class__
        self.phase_dict[phase.name] = phase

    def get_phase(self, phase):
        """
        Gets a Phase object from a BinaryPhaseDiagram using either
        the phase name (as a string) or a Phase object.

        Parameters
        ----------
        phase : str | Phase
            Phase or name of the desired phase.

        Returns
        -------
        Phase
            Phase object corresponding to the given phase

        Raises
        ------
        TypeError
            _description_
        TypeError
            _description_
        """

        if isinstance(phase, Phase):
            return self.phase_dict[phase.name]
        if isinstance(phase, str):
            phase_str = phase
            phase = self.phase_dict.get(phase_str, None)
            if phase:
                return phase
            else:
                raise mdbex.PhaseNotFound(self, phase_str)
        else:
            raise TypeError("The given phase object is not a Phase-like object.")

    def __repr__(self):
        """
        Override attribute lookup.
        Allows accessing phases by their names as attributes.

        Parameters:
            name (str): The name of the attribute.

        Returns:
            Phase: The Phase object with the specified name.

        Raises:
            AttributeError: If the attribute is not found.

        """
        repr_str = (
            f"{self.material} phase diagram with phases:"
            f" {[phase.name for phase in self.phases]}"
        )

        return repr_str

    def __getattr__(self, name):
        if name in self.phase_dict:
            return self.phase_dict[name]
        else:
            raise AttributeError(f"'PhaseDiagram' object has no attribute '{name}'")


# TODO: Imlpement this
class TernaryPhaseDiagram:
    ...


class Phase:
    """Class representing a phase in a phase diagram.

    Parameters:
    name: str
        The name of the phase.
    base_elem:
        The base element of the phase.
    base_elem_comp_max: float
        The maximum composition of the base element in the phase.
    base_elem_comp_min: float
        The minimum composition of the base element in the phase.
    prototype: str
        The prototype of the phase.
    offset: float
        The offset value of the phase.
    phase_diagram: PhaseDiagram
        The parent PhaseDiagram object that the phase belongs to.

    """

    def __init__(
        self,
        name: str,
        base_elem,
        base_elem_comp_max: float,
        base_elem_comp_min: float,
        prototype: str,
        offset: float = 0,
        phase_diagram: "BinaryPhaseDiagram" = None,
    ):
        self.name = name
        self.base_elem = Element(base_elem)
        self.base_elem_comp_max = float(base_elem_comp_max)
        self.base_elem_comp_min = float(base_elem_comp_min)
        self.prototype = prototype
        self.offset = float(offset)
        self.phase_diagram = phase_diagram

        # if phase_diagram is not None:
        # phase_diagram.add_phase(self)
        # self.phase_diagram = phase_diagram.__name__

    def __str__(self):
        """Return a string representation of the phase.

        Returns:
            str: The string representation of the phase.

        """
        repr_string = (
            f"Phase '{self.name}', {self.base_elem_comp_min*100:.1f}% {self.base_elem} - "
            f"{self.base_elem_comp_max*100:.1f}% {self.base_elem} (± {self.offset*100:.1f}%)"
        )

        if self.phase_diagram is not None:
            # repr_string += f" (belongs to {self.phase_diagram.material})"
            repr_string += f" (belongs to {self.phase_diagram})"

        return repr_string

    def __key(self):
        return (
            self.name,
            self.base_elem,
            self.base_elem_comp_max,
            self.base_elem_comp_min,
            self.prototype,
            self.offset,
            self.phase_diagram,
        )

    def __eq__(self, other):
        if not isinstance(other, Phase):
            # Do not try to compare against different types
            return NotImplemented
        return self.__key() == other.__key()

    def __hash__(self):
        return hash(self.__key())

    def perc_in_phase(self, perc: float, offset: bool = True) -> bool:
        if perc > 1:
            perc /= 100

        if offset:
            offset = self.offset
        else:
            offset = 0

        inPhase = (
            (self.base_elem_comp_min - offset)
            < perc
            < (self.base_elem_comp_max + offset)
        )

        if inPhase:
            return True
        else:
            return False


class InitialDatabase:
    """
    Object that creates an initial database where structures will be
    stored. Materials are downloaded using the materials project API.
    For it to work, a 'secrets.json' file should be located in the
    same directory.

    Attributes
    ----------
    df : pd.Dataframe
        Dataframe containing the structures for the initial database.
    secrets : dict
        Object containing secrets related to the materials project database.
    database_name : str
        Orientative name for the database. Will be used for saving it into a file.
    max_num_atoms: int
        Maximum number of atoms present in any structure generated.


    Notes
    -----
        The json file should have the following structure:

        {
            "API_KEY": "XXXXXX"
        }


    """

    # Boltzmann constant in J/(Da*K)
    kB = 8.314

    # Sourced from CODATA 2018
    Bohr2Ang = 0.5291772109030
    Ang2Bohr = 1 / Bohr2Ang
    Eh2eV = 27.211386245988
    eV2Eh = 1 / Eh2eV

    def __init__(self, database_name: str, max_num_atoms: int = 64) -> None:
        # Name of the database
        self.database_name = database_name

        # Setting the maximum number of atoms of any generated structure.
        self.max_num_atoms = max_num_atoms

        # Checking if a database is already found in the cwd
        check_flag = self._check_database()

        # Create the database if it does not exists
        # Load it if otherwise.
        if not check_flag:
            self.df = self._create_database()
        else:
            self.df = self._load_database()

        # Loading materials project API key from a json file
        self.secrets = gather_secrets()

    def __repr__(self):
        # Fields on the dataframe
        # fields = [field for field in self.df.columns]

        # Getting the class name
        class_name = self.__class__.__name__

        # Getting how many rows
        count = len(self.df.count(axis=1))

        repr_string = (
            f"{class_name} named '{self.database_name}' containing {count} structures."
        )

        return repr_string

    def _load_database(self) -> pd.DataFrame:
        """
        Load a database from a pickle file on the cwd or a specific path.

        Returns
        -------
        pd.DataFrame
            Dataframe containing structure data for the initial database
        """

        db_path = pathlib.PurePath(self.database_name)

        if len(db_path.suffixes) == 0:
            suffix = ".pkl"
        else:
            suffix = ""

        database = pd.read_pickle(f"{self.database_name}{suffix}")
        ut.custom_print(f"Loaded '{self.database_name}{suffix}'", "info")

        self.database_name = db_path.name.replace(db_path.suffix, "")

        return database

    def _check_database(self) -> bool:
        """
        This method check if a database with the name 'self.database_name'
        exists in the current working directory or is a path to a existing
        database.

        Returns
        -------
        bool
            True if the database exists, False if not does not.
        """

        # Checking if dataframe already exists on the cwd.
        file_exists = False
        file_check = [file for file in os.listdir() if self.database_name in file]
        if len(file_check) > 0:
            file_exists = True

        # If the database name is a path
        name_as_path = pathlib.Path(self.database_name)

        if name_as_path.exists():
            file_exists = True

        return file_exists

    def _create_database(self) -> pd.DataFrame:
        """
        Create an empty  dataframe in order to be used in the class

        Returns
        -------
        pd.DataFrame
            Empty dataframe containing the necessary columns to be used in the main
            database.
        """
        # Creating a pandas dataframe to store the structures
        df = pd.DataFrame(
            columns=[
                "material_name",
                "material_id",
                "structure",
                "phase",
                "formula",
                "symmetry",
                "base",
                "perturb",
                "unique_id",
                "supercell",
                "surface",
                "bulk",
                "cluster",
                "temperature",
                "magnetic_properties",
                "calc_energy",
                "calc_energy_per_atom",
                "calc_energy_toten",
                "calc_performed",
                "calc_type",
                "calc_output",
            ]
        )

        ut.custom_print(f"Created database '{self.database_name}'.", "done")

        return df

    def _find_supercell_indices(
        self,
        structure,
        get_different_supercells,
        max_atoms,
        initial_supercell_size=5,
        verbose=True,
    ):
        # Initial supercell size
        idx = initial_supercell_size

        # Copying structure
        new_structure = structure.copy(sanitize=True)

        # Setting different supercell geometry for slabs and bulks.
        if isinstance(structure, Slab):
            supercell_vec = [idx, idx, 1]
        else:
            supercell_vec = [idx, idx, idx]

        new_structure.make_supercell(supercell_vec, to_unit_cell=False)

        # Number of atoms of the supercell
        struct_size = len(new_structure.species)
        while struct_size > max_atoms:
            new_structure = structure.copy(sanitize=True)
            idx -= 1

            if isinstance(structure, Slab):
                supercell_vec = [idx, idx, 1]
            else:
                supercell_vec = [idx, idx, idx]

            new_structure.make_supercell(supercell_vec, to_unit_cell=False)
            struct_size = len(new_structure.species)

        structure_list = []
        idx_list = []
        supercell_vec_list = []
        structure_list.append(new_structure)
        idx_list.append(idx)
        supercell_vec_list.append(supercell_vec)

        if verbose:
            ut.custom_print(
                f"Supercell generated - total atoms: {len(new_structure.species)}",
                "debug",
            )

        if get_different_supercells:
            for idx_smaller in range(idx - 1, 0, -1):
                new_structure = structure.copy(sanitize=True)

                if isinstance(structure, Slab):
                    supercell_vec = [idx_smaller, idx_smaller, 1]
                else:
                    supercell_vec = [idx_smaller, idx_smaller, idx_smaller]

                new_structure.make_supercell(supercell_vec, to_unit_cell=False)
                structure_list.append(new_structure)
                idx_list.append(idx_smaller)
                supercell_vec_list.append(supercell_vec)

                if verbose:
                    ut.custom_print(
                        f"Supercell generated - total atoms: {len(new_structure.species)}",
                        "debug",
                    )

        return structure_list, idx_list, supercell_vec_list

    def _check_repeat_struct(self, curr_phase, curr_struct: Structure):
        # ut.custom_print("checking for duplicates", "warn")
        structure_list = self.df.loc[self.df.phase == curr_phase].structure.values
        species_list = set([a.symbol for a in curr_struct.species])

        r_cut = 6
        n_max = 8
        l_max = 6

        soap_structs = []

        soap = SOAP(
            species=species_list,
            periodic=True,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            # average="inner",
            sparse=False,
        )

        for pym_struct in structure_list:
            ase_struct = AseAtomsAdaptor().get_atoms(pym_struct)

            # Create output for multiple system in parallel
            struct_soap = soap.create(ase_struct, n_jobs=-1, verbose=False)

            curr_feat_sum = struct_soap.sum()

            soap_structs.append(curr_feat_sum)

        curr_ase_struct = AseAtomsAdaptor().get_atoms(curr_struct)
        curr_struct_soap = soap.create(curr_ase_struct, n_jobs=-1, verbose=False)
        curr_soap_sum = curr_struct_soap.sum()

        total_soap_arr = np.array(soap_structs)

        comp_arr = np.isclose(curr_soap_sum, total_soap_arr, rtol=7.5e-04, atol=5e-05)

        if np.count_nonzero(comp_arr) > 0:
            ut.custom_print("duplicate found!!", "warn")
            return True

        else:
            return False

    def find_repeat_structures(
        self,
        delete=False,
    ):
        # Getting the unique phases in the dataframe
        phase_list = self.df.phase.unique()

        species = CuZnInitialDatabase.ALLOY_SET
        species_str_list = [spec.symbol for spec in species]
        r_cut = 6
        n_max = 8
        l_max = 6

        tot_unique_name_list = []
        df_total_struct_list = len(self.df.structure.values)

        for curr_phase in phase_list:
            # Setting up the SOAP descriptor
            soap = SOAP(
                species=species_str_list,
                periodic=True,
                r_cut=r_cut,
                n_max=n_max,
                l_max=l_max,
                # average="inner",
                sparse=False,
            )

            ut.custom_print(
                f"Checking for repeated structures for phase '{curr_phase.name}'...",
                "info",
            )

            soap_structs = []

            structure_list = self.df.loc[self.df.phase == curr_phase].structure.values
            name_list = self.df.loc[self.df.phase == curr_phase].material_name.values

            tot_structures = len(structure_list)
            tot_equival = 0

            for pym_struct in structure_list:
                ase_struct = AseAtomsAdaptor().get_atoms(pym_struct)

                # Create output for multiple system in parallel
                struct_soap = soap.create(ase_struct, n_jobs=-1, verbose=True)

                curr_feat_sum = struct_soap.sum()

                soap_structs.append(curr_feat_sum)

            soap_arr = np.array(soap_structs)
            dupl_list = []
            for soap in soap_structs:
                curr_soap_arr = np.array(soap)
                comp_arr = np.isclose(curr_soap_arr, soap_arr, rtol=7.5e-04, atol=5e-05)

                if np.count_nonzero(comp_arr) > 1:
                    dupl_list.append(soap)
                soap_arr = soap_arr[~comp_arr]

            tot_equival = len(dupl_list)
            dupl_names = []

            dupl_list = set(dupl_list)
            for dup in dupl_list:
                soap_1_name = name_list[soap_structs.index(dup)]
                dupl_names.append(soap_1_name)
                tot_unique_name_list.append(soap_1_name)

            ut.custom_print(
                f"Total structures: {tot_structures}, not equivalent: {tot_equival} "
                f"({(tot_equival/tot_structures)*100:.2f}%)",
                "debug",
            )
            ut.custom_print(
                f"{tot_structures - len(dupl_list)} structures marked for deletion.",
                "debug",
            )

        unique_structs = list(
            set([name for name in tot_unique_name_list if "super" in name])
        )

        if delete:
            base_struct_names = list(
                set(self.df.loc[self.df.base].material_name.values)
            )
            unique_structs.extend(base_struct_names)

            unique_structures_df = self.df[self.df.material_name.isin(unique_structs)]

            unique_structures_df_drop = unique_structures_df.drop_duplicates(
                subset=["material_name"],
                keep="first",
                # inplace=True,
            )

            # print("unique_structures_df: ", unique_structures_df)

            self.df = unique_structures_df_drop
            ut.custom_print(
                f"Deleted {df_total_struct_list - len(unique_structs)} structures.",
                "warn",
            )

        else:
            ut.custom_print(
                f"{len(unique_structs)} repeated structures found. "
                "Database untouched as 'delete' is set to False.",
                "info",
            )

    def gather_base_structures(self, target_structures):
        # Checking which materials are already on the database
        missing_mat = set(target_structures) - set(self.df["material_id"].values)

        # Querying materials project database.
        with MPRester(self.secrets["API_KEY"]) as mpr:
            query_result = mpr.summary.search(material_ids=missing_mat)
            for material in query_result:
                for phase in CuZnInitialDatabase.CUZN_PHASES.phases:
                    if phase.prototype == material.material_id:
                        curr_phase = phase.name
                    else:
                        curr_phase = np.nan

                curr_struct = mdf_struct.Bulk(
                    material_id=str(material.material_id),
                    structure=material.structure,
                    temperature=np.nan,
                    perturb=False,
                    formula=material.composition_reduced,
                    symmetry=material.get_space_group_info(),
                    base=True,
                    phase=curr_phase,
                    magnetic_properties=material.total_magnetization,
                    energy_per_atom=material.energy_per_atom,
                )

                self.df = curr_struct.save_to_db(self.df)

        self.df.set_index("material_id", inplace=True, drop=False)

    def read_base_structures(self, path: str, target_structures=None):
        ut.custom_print("Reading relaxed structures...")

        # Getting the path where the calculations will be searched for.
        if path:
            read_path = pathlib.Path(path)
        else:
            read_path = pathlib.Path()

        # Getting the list of directories containing the simulations
        # list_dir = os.walk(read_path, followlinks=True)

        if target_structures:
            selection_criteria = target_structures
        else:
            selection_criteria = CuZnInitialDatabase.CUZN_PHASES.keys()

        folders = read_path.glob("./*")
        list_dir = [
            fold
            for fold in folders
            if pathlib.PurePath(fold).name in selection_criteria
        ]

        for calc_fold in list_dir:
            # Getting information about the current calculation
            curr_phase = pathlib.PurePath(calc_fold).name
            ut.custom_print(
                f"Loading calculation for '{curr_phase}' as a base structure.", "debug"
            )

            # Loading current calculation info
            xml_path = pathlib.Path(calc_fold, "vasprun.xml")
            curr_run = vasp.Vasprun(xml_path, parse_potcar_file=False)

            # Gathering phase information
            for phase in CuZnInitialDatabase.CUZN_PHASES.phase_names:
                for folder in xml_path.parts:
                    if slugify(folder) == slugify(phase):
                        curr_phase = phase
                        curr_phase = CuZnInitialDatabase.CUZN_PHASES.get_phase(phase)
                        curr_mat_id = curr_phase.prototype
                        curr_name = f"base_relax_{curr_phase.name}_MP"

            # Creating the structure object
            curr_struct = mdf_struct.Structure().from_vasprun(
                vasprun=curr_run,
                base=True,
                phase=curr_phase,
                material_name=curr_name,
                bulk=True,
                perturb=False,
                cluster=False,
                surface=False,
                material_id=curr_mat_id,
            )

            # Saving the structure to the database
            self.df = curr_struct.save_to_db(self.df)

    def save_database(self, path: str = None, suffix: str = None):
        """
        Saves the database dataframe into a pkl object.

        Parameters
        ----------
        path : str, optional

                Location where the pickle object will be saved,
                by default None, which defaults to storing the file in the CWD.
        suffix : str, optional

                String that will be added at the beginning of the filename.
        """
        if suffix:
            filename = self.database_name + f"_{suffix}.pkl"
        else:
            filename = self.database_name + ".pkl"

        if not path:
            path = ""

        file_path = pathlib.Path(path, filename)

        self.df.to_pickle(path=file_path)
        ut.custom_print(f"Database saved in {file_path}", "info")

    def perturb_gauss(self, center: float = 0.04, repeat: int = 5):
        # Getting all structures which are not perturbed
        target_entries = self.df.loc[
            (~self.df.material_name.str.contains("_perturb"))
            & (self.df.material_name.str.contains("_super"))
        ]

        # Applying displacement to all perturbed structures
        for idx, entry in target_entries.iterrows():
            str_matid = entry.material_id
            str_phase = entry.phase
            curr_str = entry.structure
            extra_info = ""
            if entry.supercell:
                extra_info += f"_super-{self._get_miller_index_str(entry.supercell)}"
            if entry.replacement:
                extra_info += (
                    f"_repl-{entry.replacement_ind[0]}-{entry.replacement_ind[1]}"
                )

            for perturb_repeat_idx in range(repeat):
                # Applying displacement
                new_struct_perturb = self._apply_gauss_perturb(
                    center=center, structure=curr_str
                )

                mat_str = f"{str_matid}_{str_phase.name}{extra_info}_perturb_gauss_{perturb_repeat_idx+1}"

                # Creating a new Structure from the perturbed structure structure
                curr_struct = mdf_struct.Structure(
                    material_name=mat_str,
                    structure=new_struct_perturb,
                    material_id=str_matid,
                    phase=str_phase,
                    base=False,
                    perturb=True,
                    supercell=entry.supercell,
                    replacement=entry.replacement,
                    formula=new_struct_perturb.formula,
                    symmetry=new_struct_perturb.get_space_group_info(),
                    temperature=entry.temperature,
                    calc_performed=False,
                )

                # Converting the structure to the appropiate type
                if entry.bulk:
                    curr_struct_conv = mdf_struct.Bulk().from_mdb_structure(curr_struct)
                elif entry.surface:
                    curr_struct_conv = mdf_struct.Surface().from_mdb_structure(
                        curr_struct
                    )
                else:
                    raise NotImplementedError(
                        "This perturbation strategy is not implemented "
                        "for the current structure type."
                    )

                # Saving the bulk to the db.
                self.df = curr_struct_conv.save_to_db(self.df)

    def _apply_gauss_perturb(self, structure: Structure, center: float = 0.04):
        new_structure = structure.copy()

        new_structure.perturb(distance=0.08, min_distance=0.02)
        return new_structure

    def perturb_min_displacement(
        self, frac_max: float = 0.05, frac_min: float = 0.01, repeat=5
    ):
        # Getting all relaxed structures
        target_entries = self.df.loc[self.df.base]

        # Applying displacement to all perturbed structures
        for idx, entry in target_entries.iterrows():
            # Getting some parameters from the current perturb structure.
            str_matid = entry.material_id
            str_phase = entry.phase
            curr_str = entry.structure

            # Applying the perturbation 'repeat' times.
            for perturb_repeat_idx in range(repeat):
                # Applying displacement,
                new_struct_perturb = self._apply_min_perturbation(structure=curr_str)

                mat_str = (
                    f"{str_matid}_{str_phase.name}_perturb_min_{perturb_repeat_idx+1}"
                )

                # Creating a new Structure from the perturbed structure structure
                curr_struct = mdf_struct.Structure(
                    material_name=mat_str,
                    structure=new_struct_perturb,
                    material_id=str_matid,
                    phase=str_phase,
                    base=False,
                    perturb=True,
                    supercell=entry.supercell,
                    replacement=entry.replacement,
                    formula=entry.formula,
                    symmetry=new_struct_perturb.get_space_group_info(),
                    temperature=entry.temperature,
                    calc_performed=False,
                )

                # Converting the structure to the appropiate type
                if entry.bulk:
                    curr_struct_conv = mdf_struct.Bulk().from_mdb_structure(curr_struct)
                elif entry.surface:
                    curr_struct_conv = mdf_struct.Surface().from_mdb_structure(
                        curr_struct
                    )
                else:
                    raise NotImplementedError(
                        "This perturbation strategy is not implemented "
                        "for the current structure type."
                    )

                # Saving the bulk to the db.
                self.df = curr_struct_conv.save_to_db(self.df)

    def _apply_min_perturbation(
        self, structure: Structure, frac_max: float = 0.05, frac_min: float = 0.01
    ):
        perturb_structure = structure.copy()
        # Making a copy of the current structure lattice which can be modified
        matrix = np.copy(perturb_structure.lattice.matrix)

        # Select non-zero indices
        non_zero_mask = np.abs(matrix) > 0.01

        # Compute perturbations for all non-zero values
        fraction = (frac_max - frac_min) * np.random.ranf(size=matrix.shape) + frac_min

        # Applying perturbation as a mask
        displacements = matrix * fraction
        displacements = np.where(non_zero_mask, displacements, 0)

        # Randomly add or subtract perturbations
        signs = np.random.choice([1, -1], size=len(non_zero_mask))

        # Apply perturbations
        matrix += signs * displacements

        # Updating perturb_structure with displaced matrix
        perturb_structure.lattice = matrix

        return perturb_structure


class CuZnInitialDatabase(InitialDatabase):
    """
    Object representing a initial database for a CuZn alloy intented
    to prepare a NNP. The database is stored as a pandas dataframe.
    Contains methods related to gathering, preparing and modifying
    the initial database.

    Returns
    -------
    CuZnInitialDatabase
        Object containing the database and methods.

    Parameters
    ----------
    database_name : str
        Name for the database. Will be used for internal reference and as a filename for
        saving the dataframe.
    use_offset : bool, optional
        Use an offset for the phase ratios to allow them to overlap, by default True.
    max_num_atoms : int
        Maximum number of atoms present in any structure generated, by default 64.
    secrets : dict
        Object containing secrets related to the materials project database.

    Raises
    ------
    KeyError
        This error is raised when a wrong phase is given.
    """

    # CuZn alloy phase diagram data
    alpha_phase = Phase(
        name="alpha",
        base_elem="Zn",
        base_elem_comp_min=0,
        base_elem_comp_max=0.3895,
        prototype="mp-30",
        offset=0.03,
    )
    m1 = Phase(
        name="m1",
        base_elem="Zn",
        base_elem_comp_min=0.3895,
        base_elem_comp_max=0.45,
        prototype="mp-30",
        offset=0.03,
    )
    beta_prime = Phase(
        name="beta-prime",
        base_elem="Zn",
        base_elem_comp_min=0.455,
        base_elem_comp_max=0.507,
        prototype="mp-987",
        offset=0.05,
    )
    m2 = Phase(
        name="m2",
        base_elem="Zn",
        base_elem_comp_min=0.51,
        base_elem_comp_max=0.577,
        prototype="mp-987",
        offset=0.05,
    )
    gamma = Phase(
        name="gamma",
        base_elem="Zn",
        base_elem_comp_min=0.577,
        base_elem_comp_max=0.706,
        prototype="mp-1368",
        offset=0.03,
    )
    m3 = Phase(
        name="m3",
        base_elem="Zn",
        base_elem_comp_min=0.706,
        base_elem_comp_max=0.785,
        prototype="mp-1216020",
        offset=0.05,
    )
    delta = Phase(
        name="delta",
        base_elem="Zn",
        base_elem_comp_min=0.7302,
        base_elem_comp_max=0.765,
        prototype="mp-1215518",
        offset=0.05,
    )
    epsilon = Phase(
        name="epsilon",
        base_elem="Zn",
        base_elem_comp_min=0.785,
        base_elem_comp_max=0.883,
        prototype="mp-972042",
        offset=0.05,
    )
    m4 = Phase(
        name="m4",
        base_elem="Zn",
        base_elem_comp_min=0.883,
        base_elem_comp_max=0.9725,
        prototype="mp-79",
        offset=0.01,
    )
    eta = Phase(
        name="eta",
        base_elem="Zn",
        base_elem_comp_min=0.9725,
        base_elem_comp_max=1,
        prototype="mp-79",
        offset=0.01,
    )

    CUZN_PHASES = BinaryPhaseDiagram(
        "CuZn", alpha_phase, m1, beta_prime, m2, gamma, m3, delta, epsilon, m4, eta
    )

    # Which atoms are involved in the alloy.
    ALLOY_SET = {Element("Cu"), Element("Zn")}

    def __init__(self, database_name, use_offset=True, **kwargs):
        # Initializing the class' parent
        super().__init__(database_name, **kwargs)

        # Using the offset
        if use_offset:
            ut.custom_print(
                "Using an offset for computing the phases concentrations.", "info"
            )
            self.use_offset = use_offset

    def gather_base_structures(self, target_structures):
        # Gathering base structures using the method from the superclass
        super().gather_base_structures(target_structures)

        # Assigning assumed phase to the base structure
        # for idx in self.df["material_id"]:
        #     curr_phase = self._get_phase_from_id(idx)
        #     self.df.at[idx, "phase"] = curr_phase

    def _get_phase_from_id(self, idx: str) -> str:
        """
        Searches for the corresponding phase in the self.CUZN_PHASES
        dict to a given a material projects id.

        Parameters
        ----------
        idx : str
            Materials project id

        Returns
        -------
        str
            A phase of the CuZn phase diagram
        """

        # Creating a list of the phase diagram phase names
        phase_list = [
            phase.name for phase in self.CUZN_PHASES.phases if phase.prototype == idx
        ]

        # Returning the key if found
        if len(phase_list) > 0:
            return phase_list[0]
        else:
            return None

    def _convert_prototype_structure(
        self, structure: Structure, phase: str
    ) -> Structure:
        """
        Convert a provided prototype structure to a structure compatible
        with the CuZn alloy preparation by replacing all atoms with Cu.

        Parameters
        ----------
        structure : Structure
            Prototype structure to be modified

        Returns
        -------
        Structure
            Structure which may have all its atoms replaced by Cu if it contains
            anything other than Cu or Zn atoms.
        """
        # Checking if there are any other atoms than Cu or Zn in the structure
        if len(set(structure.symbol_set) - self.ALLOY_SET) > 0:
            # Creating a new structure using the base one as a template
            new_structure = structure.copy(sanitize=True)

            # Replacing base_elem atoms in the structures
            for ind in range(len(structure.species)):
                new_structure.replace(
                    ind,
                    # Species(self.CUZN_PHASES.get_phase(phase).base_elem),
                    Element(self.CUZN_PHASES.get_phase(phase).base_elem),
                )

            # Returning new structure with atoms replaced with Cu
            return new_structure

        else:
            # If everything is already either Cu or Zn, leave the structure as is.
            return structure

    def _gather_prototype_structure(
        self,
        prototype: str,
        phase: str,
        get_different_supercells: bool,
        read: bool,
    ):
        """
        Gather the structure for a prototype from the materials project database,
        while checking that the phase given for the material is correct

        Parameters
        ----------
        prototype : str
            Materials project id for the prototype structure.
        phase : str
            Name of the phase to be generated.
        get_different_supercells: bool
            Wether to get one or more supercells from the base structure.
        read: bool
            Wether to read the structure from the database or to use the MP API.


        Returns
        -------
        Structure
            Prototype structure
        emmet.core.summary.SummaryDoc
            Object containing information for the queried structure

        Raises
        ------
        KeyError
            Raised if the given phase is not found. All of the available phases
            are given on the self.CUZN_PHASES dictionary. More phases could be added
            there if necessary.
        """
        # Checking for correct phase input
        phase_name = slugify(phase.name)
        if not self.CUZN_PHASES.get_phase(phase_name):
            raise KeyError(
                "Wrong phase given. "
                f"Please introduce one of: {[k for k in self.CUZN_PHASES.phases]}"
            )

        # Reading structure from database
        if read:
            ut.custom_print("Using structure from the db as template...", "debug")
            query_result = self.df.loc[self.df.phase == phase]
            material_id_prefix = query_result.material_id.values[0]
            structure = query_result.structure.values[0]

        # Querying CuZn alpha prototype structure
        else:
            ut.custom_print("Querying the MP API...", "debug")
            with MPRester(self.secrets["API_KEY"]) as mpr:
                query_result = mpr.summary.search(material_ids=[prototype])[0]
                structure = query_result.structure
                material_id_prefix = query_result.material_id

        # TODO: Add a toggle so that the user can choose to use it.
        # Converting all of the atoms from the prototype cell to the base atom type if necessary
        # structure = self._convert_prototype_structure(structure=structure, phase=phase)

        # Getting conventional cell for the replaced structure
        sga = SpacegroupAnalyzer(structure)
        structure = sga.get_conventional_standard_structure()

        # Create supercells for the replaced structure
        # This supercell will result in a Cu64 cell
        # This can return either 1 or more supercells of the
        # same structure, depending on the 'get_different_supercells' flag.
        structure_list, idx_list, supercells = self._find_supercell_indices(
            structure, get_different_supercells, max_atoms=self.max_num_atoms
        )

        struct_obj_list = []
        # Saving all the generated supercells as separate bulk structures
        for structure, idxs in zip(structure_list, supercells):
            # Getting the supercell vector as a string for naming
            idxs_str = "".join(map(str, idxs))

            # Creating a new bulk from the supercell
            curr_bulk = mdf_struct.Bulk(
                material_name=f"{material_id_prefix}_{phase.name}_super-{idxs_str}",
                material_id=material_id_prefix,
                structure=structure,
                temperature=query_result.temperature.values[0],
                perturb=False,
                surface=False,
                base=False,
                cluster=False,
                calc_performed=False,
                supercell=idxs,
                phase=phase,
            )

            # Saving the bulk to the db.
            self.df = curr_bulk.save_to_db(self.df)

            struct_obj_list.append(curr_bulk)

        return struct_obj_list, query_result, idx_list

    def _create_symmetrical_prototype(
        self,
        structure: Structure,
        phase: Phase,
        query_result: emmet.core.summary.SummaryDoc,
        read: bool,
    ):
        curr_phase_atom = self.CUZN_PHASES.get_phase(phase).base_elem
        base_atom_set = list(self.ALLOY_SET - {curr_phase_atom})

        new_structure = structure.copy(sanitize=True)

        # Replacing atoms in the structures
        ind = 2
        sum_ind = 0
        sum_list = (2, 1, 2, 3)

        while ind < structure.num_sites:
            # new_structure.replace(ind - 1, Species(base_atom_set[0]))
            new_structure.replace(ind - 1, Element(base_atom_set[0]))
            ind = ind + sum_list[sum_ind]

            if sum_ind == 3:
                sum_ind = 0
            else:
                sum_ind += 1

        if read:
            material_id_prefix = query_result.material_id.values[0]
        else:
            material_id_prefix = query_result.material_id

        # Generating the symmetrized structure
        new_struct_symm = mdf_struct.Structure(
            material_name=f"{material_id_prefix}_{phase.name}_symm",
            material_id=material_id_prefix,
            structure=structure,
            temperature=query_result.temperature.values[0],
            perturb=False,
            surface=False,
            base=False,
            cluster=False,
            calc_performed=False,
            supercell=query_result.supercell.values[0],
            phase=phase,
        )

        if query_result.bulk.values[0]:
            final_struct = mdf_struct.Bulk().from_mdb_structure(
                mdb_structure=new_struct_symm,
                new_structure=structure,
            )

        # TODO: Make this work for surfaces and clusters.
        else:
            raise NotImplementedError("Current function only implemented for bulks.")

        self.df = final_struct.save_to_db(self.df)

        return structure

    def _gen_base_elem_perc(self, phase, num_struct):
        # Computing base_elem percentages using offset
        if self.use_offset:
            # Getting offset. If not found set to 0.
            offset = phase.offset

            # Randomly generating base_elem percentages for the new structures
            max_base_elem = (phase.base_elem_comp_min) + offset
            if max_base_elem > 1:
                max_base_elem = 1

            min_base_elem = (phase.base_elem_comp_max) - offset
            if min_base_elem < 0:
                min_base_elem = 0

            subst_base_elem_perc = (min_base_elem - max_base_elem) * np.random.ranf(
                size=num_struct
            ) + max_base_elem

        # Computing base element percentages without offset.
        else:
            max_base_elem = phase.base_elem_comp_min
            min_base_elem = phase.base_elem_comp_max
            subst_base_elem_perc = (min_base_elem - max_base_elem) * np.random.ranf(
                size=num_struct
            ) + max_base_elem

        return subst_base_elem_perc

    def _gen_perc_surfaces(self, phase, num_struct, current_perc):
        print("\ncurrent_perc Zn: ", current_perc)

        # Getting offset. If not found set to 0.
        offset = phase.offset

        # Randomly generating base_elem percentages for the new structures
        max_base_elem = (phase.base_elem_comp_min) + offset
        if max_base_elem > 1:
            max_base_elem = 1

        min_base_elem = (phase.base_elem_comp_max) - offset
        if min_base_elem < 0:
            min_base_elem = 0

        subst_base_elem_perc = (min_base_elem - max_base_elem) * np.random.ranf(
            size=num_struct
        ) + max_base_elem

        print("subst_base_elem_perc: ", subst_base_elem_perc)
        adjusted_perc = [(per - current_perc) for per in subst_base_elem_perc]
        print("adjusted_perc: ", adjusted_perc)

        return adjusted_perc

    def _fit_replacements_phase(
        self,
        phase,
        structure,
        subst_base_elem_perc,
    ):
        curr_comp = structure.composition
        base_elem = phase.base_elem
        (other_elem,) = CuZnInitialDatabase.ALLOY_SET - {base_elem}

        tot_base_at_struct = curr_comp[base_elem]

        structure_len = len(structure.species)
        offset_min = phase.base_elem_comp_min - phase.offset
        offset_max = phase.base_elem_comp_max + phase.offset

        n_at_replacement_upd = []
        for str_ind, curr_perc in enumerate(subst_base_elem_perc):
            inPhase = phase.perc_in_phase(curr_perc)

            single_at_perc = 1 / structure_len
            perc_range = offset_max - offset_min

            # Skip this offset if changing one atom always results
            # in going over the maximum or minimum.
            if single_at_perc >= perc_range:
                inPhase = True

            while not inPhase:
                perc = (tot_base_at_struct + abs(curr_perc)) / structure_len

                if perc >= offset_max:
                    curr_perc -= single_at_perc
                elif perc <= offset_min:
                    curr_perc += single_at_perc
                else:
                    inPhase = phase.perc_in_phase(curr_perc)

            new_n_at = int(round(curr_perc * structure_len, 0))
            n_at_replacement_upd.append(new_n_at)

        return n_at_replacement_upd

    def _apply_replacement(self, structure, phase, structure_len, n_atoms, rng):
        # Getting current structure composition information
        # The current procedure assumes that all of the atom species in the structure
        # will have been replaced beforehand with the base atom.
        # Although this results in more randomness.
        curr_comp = structure.composition
        base_elem = phase.base_elem
        (other_elem,) = CuZnInitialDatabase.ALLOY_SET - {base_elem}

        # Getting how many base atoms must be changed in order for the
        # structure to meet the current percentage requirements
        target_atoms_base = n_atoms - curr_comp[base_elem]

        # base_atom_change = int(curr_comp[base_elem] + target_atoms_base)

        # Getting how many atoms of the other element must be changed
        other_atom_change = int(curr_comp[other_elem] - target_atoms_base)

        # Choosing which species of the structure to change with the other atom.
        other_elem_choices = rng.choice(
            a=len(structure.species),
            size=abs(int(other_atom_change)),
            replace=False,
            shuffle=True,
        )

        # Creating a new pymatgen structure using the base one as a template
        new_structure = structure.copy(sanitize=True)

        # Replacing atoms in the structures
        for ind in other_elem_choices:
            new_structure.replace(ind, other_elem)

        return new_structure

    def generate_bulk_structures(
        self,
        prototype: str,
        phase: str,
        num_struct: int,
        num_repeats: int,
        get_different_supercells: bool,
        # perturb: list,
        read: bool = True,
    ):
        """
        This method allows to create several variations of a certain phase
        structure by randomly replacing atoms in the base structure.

        Parameters
        ----------
        prototype : str
            Materials project id of the prototype structure to be used as template.
        phase : str
            Name of the phase to be used.
        num_struct : int
            Number of different atomic compositions to be generated.
        num_repeats : int
            Number of random replacements done for each atomic composition.
        get_different_supercells : bool
            Whether to store just a single supercell or several of them.
            If False, a single supercell is chosen so that the resulting structure
            has a total number of atoms under a certain threshold.
            If True, the same structure is chosen, but additionally, any possible
            structure with smaller supercells is also added.
        read: bool
            Whether to read structures from the db or use the MP API to get them, by
            default True.

        Raises
        ------
        KeyError
            Raised if the given phase is not found. All of the available phases
            are given on the self.CUZN_PHASES dictionary. More phases could be added
            there if necessary.
        """
        # Instantiating RNG
        rng = np.random.default_rng()

        # Getting the prototype structure
        # First, the structure is either gathered from the MP or the initial database,
        # then all atoms are replaced with Cu. Next, the conventional cell is obtained
        # and finally a supercell is created. Depending on the setting, one or more
        # supercells can be returned.
        structure_list, query_result, idx_list = self._gather_prototype_structure(
            get_different_supercells=get_different_supercells,
            prototype=prototype,
            phase=phase,
            read=read,
        )

        for structure_obj, supr_idx in zip(structure_list, idx_list):
            curr_structure = structure_obj.structure

            # TODO: Add a toggle so that the user can choose to use it.
            # Converting all of the atoms from the prototype cell to the base atom type if necessary
            structure = self._convert_prototype_structure(
                structure=curr_structure, phase=phase
            )

            # Replacing some atoms using symmetry
            structure = self._create_symmetrical_prototype(
                structure=structure,
                phase=phase,
                query_result=query_result,
                read=read,
            )
            structure_len = len(structure.species)

            # Preparing an array of randomly generated base elem percentages
            # for the new structures
            subst_base_elem_perc = self._gen_base_elem_perc(phase, num_struct)

            # TODO: Is this true?
            # Choosing the amount of atoms to replace with the base element in the
            # struct which at this point will be completely replaced by atoms
            # of the remaining species of the alloy.
            # n_at_replacement = [
            #     int(round(structure_len * stct, 0)) for stct in subst_base_elem_perc
            # ]

            # Attempting to fix any percentages outside of the
            # current phase ratios.
            n_at_replacement_upd = self._fit_replacements_phase(
                phase, structure, subst_base_elem_perc
            )

            # Replacing the atoms and generate 'num_replacements'
            # structures for each percentage

            for str_ind, n_atoms in enumerate(n_at_replacement_upd):
                for repl in range(num_repeats):
                    # Applying the replacement
                    new_structure = self._apply_replacement(
                        structure, phase, structure_len, n_atoms, rng
                    )

                    # Getting the supercell vector
                    supercell_vec_str = "".join(map(str, structure_obj.supercell))

                    # Creating a new Bulk object for the structure with replacement
                    new_struct_symm = mdf_struct.Bulk(
                        material_name=f"{prototype}_{phase.name}_super-{supercell_vec_str}-{supr_idx}_{str_ind+1}_{repl+1}",
                        material_id=prototype,
                        structure=new_structure,
                        temperature=query_result.temperature.values[0],
                        perturb=False,
                        surface=False,
                        replacement=True,
                        replacement_ind=(str_ind + 1, repl + 1),
                        base=False,
                        cluster=False,
                        calc_performed=False,
                        supercell=query_result.supercell.values[0],
                        phase=phase,
                    )

                    self.df = new_struct_symm.save_to_db(self.df)

    def _get_miller_index_str(self, miller_source):
        """
        Generate a miller index string from several sources,
        either a Slab structure, a numpy array with the indices
        or a string.
        The intended use of this string is for labeling structures
        and helping identification.

        Parameters
        ----------
        miller_source : Slab | np.ndarray | str
            Information about the miller indices used to
            generate the string.

        Returns
        -------
        str
            Miller indices coded as a string, without including brackets.
            Negative signs are added in front of the symbols.

        """
        if isinstance(miller_source, Slab):
            curr_miller = str(miller_source.miller_index)
        elif isinstance(miller_source, np.ndarray):
            curr_miller = str(miller_source)
        elif isinstance(miller_source, str):
            curr_miller = miller_source
        else:
            # Return None if the given structure is not a surface.
            return None

        replace_chars = ["'", ",", " ", "(", ")", "[", "]"]
        for char in replace_chars:
            curr_miller = curr_miller.replace(char, "")

        return curr_miller

    def _slab_to_bottom(
        self,
        slab: Union[Slab, Structure],
        offset: int = 2,
    ) -> Structure:
        """
        Move the slab towards the bottom of the cell, leaving a
        offset wide margin at the bottom.

        Parameters
        ----------
        slab : Union[Slab, Structure]
            Target slab to move to the bottom.
        miller: tuple
            Miller index of the slab
        offset : int, optional
            Separation to be left between the bottom of the cell
            and the slab, by default 2, in Angstrom.


        Returns
        -------
        Structure
            Pymatgen structure containing slab placed on the bottom,
            with the same attributes as the original.
        """

        # Getting the position closest to the bottom
        bottom = min(slab.cart_coords[:, 2])
        bottom_arr = np.zeros(shape=slab.cart_coords.shape)

        # Applying the offset
        bottom_arr[:, 2] += bottom - offset

        # Substracting the bottom position from the slab plus an offset
        modified_coords = slab.cart_coords - bottom_arr

        new_slab = Structure(
            lattice=slab.lattice,
            species=slab.species,
            coords=modified_coords,
            coords_are_cartesian=True,
        )

        return new_slab

    def _check_correct_vacuum_size(
        self,
        slab: Union[Slab, Structure],
        vacuum_size: float,
        tolerance: float = 0.5,
    ) -> bool:
        # Getting 'c' size for the cell
        vec_c = slab.lattice.c

        # Getting position of the topmost layer
        z_axis_max = max(slab.cart_coords[:, 2])

        # Getting vacuum layer thickness by substracting
        vac_layer_thickness = vec_c - z_axis_max

        # Checking if layer is greater or equal than vacuum_size
        if abs(vac_layer_thickness - vacuum_size) <= tolerance:
            return True
        else:
            return False

    def _adjust_vacuum(self, slab: Slab, vacuum_size: float) -> Slab:
        # Getting 'c' vector for the cell
        vec_c = slab.lattice.c

        # Getting position of the topmost layer
        z_axis_max = max(slab.cart_coords[:, 2])

        #
        current_vacuum_size = vec_c - z_axis_max

        # Computing correct slab size
        corr_slab_size = z_axis_max + vacuum_size

        # Computing the difference between the correct slab
        diff = vec_c - corr_slab_size
        # print('vec_c: ', vec_c)
        # print('top layer: ', z_axis_max)
        # print('calculated distance:',vec_c-z_axis_max)
        # print('corr_slab_size: ', corr_slab_size)

        # Changing the 'c' vector
        if current_vacuum_size > vacuum_size:
            new_vec_c = vec_c - diff
        elif current_vacuum_size < vacuum_size:
            new_vec_c = vec_c + diff

        # Creating a new abc vector
        new_latt_abc = np.array(slab.lattice.abc)
        new_latt_abc[-1] = new_vec_c

        # Converting the abc vector into a 3x3 matrix
        new_latt_matrix = np.zeros([3, 3])
        diag = np.diag_indices(3)
        new_latt_matrix[diag] = new_latt_abc

        # Creating a lattice use the new matrix
        new_lattice = Lattice(matrix=new_latt_matrix)

        # Using the lattice to create a new slab
        new_slab = Structure(
            lattice=new_lattice,
            species=slab.species,
            coords=slab.cart_coords,
            coords_are_cartesian=True,
        )
        return new_slab

    def _make_clean_surf(
        self,
        bulk: Union[Structure, Slab],
        max_num_at: float,
        n_layers: int,
        miller_list: list,
        fixed: int,
    ):
        img_miller = []
        images = []

        for miller in miller_list:
            gen = cts.SlabGenerator(
                bulk,
                miller_index=(miller),
                layers=n_layers,
                layer_type="angs",
                fixed=fixed,
                standardize_bulk="True",
                vacuum=7.5,
            )

            # Getting unique terminations for the current surface
            termination = gen.get_unique_terminations()

            for ind, t in enumerate(termination):
                img_miller.append(miller)
                imgsize = gen.get_slab(iterm=ind).get_global_number_of_atoms()
                slab_rep = int(max_num_at / imgsize)

                try:
                    slab = gen.get_slab(iterm=ind, size=slab_rep)
                except Exception:
                    break

                images.append(slab)

        return images, img_miller

    def _gen_slab_pool(self, miller, bulk, max_num_at, n_layers, fixed):
        img_miller = []
        images = []

        gen = cts.SlabGenerator(
            bulk,
            miller_index=(miller),
            layers=n_layers,
            layer_type="angs",
            fixed=fixed,
            standardize_bulk="True",
            vacuum=7.5,
        )

        # Getting unique terminations for the current
        # surface
        termination = gen.get_unique_terminations()

        for ind, t in enumerate(termination):
            img_miller.append(miller)
            imgsize = gen.get_slab(iterm=ind).get_global_number_of_atoms()
            slab_rep = int(max_num_at / imgsize)

            try:
                slab = gen.get_slab(iterm=ind, size=slab_rep)
            except Exception:
                break

            images.append(slab)

        return images, img_miller

    def _make_clean_surf_mp(
        self,
        bulk: Union[Structure, Slab],
        max_num_at: float,
        n_layers: int,
        miller_list: list,
        fixed: int,
    ):
        img_miller = []
        images = []

        # gen = make_generator_slab
        # Original parameters:
        # fixed = 3

        with Pool() as p:
            slabs_worker = p.starmap(
                self._gen_slab_pool,
                zip(
                    miller_list,
                    it.repeat(bulk),
                    it.repeat(max_num_at),
                    it.repeat(n_layers),
                    it.repeat(fixed),
                ),
            )
            for slb, mill in slabs_worker:
                if isinstance(slb, list):
                    for i in slb:
                        images.append(i)
                if isinstance(mill, list):
                    for m in mill:
                        img_miller.append(m)

        return images, img_miller

    def __gen_curr_surface(
        self,
        phase,
        curr_bulk_ase,
        n_layers,
        n_at,
        max_miller_index,
        fixed_layers,
        get_supercells,
    ):
        # Filtering specific catkit warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        n_layers = int(n_layers)
        n_at = int(n_at)

        slabs = []
        miller = cts.get_unique_indices(
            bulk=curr_bulk_ase,
            max_index=max_miller_index,
        )

        slabs, miller_idx_slabs = self._make_clean_surf(
            bulk=curr_bulk_ase,
            n_layers=n_layers,
            max_num_at=n_at,
            miller_list=miller,
            fixed=fixed_layers,
        )

        # Will contain tuples as such: (Structure, miller_index_string)
        slabs_bottom = []
        for ind, (slab, mill) in enumerate(zip(slabs, miller_idx_slabs)):
            curr_surf_pymg = AseAtomsAdaptor().get_structure(slab)
            slab = self._slab_to_bottom(curr_surf_pymg)
            mill_str = self._get_miller_index_str(mill)

            # INFO: The _adjust_vacuum function does not work correctly.
            # As of now, the vacuum size is being defined during slab creation,
            # I suspect it is related to the way pymatgen handles lattices.
            #
            # if not self._check_correct_vacuum_size(slab, min_vacuum_size):
            #     slab = self._adjust_vacuum(slab, min_vacuum_size)

            slabs_bottom.append((slab, mill_str))

        prototype = phase.prototype

        # Getting only the slabs and their miller index whose total size
        # is smaller than the maximum given for the InitialDatabase.
        slabs_size = [
            (slab, mill)
            for slab, mill in slabs_bottom
            if len(slab.sites) < self.max_num_atoms
        ]

        # Storing the remaining slabs.
        for idx, (slab, mill) in enumerate(slabs_size):
            # Getting the current slab's miller index
            # curr_miller = self._get_miller_index_str(slab)

            print('mill: ', mill)
            mill_str = self._get_miller_index_str(mill)
            print('mill_str: ', mill_str)
            quit()
            # Preparing the structure name
            surf_name = (
                f"{prototype}_{phase.name}_pure_surface"
                f"_{n_layers}-layers_{n_at}-max-at_{mill_str}-{idx+1}"
            )

            # Creating a new surface from the supercell
            curr_strct = mdf_struct.Surface(
                material_name=surf_name,
                material_id=prototype,
                structure=slab,
                temperature=np.nan,
                perturb=False,
                base=False,
                calc_performed=False,
                phase=phase,
            )

            # Saving the bulk to the db.
            self.df = curr_strct.save_to_db(self.df)

        return len(slabs_size)

        # Getting supercells
        if get_supercells:
            for idx, (slab, mill) in enumerate(slabs_size):
                super_list, idx_list, supercells = self._find_supercell_indices(
                    structure=curr_surf_pymg,
                    max_atoms=self.max_num_atoms,
                    get_different_supercells=True,
                    initial_supercell_size=3,
                    verbose=False,
                )

                # Storing the supercells.
                for supercell, idx, sup_vec in zip(super_list, idx_list, supercells):
                    if len(supercell.sites) <= self.max_num_atoms:
                        # Dragging the slab to the bottom
                        supercell_bottom = self._slab_to_bottom(curr_surf_pymg)

                        # Preparing the structure name
                        surf_name = (
                            f"{prototype}_{phase.name}_pure_surface-"
                            f"{n_layers}-layers_{mill}-super-{self._get_miller_index_str(sup_vec)}"
                        )

                        # Creating a new surface from the supercell
                        curr_strct = mdf_struct.Surface(
                            material_name=surf_name,
                            material_id=prototype,
                            structure=supercell_bottom,
                            temperature=np.nan,
                            perturb=False,
                            base=False,
                            calc_performed=False,
                            phase=phase,
                            supercell=sup_vec,
                        )

                        # Saving the bulk to the db.
                        self.df = curr_strct.save_to_db(self.df)

        return surf_name

    def _get_structs_current_phase(self, phase):
        # Getting all of the base structures
        base_structs = self.df.loc[self.df.base]

        # Getting the structures corresponding to the current phase
        phase_mask = base_structs.phase == phase
        base_structs = base_structs.where(phase_mask, other=pd.NA)
        base_structs.dropna(how="all", inplace=True)

        return base_structs

    def generate_surfaces_pure(
        self,
        phase: Phase,
        num_repeats: int,
        max_miller_index: int = 2,
        min_slab_size: float = 3,
        max_slab_size: float = 6,
        min_vacuum_size: float = 10,
        get_supercells=False,
        fixed_layers: int = 0,
        overwrite_max_num_atoms: int = None,
    ):
        # Getting the current phase from the phase name.
        if isinstance(phase, str):
            phase = CuZnInitialDatabase.CUZN_PHASES.get_phase(phase)

        base_structs = self._get_structs_current_phase(phase)

        # Checking if there are any base structures for the current
        # phase.
        if len(base_structs) == 0:
            err_msg = (
                f"No base structure could be found for phase {phase}."
                "\nThe database must contain base structures before "
                "running this function."
            )

            raise mdbex.BaseStructureNotFound(err_msg)

        # Generating an initial random slab size close to the minimum slab thickness
        # given.
        # rng = np.random.default_rng()
        # init_value = rng.uniform(high=min_slab_size + 1, low=min_slab_size)

        # Preparing equispaced points between initial random value and the
        # maximum thickness value.
        slab_sizes = np.linspace(min_slab_size, max_slab_size, num_repeats)

        for idx, row in base_structs.iterrows():
            # Getting the current base structure
            curr_bulk = row.structure

            # Getting the number of atoms in the conventional cell
            # of the bulk
            curr_surf_nat = len(curr_bulk.species)

            # Getting a range of maximum number of atoms using the bulk
            # atom number and the max atom number specified.
            if overwrite_max_num_atoms:
                max_atom_num_list = np.linspace(
                    curr_surf_nat, overwrite_max_num_atoms, 3
                )
            else:
                max_atom_num_list = np.linspace(curr_surf_nat, self.max_num_atoms, 3)

            # Getting an ASE Atoms object
            curr_bulk_ase = AseAtomsAdaptor().get_atoms(curr_bulk)

            # Preparing the progress bar
            text_column = riprg.TextColumn(
                "        {task.description}", table_column=riprg.Column(ratio=3)
            )
            rialg.Align(text_column.render, align="right")
            bar_column = riprg.BarColumn(table_column=riprg.Column())
            time_col = riprg.TimeElapsedColumn(table_column=riprg.Column())
            spin_col = riprg.SpinnerColumn(table_column=riprg.Column())
            remaining_col = riprg.MofNCompleteColumn(table_column=riprg.Column())
            t_remaining_col = riprg.TimeRemainingColumn(table_column=riprg.Column())
            empty_col = riprg.TextColumn("", table_column=riprg.Column())

            overall_progress = riprg.Progress(
                riprg.TextColumn(
                    " [···]  {task.description}",
                    table_column=riprg.Column(ratio=3),
                ),
                bar_column,
                time_col,
                t_remaining_col,
                remaining_col,
                # expand=True,
            )
            total_slabs_gen = list(it.product(slab_sizes, max_atom_num_list[1:]))
            total_slabs = len(total_slabs_gen)
            main_task_descr = f"Generating {phase.name} slabs:"
            overall_task = overall_progress.add_task(
                main_task_descr, total=int(total_slabs)
            )

            job_progress = riprg.Progress(
                text_column,
                bar_column,
                time_col,
                spin_col,
                empty_col,
                # expand=True,
            )

            group = ricns.Group(overall_progress, job_progress)
            live = riliv.Live(group, refresh_per_second=4)

            total_slabs_generated = 0
            with live:
                while not overall_progress.finished:
                    for n_layers, n_at in total_slabs_gen:
                        sub_task = job_progress.add_task(
                            description=f"{int(n_layers)} layers, {int(n_at)} atoms:",
                            total=None,
                        )

                        slab_amount = self.__gen_curr_surface(
                            phase=phase,
                            curr_bulk_ase=curr_bulk_ase,
                            n_layers=n_layers,
                            n_at=n_at,
                            max_miller_index=max_miller_index,
                            fixed_layers=fixed_layers,
                            get_supercells=get_supercells,
                        )

                        total_slabs_generated += slab_amount
                        job_progress.update(sub_task, total=1)
                        job_progress.advance(sub_task, advance=1)

                        overall_progress.advance(overall_task, advance=1)

            ut.custom_print(f"Generated {total_slabs_generated} surfaces.", "done")

    def _get_main_elem_perc(self, phase: Phase, structure):
        """
        This function provides the percentage of the main element
        for a given structure

        Parameters
        ----------
        phase : Phase
            Object describing the current phase
        structure : _type_
            Structure for which the percentage is to be calculated

        Returns
        -------
        float
            Percentage of main element in the structure
        """
        main_species = phase.base_elem
        main_cnt = 0

        species_list = structure.species
        total_atoms = len(species_list)

        for element in species_list:
            if element.symbol == main_species:
                main_cnt += 1
                print("main_cnt: ", main_cnt)

        perc = main_cnt / total_atoms
        return perc

    def generate_surfaces_replacements(
        self,
        phase: Phase,
        num_struct: int,
        num_repeats: int,
    ):
        if isinstance(phase, str):
            phase = CuZnInitialDatabase.CUZN_PHASES.get_phase(phase)

        # Instantiating RNG
        rng = np.random.default_rng()

        # Getting the base structures
        slabs_df = self.df.loc[self.df.surface]
        slabs_df = slabs_df.loc[self.df.base]
        slabs_df = slabs_df.loc[self.df.phase == phase.name]
        print("slabs_df: ", slabs_df)

        for idx, row in slabs_df.iterrows():
            print("idx: ", idx)
            curr_surface = row.structure

            structure_len = len(curr_surface.species)

            # Getting current percentage of main element
            strct_perc = self._get_main_elem_perc(
                phase=phase,
                structure=curr_surface,
            )

            # Randomly generating base elem percentages for the new structures
            subst_base_elem_perc = self._gen_perc_surfaces(
                phase=phase,
                num_struct=num_struct,
                current_perc=strct_perc,
            )

            print("subst_base_elem_perc: ", subst_base_elem_perc)

            # Choosing the amount of atoms to replace with the base element in the
            # struct, which at this point will be completely replaced by atoms
            # of the remaining species of the alloy.
            n_at_replacement = []
            for stct in subst_base_elem_perc:
                curr_repl = int(round(structure_len * abs(stct), 0))
                if stct < 0:
                    curr_repl *= -1
                n_at_replacement.append(curr_repl)
            print("n_at_replacement: ", n_at_replacement)

            # Attempting to fix any percentages outside of the
            # current phase ratios.
            n_at_replacement_upd = self._fit_replacements_phase(
                phase, curr_surface, subst_base_elem_perc
            )

            print("n_at_replacement_upd: ", n_at_replacement_upd)

            # Adapting the replacement percentages generated to the
            # percentage of the current structure
            # n_at_replacement_final = self._adjust_replacements()

            for str_ind, n_atoms in enumerate(n_at_replacement_upd):
                print("\nreplacement: ", str_ind)
                for repl in range(num_repeats):
                    print("\nreplicate: ", repl)
                    # Replacing atoms according to the current phase from the structure.
                    new_structure = self._apply_replacement(
                        curr_surface, phase, structure_len, n_atoms, rng
                    )

                    prototype = phase.prototype
                    extra = {"surface": True}

                    # Matching the miller index from the name as it is not stored.
                    match = re.search(r"\((.*?)\)", row.material_id)
                    if match:
                        curr_miller = match.group(1)
                    else:
                        curr_miller = "???"

                    # Preparing the structure name
                    mat_id_name = f"{prototype}_{phase.name}"
                    mat_id_surf_data = f"_replacement-({curr_miller})-_{str_ind}-{repl}"
                    material_id = mat_id_name + mat_id_surf_data

                    print("new", new_structure.formula)
                    final_perc = self._get_main_elem_perc(phase, new_structure)
                    print(phase)
                    print("final_perc: ", final_perc)

                    # Saving the structure into the database.
                    self._save_row(
                        material_id=material_id,
                        phase=phase,
                        structure=new_structure,
                        extra=extra,
                    )

    def _save_row(
        self,
        material_id,
        phase,
        structure,
        extra,
        base=False,
    ):
        new_row = pd.Series(
            {
                # f"{prototype}_{phase.name}_super-"{supr_idx}_{str_ind+1}_{repl+1}",
                "material_id": material_id,
                "structure": structure,
                "temperature": None,
                "perturb": True,
                "phase": phase.name,
                "base": base,
                "formula": structure.formula,
                **extra,
            }
        )

        new_row_df = new_row.to_frame().T
        new_row_df = new_row_df.astype(
            {"perturb": "boolean", "base": "boolean", "surface": "boolean"}
        )

        self.df = self.df.astype(
            {"perturb": "boolean", "base": "boolean", "surface": "boolean"}
        )
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)

        # TODO: Converting NaN values given by some functions to None
        # self.df.fillna(None, method=None, inplace=True)

    # def gather_aiida_group_structures(self, group_name: str):
    #     # Loading aiida profile
    #     load_profile()

    #     gathered_nodes = []

    #     # Getting group
    #     group =form.load_group(label=group_name)

    #     # Storing every node contained in the group into a list
    #     for node in group.nodes:
    #         if isinstance(node,form.CalcJobNode):
    #             gathered_nodes.append(node)
    #         else:
    #             for descendant in node.called_descendants:
    #                 if isinstance(descendant,form.CalcJobNode):
    #                     gathered_nodes.append(descendant)

    #     return gathered_nodes

    # def _get_pot_energy_outcar_aiida_node(self, vasprun:vasp.Vasprun) -> float:
    #     # retrieved = node.outputs.retrieved
    #     # vasprun = retrieved.get_object_content("vasp_run.xml")

    #     # with open("/tmp/vasprun.tmp", "w") as f:
    #     #     f.write(vasprun)

    #     # vasprun = vasp.Vasprun("/tmp/vasprun.tmp")

    #     # Getting energy in eV
    #     # This energy is given in VASP as: 'free energy TOTEN'
    #     # n2p2 uses this energy
    #     energy = float(vasprun.ionic_steps[-1]['e_fr_energy'])

    #     # Converting to Ha
    #     energy *= self.eV2Eh

    #     return energy

    def _gather_n2p2_reqdata_from_node(self, node):
        # Getting calculation name
        name = node.label + "_aiida-uuid_" + node.uuid
        # WARNING:
        # Getting potential energy
        # misc = node.outputs.misc.get_dict()
        # The energy can be obtained from aiida-vasp, however, the energy provided by the
        # parser is not the one used by n2p2.
        # Getting potential energy from the aiida-vasp parser, in eV.
        # Convert the aiida-vasp energy to Ha.
        # pot_energy = misc.get("total_energies", {}).get("energy_extrapolated") * self.eV2Eh

        # Writing the vasprun.xml file to a buffer.
        retrieved = node.outputs.retrieved
        vasprun_f = retrieved.get_object_content("vasprun.xml", "rb")
        buffer = BytesIO(vasprun_f)

        # Reading the file from the buffer and closing it
        vasprun = aseio.read(buffer, format="vasp-xml", index="-1")
        buffer.close()

        # Getting properties from the vasprun
        pot_energy = vasprun.get_potential_energy(force_consistent=True) * self.eV2Eh

        # print(dir(vasprun))
        # print(vasprun.calculator)
        # quit()
        # try:
        #     vasprun = vasp.Vasprun("/tmp/parser_vasprun.tmp", parse_potcar_file=False)
        # except Exception as e:
        #     # TODO: If parsing the xml fails, and the calculation has converged,
        #     # then read the energies and forces from the OUTCAR.
        #     ut.custom_print(name, 'warn')
        #     print(e)
        # vasprun = aseio.read("/tmp/vasprun.tmp", format="vasp-xml")

        # Getting atomic positions
        # Pymatgen treats coordinates in Ang.
        # contcar_str = retrieved.get_object_content("CONTCAR")

        # If the aiida-vasp parser is disabled, get the energy from the outcar itself.
        # The energy returned by the function is already in Ha.
        # pot_energy = self._get_pot_energy_outcar_aiida_node(vasprun=vasprun)
        # pot_energy = float(vasprun.ionic_steps[-1]["e_fr_energy"]) * self.eV2Eh

        # Getting forces
        # Reading forces from vasprun.xml, in eV/Ang and converting them to Ha/Bohr
        forces = vasprun.get_forces() * self.eV2Eh * self.Bohr2Ang

        # forces = (
        #     np.array(vasprun.ionic_steps[-1]["forces"]) * self.eV2Eh * self.Bohr2Ang
        # )

        # contcar = vasp.Poscar.from_string(contcar_str)

        lattice = vasprun.get_cell() * self.Ang2Bohr
        structure = vasprun.get_positions() * self.Ang2Bohr
        symbols = vasprun.get_chemical_symbols()
        # print('structure: ', structure)

        # print('lattice: ', lattice)
        # print('atoms: ', atoms)
        # quit()

        # Setting charge to 0
        # charge = contcar.structure.charge
        charge = 0

        data_dict = {
            "name": name,
            "lattice": lattice,
            "positions": structure,
            "symbols": symbols,
            "pot_energy": pot_energy,
            "charge": charge,
            "forces": forces,
        }

        return data_dict

    def _add_entry_to_n2p2_input(self, buffer: TextIOWrapper, data_dict: dict):
        # Writing begin keyword and structure name
        buffer.write("begin\n")
        buffer.write(f'comment {data_dict.get("material_name","no name found")}\n')

        # Getting lattice parameters and converting them to Bohr
        lat_x = data_dict["lattice"][0]
        lat_y = data_dict["lattice"][1]
        lat_z = data_dict["lattice"][2]

        # Writing lattice parameters
        buffer.write(f"lattice {lat_x[0]:.6f} {lat_x[1]:.6f} {lat_x[2]:.6f}\n")
        buffer.write(f"lattice {lat_y[0]:.6f} {lat_y[1]:.6f} {lat_y[2]:.6f}\n")
        buffer.write(f"lattice {lat_z[0]:.6f} {lat_z[1]:.6f} {lat_z[2]:.6f}\n")

        # Writing information for every atom. Every atom line must contain:
        # atom <x1> <y1> <z1> <e1> <c1> <n1> <fx1> <fy1> <fz1>
        for idx, (at, frc) in enumerate(
            zip(data_dict["positions"], data_dict["forces"])
        ):
            # Getting element from the current atom
            # ele = list(at.species.get_el_amt_dict().keys())[0]
            # Preparing and writing the line
            buffer.write(
                (
                    f"atom {at[0]:.6f} {at[1]:.6f}"
                    f" {at[2]:.6f}"
                    f" {data_dict['symbols'][idx]} {0:.6f} {0:.6f}"
                    f" {frc[0]:.6f} {frc[1]:.6f} {frc[2]:.6f}\n"
                )
            )

        # writing potential energy and charge
        buffer.write(f'energy {data_dict["pot_energy"]:.8f}\n')
        buffer.write(f'charge {data_dict["charge"]:.6f}\n')

        # writing end keyword
        buffer.write("end\n")

    def generate_n2p2_input_aiida(
        self, aiida_group_list: list, filter_dict: dict, path: str = None
    ):
        from aiida import load_profile
        from aiida_vasp.calcs.vasp import VaspCalculation

        # Loading aiida profile
        load_profile()

        # Handling path
        if path and isinstance(path, str):
            path = pathlib.Path(path)
        else:
            path = pathlib.Path()

        # Adding input.data filename to path
        path = path / "input.data"

        # Gathering nodes from the given group
        ut.custom_print("Getting nodes...")

        # Preparing a query in the aiida db
        qb = orm.QueryBuilder()

        # result_nodes = []

        for group in aiida_group_list:
            qb.append(orm.Group, filters={"label": group}, tag="group")
            qb.append(VaspCalculation, with_group="group", filters=filter_dict),

        result_nodes = qb.all(flat=True)

        ut.custom_print(f"{len(result_nodes)} nodes found.", "info")

        # Writing the file
        with open(path, "w") as curr_f:
            # Checking every node
            for node in riprg.track(
                result_nodes, description=" [ ⧖ ]  Writing info..."
            ):
                # Gathering the information from each node
                data_dict = self._gather_n2p2_reqdata_from_node(node=node)

                # Writing the information to the buffer
                self._add_entry_to_n2p2_input(buffer=curr_f, data_dict=data_dict)
        ut.custom_print(f"All calculations saved in '{path}'.", "done")

    # def task(self, node, name, lock, buffer):
    #     # data_dict = self._gather_n2p2_reqdata_from_node(node=node)

    #     with lock:
    #         # Getting calculation name
    #         # name = node.label + "_aiida-uuid_" + node.uuid

    #         # Writing the vasprun.xml file to a temporary file.
    #         retrieved = node.outputs.retrieved
    #         vasprun_str = retrieved.get_object_content("vasprun.xml")
    #         with open("/tmp/parser_vasprun.tmp", "w") as f:
    #             f.write(vasprun_str)

    #         # Reading the written vasprun
    #         vasprun = vasp.Vasprun("/tmp/parser_vasprun.tmp", parse_potcar_file=False)
    #         # vasprun = aseio.read("/tmp/vasprun.tmp", format="vasp-xml")

    #     # If the aiida-vasp parser is disabled, get the energy from the outcar itself.
    #     # The energy returned by the function is already in Ha.
    #     # pot_energy = self._get_pot_energy_outcar_aiida_node(vasprun=vasprun)
    #     pot_energy = float(vasprun.ionic_steps[-1]["e_fr_energy"]) * self.eV2Eh

    #     # Getting forces
    #     # Reading forces from vasprun.xml, in eV/Ang and converting them to Ha/Bohr
    #     forces = (
    #         np.array(vasprun.ionic_steps[-1]["forces"]) * self.eV2Eh * self.Bohr2Ang
    #     )

    #     structure = vasprun.ionic_steps[-1]["structure"]
    #     lattice = structure.lattice
    #     atoms = structure.sites

    #     # Setting charge to 0
    #     charge = 0

    #     data_dict = {
    #         "material_name": name,
    #         "lattice": lattice,
    #         "atoms": atoms,
    #         "pot_energy": pot_energy,
    #         "charge": charge,
    #         "forces": forces,
    #     }

    #     # Writing begin keyword and structure name
    #     with lock:
    #         buffer.write("begin\n")
    #         buffer.write(f'comment {data_dict.get("material_name","no name found")}\n')

    #         # Getting lattice parameters and converting them to Bohr
    #         lat_x = data_dict["lattice"].matrix[0] * self.Ang2Bohr
    #         lat_y = data_dict["lattice"].matrix[1] * self.Ang2Bohr
    #         lat_z = data_dict["lattice"].matrix[2] * self.Ang2Bohr

    #         # Writing lattice parameters
    #         buffer.write(f"lattice {lat_x[0]:.6f} {lat_x[1]:.6f} {lat_x[2]:.6f}\n")
    #         buffer.write(f"lattice {lat_y[0]:.6f} {lat_y[1]:.6f} {lat_y[2]:.6f}\n")
    #         buffer.write(f"lattice {lat_z[0]:.6f} {lat_z[1]:.6f} {lat_z[2]:.6f}\n")

    #         # Writing information for every atom. Every atom line must contain:
    #         # atom <x1> <y1> <z1> <e1> <c1> <n1> <fx1> <fy1> <fz1>
    #         for at, frc in zip(data_dict["atoms"], data_dict["forces"]):
    #             # Getting element from the current atom
    #             ele = list(at.species.get_el_amt_dict().keys())[0]
    #             # Preparing and writing the line
    #             buffer.write(
    #                 (
    #                     f"atom {at.x*self.Ang2Bohr:.6f} {at.y*self.Ang2Bohr:.6f}"
    #                     f" {at.z*self.Ang2Bohr:.6f}"
    #                     f" {ele} {0:.6f} {0:.6f} {frc[0]:.6f} {frc[1]:.6f} {frc[2]:.6f}\n"
    #                 )
    #             )

    #         # writing potential energy and charge
    #         buffer.write(f'energy {data_dict["pot_energy"]:.6f}\n')
    #         buffer.write(f'charge {data_dict["charge"]:.6f}\n')

    #         # writing end keyword
    #         buffer.write("end\n")


def gather_secrets():
    """
    Gather Materials project API key from a secret.json file.

    Notes
    -----
        The json file should have the following structure:

        {
            "API_KEY": "XXXXXX"
        }


    Returns
    -------
    dict
        object containing the api key
    """
    initial_db_path = pathlib.Path(__file__).parent

    if pathlib.Path("secrets.json").exists():
        with open("secrets.json", "r") as f:
            secrets = js.load(f)

    elif pathlib.Path(initial_db_path, "secrets.json").exists():
        path = pathlib.Path(initial_db_path, "secrets.json")
        with open(path, "r") as f:
            secrets = js.load(f)

    else:
        raise FileNotFoundError(
            f"'secrets.json' not found!\nPlease, add a 'secrets.json' file in the"
            f" following directory: '{initial_db_path}'. "
        )
        secrets = None

    return secrets


def check_incorrect_ratios(df, curr_phase_diag):
    for id, row in df.iterrows():
        if not row.base and not row.material_id.endswith("_symm"):
            strct = row.structure.get_sorted_structure()
            name = row.material_id
            phase = curr_phase_diag.get_phase(row.phase)

            tot_atoms = len(strct.species)
            tot_cu = strct.species.count(Species("Cu")) + strct.species.count(
                Element("Cu")
            )
            tot_zn = strct.species.count(Species("Zn")) + strct.species.count(
                Element("Zn")
            )

            # Checking the total atom number
            assert (
                tot_cu + tot_zn == tot_atoms
            ), f"""Total count does not match.
            tot_cu: {tot_cu}, tot_zn: {tot_zn}, total: {tot_atoms}.
            Species: {set(strct.species)}"""

            perc = tot_zn / tot_atoms

            offset_min = phase.base_elem_comp_min - phase.offset
            offset_max = phase.base_elem_comp_max + phase.offset

            # Checking if the current structure is between the phase ratio
            # percentages.
            if not (offset_min <= perc <= offset_max):
                ut.custom_print(
                    f"{name}: {perc:.2f} Zn out of {offset_min:.2f} - {offset_max:.2f}",
                    "error",
                )
