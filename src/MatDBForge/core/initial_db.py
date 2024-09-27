"""
Generates a pandas dataframe containing a set of
base (unperturbed) structures and a certain number of structures
with an applied perturbation with respect to the temperature.
"""

import itertools as it
import lzma
import os
import pathlib
import pathlib as pl
import pickle
import re
import warnings
from io import BytesIO, TextIOWrapper

import ase.io as aseio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymatgen.io.vasp as vasp
import rich.progress as riprg
from aiida import load_profile, orm
from aiida_vasp.calcs.vasp import VaspCalculation
from ase import Atoms as ase_atoms
from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel
from mp_api.client import MPRester
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from slugify import slugify

import MatDBForge as mdb
import MatDBForge.core.clusters as mdb_clust
import MatDBForge.core.exceptions as mdb_exc
import MatDBForge.core.phase_diagram as mdb_pd
import MatDBForge.core.structure as mdb_struct
import MatDBForge.core.surfaces as mdb_surf
from MatDBForge.core import initial_db as indb
from MatDBForge.core import utils as ut

# Filtering certain warnings
warnings.filterwarnings("ignore", category=vasp.outputs.UnconvergedVASPWarning)


class InitialDatabase:
    """
    Object that creates an initial database where structures will be
    stored. Materials are downloaded using the materials project API.
    For it to work, a 'secrets.json' file should be located in the
    same directory. The database is stored as a pandas dataframe.
    Contains methods related to gathering, preparing, visualizing
    and modifying the initial database.

    Attributes
    ----------
    df : pd.Dataframe
        Dataframe containing the structures for the initial database.
    secrets : dict
        Object containing secrets related to the materials project database.
    database_name : str
        Orientative name for the database. Will be used for saving it into a file.
    database_path : str | pl.Path
        Path where the database will be saved.
    max_num_atoms: int
        Maximum number of atoms present in any structure generated.
    max_num_atoms : int
        Maximum number of atoms present in any structure generated, by default 64.
    secrets : dict
        Object containing secrets related to the materials project database.
    use_offset : bool, optional
        Use an offset for the phase ratios to allow them to overlap, by default True.

    Returns
    -------
    InitialDatabase
        Object containing the database and methods.

    Raises
    ------
    KeyError
        This error is raised when a wrong phase is given.

    Notes
    -----
        The json file containing the secrets should have the following structure:

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

    def __init__(
        self,
        database_name: str,
        max_num_atoms: int = 64,
        load_db: bool = True,
        phase_diagram: mdb_pd.BinaryPhaseDiagram = None,
        use_offset: bool = True,
        database_path: str | pl.Path = None,
    ) -> None:
        self.db_version = mdb.__version__

        # Name of the database
        self.database_name = database_name

        # Path for the database
        self.database_path = pl.Path(database_path)

        # Setting the maximum number of atoms of any generated structure.
        self.max_num_atoms = max_num_atoms

        # Setting the phase diagram
        self.phase_diagram = phase_diagram

        # Using the offset
        if use_offset:
            ut.custom_print(
                "Using an offset for computing the phases concentrations.", "info"
            )
            self.use_offset = use_offset

        # Create the database if it does not exists
        # Load it if otherwise.
        check_flag = self._check_database()

        if not check_flag or not load_db:
            self.df = self._create_database()
        else:
            self.df = self._load_database()

    def __repr__(self):
        # Getting the class name
        class_name = self.__class__.__name__

        # Getting the amount of entries in the database
        count = len(self.df.count(axis=1))

        repr_string = (
            f"{class_name} named '{self.database_name}' containing {count} structures."
        )

        return repr_string

    def _adapt_old_db(self, database_old):
        columns_to_add = {
            "bulk": bool,
            "surface": bool,
            "cluster": bool,
            "calc_performed": bool,
            "replacement": bool,
        }

        for col in columns_to_add:
            database_old[col] = None

        return database_old

    def _load_database(self) -> pd.DataFrame:
        """
        Load a database from a pickle file on the cwd or a specific path.

        Returns
        -------
        pd.DataFrame
            Dataframe containing structure data for the initial database.
        """
        db_path = pl.Path(self.database_path) / pl.Path(self.database_name)
        ut.custom_print(f"Loading database: '{self.database_name}'", "debug")
        self.database_name = db_path.name.replace(db_path.suffix, "")

        # If no suffixes are present, add the default one.
        if len(db_path.suffixes) == 0:
            suffix = ".xz"
            db_path = db_path.with_suffix(suffix)

        # Compatibility with the old version of the database
        if ".pkl" in db_path.suffixes:
            suffix = ".pkl"
            ut.custom_print(
                "Using outdated version of database. Adding missing columns.",
                "debug",
            )
            database = pd.read_pickle(db_path)
            database = self._adapt_old_db(database)

            return database

        # Loading the database
        elif ".xz" in db_path.suffixes or suffix == ".xz":
            suffix = ".xz"
            with lzma.open(db_path, "rb") as f:
                database = pickle.load(f)

                # Setting parameters from InitialDatabase
                self.database_name = database.database_name
                self.max_num_atoms = database.max_num_atoms
                self.db_version = database.db_version

                ut.custom_print(f"Using database version {self.db_version}.", "debug")

            return database.df

        ut.custom_print(f"Loaded '{self.database_name}{suffix}'", "info")
        ut.custom_print(f"Path: {db_path}", "debug")

    def _check_database(self) -> bool:
        """
        Check if a database with the name 'self.database_name'
        exists in the current working directory or is a path to a existing
        database.

        Returns
        -------
        bool
            True if the database exists, False if not does not.
        """
        # Checking if dataframe already exists on the cwd.
        file_exists = False
        file_check = [
            f for f in os.listdir(path=self.database_path) if self.database_name in f
        ]
        if len(file_check) > 0:
            file_exists = True

        # If the database name is a path
        name_as_path = pathlib.Path(self.database_name)

        if name_as_path.exists():
            file_exists = True

        ut.custom_print(f"Database found: {file_exists}.", "debug")

        return file_exists

    def _create_database(self) -> pd.DataFrame:
        """
        Create an empty  dataframe in order to be used in the class.

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
                "replacement",
            ]
        )

        df.attrs["db_version"] = self.db_version
        ut.custom_print(f"Created database '{self.database_name}'.", "done")

        return df

    def _find_supercell_indices(
        self,
        structure,
        get_different_supercells,
        min_atoms,
        max_atoms,
        initial_supercell_size=5,
        verbose=True,
    ):
        # Initial supercell size
        idx = initial_supercell_size

        # Copying structure
        try:
            new_structure = structure.copy(sanitize=True)
        except TypeError:
            new_structure = structure.copy()

        # Setting different supercell geometry for slabs and bulks.
        if isinstance(structure, Slab):
            supercell_vec = [idx, idx, 1]
        else:
            supercell_vec = [idx, idx, idx]

        new_structure.make_supercell(supercell_vec, to_unit_cell=False)

        # Number of atoms of the supercell
        struct_size = len(new_structure.species)
        while (
            struct_size > max_atoms or struct_size < min_atoms
        ) and supercell_vec != [1, 1, 1]:
            try:
                new_structure = structure.copy(sanitize=True)
            except TypeError:
                new_structure = structure.copy()

            if isinstance(structure, Slab):
                supercell_vec = [idx, idx, 1]
            else:
                supercell_vec = [idx, idx, idx]

            new_structure.make_supercell(supercell_vec, to_unit_cell=False)
            struct_size = len(new_structure.species)
            idx -= 1

        structure_list = []
        idx_list = []
        supercell_vec_list = []
        structure_list.append(new_structure)
        idx_list.append(idx)
        supercell_vec_list.append(supercell_vec)

        if verbose:
            ut.custom_print(
                f"Supercell generated {supercell_vec}"
                f" - total atoms: {len(new_structure.species)}",
                "debug",
            )

        if get_different_supercells:
            # Generating all possible combinations of supercells up to a given size
            possible_supercells = it.combinations_with_replacement(
                range(2, initial_supercell_size + 1), r=3
            )

            for idx_smaller in possible_supercells:
                try:
                    new_structure = structure.copy(sanitize=True)
                except TypeError:
                    new_structure = structure.copy()

                # Slabs must not be repeated on z axis
                if isinstance(structure, Slab):
                    supercell_vec = [idx_smaller[0], idx_smaller[1], 1]

                    # Removing slabs if already on the list
                    if supercell_vec in supercell_vec_list:
                        continue

                # Bulks and clusters can be repeated on all axis
                else:
                    supercell_vec = idx_smaller

                # Creating and adding the supercell if it is within
                # the desired size range
                if supercell_vec != [1, 1, 1]:
                    new_structure.make_supercell(supercell_vec, to_unit_cell=False)
                    struct_size = len(new_structure.species)
                    if struct_size < max_atoms and struct_size > min_atoms:
                        structure_list.append(new_structure)
                        idx_list.append(idx_smaller)
                        supercell_vec_list.append(supercell_vec)

                        if verbose:
                            ut.custom_print(
                                (
                                    f"Supercell generated (diff.) {supercell_vec} "
                                    f"- total atoms: {struct_size}"
                                ),
                                "debug",
                            )

        return structure_list, idx_list, supercell_vec_list

    def _check_repeat_struct(self, curr_phase, curr_struct: Structure):
        structure_list = self.df.loc[self.df.phase == curr_phase].structure.values

        # species_list = set([a.symbol for a in curr_struct.species])
        species_list = [el.Z for el in self.phase_diagram.alloy_set]

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
            sparse=False,
        )

        for pym_struct in structure_list:
            ase_struct = AseAtomsAdaptor().get_atoms(pym_struct)

            # Create output for multiple system in parallel
            struct_soap = soap.create(ase_struct, n_jobs=-1, verbose=False)
            curr_feat_sum = struct_soap.sum()
            soap_structs.append(curr_feat_sum)

        # TODO: Should this be here twice?
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

    def _get_structure_type_row(self, bulk: bool, surface: bool, cluster: bool) -> str:
        """
        Determines the type of a structure based on boolean flags.

        Parameters
        ----------
        - bulk (bool): True if the structure is a bulk.
        - surface (bool): True if the structure is a surface.
        - cluster (bool): True if the structure is a cluster.

        Returns
        -------
        - str: The type of the structure ("bulk", "surface", or "cluster").
        """
        if bulk:
            return "bulk"
        elif surface:
            return "surface"
        elif cluster:
            return "cluster"
        else:
            return "unknown"  # handle the case where no flags are True

    def get_structure_list(self) -> list[ase_atoms]:
        """
        Retrieve all structures from the dataframe and return them as a list.

        Returns
        -------
        list[ase.Atoms]
            A list of ASE Atoms objects representing the structures in the dataframe.
        """
        structure_list = []

        for _, row in self.df.iterrows():
            # Get ASE structure
            pmg_curr_struct = row["structure"]
            ase_curr_struct = AseAtomsAdaptor().get_atoms(pmg_curr_struct)

            # Populate structure with information
            ase_curr_struct.info["mdb_struct_type"] = self._get_structure_type_row(
                bulk=row["bulk"],
                surface=row["surface"],
                cluster=row["cluster"],
            )
            ase_curr_struct.info["aiida_uuid"] = str(row["unique_id"])
            ase_curr_struct.info["struct_name"] = row["material_name"]

            structure_list.append(ase_curr_struct)

        return structure_list

    def export_db(
        self,
        out_format: str = "extxyz",
        file_name: str = None,
        file_path: str | pl.Path = None,
    ):
        struct_list = []
        for _, row in self.df.iterrows():
            # Get ASE structure
            pmg_curr_struct = row["structure"]
            ase_curr_struct = AseAtomsAdaptor().get_atoms(pmg_curr_struct)

            # Populate structure with information
            ase_curr_struct.info["mdb_struct_type"] = self._get_structure_type_row(
                bulk=row["bulk"],
                surface=row["surface"],
                cluster=row["cluster"],
            )

            ase_curr_struct.info["aiida_uuid"] = str(row["unique_id"])
            ase_curr_struct.info["struct_name"] = row["material_name"]
            ase_curr_struct.info["perturb"] = row["perturb"]
            ase_curr_struct.info["replacement"] = row["replacement"]
            ase_curr_struct.info["base"] = row["base"]
            ase_curr_struct.info["bulk"] = row["bulk"]
            ase_curr_struct.info["cluster"] = row["cluster"]
            ase_curr_struct.info["surface"] = row["surface"]
            ase_curr_struct.info["surface_miller"] = row["surface_miller"]
            ase_curr_struct.info["symmetry"] = row["symmetry"]
            ase_curr_struct.info["calc_type"] = row["calc_type"]
            ase_curr_struct.info["calc_performed"] = row["calc_performed"]

            struct_list.append(ase_curr_struct)

        # Use database path and name if not specified
        if file_name is None:
            file_name = self.database_name
        if file_path is None and self.database_path:
            file_path = self.database_path
        if file_path is None and not self.database_path:
            file_path = pl.Path()

        file_path = pl.Path(file_path) / file_name
        file_path = file_path.with_suffix(f".{out_format}")

        aseio.write(filename=file_path, images=struct_list, format=out_format)
        ut.custom_print(f"Database exported to {file_path}", "info")

    def find_repeat_structures(
        self,
        delete=False,
        filters: list = None,
        phase: mdb_pd.Phase = None,
    ):
        # Filtering the dataframe
        # Filters allow to select certain subsets of structures
        # from the database.
        filtered_df = self.df
        # remaining_df = self.df

        filtered_df, remaining_df, phase_list = ut.apply_filters_db(
            db_obj=filtered_df, phase=phase, filters=filters
        )

        # Getting the species from the current phase diagram
        species = self.phase_diagram.alloy_set
        species_str_list = [spec.symbol for spec in species]

        # Setting SOAP related parameters
        r_cut = 4
        n_max = 4
        l_max = 4

        ut.custom_print(
            f"Setting up SOAP with: r_cut = {r_cut}, n_max = {n_max}, l_max = {l_max}",
            "debug",
        )

        # Setting up the SOAP descriptor
        soap = SOAP(
            species=species_str_list,
            periodic=True,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sparse=False,
        )

        # The uuids of the repeated structues will be stored.
        tot_duplicate_uuid_list = []

        # Computing SOAP for every phase
        for curr_phase in phase_list:
            # This list will contain the descriptors for every structure
            soap_structs = []

            # Getting the current structures
            structure_list = filtered_df.structure.values

            # Getting the names fo the current structures
            uuid_list = filtered_df[filtered_df.phase == curr_phase]["unique_id"].values

            # Getting the total structure count
            tot_structures = len(structure_list)

            # Counter for the total equivalent structure number
            tot_equival = 0

            for pym_struct in structure_list:
                # Converting to ase structure
                ase_struct = AseAtomsAdaptor().get_atoms(pym_struct)

                # Create soap descriptors for current system and storing it
                struct_soap = soap.create(ase_struct, n_jobs=-1, verbose=True)
                soap_structs.append(struct_soap)

            # Calculating similarity with an average kernel and a gaussan metric. The
            # result will be a full similarity matrix.
            kernel = AverageKernel(metric="rbf", gamma=1)
            simi_matrix = kernel.create(soap_structs)

            # Checking every structure in the similarity matrix.
            # Similarity goes from 0 to 1. If a structure is very close to 1, it
            # will return a True in the mask. There will be always one matching
            # structure, as the algorithm compares the structure with itself.

            for struct_idx, row in enumerate(simi_matrix):
                row_n_repeat = np.count_nonzero(np.isclose(row, 1)) - 1

                # If the structure is repeated, get its name
                if row_n_repeat > 0:
                    tot_duplicate_uuid_list.append(uuid_list[struct_idx])
                    tot_equival += 1

            ut.custom_print(
                (
                    f"Phase '{curr_phase.name}' - Total selected structures:"
                    f" {tot_structures}, equivalent: {tot_equival}"
                    f" ({(tot_equival/tot_structures)*100:.2f}%)"
                ),
                "debug",
            )

        duplicate_structure_names = tot_duplicate_uuid_list

        # If the deletion flag is set, the function will delete the duplicate stuctures.
        if delete:
            ut.custom_print(
                f"{len(duplicate_structure_names)} structures marked for deletion.",
                "debug",
            )

            # Getting the dataframe entries that match the stored uuids
            mat_name_match_mask = filtered_df.isin(
                {"unique_id": duplicate_structure_names}
            )["unique_id"]
            duplicate_structures_df = filtered_df[mat_name_match_mask]

            # Dropping the matching entries
            filtered_df = filtered_df.drop(duplicate_structures_df.index)

            init_df_after_removal = pd.concat([remaining_df, filtered_df])

            self.df = init_df_after_removal

            ut.custom_print(
                f"Deleted {len(duplicate_structures_df)} structures.",
                "warn",
            )

            ut.custom_print(f"Dataframe shape after deleting: {self.df.shape}", "debug")

        else:
            ut.custom_print(
                (
                    f"{len(duplicate_structure_names)} repeated structures found. "
                    "Database untouched as 'delete' is set to False."
                ),
                "info",
            )

    def gather_base_structures(self, target_structures):
        prototypes = [phase.prototype for phase in target_structures]

        # Checking which materials are already on the database
        missing_mat = set(prototypes) - set(self.df["material_id"].values)

        if not missing_mat:
            ut.custom_print("All base structures are already in the database.", "info")
            return

        # Querying materials project database.
        ut.custom_print("Querying the MP API...", "info")
        report_replacements = True
        with MPRester(ut.gather_secrets()["API_KEY"], mute_progress_bars=True) as mpr:
            query_result = mpr.summary.search(material_ids=missing_mat)
            for material in query_result:
                for phase in self.phase_diagram.phases:
                    if phase.prototype == material.material_id:
                        curr_phase = phase.name
                        break
                    else:
                        curr_phase = np.nan

                try:
                    material_symmetry = material.get_space_group_info()
                except Exception:
                    material_symmetry = material.symmetry.symbol

                # Replacing elements
                if phase.replace_dict.get("replace"):
                    replace_with = phase.replace_dict.get(
                        "replace_with", phase.base_elem
                    )

                    replace_dict = {}
                    for element in phase.replace_dict["element_list"]:
                        replace_dict[element] = str(replace_with)

                    if report_replacements:
                        ut.custom_print(
                            (
                                f"Applying substitutions to "
                                f"base structures: {replace_dict}..."
                            ),
                            "info",
                        )
                        report_replacements = False

                    material.structure.replace_species(
                        species_mapping=replace_dict, in_place=True
                    )

                curr_struct = mdb_struct.Bulk(
                    material_id=str(material.material_id),
                    structure=material.structure,
                    temperature=np.nan,
                    perturb=False,
                    formula=material.composition_reduced,
                    symmetry=material_symmetry,
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
        read_path = pathlib.Path(path) if path else pathlib.Path()

        if target_structures:
            selection_criteria = target_structures
        else:
            selection_criteria = self.phase_diagram.keys()

        selection_criteria = [crit.name for crit in selection_criteria]

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
            for phase in self.phase_diagram.phase_names:
                for folder in xml_path.parts:
                    if slugify(folder) == slugify(phase):
                        curr_phase = phase
                        curr_phase = self.phase_diagram.get_phase(phase)
                        curr_mat_id = curr_phase.prototype
                        curr_name = f"base_relax_{curr_phase.name}_MP"

            # Creating the structure object
            curr_struct = mdb_struct.Structure().from_vasprun(
                vasprun=curr_run,
                base=True,
                phase=curr_phase,
                material_name=curr_name,
                bulk=True,
                perturb=False,
                cluster=False,
                surface=False,
                material_id=curr_mat_id,
                replacement=False,
            )

            # Saving the structure to the database
            self.df = curr_struct.save_to_db(self.df)

    def save_database(self, path: str = None, suffix: str = None):
        """
        Saves the database dataframe into a pkl object.

        Parameters
        ----------
        path : str, optional.
            Location where the pickle object will be saved,
            by default None, which defaults to storing the file in the CWD.
        suffix : str, optional.
            String that will be added at the end of the filename.

        """
        if suffix:
            filename = self.database_name + f"_{suffix}.xz"
        else:
            filename = self.database_name + ".xz"

        if not path:
            path = ""

        file_path = pathlib.Path(path, filename)

        with lzma.open(file_path, "wb") as f:
            pickle.dump(self, f)

        ut.custom_print(f"Database saved in {file_path}", "info")

    def _apply_user_filters(self, filters: list, target_entries: pd.DataFrame):
        # Creating a empty DataFrame with the same column dtypes but no entries.
        target_entries_filter = target_entries[0:0]

        # Iterating over every filter type and getting each related structure,
        # which will get concatenated to the empty dataframe
        for fil in filters:
            filter_entries = target_entries.loc[target_entries[fil]]
            target_entries_filter = pd.concat((target_entries_filter, filter_entries))

        return target_entries_filter

    def perturb_gauss(
        self, center: float = 0.04, repeat: int = 5, filters: list = None
    ):
        # Getting all structures which are not perturbed
        target_entries = self.df.loc[(~self.df.material_name.str.contains("_perturb"))]

        # Applying user specified filters
        if filters:
            target_entries = self._apply_user_filters(filters, target_entries)

        ut.custom_print(f"Applying filters {filters} for perturbation.")
        ut.custom_print(
            f"{len(target_entries)*repeat} perturbed entries will be added.", "debug"
        )

        # Applying displacement to all perturbed structures
        for _, entry in target_entries.iterrows():
            # Getting information from the current entry
            str_matid = entry.material_id
            str_phase = entry.phase
            curr_str = entry.structure

            # Generating string for naming the current structure
            extra_info = ""
            if entry.supercell:
                extra_info += f"_super-{mdb_surf.get_miller_index_str(entry.supercell)}"
            if entry.replacement:
                extra_info += (
                    f"_repl-{entry.replacement_ind[0]}-{entry.replacement_ind[1]}"
                )

            for perturb_repeat_idx in range(repeat):
                # Applying displacement
                new_struct_perturb = self._apply_gauss_perturb(
                    center=center, structure=curr_str
                )

                mat_str = f"{entry.material_name}_perturb_gauss_{perturb_repeat_idx+1}"

                # Creating a new Structure from the perturbed structure structure
                curr_struct = mdb_struct.Structure(
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
                    curr_struct_conv = mdb_struct.Bulk().from_mdb_structure(curr_struct)
                elif entry.surface:
                    curr_struct_conv = mdb_struct.Surface().from_mdb_structure(
                        curr_struct,
                        entry.surface_miller,
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
        new_structure.perturb(distance=center * 2, min_distance=center / 2)
        return new_structure

    def perturb_min_displacement(
        self,
        frac_max: float = 0.05,
        frac_min: float = 0.01,
        repeat=5,
    ):
        """
        Apply small displacements to the lattice parameters of relaxed structures.

        This method perturbs the lattice parameters of relaxed structures by applying
        small random displacements to their lattice matrix elements.
        The perturbations are repeated a specified number of times, creating multiple
        perturbed structures for each initial structure. This helps in generating
        structures with slightly higher energies and forces, useful for generating
        training data for neural network potentials (NNP) intended for
        molecular dynamics (MD) simulations.

        Parameters
        ----------
        frac_max : float, optional
            Maximum fraction of the lattice parameter perturbation, by default 0.05.
        frac_min : float, optional
            Minimum fraction of the lattice parameter perturbation, by default 0.01.
        repeat : int, optional
            Number of times to apply the perturbation to each structure, by default 5.

        Raises
        ------
        NotImplementedError
            If the perturbation strategy is applied to an unsupported structure type.

        Notes
        -----
        The perturbed structures are then converted to the appropriate type
        (Bulk or Surface) and saved to the database.

        Example
        -------
        >>> initial_db = InitialDatabase()
        >>> initial_db.perturb_min_displacement(frac_max=0.05, frac_min=0.01, repeat=5)

        """
        # Getting all relaxed structures
        target_entries = self.df.loc[self.df.base]

        # Applying displacement to all perturbed structures
        for _, entry in target_entries.iterrows():
            # Getting some parameters from the current perturb structure.
            str_matid = entry.material_id
            if isinstance(entry.phase, str):
                str_phase = self.phase_diagram.get_phase(entry.phase)
            curr_str = entry.structure

            # Applying the perturbation 'repeat' times.
            for perturb_repeat_idx in range(repeat):
                # Applying displacement,
                new_struct_perturb = self._apply_min_perturbation(
                    structure=curr_str,
                    frac_max=frac_max,
                    frac_min=frac_min,
                )

                mat_str = (
                    f"{str_matid}_{str_phase.name}_perturb_min_{perturb_repeat_idx+1}"
                )

                # Creating a new Structure from the perturbed structure structure
                curr_struct = mdb_struct.Structure(
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
                    curr_struct_conv = mdb_struct.Bulk().from_mdb_structure(curr_struct)
                elif entry.surface:
                    curr_struct_conv = mdb_struct.Surface().from_mdb_structure(
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
        # Making a copy of the current structure lattice which can be modified
        perturb_structure = structure.copy()
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

    def display_db_ase(self):
        """Display the the structures in the database using ase.visualize.view."""
        structures = self.df.structure
        ut._display_indb_dataframe(structures)

    def _get_phase_from_id(self, idx: str) -> str:
        """
        Searches for the corresponding phase in the self.phase_diagram
        dict to a given a material projects id.

        Parameters
        ----------
        idx : str
            Materials project id

        Returns
        -------
        str
            A phase of the phase diagram
        """
        # Creating a list of the phase diagram phase names
        phase_list = [
            phase.name for phase in self.phase_diagram.phases if phase.prototype == idx
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
        with the alloy preparation by replacing all atoms with the base_elem.

        Parameters
        ----------
        structure : Structure
            Prototype structure to be modified

        Returns
        -------
        Structure
            Structure which may have all its atoms replaced by base_elem if it contains
            anything other than the alloy_set atoms.
        """
        # Checking if there are any other atoms than ones in alloy_set in the structure
        if len(set(structure.symbol_set) - self.phase_diagram.alloy_set) > 0:
            # Creating a new structure using the base one as a template
            new_structure = structure.copy(sanitize=True)

            # Replacing base_elem atoms in the structures
            for ind in range(len(structure.species)):
                new_structure.replace(
                    ind,
                    # Species(self.phase_diagram.get_phase(phase).base_elem),
                    Element(self.phase_diagram.get_phase(phase).base_elem),
                )

            # Returning new structure with atoms replaced with base_elem
            return new_structure

        else:
            # If everything is already either atoms from alloy_set,
            # leave the structure as is.
            return structure

    def _gather_prototype_structure(
        self,
        prototype: str,
        phase: str,
        get_different_supercells: bool,
        read: bool,
        supercell_max_idx: int,
        num_min_atoms: int = 1,
    ):
        """
        Gather the structure for a prototype from the materials project database,
        while checking that the phase given for the material is correct.

        Parameters
        ----------
        prototype : str
            Materials project id for the prototype structure.
        phase : str
            Name of the phase to be generated.
        get_different_supercells: bool
            Whether to get one or more supercells from the base structure.
        num_min_atoms: int
            Minimum number of atoms in a generated supercell in order to be considered.
        read: bool
            Whether to read the structure from the database or to use the MP API.


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
            are given on the self.phase_diagram dictionary.
            More phases could be added there if necessary.
        """
        # Checking for correct phase input
        if isinstance(phase, mdb_pd.Phase):
            phase_name = slugify(phase.name)
        elif isinstance(phase, str):
            phase_name = slugify(phase)

        if not self.phase_diagram.get_phase(phase_name):
            raise KeyError(
                "Wrong phase given. "
                f"Please introduce one of: {[k for k in self.phase_diagram.phases]}"
            )

        # Reading structure from database
        if read:
            ut.custom_print("Using structure from the db as template...", "debug")
            try:
                query_result = self.df.loc[self.df.phase == phase]
                material_id_prefix = query_result.material_id.values[0]
                structure = query_result.structure.values[0]
            except IndexError:
                raise mdb_exc.BaseStructureNotFound() from None

        # Querying alloy prototype structure
        else:
            ut.custom_print("Querying the MP API...", "info")
            with MPRester(
                ut.gather_secrets()["API_KEY"], mute_progress_bars=True
            ) as mpr:
                query_result = mpr.summary.search(material_ids=[prototype])[0]
                structure = query_result.structure
                material_id_prefix = query_result.material_id

        # TODO: Add a toggle so that the user can choose to use it.
        # Converting all of the atoms from the prototype cell to
        # the base atom type if necessary
        # structure = self._convert_prototype_structure(structure=structure, phase=ph)

        # Getting conventional cell for the replaced structure
        sga = SpacegroupAnalyzer(structure)
        structure = sga.get_conventional_standard_structure()

        # Create supercells for the replaced structure
        # This can return either 1 or more supercells of the
        # same structure, depending on the 'get_different_supercells' flag.
        structure_list, idx_list, supercells = self._find_supercell_indices(
            structure,
            get_different_supercells,
            max_atoms=self.max_num_atoms,
            min_atoms=num_min_atoms,
            initial_supercell_size=supercell_max_idx,
        )

        struct_obj_list = []
        report_replacements = True
        # Saving all the generated supercells as separate bulk structures
        for structure, idxs in zip(structure_list, supercells):
            # Ignore small structures
            # if len(structure) < num_min_atoms:
            #     continue

            # Replacing elements
            if phase.replace_dict.get("replace"):
                replace_with = phase.replace_dict.get("replace_with", phase.base_elem)

                replace_dict = {}
                for element in phase.replace_dict["element_list"]:
                    replace_dict[element] = str(replace_with)

                if report_replacements:
                    ut.custom_print(
                        (
                            f"Applying substitutions to "
                            f"base structures: {replace_dict}..."
                        ),
                        "info",
                    )
                    report_replacements = False

                structure.replace_species(species_mapping=replace_dict, in_place=True)

            # Getting the supercell vector as a string for naming
            idxs_str = "".join(map(str, idxs))

            try:
                bulk_temp = query_result.temperature.values[0]
            except Exception:
                bulk_temp = np.nan
            # Creating a new bulk from the supercell
            curr_bulk = mdb_struct.Bulk(
                material_name=f"{material_id_prefix}_{phase.name}_super-{idxs_str}",
                material_id=material_id_prefix,
                structure=structure,
                temperature=bulk_temp,
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
        phase: mdb_pd.Phase,
        structure_obj: mdb_struct.Structure,
    ):
        phase = structure_obj.phase
        # curr_phase_atom = self.phase_diagram.get_phase(phase).base_elem
        # base_atom_set = list(self.phase_diagram.alloy_set - {curr_phase_atom})

        if isinstance(structure, (mdb_struct.Surface, Slab)):
            new_structure = structure.get_sorted_structure()
        else:
            new_structure = structure.copy(sanitize=True)

        # Replacing atoms in the structures
        ind = 2
        sum_ind = 0
        sum_list = (2, 1, 2, 3)

        while ind < structure.num_sites:
            # new_structure.replace(ind - 1, Species(base_atom_set[0]))
            new_structure.replace(ind - 1, Element(phase.base_elem))
            ind = ind + sum_list[sum_ind]

            if sum_ind == 3:
                sum_ind = 0
            else:
                sum_ind += 1

        material_id_prefix = structure_obj.material_id

        # Generating the symmetrized structure
        new_struct_symm = mdb_struct.Structure(
            material_name=f"{material_id_prefix}_{phase.name}_symm",
            material_id=material_id_prefix,
            structure=structure,
            temperature=structure_obj.temperature,
            perturb=False,
            surface=False,
            base=False,
            cluster=False,
            calc_performed=False,
            supercell=structure_obj.supercell,
            phase=phase,
        )

        if structure_obj.bulk:
            final_struct = mdb_struct.Bulk().from_mdb_structure(
                mdb_structure=new_struct_symm,
                new_structure=structure,
            )
        elif structure_obj.surface:
            final_struct = mdb_struct.Surface().from_mdb_structure(
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
            max_base_elem = min((phase.base_elem_comp_max + offset), 1)
            min_base_elem = max(phase.base_elem_comp_min - offset, 0)

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

    def _fit_replacements_phase(
        self,
        phase,
        structure,
        subst_base_elem_perc,
    ):
        curr_comp = structure.composition
        base_elem = phase.base_elem
        # tot_base_at_struct = curr_comp[base_elem]
        structure_len = len(structure.species)
        offset_min = phase.base_elem_comp_min - phase.offset
        offset_max = phase.base_elem_comp_max + phase.offset

        n_at_replacement_upd = []
        for _, curr_perc in enumerate(subst_base_elem_perc):
            inPhase = phase.perc_in_phase(curr_perc)

            single_at_perc = 1 / structure_len
            perc_range = offset_max - offset_min

            # Skip this offset if changing one atom always results
            # in going over the maximum or minimum.
            if single_at_perc >= perc_range:
                inPhase = True

            while not inPhase:
                perc = curr_comp.get_atomic_fraction(base_elem)
                # perc = (tot_base_at_struct + abs(curr_perc)) / structure_len

                if perc >= offset_max:
                    curr_perc -= single_at_perc
                elif perc <= offset_min:
                    curr_perc += single_at_perc
                else:
                    inPhase = phase.perc_in_phase(curr_perc)

            # print("_fit_repl curr_perc 2: ", curr_perc)
            new_n_at = int(round(curr_perc * structure_len, 0))
            # print("_fit_repl new_n_at: ", new_n_at)
            n_at_replacement_upd.append(new_n_at)

        return n_at_replacement_upd

    def _apply_replacement(
        self, structure: Structure, phase, n_target_at: int | float, rng=None
    ):
        if not rng:
            rng = np.random.default_rng()

        if isinstance(
            structure,
            (
                mdb_struct.Structure,
                mdb_struct.Surface,
                mdb_struct.Bulk,
            ),
        ):
            structure = structure.structure

        structure_len = len(structure.species)
        curr_comp = structure.composition

        # We assume that if the n_atoms is a fractional number, it must
        # represent the ratio of atoms in the structure, so we convert
        # that to a number of atoms.
        # print("n_target_at: ", n_target_at)
        if isinstance(n_target_at, float) and n_target_at < 1:
            n_target_at = int(n_target_at * structure_len)

        # If no replacements are going to be made, this is probably due to
        # a low percentage being rounded to 0, thus we attempt to make at
        # least one replacement.
        if n_target_at == 0:
            n_target_at = 1

        curr_n_base_atoms = int(curr_comp[phase.base_elem])
        replacement_type = "add" if n_target_at > curr_n_base_atoms else "sub"

        # Getting current structure composition information
        # The current procedure assumes that all of the atom species in the structure
        # will have been replaced beforehand with the base atom,
        # although this results in more randomness.
        base_elem = phase.base_elem
        if len(self.phase_diagram.alloy_set) > 1:
            (other_elem,) = self.phase_diagram.alloy_set - {base_elem.symbol}
        else:
            other_elem = list(self.phase_diagram.alloy_set)[0]

        # TODO: Check if this must be reenabled
        # If the structure only has one type of Element, and that is not the base
        # element, this changes with what to replace.
        if not curr_comp.as_dict().get(base_elem.symbol):
            base_elem = str(phase.phase_diagram.element_list[0])
            if len(self.phase_diagram.alloy_set) > 1:
                (other_elem,) = self.phase_diagram.alloy_set - {base_elem}
            else:
                other_elem = list(self.phase_diagram.alloy_set)[0]

        # Adding base atoms to match the target percentage
        if replacement_type == "add":
            n_at_diff = n_target_at - curr_n_base_atoms
            spec_to_replace = Element(base_elem)
            replacing_elem = Element(other_elem)
        # Removing base atoms to match the target percentage
        else:
            n_at_diff = curr_n_base_atoms - n_target_at
            spec_to_replace = Element(base_elem)
            replacing_elem = Element(other_elem)

        # Get atoms available to replace in the structure
        if isinstance(spec_to_replace, Element):
            replacable_sites = structure.indices_from_symbol(spec_to_replace.symbol)
        else:
            replacable_sites = structure.indices_from_symbol(spec_to_replace)

        # Randomly selecting indices to replace out of the available positions.
        replace_elem_choices = rng.choice(
            a=replacable_sites,
            size=abs(int(n_at_diff)),
            replace=False,
            shuffle=True,
        )

        if isinstance(structure, (mdb_struct.Surface, Slab)):
            new_structure = structure.get_sorted_structure()
        else:
            new_structure = structure.copy(sanitize=True)
        site_props_before = structure.site_properties

        # Replacing atoms in the structures
        for ind in replace_elem_choices:
            # print(f"Replaced idx {ind}: {new_structure.species[ind]} -> {other_elem}")
            new_structure = new_structure.replace(ind, replacing_elem)

        # Copying site properties
        if isinstance(structure, (mdb_struct.Surface, Slab)):
            new_structure = new_structure.get_sorted_structure()
        else:
            new_structure = new_structure.copy(
                sanitize=True, site_properties=site_props_before
            )

        # new_comp = new_structure.composition
        # print("new_comp: ", new_comp)
        # print(
        #     "new_comp.get_atomic_fraction(base_elem): ",
        #     new_comp.get_atomic_fraction(phase.base_elem),
        # )

        return new_structure

        # else:
        #     # Getting how many base atoms must be changed in order for the
        #     # structure to meet the current percentage requirements.
        #     print('curr_comp: ', curr_comp)
        #     print('curr_comp[base_elem]: ', curr_comp[base_elem])
        #     target_atoms_base = curr_comp[base_elem] - abs(n_target_at)
        #     print('target_atoms_base: ', target_atoms_base)

        #     # Getting how many atoms of the other element must be changed
        #     # other_atom_change = int(curr_comp[other_elem] - target_atoms_base)
        #     other_atom_change = int(curr_comp[other_elem] + target_atoms_base)
        #     other_atom_change = other_atom_change - curr_comp[other_elem]

        # print("other_atom_change: ", other_atom_change)
        # # Choosing which species of the structure to change with the other atom.

        # # Get indices of other atoms in the structure
        # if isinstance(other_elem, Element):
        #     replacable_sites = structure.indices_from_symbol(other_elem.symbol)
        # else:
        #     replacable_sites = structure.indices_from_symbol(other_elem)

        # print("replacable_sites: ", replacable_sites)

        # other_elem_choices = rng.choice(
        #     a=replacable_sites,
        #     size=abs(int(other_atom_change)),
        #     replace=False,
        #     shuffle=True,
        # )

        # print("other_elem_choices: ", other_elem_choices)
        # # Creating a new pymatgen structure using the base one as a template
        # new_structure = structure.copy(sanitize=True)
        # site_props_before = structure.site_properties
        # print("site_props_before: ", site_props_before)

        # # Replacing atoms in the structures
        # for ind in other_elem_choices:
        #     print(f"Replaced idx {ind}: {new_structure.species[ind]} -> {other_elem}")
        #     new_structure = new_structure.replace(ind, other_elem)

        # # TODO: Instead of this, create a new structure
        # # Copying site properties
        # new_structure = new_structure.copy(
        #     sanitize=True, site_properties=site_props_before
        # )
        # new_comp = new_structure.composition
        # print("new_comp: ", new_comp)
        # print(
        #     "new_comp.get_atomic_fraction(base_elem): ",
        #     new_comp.get_atomic_fraction(phase.base_elem),
        # )

        return new_structure

    def generate_bulk_structures(
        self,
        prototype: str,
        phase: str,
        num_struct: int,
        num_repeats: int,
        get_different_supercells: bool,
        min_num_atoms: int,
        supercell_max_idx: int,
        convert_to_base: bool = True,
        read: bool = True,
        seed: int = 42,
    ):
        """
        Allows to create several variations of a certain phase
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
        convert_to_base: bool
            Whether to convert all atoms from the structure to the base atom, in order
            to increase randomness when replacing.

        Raises
        ------
        KeyError
            Raised if the given phase is not found. All of the available phases
            are given on the self.phase_diagram dictionary.
            More phases could be added there if necessary.
        """
        # Instantiating RNG
        rng = np.random.default_rng(seed=seed)

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
            num_min_atoms=min_num_atoms,
            supercell_max_idx=supercell_max_idx,
        )

        for structure_obj, supr_idx in zip(structure_list, idx_list):
            structure = structure_obj.structure

            # Converting all of the atoms from the prototype cell to the
            # base atom type if necessary
            if convert_to_base:
                structure = self._convert_prototype_structure(
                    structure=structure, phase=phase
                )

            # Replacing some atoms using symmetry
            structure = self._create_symmetrical_prototype(
                structure=structure, phase=phase, structure_obj=structure_obj
            )
            # Preparing an array of randomly generated base elem percentages
            # for the new structures
            subst_base_elem_perc = self._gen_base_elem_perc(phase, num_struct)

            # Choosing the amount of atoms to replace with the base element in the
            # struct which at this point will be completely replaced by atoms
            # of the remaining species of the alloy.
            # n_at_replacement = [
            #     int(round(structure_len * stct, 0)) for stct in subst_base_elem_perc
            # ]

            # Attempting to fix any percentages outside of the
            # current phase ratios.
            # n_at_replacement_upd is a list which contains the
            # target number of base atoms in the new structure.
            n_at_replacement_upd = self._fit_replacements_phase(
                phase, structure, subst_base_elem_perc
            )
            # Replacing the atoms and generate 'num_replacements'
            # structures for each percentage
            for str_ind, n_atoms in enumerate(n_at_replacement_upd):
                for repl in range(num_repeats):
                    # Applying the replacement
                    new_structure = self._apply_replacement(
                        structure, phase, n_atoms, rng
                    )

                    # Getting the supercell vector
                    supercell_vec_str = mdb_surf.get_miller_index_str(
                        structure_obj.supercell
                    )

                    try:
                        bulk_temp = query_result.temperature.values[0]
                    except Exception:
                        bulk_temp = np.nan

                    # Creating a new Bulk object for the structure with replacement
                    new_struct_symm = mdb_struct.Bulk(
                        material_name=f"{prototype}_{phase.name}_super-{supercell_vec_str}-{supr_idx}_replacement-{str_ind+1}-{repl+1}",
                        material_id=prototype,
                        structure=new_structure,
                        temperature=bulk_temp,
                        perturb=False,
                        surface=False,
                        replacement=True,
                        replacement_ind=(str_ind + 1, repl + 1),
                        base=False,
                        cluster=False,
                        calc_performed=False,
                        supercell=structure_obj.supercell,
                        phase=phase,
                    )

                    self.df = new_struct_symm.save_to_db(self.df)

    def get_base_structs_current_phase(self, phase):
        # Getting all of the base structures
        base_structs = self.df.loc[self.df.base]

        if isinstance(phase, str):
            phase = self.phase_diagram.get_phase(phase=phase)

        # Getting the structures corresponding to the current phase
        phase_mask = base_structs.phase == phase

        # If an old version of the phase object is being used
        if not phase_mask.any():
            phase_mask = base_structs.phase == phase.name

        base_structs = base_structs.where(phase_mask, other=pd.NA)
        base_structs.dropna(how="all", inplace=True)

        return base_structs

    def _get_main_elem_perc(self, phase: mdb_pd.Phase, structure):
        """
        Get the percentage of the main element for a given structure.

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

        perc = main_cnt / total_atoms
        return perc

    def generate_surfaces_replacements(
        self,
        phase: mdb_pd.Phase,
        num_struct: int,
        num_repeats: int,
    ):
        if isinstance(phase, str):
            phase = self.phase_diagram.get_phase(phase)

        # Instantiating RNG
        rng = np.random.default_rng()

        # Getting the base structures
        slabs_df = self.df.loc[self.df.surface]
        slabs_df = slabs_df.loc[self.df.base]
        slabs_df = slabs_df.loc[self.df.phase == phase.name]

        for _, row in slabs_df.iterrows():
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

            # Choosing the amount of atoms to replace with the base element in the
            # struct, which at this point will be completely replaced by atoms
            # of the remaining species of the alloy.
            n_at_replacement = []
            for stct in subst_base_elem_perc:
                curr_repl = int(round(structure_len * abs(stct), 0))
                if stct < 0:
                    curr_repl *= -1
                n_at_replacement.append(curr_repl)

            # Attempting to fix any percentages outside of the
            # current phase ratios.
            n_at_replacement_upd = self._fit_replacements_phase(
                phase, curr_surface, subst_base_elem_perc
            )

            # Adapting the replacement percentages generated to the
            # percentage of the current structure
            # n_at_replacement_final = self._adjust_replacements()

            for str_ind, n_atoms in enumerate(n_at_replacement_upd):
                for repl in range(num_repeats):
                    # Replacing atoms according to the current phase from the structure.
                    new_structure = self._apply_replacement(
                        curr_surface, phase, structure_len, n_atoms, rng
                    )

                    prototype = phase.prototype
                    extra = {"surface": True}

                    # Matching the miller index from the name as it is not stored.
                    match = re.search(r"\((.*?)\)", row.material_id)
                    curr_miller = match.group(1) if match else "???"

                    # Preparing the structure name
                    mat_id_name = f"{prototype}_{phase.name}"
                    mat_id_surf_data = f"_replacement-({curr_miller})-_{str_ind}-{repl}"
                    material_id = mat_id_name + mat_id_surf_data

                    # Saving the structure into the database.
                    self._save_row(
                        material_id=material_id,
                        phase=phase,
                        structure=new_structure,
                        extra=extra,
                    )

    def _save_row(
        self,
        structure,
        material_id=None,
        phase=None,
        extra=None,
        base=False,
    ):
        # HACK: This should probably use __dict__ instead of a manually set list...
        # Attributes not to store in a row
        unwanted_attrs = ["save_to_db", "from_vasprun", "from_mdb_structure"]

        # If given structure is a pymatgen Structure
        if isinstance(structure, Structure):
            new_row = pd.Series(
                {
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
        # If given structure is a MatDBForge structure.
        elif isinstance(structure, mdb_struct.Structure):
            attr_list = [
                att
                for att in dir(structure)
                if not att.startswith("_") and att not in unwanted_attrs
            ]
            att_dict = {att: getattr(structure, att) for att in attr_list}

            new_row = pd.Series(att_dict)

        new_row_df = new_row.to_frame().T
        new_row_df = new_row_df.astype(
            {
                "perturb": bool,
                "base": bool,
                "bulk": bool,
                "surface": bool,
                "cluster": bool,
                "calc_performed": bool,
                "replacement": bool,
            }
        )

        self.df = self.df.astype(
            {
                "perturb": bool,
                "base": bool,
                "bulk": bool,
                "surface": bool,
                "cluster": bool,
                "calc_performed": bool,
                "replacement": bool,
            }
        )

        # Concatenating a row with boolean columns to an empty array results
        # in a warning message. In order to avoid the warning, if the
        # dataframe is empty, it is replaced by the dataframe with the complete
        # row in this first instance only, and then the concatenation is done
        # as usual for the rest of the execution.
        if self.df.shape[0] == 0:
            self.df = new_row_df
        else:
            self.df = pd.concat([self.df, new_row_df], ignore_index=True)

    def _gather_n2p2_reqdata_from_node(self, node):
        # Getting calculation name
        name = node.label + "_aiida-uuid_" + node.uuid

        # Writing the vasprun.xml file to a buffer.
        retrieved = node.outputs.retrieved
        vasprun_f = retrieved.get_object_content("vasprun.xml", "rb")
        buffer = BytesIO(vasprun_f)

        # Reading the file from the buffer and closing it
        vasprun = aseio.read(buffer, format="vasp-xml", index="-1")
        buffer.close()

        # Getting properties from the vasprun
        pot_energy = vasprun.get_potential_energy(force_consistent=True) * self.eV2Eh

        # Getting forces
        # Reading forces from vasprun.xml, in eV/Ang and converting them to Ha/Bohr
        forces = vasprun.get_forces() * self.eV2Eh * self.Bohr2Ang

        lattice = vasprun.get_cell() * self.Ang2Bohr
        structure = vasprun.get_positions() * self.Ang2Bohr
        symbols = vasprun.get_chemical_symbols()

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
            # Preparing and writing the line
            buffer.write(
                f"atom {at[0]:.6f} {at[1]:.6f}"
                f" {at[2]:.6f}"
                f" {data_dict['symbols'][idx]} {0:.6f} {0:.6f}"
                f" {frc[0]:.6f} {frc[1]:.6f} {frc[2]:.6f}\n"
            )

        # writing potential energy and charge
        buffer.write(f'energy {data_dict["pot_energy"]:.8f}\n')
        buffer.write(f'charge {data_dict["charge"]:.6f}\n')

        # writing end keyword
        buffer.write("end\n")

    def generate_n2p2_input_aiida(
        self, aiida_group_list: list, filter_dict: dict, path: str = None
    ):
        # Loading aiida profile
        load_profile()

        # Handling path
        path = pathlib.Path(path) if path and isinstance(path, str) else pathlib.Path()

        # Adding input.data filename to path
        path = path / "input.data"

        # Gathering nodes from the given group
        ut.custom_print("Getting nodes...")

        # Preparing a query in the aiida db
        qb = orm.QueryBuilder()

        for group in aiida_group_list:
            qb.append(orm.Group, filters={"label": group}, tag="group")
            (qb.append(orm.WorkChainNode, with_group="group", filters=filter_dict),)

        result_nodes = qb.all(flat=True)

        if len(result_nodes) == 0:
            for _group in aiida_group_list:
                qb.append(VaspCalculation, with_group="group", filters=filter_dict)
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

    def remove_structs_out_of_atom_count_range(
        self, min_num_atoms: int, max_num_atoms: int, remove_base=False
    ):
        remove_count = 0
        for row, struct in self.df.iterrows():
            # Do not remove structures marked as base.
            if struct.base and not remove_base:
                continue

            # Removing structures that are outside of the size range
            if (
                len(struct.structure.species) < min_num_atoms
                or len(struct.structure.species) > max_num_atoms
            ):
                self.df.drop(row, inplace=True)
                remove_count += 1
        return remove_count

    def remove_structs_out_of_cell_size_range(
        self, min_cell_size: float, remove_base=False
    ):
        remove_count = 0
        for row, struct in self.df.iterrows():
            structure_obj = struct.structure

            # Do not remove structures marked as base.
            if struct.base and not remove_base:
                continue

            if isinstance(structure_obj, Structure):
                structure_obj = AseAtomsAdaptor().get_atoms(
                    structure_obj, msonable=False
                )

            if any(structure_obj.cell.cellpar()[:3] < min_cell_size):
                self.df.drop(row, inplace=True)
                remove_count += 1

        return remove_count

    def limit_structure_number_phases(
        self,
        structure_limit: int,
        structure_types: list,
        phases_to_use: list = None,
    ):
        # Getting which phases to use
        if phases_to_use:
            num_phases = len(phases_to_use)
        else:
            phases_to_use = self.phase_diagram.phases
            num_phases = len(self.phase_diagram.phases)

        # Maximum number of structures per phase
        max_struct_phase = structure_limit // num_phases

        # Getting the current phase as a Phase object
        # If the phase is not found, omitting.
        for curr_phase in phases_to_use:
            try:
                curr_phase = self.phase_diagram.get_phase(curr_phase)
            except mdb_exc.PhaseNotFound:
                continue

            structure_list = self.df.loc[self.df.phase == curr_phase]

            # Getting the selected types to use.
            # TODO: Fix this, as now only keeps one structure type.
            # for structure_type in structure_types:
            # structure_list.loc[structure_list['channel'].isin(['sale','fullprice'])]
            # structure_list = structure_list.loc[structure_list[structure_type]]

            # If the number of structures for the selected types is larger than
            # the maximum allowed, reduce the number by sampling a certain amount.
            if structure_list.shape[0] > max_struct_phase:
                # Getting the all but the original surface used for this phase,
                # which should be maintained in the database.
                changed_structures = structure_list.loc[
                    (structure_list.phase == curr_phase)
                    & (
                        structure_list.perturb
                        | structure_list.supercell
                        | structure_list.replacement
                    )
                ]

                to_remove = structure_list.loc[
                    structure_list.unique_id.isin(changed_structures.unique_id)
                ]
                structure_list_base = structure_list.drop(to_remove.index)

                sample_amount = structure_limit - structure_list_base.shape[0]
                changed_structures_sample = changed_structures.sample(sample_amount)

                # These are the selected structures for the desired phase and type.
                # We want to keep these structures in the original dataframe.
                phase_structures = pd.concat(
                    [structure_list_base, changed_structures_sample]
                )

                # We remove all structures of the selected type
                # and phase from the original
                # dataframe, and add the selected ones.
                orig_removed = self.df.loc[
                    ~self.df.unique_id.isin(structure_list.unique_id)
                ]

                phase_result_df = pd.concat([orig_removed, phase_structures])
                self.df = phase_result_df

    def generate_clusters(
        self,
        size_range: list,
        get_replacements=False,
        get_perturbed=False,
        add_dimer=False,
        save_in_db=False,
        limit_per_phase: int = None,
        num_struct: int = 2,
        num_repeat: int = 2,
    ):
        # Getting default phase
        phase = self.phase_diagram.get_phase("alpha")

        # Generate a list of mdb_struct.Cluster
        cluster_list = []

        # Adding a dimer
        if add_dimer:
            ut.custom_print("Adding base dimers...", "debug")
            clust_obj = mdb_clust.make_clean_dimer(self, phase=phase)
            cluster_list.append(clust_obj)
            ut.custom_print("Base dimers done...", "debug")

        # Create clusters over all size range given, from smallest to largest.
        ut.custom_print(
            f"Adding base clusters with n_atoms: {size_range[0]}-{size_range[-1]}...",
            "debug",
        )
        for size in size_range:
            clust_obj = mdb_clust.make_clean_cluster(self, size=size, phase=phase)
            cluster_list.append(clust_obj)
        ut.custom_print("Base clusters done...", "debug")

        # If True, store the structures along with their information
        # into the MatDBForge InitialDatabase object
        if save_in_db:
            for _idx, cluster in enumerate(cluster_list):
                self._save_row(structure=cluster)

        # Return the cluster list in case the user just wants the clusters
        # but not storing them into the database.
        ut.custom_print(f"Generated {len(cluster_list)} base clusters.", "debug")
        return cluster_list

    def plot_database_composition(
        self,
        temperature_K: float = 273.0,
        rc_params: dict = None,
        fig_path: str | pl.Path = ".",
        fig_name: str = "database_composition",
    ):
        fig, axs = plt.subplots(
            ncols=1, nrows=2, gridspec_kw={"height_ratios": [1, 4]}, figsize=(10, 7)
        )
        hist_t_ax = axs[0]
        main_plot_ax = axs[1]

        # Plotting the phase diagram of the database
        main_plot_ax = self.phase_diagram.plot_diagram(
            rc_params=rc_params, show_plot=False, ax=main_plot_ax
        )

        # Getting base element to use in x-axis
        base_elem = self.phase_diagram.base_elem

        plot_dict = {
            "bulk": {"structs": [], "color": "#458588"},
            "surface": {"structs": [], "color": "#fe8019"},
            "cluster": {"structs": [], "color": "#d3869b"},
            "unknown": {"structs": [], "color": "#ee0000"},
        }

        # Get base element composition for every structure in the database
        for idx, row in self.df.iterrows():
            curr_comp = row.structure.composition
            curr_frac = curr_comp.get_atomic_fraction(base_elem.symbol)

            if row.bulk:
                plot_dict["bulk"]["structs"].append(curr_frac * 100)
            elif row.surface:
                plot_dict["surface"]["structs"].append(curr_frac * 100)
            elif row.cluster:
                plot_dict["cluster"]["structs"].append(curr_frac * 100)
            else:
                print("idx: ", idx)
                print("row.cluster: ", row.cluster)
                print("row.surface: ", row.surface)
                print("row.bulk: ", row.bulk)
                plot_dict["unknown"]["structs"].append(curr_frac * 100)

        plot_dict["bulk"]["structs"] = np.array(plot_dict["bulk"]["structs"])
        plot_dict["surface"]["structs"] = np.array(plot_dict["surface"]["structs"])
        plot_dict["cluster"]["structs"] = np.array(plot_dict["cluster"]["structs"])

        for key, type_dict in plot_dict.items():
            struct_comp_base_elem = type_dict["structs"]

            if len(struct_comp_base_elem) == 0:
                continue

            # The y-axis is generated at a fixed T.
            temps_K = np.ones_like(struct_comp_base_elem) + (temperature_K - 1)
            # Plotting compositions
            main_plot_ax.scatter(
                struct_comp_base_elem,
                temps_K,
                color=type_dict["color"],
                label=key,
                edgecolor="black",
                marker="o",
                s=55,
                alpha=0.7,
                linewidth=0.25,
                # zorder=10,
            )

            # Top histogram
            # now determine histogram limits by hand:
            binwidth = 1
            xymax = max(np.max(np.abs(struct_comp_base_elem)), np.max(temps_K))

            lim = (int(xymax / binwidth) + 1) * binwidth
            bins = np.arange(-lim, lim + binwidth, binwidth)

            # Top histogram
            hist_t_ax.sharex(main_plot_ax)
            hist_t_ax.hist(
                struct_comp_base_elem,
                color=type_dict["color"],
                alpha=0.5,
                bins=bins,
            )
            hist_t_ax.tick_params(axis="x", labelbottom=False)

        hist_t_ax.set_title("Database composition")
        hist_t_ax.spines["top"].set_visible(False)
        hist_t_ax.spines["right"].set_visible(False)
        main_plot_ax.set_title("")
        main_plot_ax.legend()
        fig.tight_layout()

        # Saving the figure
        fig_name = "comp_plot_" + fig_name
        chart_img_path = pl.Path(fig_path) / fig_name
        chart_img_path.with_suffix(".png")
        plt.savefig(chart_img_path, dpi=300)

        # Displaying the plot
        plt.show()


def cli_run_gen_initial_database(
    db_path,
    db_dict,
    phase_diagram_dict,
    gen_dict,
    selected_phases,
    config_dict,
):
    # If db_path is not given, the current directory is used.
    if not db_path:
        db_path = pl.Path.cwd()

    # Start logger
    log_path = pl.Path(db_path) / "logs"

    if not log_path.exists():
        log_path.mkdir(parents=True)

    ut.init_logger(source=pl.Path(__file__).stem, log_path=f"{db_path}/logs")
    ut.custom_print("Starting generation of initial database...", "info")
    print()

    # Get seed from input file or generate one
    rng_seed = int(db_dict.get("rng_seed", np.random.randint(0, 10000000)))
    ut.custom_print(f"Using seed: '{rng_seed}'.", "info")

    # Assemble phase diagram
    phase_diagram = mdb_pd.PhaseDiagram(
        material=phase_diagram_dict["material_name"],
        base_elem=phase_diagram_dict["base_element"],
        element_list=phase_diagram_dict["element_list"],
    )

    # Create phase diagram
    phases_list = []
    for curr_phase_name, phase_d in phase_diagram_dict["phase"].items():
        curr_phase = mdb_pd.Phase(
            name=curr_phase_name,
            element_list=phase_diagram_dict["element_list"],
            cluster_elem=phase_d.get("cluster_element"),
            composition=phase_d["composition"],
            prototype=phase_d["prototype"],
            offset=float(phase_d.get("offset", 0)),
            replace_dict=phase_d.get("replacements"),
            phase_diagram=phase_diagram,
        )
        phases_list.append(curr_phase)

    for phase in phases_list:
        phase_diagram.add_phase(phase)

    # Initialize the database
    structures = indb.InitialDatabase(
        database_name=db_dict["database_name"],
        database_path=db_path,
        max_num_atoms=int(db_dict["max_num_atoms"]),
        phase_diagram=phase_diagram,
        load_db=True,
    )
    ut.custom_print(structures, "info")

    read_from_db = True

    if db_dict.get("relax_struct_path"):
        # Initial structures obtained with DFT relaxation are loaded from a given path
        structures.read_base_structures(
            path=db_dict["relax_struct_path"],
            target_structures=selected_phases,
        )
    else:
        # Obtain structures from Materials Project
        structures.gather_base_structures(target_structures=phase_diagram.phases)
        read_from_db = False

    ut.custom_print("Generating structures from initial structures...", "debug")

    for phase_idx, phase in enumerate(selected_phases):
        # Line break for aesthetic purposes
        print()

        # Creating surfaces from the base structures, generating
        # different supercells and applying replacements.
        ut.custom_print(
            f"({phase_idx+1}/{len(selected_phases)}) - Current phase: {phase}", "info"
        )

        # Getting phase object
        phase = phase_diagram.get_phase(phase)

        if "bulk" in gen_dict:
            ut.custom_print("Generating bulk structures...", "info")

            # Generating bulk structures.
            structures.generate_bulk_structures(
                prototype=phase.prototype,
                phase=phase,
                num_struct=int(gen_dict["bulk"]["num_struct"]),
                num_repeats=int(gen_dict["bulk"]["num_repeat"]),
                get_different_supercells=True,
                min_num_atoms=int(db_dict["min_num_atoms"]),
                supercell_max_idx=int(gen_dict["bulk"]["supercell_max_idx"]),
                read=read_from_db,
                convert_to_base=False,
                seed=rng_seed,
            )

            ut.custom_print(structures, "info")

        if "surface" in gen_dict:
            # Generating surface structures.
            ut.custom_print("Generating surface structures...", "info")

            mdb_surf.gen_surfaces_diff_miller(
                db_obj=structures,
                phase=phase,
                min_num_atoms=int(db_dict["min_num_atoms"]),
                overwrite_max_num_atoms=int(db_dict["max_num_atoms"]),
                min_miller_index=int(gen_dict["surface"]["min_miller_index"]),
                max_miller_index=int(gen_dict["surface"]["max_miller_index"]),
                min_slab_size=float(gen_dict["surface"]["min_slab_size_ang"]),
                # num_diff_layer_size=int(gen_dict["surface"]["num_diff_layer_size"]),
                min_vacuum_size=float(gen_dict["surface"]["min_vacuum_size_ang"]),
                get_supercells=gen_dict["surface"]["get_supercells"],
                fixed_layers=int(gen_dict["surface"]["fixed_layers"]),
                limit_supercell=int(gen_dict["surface"]["max_number_supercells"]),
                save_in_db=gen_dict["surface"]["save_in_db"],
            )

            ut.custom_print(structures, "info")

        if "cluster" in gen_dict:
            raise NotImplementedError("Cluster type not implemented yet")

        # Filter small and large structures
        remove_count = structures.remove_structs_out_of_atom_count_range(
            min_num_atoms=int(db_dict["min_num_atoms"]),
            max_num_atoms=int(db_dict["max_num_atoms"]),
        )
        ut.custom_print(
            f"Removed {remove_count} structures out of atom count range.", "info"
        )
        ut.custom_print(structures, "info")

        # Lattice displacement
        if "displacement" in config_dict:
            displ_dict = config_dict["displacement"]

            ut.custom_print("Applying displacements to lattices.", "info")

            structures.perturb_min_displacement(
                frac_max=float(displ_dict["lattice_frac_displ_max"]),
                frac_min=float(displ_dict["lattice_frac_displ_min"]),
                repeat=int(displ_dict["num_repeats"]),
            )
            ut.custom_print(structures, "info")

            remove_count = structures.remove_structs_out_of_cell_size_range(
                min_cell_size=float(db_dict["min_cell_size"])
            )
            ut.custom_print(
                f"Removed {remove_count} structures out of cell size range.", "info"
            )

        if "perturbation" in config_dict:
            perturb_dict = config_dict["perturbation"]
            ut.custom_print(
                "Applying a random perturbation to the structures...",
                "info",
            )

            ut.apply_gauss_perturb_db(
                db_obj=structures,
                repeat=int(perturb_dict["num_repeats"]),
                filters=perturb_dict["filter_struct_types"],
                phase=phase,
                limit_num_structures=int(perturb_dict["limit_max_num_perturbs"]),
            )

            ut.custom_print(structures, "info")

        ut.custom_print(
            "Finishing populating structures from every phase...",
            "done",
        )
    print()
    if "adsorbates" in config_dict:
        ut.custom_print(
            "Adding adsorbates on top of the structures...",
            "info",
        )

        adsorb_dict = config_dict["adsorbates"]
        ut.add_adsorbates(
            db_obj=structures,
            repeat=int(adsorb_dict["num_repeats"]),
            filters=adsorb_dict["filter_struct_types"],
            phase=phase,
            limit_num_structures=int(adsorb_dict["limit_max_num_perturbs"]),
            adsorbate_species=adsorb_dict["adsorbate_species"],
        )

    ut.custom_print(structures, "done")
    print()

    structures.save_database(
        path=db_dict["database_path"],
    )

    # Plot the database if requested
    if db_dict.get("plot_db", {}).get("show"):
        ut.custom_print("Plotting database composition...", "info")

        # Selecting plot style settings for the plot
        params = db_dict["plot_db"].get("rc_params")

        structures.plot_database_composition(
            temperature_K=400,
            rc_params=params,
            fig_path=db_dict["database_path"],
            fig_name=db_dict["database_name"],
        )

    # Export the database if requested
    if db_dict.get("export", {}).get("export"):
        export_path = db_dict["export"].get("file_path", db_dict["database_path"])
        file_name = db_dict["export"].get("file_name", db_dict["database_name"])

        if not file_name:
            file_name = db_dict["database_name"]
        if not export_path:
            export_path = db_dict["database_path"]

        out_format = db_dict["export"].get("format")

        ut.custom_print(f"Exporting database as '{out_format}'...", "info")

        structures.export_db(
            out_format=out_format,
            file_path=export_path,
            file_name=file_name + "_structures",
        )

    # Display the database if requested
    if db_dict.get("show_db_ase", {}).get("show"):
        ut.custom_print("Displaying database in ASE...", "info")
        structures.display_db_ase()
