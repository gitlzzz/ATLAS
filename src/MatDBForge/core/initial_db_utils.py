"""
This script generates a pandas dataframe containing a set of
base (unperturbed) structures and a certain number of structures
with an applied perturbation with respect to the temperature.
"""

import json as js
import os
import pathlib
import warnings
from multiprocessing import Pool

import emmet
import numpy as np
import pandas as pd
import pymatgen.io.vasp as vasp
from dscribe.descriptors import SOAP
from mp_api.client import MPRester
from pymatgen.core.periodic_table import Element, Species
from pymatgen.core.structure import Species, Structure
from pymatgen.core.surface import generate_all_slabs
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from MatDBForge.core import utils as ut

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
        self.material = material
        self.phase_dict = {}

        for phase in phases:
            self.add_phase(phase)

    def add_phase(self, phase):
        """Add a phase to the phase diagram.

        Parameters:
            phase (Phase): The Phase object to add.

        """
        self.phases.append(phase)
        phase.phasediagram = self
        self.phase_dict[phase.name] = phase

    def get_phase(self, phase):
        if isinstance(phase, Phase):
            return self.phase_dict[phase.name]
        if isinstance(phase, str):
            phase = self.phase_dict.get(phase, None)
            if phase:
                return phase
            else:
                raise TypeError(
                    "The given phase str is not found in the phase diagram."
                )
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


class Phase:
    """Class representing a phase in a phase diagram.

    Parameters:
        name (str): The name of the phase.
        base_elem: The base element of the phase.
        base_elem_comp_max (float): The maximum composition of the base element in the phase.
        base_elem_comp_min (float): The minimum composition of the base element in the phase.
        prototype (str): The prototype of the phase.
        offset (float): The offset value of the phase.

    Attributes:
        name (str): The name of the phase.
        base_elem: The base element of the phase.
        base_elem_comp_max (float): The maximum composition of the base element in the phase.
        base_elem_comp_min (float): The minimum composition of the base element in the phase.
        prototype (str): The prototype of the phase.
        offset (float): The offset value of the phase.
        phasediagram (PhaseDiagram): The PhaseDiagram object that the phase belongs to.

    """

    def __init__(
        self,
        name: str,
        base_elem,
        base_elem_comp_max: float,
        base_elem_comp_min: float,
        prototype: str,
        offset: float = 0,
        phasediagram: "BinaryPhaseDiagram" = None,
    ):
        self.name = name
        self.base_elem = base_elem
        self.base_elem_comp_max = float(base_elem_comp_max)
        self.base_elem_comp_min = float(base_elem_comp_min)
        self.prototype = prototype
        self.offset = float(offset)
        self.phasediagram = phasediagram

        if phasediagram is not None:
            phasediagram.add_phase(self)

    def __repr__(self):
        """Return a string representation of the phase.

        Returns:
            str: The string representation of the phase.

        """
        repr_string = (
            f"{self.name}: {self.base_elem_comp_min*100:.1f}% {self.base_elem} - "
            f"{self.base_elem_comp_max*100:.1f}% {self.base_elem}"
        )

        if self.phasediagram is not None:
            repr_string += f" (belongs to {self.phasediagram.material})"

        return repr_string


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


    Notes
    -----
        The json file should have the following structure:

        {
            "API_KEY": "XXXXXX"
        }


    """

    # Boltzmann constant in J/(Da*K)
    kB = 8.314

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
                "material_id",
                "structure",
                "formula",
                "symmetry",
                "perturb",
                "temperature",
                "base",
                "phase",
                "magnetic_properties",
                "energy_per_atom",
                "surface",
            ]
        )

        # df = df.astype({"perturb": bool, "base": bool})

        ut.custom_print(f"Created database '{self.database_name}'.", "done")

        return df

    def _find_supercell_indices(
        self,
        structure,
        get_different_supercells,
        max_atoms,
    ):
        # Initial supercell size
        idx = 5

        # Copying structure
        new_structure = structure.copy(sanitize=True)

        # Creating initial supercell
        new_structure.make_supercell([idx, idx, idx], to_unit_cell=False)

        # Number of atoms of the supercell
        struct_size = len(new_structure.species)
        while struct_size > max_atoms:
            new_structure = structure.copy(sanitize=True)
            idx -= 1
            new_structure.make_supercell([idx, idx, idx], to_unit_cell=False)
            struct_size = len(new_structure.species)

        structure_list = []
        idx_list = []
        structure_list.append(new_structure)
        idx_list.append(idx)

        ut.custom_print(
            f"Supercell generated - total atoms: {len(new_structure.species)}",
            "debug",
        )

        if get_different_supercells:
            for idx_smaller in range(idx - 1, 0, -1):
                new_structure = structure.copy(sanitize=True)
                new_structure.make_supercell(
                    [idx_smaller, idx_smaller, idx_smaller], to_unit_cell=False
                )
                structure_list.append(new_structure)
                idx_list.append(idx_smaller)

                ut.custom_print(
                    f"Supercell generated - total atoms: {len(new_structure.species)}",
                    "debug",
                )

        return structure_list, idx_list

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
        phase_list = self.df.phase.unique()

        species = ["Cu", "Zn"]
        r_cut = 6
        n_max = 8
        l_max = 6

        tot_unique_name_list = []
        df_total_struct_list = len(self.df.structure.values)

        for curr_phase in phase_list:
            # Setting up the SOAP descriptor
            soap = SOAP(
                species=species,
                periodic=True,
                r_cut=r_cut,
                n_max=n_max,
                l_max=l_max,
                # average="inner",
                sparse=False,
            )

            ut.custom_print(
                f"Checking for repeated structures for phase '{curr_phase}'...",
                "info",
            )

            soap_structs = []

            structure_list = self.df.loc[self.df.phase == curr_phase].structure.values
            name_list = self.df.loc[self.df.phase == curr_phase].material_id.values

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
            base_struct_names = list(set(self.df.loc[self.df.base].material_id.values))
            unique_structs.extend(base_struct_names)

            unique_structures_df = self.df[self.df.material_id.isin(unique_structs)]

            unique_structures_df_drop = unique_structures_df.drop_duplicates(
                subset=["material_id"],
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
                new_row = pd.Series(
                    {
                        "material_id": str(material.material_id),
                        "structure": material.structure,
                        "temperature": np.nan,
                        "perturb": False,
                        "formula": material.composition_reduced,
                        "symmetry": material.symmetry,
                        "base": True,
                        "surface": False,
                        "phase": np.nan,
                        "magnetic_properties": {
                            "is_magnetic": material.is_magnetic,
                            "ordering": material.ordering,
                            "total_magnetization": material.total_magnetization,
                            "total_magnetization_normalized_vol": material.total_magnetization_normalized_vol,
                            "total_magnetization_normalized_formula_units": material.total_magnetization_normalized_formula_units,
                            "num_magnetic_sites": material.num_magnetic_sites,
                            "num_unique_magnetic_sites": material.num_unique_magnetic_sites,
                            "types_of_magnetic_species": material.types_of_magnetic_species,
                        },
                        "energy_per_atom": material.energy_per_atom,
                    }
                )
                new_row = new_row.to_frame().T.astype({"perturb": bool, "base": bool, "surface":bool})

                self.df = self.df.astype({"perturb": bool, "base": bool, "surface": bool})

                self.df = pd.concat([self.df, new_row], ignore_index=True)

        self.df.set_index("material_id", inplace=True, drop=False)

    def read_base_structures(self, path: str = "./", target_structures=None):
        ut.custom_print("Reading relaxed structures...")
        # Getting the path where the calculations will be searched for.
        read_path = pathlib.Path(path)

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

            # Storing the energy per atom
            num_atom_struct = curr_run.ionic_steps[0]["structure"].num_sites
            energy_per_atom = curr_run.final_energy / num_atom_struct

            new_row = pd.Series(
                {
                    "material_id": CuZnInitialDatabase.CUZN_PHASES.get_phase(
                        curr_phase
                    ).prototype,
                    "structure": curr_run.final_structure,
                    "temperature": np.nan,
                    "perturb": False,
                    "formula": curr_run.final_structure.formula,
                    "symmetry": curr_run.final_structure.get_space_group_info(),
                    "base": True,
                    "surface": False,
                    "phase": curr_phase,
                    "magnetic_properties": curr_run.projected_magnetisation,
                    "energy_per_atom": energy_per_atom,
                }
            )
            new_row = new_row.to_frame().T.astype({"perturb": bool, "base": bool, "surface": bool})

            self.df = self.df.astype({"perturb": bool, "base": bool, "surface": bool})
            self.df = pd.concat([self.df, new_row], ignore_index=True)

    def save_database(self, path: str = None, suffix: str = None):
        """
        Saves the initial database dataframe into a pkl object

        Parameters
        ----------
        path : str, optional
            Location where the pickle object will be saved,
            by default None, which defaults to storing the file in the CWD.
        """
        if suffix:
            filename = self.database_name + f"_{suffix}.pkl"
        else:
            filename = self.database_name + ".pkl"

        if not path:
            path = ""

        file_path = pathlib.Path(path, filename)

        self.df.to_pickle(path=file_path)
        ut.custom_print(f"Database saved in {file_path}", "warn")

    def perturb_gauss(self, center: float = 0.04, repeat=5):
        # Getting all structures which are not perturbed
        target_entries = self.df.loc[
            (~self.df.material_id.str.contains("_perturb"))
            & (self.df.material_id.str.contains("_super"))
        ]

        # Applying displacement to all perturbed structures
        for idx, entry in target_entries.iterrows():
            str_name = entry.material_id
            str_phase = entry.phase
            curr_str = entry.structure

            # print('curr_str: ', curr_str.cart_coords[0])

            for perturb_repeat_idx in range(repeat):
                # Applying displacement
                new_struct_perturb = self._apply_gauss_perturb(
                    center=center, structure=curr_str
                )
                # print('new_struct_perturb: ', new_struct_perturb.cart_coords[0])
                # new_row = entry.copy()

                new_row = pd.Series(
                    {
                        "material_id": f"{str_name}_perturb_gauss_{perturb_repeat_idx+1}",
                        "structure": new_struct_perturb,
                        "temperature": None,
                        "perturb": True,
                        "phase": str_phase,
                        "base": False,
                        "surface": False,
                        "formula": new_struct_perturb.formula,
                    }
                )

                new_row_df = new_row.to_frame().T
                new_row_df = new_row_df.astype(
                    {"perturb": bool, "base": bool, "surface": bool}
                )

                self.df = pd.concat([self.df, new_row_df], ignore_index=True)

    def _apply_gauss_perturb(self, structure: Structure, center: float = 0.04):
        new_structure = structure.copy()

        new_structure.perturb(distance=0.08, min_distance=0.02)
        return new_structure


class CuZnInitialDatabase(InitialDatabase):
    """
    Object representing a initial database for a CuZn alloy intented
    to prepare a NNP. The database is stored as a pandas dataframe.
    Contains methods related to gathering, preparing and modifying
    the initial database.

    Returns
    -------
    CuZnInitialDatabase
        Object containing the database and methods

    Raises
    ------
    KeyError
        This error is raised when a wrong phase is given.
    """

    # Phase diagram
    # Approx for Diagram at 300K?
    # alpha_phase = Phase("alpha", "Cu", 1, 0.6105, "mp-30", 0.03)
    # m1 = Phase("m1", "Cu", 0.627, 0.533, "mp-30", 0.03)
    # beta_prime = Phase("beta-prime", "Cu", 0.545, 0.493, "mp-987", 0.05)
    # m2 = Phase("m2", "Zn", 0.51, 0.404, "mp-987", 0.05)
    # gamma = Phase("gamma", "Zn", 0.423, 0.294, "mp-1368", 0.03)
    # m3 = Phase("m3", "Zn", 0.33, 0.208, "mp-1216020", 0.05)
    # delta = Phase("delta", "Zn", 1, 0.6105, "mp-1215518", 0.05)
    # epsilon = Phase("epsilon", "Zn", 0.215, 0.117, "mp-972042", 0.05)
    # m4 = Phase("m4", "Zn", 0.135, 0.018, "mp-79", 0.01)
    # eta = Phase("eta", "Zn", 0.0275, 0, "mp-79", 0.01)

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
        base_elem_comp_min=0.507,
        base_elem_comp_max=0.455,
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

    ALLOY_SET = {"Cu", "Zn"}

    def __init__(self, database_name, use_offset=True, **kwargs):
        super().__init__(database_name, **kwargs)

        if use_offset:
            ut.custom_print(
                "Using an offset for computing the phases concentrations.", "info"
            )
            self.use_offset = use_offset

    def gather_base_structures(self, target_structures):
        # Gathering base structures using the method from the superclass
        super().gather_base_structures(target_structures)

        # Assigning assumed phase to the base structure
        for idx in self.df["material_id"]:
            curr_phase = self._get_phase_from_id(idx)
            self.df.at[idx, "phase"] = curr_phase

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

        # Getting which key (corresponding to a phase) contains the
        # given materials project id
        # phase_list = [
        #     key
        #     for key in self.CUZN_PHASES.keys()
        #     if self.CUZN_PHASES[key].get("prototype") == idx
        # ]
        # print('phase_list: ', phase_list)

        phase_list = [
            phase.name for phase in self.CUZN_PHASES.phases if phase.prototype == idx
        ]
        # print('phase_list: ', phase_list)
        # quit()

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

            # Replacing Cu atoms in the structures
            for ind in range(len(structure.species)):
                new_structure.replace(
                    ind,
                    Species(self.CUZN_PHASES.get_phase(phase).base_elem),
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
        phase = phase.name.lower()
        if not self.CUZN_PHASES.get_phase(phase):
            raise KeyError(
                "Wrong phase given. "
                f"Please introduce one of: {[k for k in self.CUZN_PHASES.phases]}"
            )

        # Reading structure from database
        if read:
            ut.custom_print("Using structure from the db as template...", "debug")
            query_result = self.df.loc[self.df.phase == phase]
            material_id_prefix = query_result.material_id.values[0]
            structure = self.df.loc[self.df.phase == phase].structure.values[0]

        # Querying CuZn alpha prototype structure
        else:
            ut.custom_print("Querying the MP API...", "debug")
            with MPRester(self.secrets["API_KEY"]) as mpr:
                query_result = mpr.summary.search(material_ids=[prototype])[0]
                structure = query_result.structure
                material_id_prefix = query_result.material_id

        # Converting atoms from prototype cell to Cu if necessary
        structure = self._convert_prototype_structure(structure=structure, phase=phase)

        # Getting conventional cell
        sga = SpacegroupAnalyzer(structure)
        structure = sga.get_conventional_standard_structure()

        # Create supercell.
        # This supercell will result in a Cu64 cell
        # This can return either 1 or more supercells of the
        # same structure, depending on the 'get_different_supercells' flag.
        structure_list, idx_list = self._find_supercell_indices(
            structure, get_different_supercells, max_atoms=self.max_num_atoms
        )

        for structure, idx in zip(structure_list, idx_list):
            new_row = pd.Series(
                {
                    "material_id": f"{material_id_prefix}_{phase}_super-{idx}",
                    "structure": structure,
                    "temperature": np.nan,
                    "perturb": False,
                    "surface": False,
                    "phase": phase,
                    "base": False,
                }
            )

            new_row_df = new_row.to_frame().T
            new_row_df = new_row_df.astype(
                {"perturb": bool, "base": bool, "surface": bool}
            )
            self.df = self.df.astype(
                {"perturb": bool, "base": bool, "surface": bool, "surface": bool}
            )

            self.df = pd.concat([self.df, new_row_df], ignore_index=True)

            # with pd.option_context(
            #     "display.max_rows", None, "display.max_columns", None
            # ):
            #     print(self.df)

        return structure_list, query_result, idx_list

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
        # itertools.
        ind = 2
        sum_ind = 0
        sum_list = (2, 1, 2, 3)

        while ind < structure.num_sites:
            new_structure.replace(ind - 1, Species(base_atom_set[0]))
            ind = ind + sum_list[sum_ind]

            if sum_ind == 3:
                sum_ind = 0
            else:
                sum_ind += 1

        if read:
            material_id_prefix = query_result.material_id.values[0]
        else:
            material_id_prefix = query_result.material_id

        new_row = pd.Series(
            {
                "material_id": f"{material_id_prefix}_{phase.name}_symm",
                "structure": new_structure,
                "temperature": np.nan,
                "perturb": True,
                "base": False,
                "phase": phase.name,
            }
        )
        new_row_df = new_row.to_frame().T
        new_row_df = new_row_df.astype({"perturb": bool, "base": bool, "surface": bool})
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)
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

        for structure, supr_idx in zip(structure_list, idx_list):
            # Replacing some atoms using symmetry
            structure = self._create_symmetrical_prototype(
                structure=structure, phase=phase, query_result=query_result, read=read
            )
            structure_len = len(structure.species)

            # Randomly generating base elem percentages for the new structures
            subst_base_elem_perc = self._gen_base_elem_perc(phase, num_struct)
            # print('phase: ', phase)

            # Choosing the amount of atoms to replace with the base element in the
            # struct, which at this point will be completely replaced by atoms
            # of the remaining species of the alloy.
            n_at_replacement = [
                int(round(structure_len * stct, 0)) for stct in subst_base_elem_perc
            ]

            # Attempting to fix any percentages outside of the
            # current phase ratios.
            offset_min = phase.base_elem_comp_min - phase.offset
            offset_max = phase.base_elem_comp_max + phase.offset
            n_at_replacement_upd = []

            for str_ind, n_at in enumerate(n_at_replacement):
                inPhase = False
                while not inPhase:
                    perc = n_at / structure_len
                    if perc > offset_max:
                        n_at -= 1
                    elif perc < offset_min:
                        n_at += 1
                    else:
                        inPhase = True
                # print('perc: ', perc)

                n_at_replacement_upd.append(n_at)

            # Replacing the atoms and generate 'num_replacements'
            # structures for each percentage
            rng = np.random.default_rng()
            for str_ind, n_atoms in enumerate(n_at_replacement_upd):
                for repl in range(num_repeats):
                    # Getting which indices to replace
                    idx_replace = rng.choice(
                        a=structure_len,
                        size=(n_atoms,),
                        replace=False,
                        shuffle=True,
                    )

                    # Creating a new structure using the base one as a template
                    new_structure = structure.copy(sanitize=True)

                    # Getting which species to replace with
                    # curr_main = self.CUZN_PHASES.get_phase(phase).base_elem
                    # repl_spec = Species(list(self.ALLOY_SET - {curr_main})[0])
                    repl_spec = Species(phase.base_elem)
                    other_spec = Species(list(self.ALLOY_SET - {repl_spec})[0])

                    # Replacing everything with the opposite atom.
                    for atom in range(structure_len):
                        new_structure.replace(atom, other_spec)

                    # Replacing atoms in the structures
                    for ind in idx_replace:
                        new_structure.replace(ind, repl_spec)

                    # print(f"new structure formula: {new_structure.formula}")
                    # print(f"{prototype}_{phase.name}_super-{supr_idx}_{str_ind+1}")
                    # print(f"{len(idx_replace)}/{structure_len}")

                    self._save_row(
                        material_id=f"{prototype}_{phase.name}_super-{supr_idx}_{str_ind+1}_{repl+1}",
                        phase=phase,
                        structure=new_structure,
                    )

    def generate_surfaces_pure(
        self,
        phase: Phase,
        num_struct: int,
        num_repeats: int,
        read: bool = True,
        max_miller_index: int = 5,
        min_slab_size: float = 10,
        min_vacuum_size: float = 10,
    ):
        if isinstance(phase, str):
            phase = CuZnInitialDatabase.CUZN_PHASES.get_phase(phase)

        # Getting the base structures
        base_structs = self.df.loc[self.df.base]
        base_structs = base_structs.loc[self.df.phase == phase.name]

        for idx, row in base_structs.iterrows():
            curr_surface = row.structure

            # Generating surfaces
            slabs = generate_all_slabs(
                structure=curr_surface,
                max_index=max_miller_index,
                min_slab_size=min_slab_size,  # in A
                min_vacuum_size=min_vacuum_size,  # in A
            )
            # from pymatgen.io.vasp import Poscar
            # for cnt, slab in enumerate(slabs):
            # Poscar(slab).write_file(f'/tmp/surfaces_test/{cnt}_slab.poscar')
            # quit()

            prototype = phase.prototype
            extra = {"surface": True}
            for idx, slab in enumerate(slabs):
                curr_miller = str(slab.miller_index).replace(",", "").replace(" ", "")
                material_id = (
                    f"{prototype}_{phase.name}_pure_surface-{curr_miller}-{idx+1}"
                )
                self._save_row(
                    material_id=material_id, phase=phase, structure=slab, extra=extra
                )

    def generate_surfaces_replacements(
        self,
        phase: Phase,
        num_struct: int,
        num_repeats: int,
        get_different_supercells: bool,
        read: bool = True,
        max_miller_index: int = 5,
    ):
        """
        This method allows to create several variations of a certain surface
        structure by randomly replacing atoms in the base structure.

        Parameters
        ----------
        phase : Phase|str
            Phase or name of the phase to be used.
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
        max_miller_index: int
            Max miller index to reach.

        Raises
        ------
        KeyError
            Raised if the given phase is not found. All of the available phases
            are given on the self.CUZN_PHASES dictionary. More phases could be added
            there if necessary.
        """

        ...

    def _save_row(
        self,
        material_id,
        phase,
        structure,
        extra,
    ):
        new_row = pd.Series(
            {
                # f"{prototype}_{phase.name}_super-"{supr_idx}_{str_ind+1}_{repl+1}",
                "material_id": material_id,
                "structure": structure,
                "temperature": None,
                "perturb": True,
                "phase": phase.name,
                "base": False,
                "formula": structure.formula,
                **extra,
            }
        )

        new_row_df = new_row.to_frame().T
        new_row_df = new_row_df.astype({"perturb": bool, "base": bool, "surface": bool})

        self.df = self.df.astype({"perturb": bool, "base": bool, "surface": bool})
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)

        # TODO: Converting NaN values given by some functions to None
        # self.df.fillna(None, method=None, inplace=True)

    def perturb_min_displacement(
        self, frac_max: float = 0.05, frac_min: float = 0.01, repeat=5
    ):
        # Getting all relaxed structures
        target_entries = self.df.loc[self.df.base]

        # Applying displacement to all perturbed structures
        for idx, entry in target_entries.iterrows():
            str_name = entry.material_id
            str_phase = entry.phase
            curr_str = entry.structure

            for perturb_repeat_idx in range(repeat):
                # Applying displacement
                new_struct_perturb = self._apply_min_perturbation(structure=curr_str)

                new_row = pd.Series(
                    {
                        "material_id": f"{str_name}_{str_phase}_perturb_min_{perturb_repeat_idx+1}",
                        "structure": new_struct_perturb,
                        "temperature": None,
                        "perturb": True,
                        "phase": str_phase,
                        "base": False,
                        "formula": new_struct_perturb.formula,
                    }
                )

                new_row_df = new_row.to_frame().T
                new_row_df = new_row_df.astype({"perturb": bool, "base": bool, "surface":bool})

                self.df = pd.concat([self.df, new_row_df], ignore_index=True)

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
            ), f"Total count does not match. tot_cu: {tot_cu}, tot_zn: {tot_zn}, total: {tot_atoms}. Species: {set(strct.species)}"

            perc = tot_zn / tot_atoms

            offset_min = phase.base_elem_comp_min - phase.offset
            offset_max = phase.base_elem_comp_max + phase.offset

            # Checking if the current structure is between the phase ratio
            # percentages.
            if not (offset_min <= perc <= offset_max):
                ut.custom_print(
                    f"{name}: {perc:.2f} Zn not between {offset_min:.2f} - {offset_max:.2f}",
                    "error",
                )
