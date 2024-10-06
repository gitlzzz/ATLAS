"""Module for the Structure class."""

import inspect  # noqa
import uuid

import pandas as pd
import pymatgen.io.vasp as vasp
from pymatgen.core.units import Energy
from MatDBForge.core.code_utils import deprecated  # noqa

from MatDBForge.core import initial_db as mdb_indb


class Structure:
    """
    Wrapper around a pymatgen structure object with extra fields for MDBForge.

    Parameters
    ----------
    structure : pymatgen.core.Structure, optional
        A pymatgen Structure object representing the atomic structure, by default None
    material_name : str, optional
        The name of the material, by default None
    material_id : str or int, optional
        The unique identifier for the material, by default None
    phase : str, optional
        The phase of the material (e.g., solid, liquid, gas), by default None
    base : bool, optional
        Flag indicating if this is a base structure, by default None
    perturb : bool, optional
        Flag indicating if the structure should be perturbed, by default None
    supercell : tuple of int, optional
        The dimensions of the supercell, specified as a tuple of integers
        (e.g., (2, 2, 2)), by default None
    surface : bool, optional
        Flag indicating if this is a surface structure, by default False
    bulk : bool, optional
        Flag indicating if this is a bulk structure, by default False
    cluster : bool, optional
        Flag indicating if this is a cluster structure, by default False
    vacancy: bool, optionsl
        Flag indicating if the structure contains a vacancy, by default False
    formula : str, optional
        The chemical formula of the material, by default None
    replacement : bool, optional
        Flag indicating if an atomic replacement should be performed, by default False
    replacement_ind : int, optional
        The index of the atom to be replaced, by default None
    symmetry : str, optional
        The symmetry information of the structure, by default None
    energy_per_atom : float, optional
        The energy per atom of the structure, by default None
    temperature : float, optional
        The temperature at which the properties are calculated, by default None
    magnetic_properties : dict, optional
        A dictionary containing magnetic properties of the structure, by default None
    calc_energy_per_atom : float, optional
        The calculated energy per atom from a computational method, by default None
    calc_energy_toten : float, optional
        The total energy from a computational method, by default None
    calc_energy : float, optional
        The calculated energy of the structure, by default None
    calc_performed : bool, optional
        Flag indicating if a calculation has been performed, by default False
    calc_type : str, optional
        The type of calculation performed (e.g., DFT, MD), by default None
    calc_output : dict, optional
        The output of the calculation, by default None
    surface_miller : tuple of int, optional
        The Miller indices of the surface, specified as a tuple of integers,
        by default None
    """

    def __init__(
        self,
        structure=None,
        material_name: str = None,
        material_id=None,
        phase=None,
        base: bool = None,
        perturb: bool = None,
        displacement: bool = None,
        supercell=None,
        surface: bool = False,
        bulk: bool = False,
        cluster: bool = False,
        formula=None,
        replacement: bool = False,
        replacement_ind=None,
        vacancy: bool = False,
        symmetry=None,
        energy_per_atom=None,
        temperature: float = None,
        magnetic_properties=None,
        calc_energy_per_atom=None,
        calc_energy_toten=None,
        calc_energy=None,
        calc_performed=False,
        calc_type=None,
        calc_output=None,
        surface_miller=None,
        targeted_modification: str = None,
        al_loop_step: int = 0,
        unique_id=None,
    ):
        if unique_id:
            self.unique_id = unique_id
        else:
            self.unique_id = uuid.uuid4()
        self.material_name = material_name
        self.structure = structure
        self.material_id = material_id
        self.phase = phase
        self.base = base
        self.perturb = perturb
        self.supercell = supercell
        self.displacement = displacement
        self.surface = surface
        self.surface_miller = surface_miller
        self.bulk = bulk
        self.replacement = replacement
        self.replacement_ind = replacement_ind
        self.cluster = cluster

        # This should be the material's project db energies
        self.energy_per_atom = energy_per_atom

        if not formula and structure:
            formula = structure.formula
        self.formula = formula

        self.symmetry = symmetry
        self.temperature = temperature
        self.magnetic_properties = magnetic_properties

        # Everything prefixed with calc_ should come from a DFT
        # calculation, and not the materials project db
        self.calc_energy_per_atom = calc_energy_per_atom
        self.calc_energy_toten = calc_energy_toten
        self.calc_energy = calc_energy
        self.calc_performed = calc_performed
        self.calc_type = calc_type
        self.calc_output = calc_output
        self.vacancy = vacancy
        self.targeted_modification = targeted_modification
        self.al_loop_step = al_loop_step

    def to_bulk(self):
        # Create a Bulk instance by passing the current object's attributes
        attributes = {
            name: value
            for name, value in inspect.getmembers(self)
            if not inspect.isroutine(value) and not name.startswith("__")
        }
        return Bulk(**attributes)

    def to_surface(self):
        # Create a surface instance by passing the current object's attributes
        attributes = {
            name: value
            for name, value in inspect.getmembers(self)
            if not inspect.isroutine(value) and not name.startswith("__")
        }
        return Surface(**attributes)

    def to_cluster(self):
        # Create a cluster instance by passing the current object's attributes
        attributes = {
            name: value
            for name, value in inspect.getmembers(self)
            if not inspect.isroutine(value) and not name.startswith("__")
        }
        return Cluster(**attributes)

    def from_vasprun(
        self,
        vasprun: vasp.Vasprun,
        **kwargs,
    ):
        # Getting the structure
        structure = vasprun.structures[-1]

        # Getting energy information
        energy = vasprun.final_energy
        energy_toten = Energy(
            float(vasprun.ionic_steps[-1]["e_fr_energy"]),
            "eV",
        )
        energy_per_atom = energy / len(structure.species)

        # Getting the temperature from the vasp parameters
        # If the temperature is not set, the vasprun shows 0.0001K as T.
        # I round to the third decimal place so this value then equals to 0.
        if float(vasprun.parameters["TEBEG"]) <= 1e-4:
            temperature = 0
        else:
            temperature = float(vasprun.parameters["TEBEG"])

        generated_structure = Structure(
            energy_per_atom=None,
            material_name=kwargs.get("material_name"),
            structure=structure,
            material_id=kwargs.get("material_id"),
            phase=kwargs.get("phase"),
            base=kwargs.get("base"),
            perturb=kwargs.get("perturb"),
            supercell=kwargs.get("supercell"),
            surface=kwargs.get("surface"),
            bulk=kwargs.get("bulk"),
            cluster=kwargs.get("cluster"),
            replacement=kwargs.get("replacement"),
            formula=structure.formula,
            symmetry=structure.get_space_group_info(),
            temperature=temperature,
            magnetic_properties=vasprun.projected_magnetisation,
            calc_energy=energy,
            calc_energy_per_atom=energy_per_atom,
            calc_energy_toten=energy_toten,
            calc_performed=True,
            calc_type=vasprun.run_type,
            calc_output=vasprun,
        )

        if not generated_structure.replacement:
            generated_structure.replacement = False

        return generated_structure

    def __repr__(self):
        repr_str = ""
        spc = " " * 2

        # Gathering name information
        if self.material_name:
            repr_str += f"MatDBForge {self.__class__.__name__}: {self.material_name}\n"
        else:
            repr_str += f"MatDBForge {self.__class__.__name__}: (no name)\n"

        repr_str += f"{spc}ID: {self.unique_id}\n"

        # Gathing formula and phase
        if self.formula:
            repr_str += f"{spc}Formula: {self.formula}\n"
        if self.phase:
            repr_str += f"{spc}{self.phase}\n"
        else:
            repr_str += f"{spc}Phase: unknown phase\n"

        repr_str += f"{spc}Status flags: "
        # Gathering if the structure is a base or structure phase
        props = []
        if self.base:
            props.append("base")
        elif self.perturb:
            props.append("+atom_positions_perturbed")
        elif self.displacement:
            props.append("+lattice_displaced")
        elif self.vacancy:
            props.append("+vacancies")

        # Gathering the type of structure
        if self.bulk:
            props.append("bulk")
        elif self.surface:
            props.append("surface")
        elif self.cluster:
            props.append("cluster")

        repr_str += " ".join(props)

        # Gathering DFT data
        if self.calc_performed:
            repr_str += "\n"
            repr_str += f"Obtained with DFT {self.calc_type} calculation:\n"
            repr_str += f"\t - Energy {self.calc_energy}\n"
            repr_str += f"\t - Free energy {self.calc_energy_toten}\n"
            repr_str += f"\t - Energy per atom: {self.calc_energy_per_atom}"

        return repr_str

    # def __dict__(self):
    #     dict_obj = {}
    #     att_names = [att for att in dir(self) if not att.startswith('__')]
    #     for att in att_names:
    #         dict_obj[att] = self.getattr(att)
    #     return dict_obj

    def save_to_db(self, db_obj):
        new_row = pd.Series(
            {
                "material_id": str(self.material_id),
                "structure": self.structure,
                "temperature": self.temperature,
                "perturb": self.perturb,
                "formula": self.formula,
                "symmetry": self.symmetry,
                "base": self.base,
                "surface": self.surface,
                "surface_miller": self.surface_miller,
                "phase": self.phase,
                "magnetic_properties": self.magnetic_properties,
                "energy_per_atom": None,
                "unique_id": self.unique_id,
                "material_name": self.material_name,
                "replacement": self.replacement,
                "replacement_ind": self.replacement_ind,
                "supercell": self.supercell,
                "bulk": self.bulk,
                "cluster": self.cluster,
                "calc_energy": self.calc_energy,
                "calc_energy_per_atom": self.calc_energy_per_atom,
                "calc_energy_toten": self.calc_energy_toten,
                "calc_performed": self.calc_performed,
                "calc_type": self.calc_type,
                "calc_output": self.calc_output,
                "vacancy": self.vacancy,
                "targeted_modification": self.targeted_modification,
                "displacement": self.displacement,
                "al_loop_step": self.al_loop_step,
            }
        )
        bool_columns = {
            "perturb": bool,
            "displacement": bool,
            "base": bool,
            "bulk": bool,
            "surface": bool,
            "cluster": bool,
            "calc_performed": bool,
            "replacement": bool,
            "vacancy": bool,
        }
        new_row = new_row.to_frame().T.astype(bool_columns)

        is_InitialDatabase = isinstance(db_obj, mdb_indb.InitialDatabase)

        struct_df = db_obj.df if is_InitialDatabase else db_obj

        with pd.option_context("future.no_silent_downcasting", True):
            struct_df = struct_df.fillna(value=False).infer_objects(copy=False)
        struct_df = struct_df.astype(bool_columns)

        if struct_df.shape[0] == 0:
            struct_df = new_row
        else:
            struct_df = pd.concat([struct_df, new_row], ignore_index=True)

        if is_InitialDatabase:
            db_obj.df = struct_df
        else:
            db_obj = struct_df

        return db_obj

    @deprecated(reason="Use to_bulk/to_surface/to_cluster instead.", since_ver="0.6.2")
    def from_mdb_structure(
        self,
        mdb_structure,
        new_structure=None,
        material_name=None,
    ):
        """
        Convert a structure from the Structure to a different type.

        Note
        ----------
        This function is deprecated and will be removed in a future version.

        Use `to_bulk`/`to_surface`/`to_cluster` instead.

        Deprecated since version `0.6.2`.


        Parameters
        ----------
        mdb_structure : Structure
            The structure to be converted.
        new_structure : Structure, optional
            Either do the conversion in place or use a new structure, by default None.
        material_name : str, optional
            New name to give to the structure, by default None.

        Returns
        -------
        Bulk, Surface, Cluster
            The converted structure.
        """
        if material_name:
            self.material_name = material_name
        else:
            self.material_name = mdb_structure.material_name

        if new_structure:
            self.structure = new_structure
        else:
            self.structure = mdb_structure.structure

        self.material_id = mdb_structure.material_id
        self.phase = mdb_structure.phase
        self.base = mdb_structure.base
        self.perturb = mdb_structure.perturb
        self.vacancy = mdb_structure.vacancy
        self.supercell = mdb_structure.supercell
        self.surface = mdb_structure.surface
        self.cluster = mdb_structure.cluster
        self.formula = mdb_structure.formula
        self.symmetry = mdb_structure.symmetry
        self.temperature = mdb_structure.temperature
        self.replacement = mdb_structure.replacement
        self.replacement_ind = mdb_structure.replacement_ind
        self.magnetic_properties = mdb_structure.magnetic_properties
        self.calc_energy = mdb_structure.calc_energy
        self.calc_energy_per_atom = mdb_structure.calc_energy_per_atom
        self.calc_energy_toten = mdb_structure.calc_energy_toten
        self.calc_performed = mdb_structure.calc_performed
        self.calc_type = mdb_structure.calc_type
        self.calc_output = mdb_structure.calc_output
        self.energy_per_atom = mdb_structure.energy_per_atom

        if mdb_structure.bulk:
            self.bulk = True
            self.surface = False
            self.cluster = False
        elif mdb_structure.surface:
            self.surface_miller = mdb_structure.surface_miller
            self.surface = True
            self.bulk = False
            self.cluster = False
        elif mdb_structure.cluster:
            self.cluster = True
            self.bulk = False
            self.surface = False

        return self


class Bulk(Structure):
    """Class for bulk structures."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Setting the bulk property as True.
        self.bulk = True
        self.surface = False
        self.cluster = False


class Surface(Structure):
    """Class for slab structures."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Setting the surface property as True.
        self.surface = True
        self.cluster = False
        self.bulk = False


class Cluster(Structure):
    """Class for cluster structures."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Setting the cluster property as True.
        self.cluster = True
        self.surface = False
        self.bulk = False
