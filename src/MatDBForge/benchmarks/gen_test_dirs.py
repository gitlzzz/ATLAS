import copy
import os
import pathlib
import uuid

import ase.data as ad
import numpy as np
import pandas as pd
import pymatgen.io.vasp as vasp
import initial_db_p2.target_structures.initial_db_utils as indb
from iciq_utils import custom_print
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import DictSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

VDW_DATA_PATH = "/home/psanz/Documents/aiida_scripts/tim_workchain_vasp/relax_workchain/aiida-workchains/aiida_tim/data/vdw-data"

INFO_DICT = indb.CuZnInitialDatabase.CUZN_PHASES.keys()


calc_types = [
    "alpha",
    # "m1",
    "beta-prime",
    # "m2",
    "gamma",
    "m3",
    "epsilon",
    # "m4",
    "eta",
]

df_columns = [
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
]

PHASES = [
    "alpha",
    # "m1",
    "beta-prime",
    # "m2",
    "gamma",
    "m3",
    "epsilon",
    # "m4",
    "eta",
]

INCAR = {
    ## general:
    "istart": 0,
    "icharg": 2,
    "gga": "PE",
    "ispin": 1,
    "lorbit": 11,
    ## electronic steps:
    # "prec": "NORMAL",
    "encut": 600,
    "KSPACING": 0.1256637,
    "ediff": 1e-6,
    "ismear": 0,
    "sigma": 0.03,
    "algo": "Fast",
    "lreal": "Auto",
    "nelm": 100,
    ## ionic steps:
    "ibrion": 2,
    "nsw": 350,
    "ediffg": -0.03,
    "isif": 3,
    "potim": 0.3,
    # "nfree": 3,
    ## files to write:
    "lwave": False,
    "lcharg": False,
    # "nwrite": 0,
    ## parallelization:
    # "npar": 4,
    "ncore": 4,
    "kpar": 4,
    ## dipole correction
    # "ldipol": True,
    # "idipol": 3,
    "lelf": False,
    # "lvhar": False,
    # "lvtot": False,
    # "ivdw": 1,
    ## van der Waals:
    "lvdw": True,
    # "vdw_version": 2,
    "vdw_radius": 40,
    "vdw_scaling": 0.75,
    # "maxmix": -45,
    # "nelmdl": -7,
    # "nelmin": 5,
    # "parchg": False,
}

INCAR_SP = {
    ## general:
    "istart": 0,
    "icharg": 2,
    "gga": "PE",
    "ispin": 1,
    "lorbit": 11,
    ## electronic steps:
    # "prec": "NORMAL",
    "encut": 450,
    "ediff": 1e-6,
    "ismear": 0,
    "sigma": 0.03,
    "algo": "Fast",
    "lreal": "Auto",
    "nelm": 100,
    ## ionic steps:
    "ibrion": -1,
    "nsw": 3,
    "ediffg": -0.03,
    "isif": 2,
    "potim": 0.3,
    # "nfree": 3,
    ## files to write:
    "lwave": False,
    "lcharg": False,
    # "nwrite": 0,
    ## parallelization:
    # "npar": 4,
    "ncore": 4,
    "kpar": 4,
    ## dipole correction
    # "ldipol": True,
    # "idipol": 3,
    "lelf": False,
    # "lvhar": False,
    # "lvtot": False,
    "ivdw": 11,
    ## van der Waals:
    # "vdw_version": 2,
    # "vdw_radius": 40,
    # "vdw_scaling": 0.75,
    # "maxmix": -45,
    # "nelmdl": -7,
    # "nelmin": 5,
    # "parchg": False,
}


k_point_range = range(-1, 25)
# print(list(k_point_range))
# quit()
encut_range = range(450, 600, 50)
ispin_values = [1, 2]

DICT_TESTS = {
    "kpoints": {"values": k_point_range},
    # "encut": {"values": encut_range},
    # "ispin": {"values": ispin_values},
}


def read_df(path):
    base_structs = pd.read_pickle(path)

    return base_structs


def get_vdw_params(structure, incar):
    # Creating a list to contain the ordered elements
    elem_list = []

    # Gathering sites from the pymatgen structure
    sites = structure.sites
    sites.reverse

    # Adding sites to the list on blocks
    old_el = sites[0].specie.symbol
    elem_list.append(old_el)
    for site in sites:
        curr_el = site.specie.symbol

        if curr_el != old_el:
            elem_list.append(curr_el)

        old_el = curr_el

    # Copying the incar to avoid modifying the original
    new_incar = copy.deepcopy(incar)

    # Creating empty lists for the vdw parameters
    c6_ele_list = []
    r0_ele_list = []

    try:
        # Gathering the vdw parameters for each element
        for element in elem_list:
            with open(VDW_DATA_PATH + "/" + element) as f:
                param_file = f.readlines()
            c6_ele_list.append(float(param_file[-2].strip()))
            r0_ele_list.append(float(param_file[-1].strip()))

        # Adding the vdw parameters to the copied incar
        new_incar["vdw_c6"] = c6_ele_list
        new_incar["vdw_r0"] = r0_ele_list
    except Exception():
        pass

    return new_incar


def generate_potential_mapping() -> dict:
    """Generate a dictionary specifying the potential mapping for vasp.
    As of now, this function only assigns the default potential for every
    atom.

    Returns
    -------
    dict
        Dictionary containing the potential assignation for each atom of
        the periodic table, with the shape:
        {'H': 'H', 'He': 'He', ...}
    """

    # Creating empty dict for the potential mapping
    potential_mapping = {}

    # Mapping every symbol on the periodic table to itself
    for symbol in ad.chemical_symbols[1:]:
        potential_mapping[symbol] = symbol


    return potential_mapping


def load_relax_calcs(target_dir):
    # Creating a new pandas df
    structs_df = pd.DataFrame()

    # Gathering all folders from run directory
    list_dir = os.walk(target_dir)

    # Iterating over every folder, checking for the basename to check
    # if a folder corresponds with a calculation folder
    for path in list_dir:
        path = os.path.abspath(path[0])
        if os.path.basename(path) in calc_types:
            try:
                xml_path = os.path.join(path, "vasprun.xml")
            except FileNotFoundError:
                custom_print(
                    "The current calculation has not finished correctly.", "warn"
                )

            # Getting information about the current phase and
            # test
            test_info = os.path.split(path)
            curr_phase = test_info[1]
            curr_test = os.path.basename(test_info[0])

            # Loading finished vasp run
            curr_run = vasp.Vasprun(xml_path)

            # Loading the generated structure
            curr_struct = curr_run.final_structure

            # Storing the structrure on the dataframe
            new_row = pd.Series(
                {
                    "material_id": pd.NA,
                    "structure": curr_struct,
                    "temperature": pd.NA,
                    "perturb": pd.NA,
                    "formula": pd.NA,
                    "symmetry": pd.NA,
                    "base": pd.NA,
                    "phase": curr_phase,
                    "magnetic_properties": {
                        "is_magnetic": pd.NA,
                        "ordering": pd.NA,
                        "total_magnetization": pd.NA,
                        "total_magnetization_normalized_vol": pd.NA,
                        "total_magnetization_normalized_formula_units": pd.NA,
                        "num_magnetic_sites": pd.NA,
                        "num_unique_magnetic_sites": pd.NA,
                        "types_of_magnetic_species": pd.NA,
                    },
                    "energy_per_atom": pd.NA,
                }
            )

            new_row = new_row.to_frame().T.astype(
                {"perturb": "boolean", "base": "boolean"}
            )
            structs_df = pd.concat([structs_df, new_row], ignore_index=True)


            # structs_df.loc[
            #     structs_df["phase"] == curr_phase, "structure"
            # ] = curr_struct.to_json()

            # Returning a flag that lets the program know that json is being used
            # json_flag = True

    return structs_df


if __name__ == "__main__":
    # TODO: TENGO QUE AÑADIR UNA FUNCTION que me PERMITA
    # LEER LOS CALCULOS DE UNA CARPETA QUE HAYAN TERMINADO
    # Y METER LO QUE USABA AHORA DEL DF EN OTRA FUNCIÓN ASÍ PUEDO ESCOGER QUE MÉTODO USO

    target_path = (
        # "/home/psanz/Documents/phd-iciq/Projects/P2-Cu/initial_db_p2/base_structs_new.pkl"
        # "/home/psanz/Documents/phd-iciq/Projects/P2-Cu/initial_db_p2/base_structs_111.pkl"
        # "/home/psanz/Documents/phd-iciq/Projects/P2-Cu/initial_db_p2/base_structs_m3double.pkl"
        # "/home/psanz/Documents/phd-iciq/Projects/P2-Cu/initial_db_p2/base_structs_111.pkl"
        "/home/psanz/teklahome/projects/p2-CuZn/relaxed_structures_initialdb/relaxed_structures"
    )

    # Reading dataframe with the structures.
    # base_structs = read_df(target_path)

    # Generating dataframe from relaxation calculations
    base_structs = load_relax_calcs(target_path)

    # Getting important paths
    file_path = os.path.realpath(__file__)
    test_dir = str(pathlib.Path(file_path).parent)

    # Generating random folder name
    test_folder_name = f"tests_dir_{str(uuid.uuid4().hex)}"
    custom_print(f"Generated folder {test_folder_name}", "info")

    # Creating folder
    test_dir = os.path.join(test_dir, test_folder_name)
    os.mkdir(test_dir)

    # Iterating over every type of test that needs to be run
    for test_name in DICT_TESTS.keys():
        # Getting the path of the folder for the current test
        curr_test = os.path.join(test_dir, test_name)
        # Creating a folder for that test
        os.mkdir(curr_test)

        # Getting the different values for a parameter
        # that the test will use
        values = DICT_TESTS.get(test_name).get("values")

        for test_value in values:
            new_test_path = os.path.join(curr_test, f"{test_name}_{test_value}")
            os.mkdir(new_test_path)
            for phase in PHASES:
                phase_dir = os.path.join(new_test_path, phase)

                # Creating directory for the current structure
                os.mkdir(phase_dir)

                # Getting current structure
                curr_struct = base_structs.loc[
                    base_structs["phase"] == phase
                ].structure.values[0]

                # Storing current test parameters
                latt_vector = np.array(curr_struct.lattice.parameters[:3])

                # Computing k-points vector, assuming the material is a metal.
                # 30 = num_kpoints / lattice_param (for each axis)
                k_point_vec = np.around(30 / latt_vector).astype(int)

                calc_kpoints = Kpoints().gamma_automatic(
                    kpts=k_point_vec, shift=(0, 0, 0)
                )

                if test_name == "kpoints":
                    k_point_vec = k_point_vec + test_value
                    calc_kpoints = Kpoints.gamma_automatic(
                        kpts=k_point_vec, shift=(0, 0, 0)
                    )

                if test_name == "ispin":
                    INCAR["ispin"] = test_value
                if test_name == "encut":
                    INCAR["encut"] = test_value

                # Adding dft-d2 vdw parameters
                # INCAR_SP = get_vdw_params(curr_struct, INCAR_SP)

                # Converting to conventional cell for testing.
                #sga = SpacegroupAnalyzer(curr_struct)
                #curr_struct = sga.get_conventional_standard_structure()

                # Preparing the vasp calculation files using pymatgen
                vasp_calc = DictSet(
                    curr_struct,
                    config_dict={
                        "INCAR": INCAR_SP,
                        "KPOINTS": calc_kpoints,
                        "POTCAR": generate_potential_mapping(),
                    },
                    # user_incar_settings=user_dict,
                    vdw="dftd3",
                    user_kpoints_settings=calc_kpoints,
                )

                vasp_calc.write_input(f"{phase_dir}")

                # os.chdir(curr_test)
            # os.chdir(test_dir)
