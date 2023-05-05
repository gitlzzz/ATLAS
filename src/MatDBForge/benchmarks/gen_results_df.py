import os
from collections import defaultdict
from iciq_utils import custom_print
import pandas as pd
import xml
import multiprocessing as mp
import numpy as np
import pymatgen.io.vasp as vasp
import warnings

warnings.filterwarnings("ignore", category=vasp.outputs.UnconvergedVASPWarning)

CWD = os.path.abspath(
    # "/home/psanz/teklahome/tests_dir_4c898d1cba504efc860ab75a6dc20919"
    # "/home/psanz/teklahome/projects/p2-CuZn/initial_tests/tests_dir_4678bcb9da984d13a824a4426f3a66c8_test_batch_2"
    # "/home/psanz/teklahome/projects/p2-CuZn/initial_tests/test_batch_4"
    "/home/psanz/teklahome/projects/p2-CuZn/initial_tests/final_tests_plot"
)

FILENAME = "result_df_test007_new2.pkl"
res_dict = defaultdict(dict)
test_types = [
    "ispin",
    "encut",
    "kpoints_conv_cell",
    "kpoints_old",
    "kpoints_Zn_pv_2005",
    "kpoints_Zn_2000",
    "kpoints_encut_500",
    "kpoint_testing_newstructures",
]

calc_types = [
    "alpha",
    "m1",
    "beta-prime",
    "m2",
    "gamma",
    "m3",
    "epsilon",
    "m4",
    "eta",
    "m3_old",
]


def iterate_and_read_vasp_multi(df, path):
    path = path[0]
    if os.path.basename(path) in calc_types:
        xml_path = os.path.join(path, "vasprun.xml")

        # Getting information about the current phase and
        # test
        test_info = os.path.split(path)
        curr_phase = test_info[1]
        curr_test = os.path.basename(test_info[0])

        curr_test = check_path_exception(path, curr_test)

        # Checking if the df has a valid value for that phase
        # or if the dataframe has that key
        try:
            row = df.loc[df["phase"] == curr_phase]
            curr_test_df_data = row[curr_test].values[0]

        except KeyError:
            curr_test_df_data = pd.NA

        # If the dataframe is missing this data, try to load it
        # from the files.
        if pd.isnull(curr_test_df_data):
            try:
                curr_run = vasp.Vasprun(xml_path)

                # Storing the energy per atom
                num_atom_struct = curr_run.ionic_steps[0]["structure"].num_sites
                energy_per_atom = curr_run.final_energy / num_atom_struct
                res_dict[curr_test][curr_phase] = energy_per_atom

                # Getting the kpoint density
                # Getting array of kpoints
                arr_kpt_run = np.array(curr_run.kpoints.kpts[0])

                # Getting lattice vectors
                l_mat = curr_run.structures[0].lattice.matrix

                # Getting volume of the reciprocal cell
                v_mat = np.dot(np.cross(l_mat[0, :], l_mat[1, :]), l_mat[2, :])

                # Computing values for each axis
                a_rcpr = np.linalg.norm((np.cross(l_mat[1, :], l_mat[2, :])) / v_mat)
                b_rcpr = np.linalg.norm((np.cross(l_mat[0, :], l_mat[2, :])) / v_mat)
                c_rcpr = np.linalg.norm((np.cross(l_mat[0, :], l_mat[1, :])) / v_mat)

                # Computing the kpt density values in an array
                kpt_dens_arr = np.array((a_rcpr, b_rcpr, c_rcpr)) * ((1 / arr_kpt_run))

                # Getting the maximum value
                kpt_dens_max = np.max(kpt_dens_arr)

                # Adding the density to the dataframe.
                # res_dict[curr_test + "_kpt_density"][curr_phase] = kpt_dens_max

                # Adding the kpoint vector to the dataframe
                # res_dict[curr_test + "_kpt_vector"][curr_phase] = arr_kpt_run

            # Some vasp calculations, with large numbers of kpoints
            # even without raising an errror, output an incomplete
            # xml result file, resulting in an error being raised.
            # This exception catches this error and just does not add
            # that file to the df.
            except xml.etree.ElementTree.ParseError as e:
                custom_print(f"Skipping: {xml_path}", "warn")
                print(e)
                energy_per_atom = np.nan
                kpt_dens_arr = np.nan
                kpt_dens_max = np.nan
                arr_kpt_run = np.nan
                res_dict[curr_test][curr_phase] = np.nan
                # custom_print(e, "\n")

            changes_made = True
            pid = os.getpid()
            custom_print(
                f"Loaded '{curr_test}_{curr_phase} on process {pid}'.", "debug"
            )

        else:
            custom_print(f"Calc '{curr_test}_{curr_phase}' is already loaded.", "debug")

        return (
            curr_test,
            curr_phase,
            energy_per_atom,
            kpt_dens_max,
            arr_kpt_run,
            changes_made,
        )


def check_path_exception(path, curr_test):
    try:
        if "newstructures" in path:
            curr_test = (
                curr_test.split("_")[0] + "_newstructures_" + curr_test.split("_")[1]
            )

        if "Zn_2000" in path:
            curr_test = (
                curr_test.split("_")[0] + "_pot_Zn_2000_" + curr_test.split("_")[1]
            )

        if "Zn_pv" in path:
            curr_test = (
                curr_test.split("_")[0] + "_pot_Zn_pv_2005_" + curr_test.split("_")[1]
            )

        if "_conv_cell" in path:
            curr_test = (
                curr_test.split("_")[0] + "_conv_cell_" + curr_test.split("_")[1]
            )

        if "_old" in path:
            curr_test = curr_test.split("_")[0] + "_old_" + curr_test.split("_")[1]

        if "kpoints_encut_" in path:
            curr_test = (
                curr_test.split("_")[0] + "_encut_500_" + curr_test.split("_")[1]
            )

    except IndexError:
        print("curr_test: ", curr_test)
        print("path: ", path)

        curr_test = "error"

    return curr_test


def gen_res_df():
    # Trying to load an existing result dataframe.
    try:
        df = pd.read_pickle(os.path.join(CWD, FILENAME))
        custom_print(f"Dataframe '{FILENAME}' loaded.", "info")
    except FileNotFoundError:
        # If the file does not exist create an empty dataframe.
        custom_print("Result dataframe not found, creating an empty dataframe.", "warn")
        df = pd.DataFrame()

    # Gathering all folders from run a directory
    # All simulation data can be aggegrated into a single
    # folder as symbolic links so files don't have to be copied
    list_dir = os.walk(CWD, followlinks=True)

    # Flag to see if changes have been made
    changes_made = False
    # changes_made = iterate_and_read_vasp(df, list_dir)
    res_dict = defaultdict(dict)

    changes_list = []
    from itertools import repeat

    with mp.Pool() as pool:
        results = pool.starmap(iterate_and_read_vasp_multi, zip(repeat(df), list_dir))

    for entry in results:
        # print('entry: ', entry)
        if entry:
            # print("entry: ", entry)
            res_dict[entry[0]][entry[1]] = entry[2]
            res_dict[entry[0] + "_kpt_density"][entry[1]] = entry[3]
            res_dict[entry[0] + "_kpt_vector"][entry[1]] = entry[4]
            changes_list.append(entry[5])

    changes_made = np.any(np.array(changes_list))
    df = pd.DataFrame.from_dict(res_dict)
    df.reset_index(inplace=True)
    df = df.rename(columns={"index": "phase"})

    return df, changes_made


if __name__ == "__main__":
    # Generating or reading the results dataframe
    df, changes_made = gen_res_df()
    print("df: ", df)

    # Getting the file save path
    save_path = os.path.join(CWD, FILENAME)

    if changes_made:
        # Saving the file
        df.to_pickle(save_path)
        custom_print(f"Dataframe saved to {save_path}", "done")
    else:
        custom_print("Dataframe unchanged.", "done")
