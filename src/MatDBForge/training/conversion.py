import pathlib
import time
from enum import Enum
from io import BytesIO, TextIOWrapper
from typing import Union

import ase.io as aseio
import MatDBForge.core.initial_db as mdb_indb
import MatDBForge.core.structure as mdb_strc
import MatDBForge.core.utils as mdb_ut
import numpy as np
import rich.progress as riprg
from aiida import load_profile, orm
from aiida_vasp.calcs.vasp import VaspCalculation
from ase.atoms import Atoms


class Units(Enum):
    # Boltzmann constant in J/(Da*K)
    kB = 8.314

    # Sourced from CODATA 2018
    Bohr2Ang = 0.5291772109030
    Ang2Bohr = 1 / Bohr2Ang
    Eh2eV = 27.211386245988
    eV2Eh = 1 / Eh2eV


def mdb_database_to_mace_train(mdb_database: "mdb_indb.InitialDatabase"):
    # Gathering all structures in an InitialDatabase.
    # Generate an entry for every structure.
    # Write the entry into a file. Use Multithread?
    ...


def _vasprun_to_extended_xyz(structure: "mdb_strc.Structure"):
    ...


def _structure_to_extended_xyz(structure: "mdb_strc.Structure"):
    ...


def _add_entry_to_mace_input(
    vasprun,
    node,
    remove_dipole,
    remove_stress,
    remove_kinetic=True,
    # remove_energy=True,
    buffer: TextIOWrapper = None,
    to_file=True,
):
    # The training data is in extxyz format.
    # The parser from ase can be used to read the vasprun directly
    # and convert it to the correct format, which will have the
    # positions, energies, forces and stresses included
    # extxyz.write_extxyz(buffer, data_dict["atoms_obj"])

    if isinstance(vasprun, dict):
        # print("vasprun: ", vasprun["forces"].shape)
        vasprun = Atoms.fromdict(vasprun)

    name = node.label
    if not name:
        name = "unknown"

    # Adding structure type information to the dataset
    if "mdb_struct_type" not in vasprun.info.keys():
        vasprun.info["mdb_struct_type"] = get_struct_type(vasprun)
    if "struct_name" not in vasprun.info.keys():
        vasprun.info["struct_name"] = name
    if "aiida_uuid" not in vasprun.info.keys():
        vasprun.info["aiida_uuid"] = node.uuid

    # HACK: This aseio writer writes all the properties from
    # vasprun, included dipole_moments. However, some calculations
    # do not contain dipole information, and the files are written without
    # dipole, which leads to an error in training.
    # One solution is to remove dipole and stress if not needed.
    if vasprun.calc:
        if remove_dipole and "dipole" in vasprun.calc.results.keys():
            vasprun.calc.results.pop("dipole")
        if remove_stress and "stress" in vasprun.calc.results.keys():
            vasprun.calc.results.pop("stress")

        # HACK: Removing energy (without entropy, as it is not used to calculate
        # the forces) and kinetic energy.
        # if remove_energy and "energy" in vasprun.calc.results.keys():
        #     vasprun.calc.results.pop("energy")
        if remove_kinetic and "kinetic_energy" in vasprun.calc.results.keys():
            vasprun.calc.results.pop("kinetic_energy")

    if to_file:
        aseio.write(buffer, images=vasprun, format="extxyz")
    else:
        return vasprun


def _gather_mace_req_calc_data_from_node(node):
    # Getting calculation name
    # name = node.label + "_aiida-uuid_" + node.uuid

    # Writing the vasprun.xml file to a buffer.
    retrieved: orm.NodeRepository = node.outputs.retrieved

    # Reading the file from the buffer and closing it
    with retrieved.open("vasprun.xml", "rb") as f:
        vasprun = aseio.read(f, format="vasp-xml", index="-1")
    return vasprun


def gather_calc_data_from_node(node, units="atomic"):
    if units == "atomic":
        length_unit = Units.Ang2Bohr.value
        energy_unit = Units.eV2Eh.value
    elif units == "mace":
        length_unit = 1
        energy_unit = 1

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
    # Free energy can be gathered with self.get_potential_energy(force_consistent=True)
    # energy_zero can be gathered with self.get_potential_energy(force_consistent=False)
    pot_energy = vasprun.get_potential_energy(force_consistent=False) * energy_unit
    tot_energy = vasprun.get_total_energy() * energy_unit

    # Getting forces
    # Reading forces from vasprun.xml, in eV/Ang and converting them to Ha/Bohr
    forces = vasprun.get_forces() * energy_unit * 1 / length_unit

    lattice = vasprun.get_cell() * length_unit
    pbc = vasprun.get_pbc()
    structure = vasprun.get_positions() * length_unit
    symbols = vasprun.get_chemical_symbols()
    numbers = vasprun.get_atomic_numbers()
    dipole = vasprun.get_dipole_moment()

    # voigt=False is needed to get a 3x3 array, which gets used by the
    # extxyz format
    stress = vasprun.get_stress(voigt=False)

    # Setting charge to 0
    # charge = contcar.structure.charge
    charge = 0

    struct_type = get_struct_type(vasprun)

    # MACE by default checks the 'energy' key for the energies in the training files.
    # However, we decided to use free_energy to store the energy.
    # Which key is used by MACE training can be set on the launch arguments.
    # TODO: Re-add dipole and potential_energy.
    data_dict = {
        "name": name,
        "lattice": lattice,
        "positions": structure,
        "symbols": symbols,
        "numbers": numbers,
        # "pot_energy": pot_energy,
        "free_energy": tot_energy,
        "charge": charge,
        "pbc": pbc,
        "stress": stress,
        # "dipole": dipole,
        "forces": forces,
        "atoms_obj": vasprun,
        "struct_type": struct_type,
    }

    return data_dict


def get_struct_type(vasprun):
    # HACK: Structure type is now inferred from calc settings.
    # TODO: Add a calc type identifier to aiida dft calculations and use
    # that instead.

    run_params = vasprun.calc.parameters
    # kpt_arr = run_params["kpoints_generation"]["divisions"]

    if not run_params["ldipol"]:
        struct_type = "bulk"
    # HACK: This won't be always like this
    elif run_params["dipol"] == [0.5, 0.5, 0.5]:
        struct_type = "cluster"
    elif run_params["idipol"] == 3:
        struct_type = "surface"
    else:
        struct_type = "unknown"

    return struct_type


def _gather_result_nodes_aiida(path, aiida_group_list, filter_dict):
    # Loading aiida profile
    load_profile()

    # Gathering nodes from the given group
    mdb_ut.custom_print("Getting nodes...")

    # Preparing a query in the aiida db for every group
    result_nodes_list = []
    for idx, group in enumerate(aiida_group_list):
        # Querying for WorkChainNode objects
        qb = orm.QueryBuilder()
        qb.append(orm.Group, filters={"label": group}, tag="group")
        qb.append(orm.WorkChainNode, with_group="group", filters=filter_dict)
        result_nodes = qb.all(flat=True)

        # Old versions will result in VaspCalculations being stored in the
        # aiida groups instead and therefore won't show in the first query.
        # In those cases, an additional query for VaspCalculations is prepared.
        if len(result_nodes) == 0:
            qb = orm.QueryBuilder()
            qb.append(orm.Group, filters={"label": group}, tag="group")
            qb.append(VaspCalculation, with_group="group", filters=filter_dict)
            result_nodes = qb.all(flat=True)

        result_nodes_list.extend(result_nodes)

    mdb_ut.custom_print(f"{len(result_nodes_list)} nodes found.", "info")

    return result_nodes_list


def gen_mace_train_aiida(
    aiida_group_list: list,
    filter_dict: dict,
    path: str = None,
    remove_dipole=False,
    remove_stress=False,
):
    # Gathering aiida nodes containing the desired calculation results
    result_nodes = _gather_result_nodes_aiida(path, aiida_group_list, filter_dict)

    # Handling path
    if path and isinstance(path, str):
        path = pathlib.Path(path)
    else:
        path = pathlib.Path()

    ctime = time.strftime("%Y%m%dT%H%M%S")

    # Adding input.data filename to path
    path = path / f"mace_training_data_{ctime}.xyz"

    # Writing the file
    with open(path, "w") as curr_f:
        # Checking every node
        for node in riprg.track(result_nodes, description=" [ ⧖ ]  Writing info..."):
            # Gathering the information from each node
            vasprun = _gather_mace_req_calc_data_from_node(node=node)

            # Writing the information to the buffer
            _add_entry_to_mace_input(
                buffer=curr_f,
                vasprun=vasprun,
                node=node,
                remove_dipole=remove_dipole,
                remove_stress=remove_stress,
            )

        final_size = curr_f.tell() * 1e-06

    mdb_ut.custom_print(
        f"All calculations saved in '{path}' ({final_size:.2f} MB).", "done"
    )


def _add_entry_to_n2p2_input(buffer: TextIOWrapper, data_dict: dict):
    # Writing begin keyword and structure name
    write_name = f'{data_dict.get("struct_type", "unkw")}_{data_dict.get("name", "no name found")}'
    buffer.write("begin\n")
    buffer.write(f"comment {write_name}\n")

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
    for idx, (at, frc) in enumerate(zip(data_dict["positions"], data_dict["forces"])):
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


def gen_n2p2_train_aiida(aiida_group_list: list, filter_dict: dict, path: str = None):
    # Gathering aiida nodes containing the desired calculation results
    result_nodes = _gather_result_nodes_aiida(path, aiida_group_list, filter_dict)

    # Handling path
    if path and isinstance(path, str):
        path = pathlib.Path(path)
    else:
        path = pathlib.Path()

    ctime = time.strftime("%Y%m%dT%H%M%S")

    # Adding input.data filename to path
    path = path / f"n2p2_training_data_{ctime}.input"

    # Writing the file
    with open(path, "w") as curr_f:
        # Checking every node
        for node in riprg.track(result_nodes, description=" [ ⧖ ]  Writing info..."):
            # Gathering the information from each node
            data_dict = gather_calc_data_from_node(node=node)

            # Writing the information to the buffer
            _add_entry_to_n2p2_input(buffer=curr_f, data_dict=data_dict)

        final_size = curr_f.tell() * 1e-06

    mdb_ut.custom_print(
        f"All calculations saved in '{path}' ({final_size:.2f} MB).", "done"
    )


def gen_mace_train_structure_list(
    path: Union[str, pathlib.Path],
    structure_list: list,
    disable=False,
    skip_dipole=True,
    skip_stress=True,
    skip_free_energy=False,
):
    # Handling path
    if path and isinstance(path, str):
        path = pathlib.Path(path).resolve()
    else:
        path = pathlib.Path().resolve()

    # Creating path if does not exist
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

        ase_structs = []
        # Converting into ase atoms object
        # len_struct = len(structure_list)
        structure_list = structure_list.get_list()
        for idd, struct in enumerate(structure_list):
            new_struct = {}

            dict_keys_set = set(list(struct.keys()))
            info_keys_set = set(list(struct["info"].keys()))
            dict_to_array_set = set(
                [
                    "pbc",
                    "cell",
                    "forces",
                    "positions",
                    "energy",
                    "numbers",
                    "free_energy",
                ]
            )

            # List containing possible keys in atoms.info
            info_list = [
                "stress",
                "dipole",
                "struct_name",
                "energy",
                "aiida_uuid",
                "free_energy",
                "mdb_struct_type",
            ]

            # Whether to keep or remove stress, dipole and energy
            if skip_stress:
                info_list.remove("stress")
            if skip_free_energy:
                info_list.remove("free_energy")
            if skip_dipole:
                info_list.remove("dipole")
            info_to_array_set = set(info_list)

            for arr_key in dict_to_array_set.intersection(dict_keys_set):
                new_struct[arr_key] = np.array(struct.get(arr_key))

            # Storing keys in atoms.info
            new_struct["info"] = {}
            for arr_key in info_to_array_set.intersection(info_keys_set):
                value = struct.get("info").get(arr_key)

                # If not converted to array will be written incorrectly
                if arr_key in ["stress", "dipole"]:
                    value = np.array(value)

                new_struct["info"][arr_key] = value

            ase_structs.append(Atoms.fromdict(new_struct))

        # Writing the file
        aseio.write(path, ase_structs, "extxyz")

    # REMOVE
    else:
        print("Path already exists, not overwriting final db.")
