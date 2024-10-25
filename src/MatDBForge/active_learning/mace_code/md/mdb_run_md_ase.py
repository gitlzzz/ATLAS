"""Script to run MACE MD simulations, descriptor generation and extrapolation checks."""

from json import load

import numpy as np
from ase import units
from ase.io import read as ase_read
from ase.md import MDLogger
from ase.md.langevin import Langevin
from mace.calculators import MACECalculator


def write_frame():
    dyn.atoms.write("trajectory_final.xyz", append=True)


# Function to compute the temperature ramp
def temperature_ramp(step, total_steps, T_start, T_end):
    # Linear ramp from T_start to T_end
    return T_start + (T_end - T_start) * step / total_steps


def generate_descriptors_mace(model_path: str, database):
    calculator = MACECalculator(
        model_paths=model_path, device="cpu", default_dtype="float32"
    )
    descriptor_dict = {}
    descriptor_list = []
    for struct in database:
        descriptor_dict[struct.info["aiida_uuid"]] = []

    for struct in database:
        curr_struct_descriptors = calculator.get_descriptors(struct)
        descriptor_list.append(curr_struct_descriptors)
        descriptor_dict[struct.info["aiida_uuid"]].append(curr_struct_descriptors)

    descriptor_arr = np.vstack(descriptor_list)
    return descriptor_dict, descriptor_arr


if __name__ == "__main__":

    # Reading the JSON file with the settings
    with open("md_settings.json") as f:
        settings = load(f)

    total_steps = settings["num_steps"]
    T_start = settings["temperature_K"]
    T_multiplier = settings["max_temp_multiplier"]
    T_end = T_start * T_multiplier

    # Reading structure
    init_conf = ase_read("curr_structure.xyz", format="extxyz")

    # Load the trained model as an ASE calculator and attach it to the atoms object
    calculator = MACECalculator(
        model_paths="curr_model.model",
        device=settings["device"],
        default_dtype=settings["dtype"],
    )
    init_conf.set_calculator(calculator)

    print("Starting MD...\n")

    # Define the Langevin dynamics
    dyn = Langevin(
        init_conf,
        settings["tstep_size"] * units.fs,
        temperature_K=T_start,
        friction=settings["tstep_size"] * 100,
    )

    # Attach the write_frame function to write the trajectory
    dyn.attach(write_frame, interval=settings["frame_save_interval"])
    dyn.attach(
        MDLogger(dyn, init_conf, "-", header=True, stress=False, mode="a"),
        interval=settings["log_save_interval"],
    )

    # Loop through the steps and manually adjust the temperature
    for step in range(total_steps):
        # Update the temperature using the ramp function
        current_temperature = temperature_ramp(step, total_steps, T_start, T_end)

        # Set the temperature in units of energy
        dyn.set_temperature(temperature_K=current_temperature)

        dyn.run(1)  # Run one step at a time

    print("MD finished!")

    # Read trajectory
    md_traj = ase_read("trajectory_final.xyz", format="extxyz", index=":")

    # Get the step, energy and forces and save into an array
    step_E_F_list = []
    for idx, frame in md_traj:
        e_pot = frame.get_potential_energy()
        fmax = frame.get_forces().max()
        step_E_F_list.append([idx, e_pot, fmax])

    step_E_F_arr = np.stack(step_E_F_list, axis=10)
    np.save(file="step_E_F_arr.npy", arr=step_E_F_arr)

    # Get the descriptors
    # Only MACE is supported for now
    if settings["descriptor_type"]:
        descriptor_dict, descriptor_arr = generate_descriptors_mace(
            "curr_model.model", md_traj
        )
        descriptor_arr.write("descriptors_final_vstack.npy")
