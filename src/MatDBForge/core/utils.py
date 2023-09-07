# General utilities
import json as js
import logging
import pathlib

from pymatgen.core.periodic_table import Element, Species

# logging.basicConfig(level='CRITICAL')


def custom_print(string: str, print_type: str = "default", end="\n"):
    """Prints a string using different formatting styles for
    easier debugging.

    Parameters
    ----------
    string : str
        Text to be printed
    print_type : str, optional, `default=info`
        Style to use when printing. Available styles are:
        - `info/default`: prefixes [i] before the string
        - `warning`: prefixes [!] before the string
        - `debug`: prefixes [...] before the string
        - `done`: prefixes [ ✔ ] before the string
    """
    normal = "\u001b[0m"

    if print_type in ["info", "default"]:
        prefix = "\u001b[38;5;33m [ i ]"
        print(f"{prefix}{normal}\t{string}", end=end)
        logging.info(string)
    elif print_type in ["warn", "warning"]:
        prefix = "\u001b[38;5;220m [ ! ]"
        print(f"{prefix}\t{string}{normal}", end=end)
        logging.warning(string)
    elif print_type in ["warn-soft", "warning-soft"]:
        prefix = "\u001b[38;5;220m [ ! ]"
        print(f"{prefix}{normal}\t{string}", end=end)
        logging.warning(string)
    elif print_type in ["extra", "debug"]:
        prefix = "\u001b[38;5;8m [···]"
        print(f"{prefix}\t{string}{normal}", end=end)
        logging.debug(string)
    elif print_type in ["done"]:
        prefix = "\u001b[38;5;46m [ ✔ ]"
        print(f"{prefix}{normal}\t{string}", end=end)
        logging.info(string)
    if print_type in ["error", "problem"]:
        prefix = "\u001b[38;5;1m [ X ]"
        print(f"{prefix}{normal}\t{string}", end=end)
        logging.error(string)


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
        if not row.base and not row.material_name.endswith("_symm"):
            strct = row.structure.get_sorted_structure()
            name = row.material_name
            phase = curr_phase_diag.get_phase(row.phase)
            offset = phase.offset
            tot_atoms = len(strct.species)
            one_at_perc = 1 / tot_atoms

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

            perc = round(tot_zn / tot_atoms, 2)

            offset_min = round(phase.base_elem_comp_min - offset, 2)
            if offset_min < 0:
                offset_min = 0

            offset_max = round(phase.base_elem_comp_max + offset, 2)
            if offset_max > 1:
                offset_max = 1

            # Checking if the current structure is between the phase ratio
            # percentages.
            if not (offset_min <= perc <= offset_max):
                # If the structure could be fixed by adding or removing an atom
                if (offset_min <= perc + one_at_perc <= offset_max) or (
                    offset_min <= perc - one_at_perc <= offset_max
                ):
                    custom_print(
                        f"{name}: {perc:.2f} Zn outside of ({offset_min:.2f} - {offset_max:.2f}) range",
                        "error",
                    )
                else:
                    pass
