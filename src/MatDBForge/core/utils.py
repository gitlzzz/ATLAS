# General utilities


def custom_print(string: str, print_type: str = "default"):
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
        print(f"{prefix}{normal}\t{string}")
    elif print_type in ["warn", "warning"]:
        prefix = "\u001b[38;5;220m [ ! ]"
        print(f"{prefix}\t{string}{normal}")
    elif print_type in ["extra", "debug"]:
        prefix = "\u001b[38;5;8m [···]"
        print(f"{prefix}\t{string}{normal}")
    elif print_type in ["done"]:
        prefix = "\u001b[38;5;46m [ ✔ ]"
        print(f"{prefix}{normal}\t{string}")
