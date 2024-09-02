"""Launch a dashboard to monitor the active learning loop."""

import argparse
import time
from argparse import RawTextHelpFormatter

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from MatDBForge.core.command_line.command_line_utils import MDB_LOGO


def console_output(url: str, port: str, dash_pid: str, process_pk: str, log_path: str):
    """Prints a console output with the dashboard information using the rich library."""
    console = Console(record=True)

    store_output_file = f"mdb_dashboard_run_{process_pk}_info.out"

    # Creating Text containing the process information
    process_text = Text.assemble(
        (
            f"{MDB_LOGO}",
            "bold magenta",
        ),
        justify="full",
    )
    process_text.append(
        Text.assemble(
            "\n\nThe active learning loop has been launched (PK: ",
            (f"{process_pk}", "bold blue"),
            ") \nusing a dashboard interface in a separate process.\n",
            "To access the dashboard visit: ",
            (f"http://{url}:{port}.\n", "bold blue"),
            "\n\nUse ",
            (f"kill {dash_pid}", "bold red"),
            " to stop the dashboard.\n",
            "You can find this information again in the file ",
            (f"{store_output_file}", "bold blue"),
            " in the CWD.",
            "\n",
            justify="full",
        ),
    )

    # Create a table
    table = Table(
        show_header=True,
        header_style="bold magenta",
        # highlight=True,
        title="Active Learning Dashboard Info",
    )
    table.add_column("Info Type", style="dim", width=20)
    table.add_column("Details", style="bold")

    # Add rows with the information
    table.add_row("Status", "[green]Running :heavy_check_mark:")
    table.add_row("Process PK", f"{process_pk}")
    table.add_row("Dashboard URL", f"http://{url}:{port}")
    table.add_row("Dashboard log file", "[dim].mdb_dashboard_output.log")
    table.add_row("PID File", "[dim].mdb_dashboard.pid")
    table.add_row("Dashboard PID", f"[dim]{dash_pid}")
    table.add_row(
        "MDB Log File",
        f"[bold]{log_path}",
    )

    # Create a group with the text and the table
    grp = Group(process_text, table)

    # Create a panel with the group inside
    panel = Panel(
        grp,
        title="MatDBForge Active Learning",
        border_style="magenta",
        expand=False,
        padding=(2, 2),
    )

    # Print the panel
    console.print(panel)
    console.save_text(f"mdb_run_{process_pk}_info.out")


def run_dashboard_app(process_id, port, update_interval, debug, online):
    import subprocess as sb

    from aiida import load_profile
    from aiida.orm import load_node

    load_profile()

    # Get url from settings
    url = "0.0.0.0" if online else "127.0.0.1"

    # Enable debug mode
    debug = ["--log-level", "debug"] if debug else ["--log-level", "info"]

    # print('reload_mode: ', reload_mode)
    # Laucnh the dashboard in a separate process
    sb.call(
        [
            "gunicorn",
            "--daemon",
            "--bind",
            f"{url}:{port}",
            # reload_mode,
            *debug,
            "--capture-output",
            "--error-logfile",
            "./.mdb_dashboard_output.log",
            "--pid",
            "./.mdb_dashboard.pid",
            "--name",
            "mdb_dashboard",
            f"MatDBForge.active_learning.dashboard.training_dashboard_flask:run_training_dashboard(workchain_node_id='{process_id}',refresh_interval='{update_interval}',port='{port}')",
        ]
    )

    # Wait for the dashboard to start and get the pid
    time.sleep(5)
    with open("./.mdb_dashboard.pid") as f:
        pid = f.readline().strip()

    # Get the log path
    log_path = load_node(process_id).inputs.log_path.value

    # Print the dasboard information in the console.
    console_output(
        url="127.0.0.1",
        port=port,
        dash_pid=pid,
        process_pk=process_id,
        log_path=log_path,
    )


def monitor_al_loop():
    parser = argparse.ArgumentParser(
        prog="monitor_al_loop",
        description="Monitor a MDB active learning loop.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--process_id",
        help=("Process id (pk/uuid) of the WorkChain to monitor.\n"),
        type=str,
        metavar="UUID/PK",
    )
    parser.add_argument(
        "--update_interval",
        help=("Refresh time interval in seconds"),
        type=int,
        default=60,
        metavar="n_sec",
    )
    parser.add_argument(
        "--port",
        help=("Port to use for the webapp"),
        type=int,
        default=8000,
        metavar="port",
    )
    parser.add_argument(
        "--debug",
        help=("Enable Flask debug"),
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--online",
        help=("Enable online"),
        action="store_const",
        const=True,
        default=False,
    )

    # Getting CLI arguments
    args = parser.parse_args()

    run_dashboard_app(
        process_id=args.process_id,
        port=args.port,
        update_interval=args.update_interval,
        debug=args.debug,
        online=args.online,
    )
