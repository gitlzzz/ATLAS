"""Launch a dashboard to monitor the active learning loop."""

import argparse
import multiprocessing as mp
import time
from argparse import RawTextHelpFormatter

from aiida.orm import load_node
from gunicorn.app.wsgiapp import WSGIApplication
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from MatDBForge.core.command_line.command_line_utils import MDB_LOGO


def console_output(url: str, port: str, dash_pid: str, process_pk: str, log_path: str):
    """Prints a console output with the dashboard information using the rich library."""
    console = Console()

    # Creating Text and Panel for each section
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
            "You can find the pid again in the file ",
            (".mdb_dashboard.pid", "bold blue"),
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

    grp = Group(process_text, table)

    # Create a panel with the table inside
    panel = Panel(
        grp,
        title="MatDBForge",
        border_style="magenta",
        expand=False,
        padding=(2, 2),
    )

    # Print the panel
    console.print(panel)


class MDBDashboardApp(WSGIApplication):
    """Gunicorn application for the MDB dashboard."""

    def __init__(self, app_uri, options=None):
        self.options = options or {}
        self.app_uri = app_uri
        super().__init__()

    def load_config(self):
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)


def run_dashboard_process(process_id, port, update_interval, online, debug):
    app = MDBDashboardApp(
        f"MatDBForge.active_learning.dashboard.training_dashboard_flask"
        f":run_training_dashboard(workchain_node_id={process_id}, "
        f"refresh_interval={update_interval}, port={port})",
    )
    app.options["pidfile"] = ".mdb_dashboard.pid"
    app.options["daemon"] = "true"
    app.options["capture_output"] = "true"
    app.options["errorlog"] = "./.mdb_dashboard_output.log"
    app.options["worker_connections"] = 1
    app.options["keepalive"] = 5
    app.options["proc_name"] = "mdb_dashboard"
    app.options["preload_app"] = "true"
    if online:
        app.options["bind"] = f"0.0.0.0:{port}"
    else:
        app.options["bind"] = f"127.0.0.1:{port}"
    if debug:
        print("Debug mode enabled.")
        app.options["reload"] = "true"
        app.options["loglevel"] = "debug"

    # print("Logging dashboard status in '.mdb_dashboard_output.log'")
    app.load_config()
    app.run()


def run_dashboard_app(process_id, port, update_interval, debug, online):
    process = mp.Process(
        target=run_dashboard_process,
        args=(process_id, port, update_interval, online, debug),
        name="mdb_dashboard",
        daemon=True,
    )
    process.start()
    time.sleep(2)
    with open("./.mdb_dashboard.pid") as f:
        pid = f.readline().strip()

    log_path = load_node(process_id).inputs.log_path.value

    console_output(
        url="127.0.0.1",
        port=port,
        dash_pid=pid,
        process_pk=process_id,
        log_path=log_path,
    )
    process.join()


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
