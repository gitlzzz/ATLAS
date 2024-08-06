"""Launch a dashboard to monitor the active learning loop."""

import argparse
import multiprocessing as mp
import time
from argparse import RawTextHelpFormatter

from gunicorn.app.wsgiapp import WSGIApplication


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
    app.options["errorlog"] = "./mdb_dashboard_output.log"
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

    print("Logging in 'mdb_dashboard_output.log'")
    app.load_config()
    app.run()


def run_dashboard_app(process_id, port, update_interval, debug, online):
    print(
        f"\nRunning dashboard to monitor process: {process_id}.\n"
        f"To access the dashboard visit: http://127.0.0.1:{port}."
    )
    # if debug:
    #     app = run_training_dashboard(
    #         workchain_node_id=process_id,
    #         refresh_interval={update_interval},
    #         port={port},
    #     )
    #     host = "0.0.0.0" if online else "127.0.0.1"

    #     print("Pres Ctrl+C to stop the dashboard.")
    #     app.run(debug=True, port=port, host=host)
    # else:
    print("\nStarting dashboard in a separate process.")
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
    print(
        f"Use 'kill {pid}' to stop the dashboard.\n"
        "You can find the pid again in the file '.mdb_dashboard.pid'."
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
