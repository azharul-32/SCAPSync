"""SCAPSync.exe entry point.

Double-click:              starts the app on this PC and opens it in the browser.
SCAPSync.exe run x.json  runs a study (the app starts studies this way in the background).
Any other fermipilot command (inspect, params, boundary, verify) works the same way.

The app runs only on this PC (http://localhost); nothing is sent anywhere.
"""
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

PORT = 8765
URL = f"http://localhost:{PORT}"
CLI = {"run", "inspect", "params", "boundary", "verify", "where", "whyfail", "design"}


def _bundle():
    """Never called: lists the libraries the app script needs, so the packager includes them
    (the app script itself is shipped as a data file and not scanned)."""
    import altair, matplotlib, numpy, openpyxl, pandas, streamlit  # noqa: F401
    import matplotlib.pyplot  # noqa: F401
    import sklearn.ensemble  # noqa: F401
    import sklearn.gaussian_process  # noqa: F401
    from fermipilot import (active, analysis, catalogue, defmodel, design, graphs, jobs, paths, results,  # noqa: F401
                            runner, sbf, session, speedcheck, study, verify, workers)


def here():
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def already_running():
    try:
        with urllib.request.urlopen(URL + "/_stcore/health", timeout=2) as r:
            return r.status == 200
    except Exception:                           # noqa: BLE001
        return False


def port_free():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", PORT)) != 0


def open_when_ready():
    for _ in range(240):                        # up to 2 minutes for the first start
        if already_running():
            webbrowser.open(URL)
            print(f"\nSCAPSync is open in your browser: {URL}")
            print("Keep this window open while you use SCAPSync. Close it to quit.\n", flush=True)
            return
        time.sleep(0.5)
    print(f"SCAPSync did not start. Open {URL} by hand, or close this window and try again.", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in CLI:
        sys.path.insert(0, here())
        from fermipilot.__main__ import main as cli_main
        sys.exit(cli_main(sys.argv[1:]) or 0)

    print("Starting SCAPSync ... (the first start can take up to a minute)", flush=True)
    if already_running():                       # second double-click: just show the open app
        webbrowser.open(URL)
        return
    if not port_free():
        print(f"Port {PORT} is used by another program; SCAPSync cannot start.", flush=True)
        input("Press Enter to close.")
        return
    threading.Thread(target=open_when_ready, daemon=True).start()
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", os.path.join(here(), "fermipilot_app.py"),
                f"--server.port={PORT}", "--server.address=localhost", "--server.headless=true",
                "--global.developmentMode=false", "--browser.gatherUsageStats=false",
                "--server.fileWatcherType=none", "--client.toolbarMode=minimal"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
