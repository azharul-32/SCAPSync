"""Run one SCAPS job headless and report what happened."""
import os
import subprocess
import time

from .paths import DEFAULT_SCAPS_DIR  # noqa: F401  (kept here for older imports)


def find_exe(scaps_dir):
    for name in sorted(os.listdir(scaps_dir)):
        if name.lower().startswith("scaps") and name.lower().endswith(".exe"):
            return os.path.join(scaps_dir, name)
    raise FileNotFoundError(f"no scaps*.exe in {scaps_dir}")


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def kill_scaps(pid):
    """Kill only the SCAPS process FermiPilot started (never a SCAPS window the user has open)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/PID", str(pid), "/T"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        else:
            os.kill(pid, 9)
    except (OSError, subprocess.SubprocessError):
        pass


def start_scaps(exe, script_path, scaps_dir):
    """Start SCAPS on a script, minimized and without taking the keyboard focus (SW_SHOWMINNOACTIVE).
    Set the environment variable FERMIPILOT_SCAPS_WINDOW=normal to watch SCAPS work.
    Known limit (diagnosed 30-10-2026 with diag/window_probe.py): while calculating, SCAPS itself opens its
    Action Panel and Energy Bands Panel on screen and stops answering Windows, so they cannot be moved or
    hidden from outside, and Windows adds a '(Not Responding)' ghost copy. Tried without success: a
    hidden Windows desktop (SCAPS failed to start, 0xc0000142), minimizing / making transparent / moving
    its windows from a watcher thread, and off-screen margins in panelsize3309.txt (ignored by SCAPS)."""
    kw = {}
    if os.name == "nt" and os.environ.get("FERMIPILOT_SCAPS_WINDOW", "minimized").lower() != "normal":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 7                      # SW_SHOWMINNOACTIVE
        kw["startupinfo"] = si
    return subprocess.Popen([exe, script_path], cwd=scaps_dir, **kw)


def run_script(scaps_dir, script_path, timeout=30):
    """Start SCAPS with a script, wait until it quits by itself (or time out).
    Returns dict: elapsed_s, timed_out, returncode, errors (text from the SCAPS log, if new)."""
    exe = find_exe(scaps_dir)
    log = os.path.join(scaps_dir, "SCAPSErrorLogFile.log")
    before = _mtime(log)
    t0 = time.time()
    proc = start_scaps(exe, script_path, scaps_dir)
    timed_out = False
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_scaps(proc.pid)
        try:
            proc.kill()
        except OSError:
            pass
        rc = None
    elapsed = time.time() - t0
    errors = ""
    after = _mtime(log)
    if after is not None and after != before:
        with open(log, "rb") as f:
            errors = f.read().decode("latin-1").strip()
    return {"elapsed_s": round(elapsed, 2), "timed_out": timed_out,
            "returncode": rc, "errors": errors}
