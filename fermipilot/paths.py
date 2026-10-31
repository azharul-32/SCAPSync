"""Where things are: the SCAPS installation, FermiPilot's settings, and folders FermiPilot may write to.

For a new user nothing has to be set up:
- SCAPS is searched in the usual places (Program Files, C:\\, D:\\, Desktop, Documents, Downloads).
- The SCAPS folder, once found or chosen, is remembered in %LOCALAPPDATA%\\FermiPilot\\settings.json.
- Results, the SCAPS working copies and uploaded models go next to SCAPS when that folder can be
  written to (as on the development PC), otherwise to %LOCALAPPDATA%\\FermiPilot. A normal Windows
  user cannot write inside C:\\Program Files (x86), and SCAPS itself is never changed.
"""
import glob
import json
import os
import sys
import threading

DEFAULT_SCAPS_DIR = r"C:\Program Files (x86)\Scaps3309"


def frozen():
    """True when running from the packaged FermiPilot.exe."""
    return bool(getattr(sys, "frozen", False))


def app_data():
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, "FermiPilot")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------- settings ----------------
def _settings_path():
    return os.path.join(app_data(), "settings.json")


def load_settings():
    try:
        with open(_settings_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_setting(key, value):
    s = load_settings()
    s[key] = value
    with open(_settings_path(), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)


# ---------------- SCAPS ----------------
def scaps_exe(folder):
    """Path of scaps*.exe in `folder`, or None."""
    try:
        for name in sorted(os.listdir(folder)):
            if name.lower().startswith("scaps") and name.lower().endswith(".exe"):
                return os.path.join(folder, name)
    except OSError:
        pass
    return None


def _candidates():
    home = os.path.expanduser("~")
    roots = [os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
             os.environ.get("ProgramFiles", r"C:\Program Files"), "C:\\", "D:\\", "E:\\",
             home, os.path.join(home, "Desktop"), os.path.join(home, "Documents"), os.path.join(home, "Downloads"),
             os.path.join(home, "OneDrive", "Desktop"), os.path.join(home, "OneDrive", "Documents")]
    seen = []
    for r in roots:
        for pat in ("Scaps*", "SCAPS*", "scaps*", os.path.join("*", "Scaps*"), os.path.join("*", "SCAPS*")):
            for d in sorted(glob.glob(os.path.join(r, pat)), reverse=True):     # newest version name first
                if os.path.isdir(d) and d not in seen:
                    seen.append(d)
    return seen


def find_scaps():
    """The remembered SCAPS folder if it still has SCAPS, else the first one found, else None."""
    remembered = load_settings().get("scaps_dir")
    if remembered and scaps_exe(remembered):
        return remembered
    for d in [DEFAULT_SCAPS_DIR] + _candidates():
        if scaps_exe(d):
            return d
    return None


# ---------------- writable folders ----------------
def writable(folder):
    """True if FermiPilot can create files in `folder`. A unique probe file is used: the app checks
    from several threads at once, and a shared probe name made one check fail (seen 28-10-2026)."""
    import tempfile
    try:
        os.makedirs(folder, exist_ok=True)
        fd, probe = tempfile.mkstemp(prefix=".fermipilot_write_test_", dir=folder)
        os.close(fd)
        os.remove(probe)
        return True
    except OSError:
        return False


_cache = {}
_lock = threading.Lock()


def data_root(scaps_dir):
    """Folder that holds FermiPilot_runs and FermiPilot_workers for this SCAPS installation.
    Decided once per SCAPS folder and remembered, so results never get split between two places."""
    key = os.path.normcase(os.path.abspath(scaps_dir))
    with _lock:
        if key not in _cache:
            saved = load_settings().get("data_roots", {}).get(key)
            if saved and writable(saved):
                _cache[key] = saved
            else:
                _cache[key] = scaps_dir if writable(scaps_dir) else os.path.join(app_data(), "data")
                roots = load_settings().get("data_roots", {})
                roots[key] = _cache[key]
                save_setting("data_roots", roots)
        return _cache[key]


def runs_root(scaps_dir):
    return os.path.join(data_root(scaps_dir), "FermiPilot_runs")


def workers_root(scaps_dir):
    return os.path.join(data_root(scaps_dir), "FermiPilot_workers")


def models_dir():
    """Models (.def) the user uploaded in the app."""
    d = os.path.join(app_data(), "models")
    os.makedirs(d, exist_ok=True)
    return d


def work_dir(app_dir):
    """Where the app keeps its logs and study files: next to the app in the source version,
    in %LOCALAPPDATA%\\FermiPilot for the packaged .exe (its own folder may be read-only)."""
    if not frozen() and writable(app_dir):
        return app_dir
    return app_data()
