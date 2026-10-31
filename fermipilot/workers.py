"""Private SCAPS copies so several simulations can run at the same time.

SCAPS keeps one error log (SCAPSErrorLogFile.log) and its panel-state files in its own folder,
so two copies started from the same folder would overwrite each other's messages. Each worker
therefore gets its own folder under <scaps_dir>\\FermiPilot_workers\\w<N> holding the program
and the data folders SCAPS reads (absorption, spectrum, ...). Nothing is downloaded or shared:
it is the user's own SCAPS installation, copied on the same PC.
"""
import os
import shutil

from . import paths

# folders that are user work or FermiPilot output, never needed by a worker
SKIP_DIRS = {"fermipilot", "fermipilot_runs", "fermipilot_workers", "results", "script", "def"}
SKIP_EXT = (".log", ".pdf", ".bat")


def _signature(scaps_dir):
    """Changes when the SCAPS installation changes, so stale copies are refreshed."""
    parts = []
    for name in sorted(os.listdir(scaps_dir)):
        p = os.path.join(scaps_dir, name)
        if os.path.isfile(p) and name.lower().endswith((".exe", ".dll", ".uir")):
            parts.append(f"{name}:{os.path.getsize(p)}:{int(os.path.getmtime(p))}")
    return "|".join(parts)


def prepare(scaps_dir, n, log=print):
    """Return n folders, each a working copy of SCAPS. n == 1 uses the installation itself when its
    folder can be written to (SCAPS writes its error log there); otherwise a copy is used too."""
    if n <= 1 and paths.data_root(scaps_dir) == scaps_dir:
        return [scaps_dir]
    n = max(1, n)
    root = paths.workers_root(scaps_dir)
    sig = _signature(scaps_dir)
    dirs = []
    for i in range(1, n + 1):
        w = os.path.join(root, f"w{i}")
        marker = os.path.join(w, "fermipilot_copy.txt")
        fresh = os.path.exists(marker) and open(marker, encoding="utf-8").read() == sig
        if not fresh:
            log(f"Preparing SCAPS copy {i} of {n} (one-time) ...")
            if os.path.isdir(w):
                shutil.rmtree(w, ignore_errors=True)
            os.makedirs(w, exist_ok=True)
            for name in os.listdir(scaps_dir):
                src = os.path.join(scaps_dir, name)
                if os.path.isdir(src):
                    if name.lower() not in SKIP_DIRS:
                        shutil.copytree(src, os.path.join(w, name))
                elif not name.lower().endswith(SKIP_EXT):
                    shutil.copy2(src, os.path.join(w, name))
            os.makedirs(os.path.join(w, "def"), exist_ok=True)
            with open(marker, "w", encoding="utf-8") as f:
                f.write(sig)
        dirs.append(w)
    return dirs


def suggested(cpu=None):
    """Default number of SCAPS copies at once: 2.
    Measured on a Ryzen 7 4700U laptop (8 cores, 15 W) with the session engine, 16 runs:
    1 copy 4.8 s/run, 2 copies 2.1 s/run, 4 copies 2.7 s/run - more copies lower every core's clock.
    A machine with fewer than 4 logical CPUs gets 1."""
    cpu = cpu or os.cpu_count() or 2
    return 2 if cpu >= 4 else 1


def maximum(cpu=None):
    """More copies than physical cores only slows every run down."""
    cpu = cpu or os.cpu_count() or 2
    return max(1, cpu // 2)
