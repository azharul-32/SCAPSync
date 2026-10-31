"""One-time speed check: how many SCAPS copies at once are fastest on this PC.

More copies is not always faster: on a 15 W laptop (Ryzen 7 4700U) 2 copies gave 2.1 s/run but
4 copies 2.7 s/run, because the chip lowers every core's clock. The best number depends on the
PC, so it is measured once, with the user's own model, and stored in
<scaps_dir>\\FermiPilot_workers\\speed.json. It is measured again only if the CPU count or the
SCAPS program changes.

The check runs the unchanged model (identical runs, so the run time does not depend on the
parameter values) through the same session engine as a real study, 6 runs per copy, for
1, 2, 3, ... copies. It stops as soon as one more copy is not at least 5 % faster.
Typical cost: 2 to 3 minutes, once.
"""
import json
import os
import shutil
import time

from . import paths, workers

RUNS_PER_COPY = 6
MIN_GAIN = 0.05
MIN_STUDY = 30      # studies smaller than this never wait for the check; they use the stored or default value


def _path(scaps_dir):
    return os.path.join(paths.workers_root(scaps_dir), "speed.json")


def _key(scaps_dir):
    return {"cpu": os.cpu_count(), "scaps": workers._signature(scaps_dir)}


def stored(scaps_dir):
    """The stored measurement if it is still valid for this PC and SCAPS, else None."""
    try:
        with open(_path(scaps_dir), encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    k = _key(scaps_dir)
    if d.get("cpu") != k["cpu"] or d.get("scaps") != k["scaps"] or int(d.get("best", 0)) < 1:
        return None
    return d


def best_copies(scaps_dir):
    d = stored(scaps_dir)
    return int(d["best"]) if d else workers.suggested()


def measure(scaps_dir, base_def, meas, light=True, log=print):
    """Run the check and store the result. Returns the stored dict."""
    from . import study            # late import: study imports this module
    top = min(workers.maximum(), 4)
    log(f"One-time speed check of this PC (identical runs of your model, 1 to {top} SCAPS copies; "
        "about 2-3 minutes). The result is stored and used for every later study.")
    workers.prepare(scaps_dir, top, log)        # copying SCAPS is not part of the timing
    tried, best_k, best_t = {}, 1, None
    for k in range(1, top + 1):
        name = f"_speedcheck/copies_{k}"
        shutil.rmtree(os.path.join(paths.runs_root(scaps_dir), name), ignore_errors=True)
        n = RUNS_PER_COPY * k
        st = {"name": name, "scaps_dir": scaps_dir, "base_def": base_def, "light": light,
              "measurements": meas, "design": {"type": "list", "runs": [{} for _ in range(n)]},
              "timeout_s": 60, "retry": False, "parallel": k}
        t0 = time.time()
        study.run_study(st, log=lambda m: None)
        per_run = (time.time() - t0) / n
        ok = _count_ok(os.path.join(paths.runs_root(scaps_dir), name, "master.csv"))
        tried[k] = {"s_per_run": round(per_run, 2), "ok": ok, "runs": n}
        log(f"  {k} cop{'ies' if k > 1 else 'y'}: {per_run:.1f} s per run ({ok}/{n} ok)")
        if ok < n:
            break                               # this many copies made runs fail; do not go higher
        if best_t is None or per_run < best_t * (1 - MIN_GAIN):
            best_k, best_t = k, per_run
        else:
            break
    d = dict(_key(scaps_dir), best=best_k, s_per_run=best_t, tried=tried, measured=time.time())
    os.makedirs(os.path.dirname(_path(scaps_dir)), exist_ok=True)
    with open(_path(scaps_dir), "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    shutil.rmtree(os.path.join(paths.runs_root(scaps_dir), "_speedcheck"), ignore_errors=True)
    log(f"Speed check done: {best_k} SCAPS cop{'ies' if best_k > 1 else 'y'} at once is fastest on this PC.")
    return d


def _count_ok(path):
    import csv
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return sum(r["status"].startswith("ok") for r in csv.DictReader(f))
    except OSError:
        return 0
