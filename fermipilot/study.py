"""Plan and run a study: a list of runs, each a set of parameter values.

Study file (JSON):
{
  "name": "acceptance",
  "scaps_dir": "C:\\\\Program Files (x86)\\\\Scaps3309",       (optional)
  "base_def": "def\\\\CdSeTe Model.def",                        (relative to scaps_dir, or absolute)
  "light": true,
  "measurements": {"iv": {"start": 0, "stop": 1, "points": 41}, "eb": true, "genrec": true},
  "fixed": {"L2.mun": 100},                                      (applied to every run)
  "design": {"type": "list", "runs": [{}, {"L3.Eg@B": 1.43}]}
           | {"type": "oat",  "params": {"L2.mun": [10, 50, 100]}}
           | {"type": "grid", "params": {"L2.mun": [10, 100], "L2.NA": [1e15, 1e16]}}
           | {"type": "linked", "params": {"WP.temperature": [280, 300, 320], "L2.Eg": [1.52, 1.50, 1.48]}}
           | {"type": "lhs",  "n": 20, "seed": 1,
              "params": {"L2.mun": [10, 100, "lin"], "L2.NA": [1e14, 1e17, "log"]}},
  "timeout_s": 30,          (a run that takes longer is stopped; raised automatically to 2x the
                             typical run time if the study's normal runs are slower)
  "retry": true,
  "parallel": "auto"        (SCAPS copies running at once; "auto" = the number measured fastest on this
                             PC by a one-time speed check, see speedcheck.py; 1 = one after another)
}
Results go to <scaps_dir>\\FermiPilot_runs\\<name>\\ (or %LOCALAPPDATA%\\FermiPilot\\data\\... when the SCAPS
folder cannot be written to, see paths.py): one folder per run plus master.csv.
Re-running the same study skips runs already finished (resume).
"""
import csv
import itertools
import json
import math
import os
import random
import threading
import time

from . import catalogue, defmodel, jobs, paths, results, runner, session, speedcheck, verify, workers


# ---------------- designs ----------------
def _clean(v):
    """Round away floating-point noise (6.999999999999999e+17 -> 7e+17). File names stay as they are."""
    if isinstance(v, str):
        return v
    return float(f"{float(v):.10g}")


def plan(design):
    return [{k: _clean(v) for k, v in r.items()} for r in _plan(design)]


def _plan(design):
    t = design["type"]
    if t == "list":
        return [dict(r) for r in design["runs"]]
    if t == "oat":
        return [{k: v} for k, vals in design["params"].items() for v in vals]
    if t == "linked":
        # SCAPS batch 'simultaneous': all parameters step together, value i of each in run i
        lists = list(design["params"].values())
        if len({len(v) for v in lists}) > 1:
            raise ValueError("Parameters that change together need the same number of steps (or values).")
        keys = list(design["params"])
        return [dict(zip(keys, combo)) for combo in zip(*lists)]
    if t == "grid":
        keys = list(design["params"])
        return [dict(zip(keys, combo)) for combo in itertools.product(*design["params"].values())]
    if t == "lhs":
        return latin_hypercube(design["params"], int(design["n"]), design.get("seed", 1))
    raise ValueError(f"unknown design type '{t}'")


def latin_hypercube(params, n, seed=1):
    rng = random.Random(seed)
    cols = {}
    for k, spec in params.items():
        lo, hi = float(spec[0]), float(spec[1])
        scale = spec[2] if len(spec) > 2 else "lin"
        strata = [(i + rng.random()) / n for i in range(n)]
        rng.shuffle(strata)
        if scale == "log":
            cols[k] = [10 ** (math.log10(lo) + u * (math.log10(hi) - math.log10(lo))) for u in strata]
        else:
            cols[k] = [lo + u * (hi - lo) for u in strata]
    return [{k: cols[k][i] for k in params} for i in range(n)]


# ---------------- running ----------------
CHUNK = 8          # runs per SCAPS start in the session engine

LEDGER_FIELDS = ["run_id", "status", "attempts", "elapsed_s", "proof", "Voc_V", "Jsc_mAcm2",
                 "FF_pct", "eta_pct", "Vmpp_V", "Jmpp_mAcm2", "note", "engine", "errors", "folder"]

# what each status means (shown in the app and the docs)
STATUS_HELP = {
    "ok": "SCAPS solved the run and used exactly the requested values.",
    "ok with warnings": "Solved, but SCAPS logged a message; see the errors column.",
    "no convergence": "SCAPS tried and reported a convergence failure.",
    "not calculated": "SCAPS stopped before solving and gave no reason. Seen for unphysical values "
                      "(e.g. a band offset of more than ~1 eV).",
    "timeout": "SCAPS was still busy at the time limit and was stopped.",
    "no output": "No result was written; see the errors column.",
    "rejected (proof)": "SCAPS did not take the requested value; the result is not used.",
    "incomplete I-V": "SCAPS stopped the I-V sweep before Voc (convergence trouble). Its Voc, FF and efficiency "
                      "would be extrapolated, so they are left out; Jsc is kept.",
}


def _read_ledger(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["run_id"]: r for r in csv.DictReader(f)}


def _write_ledger(path, rows, param_keys):
    extra = []                       # band-diagram quantities (eb_...), in the order SCAPS lists them
    for r in rows.values():
        extra += [k for k in r if k.startswith("eb_") and k not in extra]
    fields = LEDGER_FIELDS[:2] + param_keys + LEDGER_FIELDS[2:-2] + extra + LEDGER_FIELDS[-2:]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for rid in sorted(rows):
            w.writerow(rows[rid])


class StudyRunner:
    """Holds everything a study needs so single runs can be requested one by one
    (used by fixed designs, the boundary finder and inverse-design verification)."""

    def __init__(self, study, log=print):
        self.study, self.log = study, log
        self.scaps_dir = study.get("scaps_dir", runner.DEFAULT_SCAPS_DIR)
        base_def = study["base_def"]
        if not os.path.isabs(base_def):
            base_def = os.path.join(self.scaps_dir, base_def)
        self.base = defmodel.DefFile.load(base_def)
        self.meas = study.get("measurements", {"iv": {}})
        self.light = study.get("light", True)
        self.fixed = study.get("fixed", {})
        self.timeout = study.get("timeout_s", 30)
        self.retry = study.get("retry", True)
        self.params = {}
        self.out_root = os.path.join(paths.runs_root(self.scaps_dir), study["name"])
        os.makedirs(self.out_root, exist_ok=True)
        with open(os.path.join(self.out_root, "study.json"), "w", encoding="utf-8") as f:
            json.dump(study, f, indent=2)
        self.ledger_path = os.path.join(self.out_root, "master.csv")
        self.ledger = _read_ledger(self.ledger_path)
        self.lock = threading.Lock()
        # where a whole-model batch (MODEL.file) looks for its .def files
        self.model_dirs = [os.path.dirname(self.base.path), os.path.join(self.scaps_dir, "def"), paths.models_dir()]
        self.parallel = 1
        self.keys = sorted({k for r in self.ledger.values() for k in r if "." in k} | set(self.fixed))

    def resolve(self, keys):
        for k in keys:
            if k not in self.params:
                self.params[k] = catalogue.resolve(k, self.base)

    def already_done(self, run_id, values):
        """True only if this run_id finished ok with the same parameter values
        (a study edited and re-run under the same name must not reuse old results).
        With "keep_failed": true in the study (datasets), a recorded failure counts as done too, so resuming
        an interrupted study never gives earlier failures a second chance the later runs do not get."""
        r = self.ledger.get(run_id)
        done = ("ok", "ok with warnings") + ((
            "no convergence", "timeout", "not calculated", "incomplete I-V") if self.study.get("keep_failed") else ())
        if not r or r.get("status") not in done:
            return False
        for k, v in values.items():
            if isinstance(v, str):
                if r.get(k) != v:
                    return False
                continue
            try:
                old = float(r.get(k, "nan"))
            except ValueError:
                return False
            if not math.isclose(old, float(v), rel_tol=1e-9, abs_tol=1e-12):
                return False
        return True

    def time_limit(self):
        """The study's time limit. With several SCAPS copies sharing the CPU each run is slower, so the
        limit grows by half for every extra copy (tested: 6 copies on a 4-core laptop made every run
        exceed 30 s). It is also raised to 2x the median of finished runs if those are slow."""
        times = sorted(float(r["elapsed_s"]) for r in self.ledger.values()
                       if r.get("status") == "ok" and r.get("elapsed_s") not in ("", None))
        typical = times[len(times) // 2] if times else 0
        base = float(self.timeout) * (1 + 0.5 * (self.parallel - 1))
        return max(base, 2 * typical)

    def run(self, values, run_id=None, scaps_dir=None):
        """Run one set of values (fixed values added). Skips a run already done with the same values.
        scaps_dir: which SCAPS copy to use (parallel workers each pass their own)."""
        values = dict(self.fixed, **{k: _clean(v) for k, v in values.items()})
        with self.lock:
            self.resolve(values)
            if run_id is None:
                run_id = f"run_{len(self.ledger) + 1:04d}"
            if self.already_done(run_id, values):
                return self.ledger[run_id]
            limit = self.time_limit()
        if scaps_dir is None:
            scaps_dir = workers.prepare(self.scaps_dir, 1, self.log)[0]
        row = run_one(run_id, values, self.params, self.base, scaps_dir, self.out_root,
                      self.meas, self.light, limit, self.retry, model_dirs=self.model_dirs)
        with self.lock:
            self.ledger[run_id] = row
            self.keys = sorted(set(self.keys) | set(values))
            _write_ledger(self.ledger_path, self.ledger, self.keys)
        return row

    def stop_requested(self):
        f = os.path.join(self.out_root, "STOP")
        if os.path.exists(f):
            os.remove(f)
            return True
        return False


def run_study(study, log=print):
    sr = StudyRunner(study, log)
    runs = [dict(sr.fixed, **r) for r in plan(study["design"])]
    # resolve every key once, so a typo stops the study before SCAPS starts
    sr.resolve({k for r in runs for k in r})
    keys = sorted({k for r in runs for k in r})
    # Tested 24-10-2026: a spectrum set by 'action spectrumfile' stays in force for the next runs of the
    # same SCAPS session, even after reloading the model and 'clear actions'. A working-point or
    # illumination value must therefore be given for every run, or for none.
    for k in keys:
        if sr.params[k].kind == "action":
            lacking = [i + 1 for i, r in enumerate(runs) if k not in r]
            if lacking:
                raise ValueError(f"{k} is given for some runs but not for run(s) {lacking[:5]}"
                                 f"{' ...' if len(lacking) > 5 else ''}: give it for every run (or for none), "
                                 "because SCAPS keeps such a setting from one run to the next.")
    sr.stop_requested()                       # clear a stale stop request
    asked = study.get("parallel", 1)
    if asked == "auto":
        # the number of copies that is fastest on this PC: measured once (speedcheck.py), else 2
        if speedcheck.stored(sr.scaps_dir) is None and len(runs) >= speedcheck.MIN_STUDY:
            _progress(sr.out_root, len(runs), 0, "one-time speed check")
            try:
                speedcheck.measure(sr.scaps_dir, sr.base.path, {"iv": sr.meas.get("iv", {})}, sr.light, log)
            except Exception as e:                  # noqa: BLE001 - a failed check must not stop the study
                log(f"Speed check failed ({e}); using the default number of SCAPS copies.")
        asked = speedcheck.best_copies(sr.scaps_dir)
    asked = int(asked)
    n_par = max(1, min(asked, workers.maximum(), len(runs) or 1))
    if n_par < asked and n_par < len(runs):
        log(f"Using {n_par} SCAPS copies, not {asked}: more copies than CPU cores slows every run down.")
    log(f"FermiPilot study '{study['name']}': {len(runs)} runs, parameters: {', '.join(keys) or 'none'}"
        + (f", {n_par} SCAPS copies at once" if n_par > 1 else ""))
    sr.parallel = n_par
    dirs = workers.prepare(sr.scaps_dir, n_par, log)
    engine = study.get("engine", "session")
    todo = [(f"run_{n:04d}", dict(sr.fixed, **{k: _clean(v) for k, v in v0.items()}))
            for n, v0 in enumerate(runs, 1)]
    state = {"queue": list(todo), "done": 0, "stopped": False, "requeued": {}, "recorded": set(),
             "session_times": []}

    def session_limit():
        """How long one run may take inside a session before it is moved to its own SCAPS start:
        5x the typical session run (at least 20 s), never more than the study's time limit."""
        t = sorted(state["session_times"])
        typical = t[len(t) // 2] if t else None
        full = sr.time_limit()
        return full if typical is None else min(full, max(20.0, 5 * typical))
    t0 = time.time()

    def record(row):
        with sr.lock:
            sr.ledger[row["run_id"]] = row
            state["recorded"].add(row["run_id"])
            sr.keys = sorted(set(sr.keys) | {k for k in row if "." in k and k != "folder"})
            _write_ledger(sr.ledger_path, sr.ledger, sr.keys)
            state["done"] += 1
            _progress(sr.out_root, len(todo), state["done"], row["run_id"])
            log(f"  {row['run_id']}: {row['status']}  eta={row.get('eta_pct', '')}  "
                f"proof={row['proof']}  {row['elapsed_s']} s")

    def take():
        """Next runs for one SCAPS start: up to CHUNK, fewer near the end so all copies finish together."""
        with sr.lock:
            if state["stopped"]:
                return None
            if sr.stop_requested():
                state["stopped"] = True
                return None
            out = []
            size = 1 if engine == "classic" else max(1, min(CHUNK, -(-len(state["queue"]) // n_par)))
            # every run is checked, not only the first: a resumed study must never redo a finished run
            # (seen 30-10-2026: a chunk starting at an unfinished run re-ran the 7 finished runs after it)
            while state["queue"] and len(out) < size:
                rid, vals = state["queue"].pop(0)
                if sr.already_done(rid, vals):
                    state["done"] += 1
                    log(f"  {rid}: already done, skipped")
                    continue
                out.append((rid, vals))
            if not out:
                return None
            return out

    def classic(rid, vals, wdir, retry_first=False):
        return run_one(rid, vals, sr.params, sr.base, wdir, sr.out_root, sr.meas, sr.light,
                       sr.time_limit(), sr.retry, retry_first=retry_first, model_dirs=sr.model_dirs)

    def worker(wdir, idx):
        """A worker never dies silently: an unexpected error puts its unfinished runs back in the
        queue once, and the second time records them as 'error' so the study still ends complete."""
        while True:
            chunk = take()
            if not chunk:
                return
            try:
                work(wdir, idx, chunk)
            except Exception as e:                      # noqa: BLE001
                log(f"  worker {idx + 1}: unexpected error ({e}); its runs are handed back")
                with sr.lock:
                    pending = [(r, v) for r, v in chunk if r not in state["recorded"]]
                    retry_ = [(r, v) for r, v in pending if state["requeued"].get(r, 0) < 2]
                    for r, _ in pending:
                        state["requeued"][r] = state["requeued"].get(r, 0) + 1
                    failed = [(r, v) for r, v in pending if (r, v) not in retry_]
                    state["queue"][0:0] = retry_
                for r, v in failed:
                    record({"run_id": r, **v, "status": "error", "attempts": 0, "elapsed_s": 0,
                            "proof": "-", "errors": str(e)[:500], "engine": engine})

    def work(wdir, idx, chunk):
        if len(chunk) == 1 and engine == "classic":
            record(classic(chunk[0][0], chunk[0][1], wdir))
            return
        jobs_ = [prepare_run(rid, vals, sr.params, sr.base, sr.out_root, sr.model_dirs) for rid, vals in chunk]
        limit = session_limit()
        done, _, rest = session.run_chunk(wdir, jobs_, sr.meas, sr.light, limit,
                                          os.path.join(sr.out_root, f"session_w{idx}.script"))
        for job, info in done:
            if info["timed_out"]:
                # Tested: a run can stall inside a session yet finish normally in a fresh SCAPS
                # (SCAPS keeps some state between runs of one session). So a stalled run gets its
                # own SCAPS start with the full time limit before it is called a timeout.
                row = classic(job["run_id"], job["values"], wdir)
                row["note"] = "; ".join(x for x in (row.get("note"), "stalled in a session, rerun alone") if x)
                record(row)
                continue
            row = finalize_run(job, info, sr.meas, sr.light, 1, limit, engine="session")
            if row["status"] == "no convergence" and sr.retry:
                row = classic(job["run_id"], job["values"], wdir, retry_first=True)
            elif row["status"].startswith("ok"):
                with sr.lock:
                    state["session_times"].append(info["elapsed_s"])
            record(row)
        back = []
        for job in rest:
            n = state["requeued"].get(job["run_id"], 0) + 1
            state["requeued"][job["run_id"]] = n
            if n >= 2:          # SCAPS twice ended before this run: give it its own SCAPS start
                record(classic(job["run_id"], job["values"], wdir))
            else:
                back.append((job["run_id"], job["values"]))
        if back:
            with sr.lock:
                state["queue"][0:0] = back

    _progress(sr.out_root, len(todo), 0, "starting")
    threads = [threading.Thread(target=worker, args=(d, i), daemon=True) for i, d in enumerate(dirs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if state["stopped"]:
        log("Stopped by user. Run the study again to resume.")
        _progress(sr.out_root, len(todo), state["done"], "stopped")
        return sr.ledger_path
    _progress(sr.out_root, len(todo), len(todo), "finished")
    log(f"Done in {time.time() - t0:.0f} s. Master dataset: {sr.ledger_path}")
    return sr.ledger_path


def _progress(out_root, total, done, current):
    with open(os.path.join(out_root, "progress.json"), "w", encoding="utf-8") as f:
        json.dump({"total": total, "done": done, "current": current, "time": time.time()}, f)


def prepare_run(run_id, values, params, base, out_root, model_dirs=()):
    """Make the run folder: its own copy of the model (with any .def edits), output paths,
    and the script-route assignments. Shared by the one-run and the session engines."""
    folder = os.path.join(out_root, run_id)
    os.makedirs(folder, exist_ok=True)
    if "MODEL.file" in values:
        # a batch over whole models (SCAPS "variation of definition files"): the file is looked up
        # in the folder of the study's model (SCAPS's def folder), or used as given if absolute
        src = values["MODEL.file"]
        if not os.path.isabs(src):
            found = [os.path.join(d, src) for d in (model_dirs or [os.path.dirname(base.path)])
                     if os.path.exists(os.path.join(d, src))]
            if not found:
                raise FileNotFoundError(f"model file {src} not found in {', '.join(model_dirs)}")
            src = found[0]
        base = defmodel.DefFile.load(src)
    model = defmodel.DefFile(base.text(), base.path)
    script_assign, all_assign = [], []
    for k, v in values.items():
        p = params[k]
        if p.kind == "model":
            continue
        all_assign.append((p, v))
        if p.route == "def":
            verify.apply_def_edit(model, p, v)
        else:
            script_assign.append((p, v))
    def_path = os.path.join(folder, "model.def")
    model.save(def_path)
    out = {k: os.path.join(folder, f"result.{k}") for k in ("iv", "cv", "cf", "qe", "eb")}
    out["genrec"] = os.path.join(folder, "result.gen")
    out["state"] = os.path.join(folder, "scaps_state.def")
    for f in list(out.values()):
        if os.path.exists(f):
            os.remove(f)
    return {"run_id": run_id, "values": values, "folder": folder, "def_path": def_path, "out": out,
            "script_assign": script_assign, "all_assign": all_assign}


def has_result(job, meas, light):
    out = job["out"]
    summary = results.iv_summary(out["iv"]) if os.path.exists(out["iv"]) else {}
    if "iv" in meas or "cv" in meas:
        # a result file can exist with no simulation in it (SCAPS writes the header even
        # when the calculation diverged), so count simulations instead of trusting the file
        ok = summary.get("n_simulations", 0) >= 1 and (not light or "Jsc_mAcm2" in summary)
    else:
        ok = True
    return ok, summary


def interfaces_um(*defs):
    """Interface positions (um from the left contact) from the layer thicknesses in the first readable
    model: SCAPS's own state file (thicknesses as used), else the run's model copy."""
    import itertools
    for p in defs:
        try:
            m = defmodel.DefFile.load(p)
            ds = [m.get_columns(s, "d")[0] * 1e6 for s in m.sections if s.kind == "layer"]
            return list(itertools.accumulate(ds))[:-1]
        except Exception:                           # noqa: BLE001
            continue
    return None


def finalize_run(job, info, meas, light, attempts, timeout, engine="one SCAPS start per run"):
    """Status, proof of input and run.json for one finished (or failed) run."""
    out, folder, all_assign = job["out"], job["folder"], job["all_assign"]
    have_iv, summary = has_result(job, meas, light)
    row = {"run_id": job["run_id"], "folder": folder, **dict(job["values"])}
    proof_txt = "no state file"
    if os.path.exists(out["state"]):
        checks = verify.prove(defmodel.DefFile.load(out["state"]), all_assign)
        bad = [c for c in checks if c["status"] not in ("ok", "not stored")]
        proof_txt = "ok" if not bad else "; ".join(f"{c['key']} {c['status']} (wanted {c['wanted']:g}, "
                                                  f"SCAPS {c['scaps'] if c['scaps'] is None else format(c['scaps'], 'g')})" for c in bad)
        with open(os.path.join(folder, "proof.json"), "w", encoding="utf-8") as f:
            json.dump(checks, f, indent=2)
    spe = job["values"].get("LIGHT.spectrum")
    if spe and have_iv:
        # a spectrum is not stored in the .def; the proof is the spectrum file named in SCAPS's .iv header
        if spe.lower() not in results.read_text(out["iv"]).lower():
            proof_txt = (f"LIGHT.spectrum: SCAPS did not report {spe}"
                         if proof_txt == "ok" else proof_txt + f"; LIGHT.spectrum: SCAPS did not report {spe}")
    errors = info["errors"]
    if info["timed_out"]:
        status = "timeout"
    elif not have_iv and "convergence failure" in errors.lower():
        status = "no convergence"
    elif not have_iv and "no iv-calculation is made yet" in errors.lower():
        status = "not calculated"
    elif not have_iv:
        status = "no output"
    elif proof_txt != "ok":
        status = "rejected (proof)"
    elif errors:
        status = "ok with warnings"
    else:
        status = "ok"
    iv_end = None
    if status.startswith("ok") and light and "iv" in meas:
        # SCAPS can stop an I-V sweep early (convergence trouble) and still print Voc, FF and eta,
        # extrapolated past the last point (seen 30-10-2026: Voc 8.6 V, FF 9.8 %). Not a usable result.
        try:
            cols, rows = results.table(out["iv"])
            if rows and rows[-1][1] < 0:
                iv_end = rows[-1][0]
                status = "incomplete I-V"
                summary = {k: v for k, v in summary.items() if k in ("Jsc_mAcm2", "n_simulations")}
        except Exception:                           # noqa: BLE001
            pass
    row.update(summary)
    if os.path.exists(out["eb"]):
        try:
            row.update(results.eb_extras(out["eb"], interfaces_um(out["state"], job["def_path"])))
        except Exception as e:                      # noqa: BLE001 - extras must never lose a run
            errors = (errors + f" | band diagram not read: {e}").strip(" |")
    note = ""
    if iv_end is not None:
        note = (f"SCAPS stopped the I-V at {iv_end:g} V, before Voc; its Voc, FF and efficiency would be "
                "extrapolated and are left out (Jsc kept)")
    elif status == "timeout":
        note = f"stopped after {timeout:.0f} s"
    elif status.startswith("ok") and light and summary.get("Jsc_mAcm2") is not None and abs(summary["Jsc_mAcm2"]) < 1:
        note = "Jsc below 1 mA/cm2: the device is blocked (check band offsets)"
    row["note"] = note
    row.update({"status": status, "attempts": attempts, "elapsed_s": info["elapsed_s"],
                "proof": proof_txt, "errors": errors.replace("\r", " ").replace("\n", " | ")[:500],
                "engine": engine})
    with open(os.path.join(folder, "run.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, default=str)
    return row


def run_one(run_id, values, params, base, scaps_dir, out_root, meas, light, timeout, retry, retry_first=False,
            model_dirs=()):
    """One run, one SCAPS start (used for single runs, the boundary finder and retries).
    retry_first: the run already failed to converge elsewhere; start with more working-point steps."""
    job = prepare_run(run_id, values, params, base, out_root, model_dirs)
    attempts, extra = (1, [(catalogue.resolve("WP.points"), 20)]) if retry_first else (0, [])
    while True:
        attempts += 1
        for f in list(job["out"].values()):
            if os.path.exists(f):
                os.remove(f)
        script = jobs.build_script(run_id, job["def_path"], job["script_assign"] + extra, meas, job["out"], light)
        script_path = os.path.join(job["folder"], "job.script")
        with open(script_path, "w", encoding="latin-1", newline="") as f:
            f.write(script)
        info = runner.run_script(scaps_dir, script_path, timeout)
        have_iv, _ = has_result(job, meas, light)
        diverged = "convergence failure" in info["errors"].lower()
        if not have_iv and diverged and retry and attempts == 1 and not extra:
            # more working-point steps is the manual's first remedy for divergence.
            # Not retried: time-outs (would double the wait) and runs SCAPS refused to start.
            extra = [(catalogue.resolve("WP.points"), 20)]
            continue
        break
    return finalize_run(job, info, meas, light, attempts, timeout)
