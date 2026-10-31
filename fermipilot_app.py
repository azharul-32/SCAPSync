"""SCAPSync - automation for SCAPS-1D. Local app.

Start: double-click SCAPSync.exe (packaged), or  py -3 launcher.py  (source), or  streamlit run fermipilot_app.py

Pages (sidebar):  New study  ·  Results  ·  Analyse  ·  Help
Design rule: ask only what changes the science. Time limits, number of SCAPS copies, retries and
file handling are decided by SCAPSync (values measured on real runs); they stay editable in the
study file for experts.
"""
import json
import math
import os
import subprocess
import sys
import time

import pandas as pd
import streamlit as st

from fermipilot import __version__, analysis, catalogue, defmodel, paths, sbf, speedcheck, study, verify, workers
from fermipilot.runner import find_exe

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = paths.work_dir(APP_DIR)      # logs and study files (AppData for the packaged .exe)
UPLOADED = " (uploaded)"
SECONDS_PER_RUN = 5           # measured (Ryzen 7 4700U): 4.8 s per I-V run, one SCAPS session
TIME_LIMIT = 30               # s per run with one copy; SCAPSync scales it for parallel copies
RED = "#e60000"

st.set_page_config(page_title="SCAPSync", layout="wide")


# ======================================================================
# helpers
# ======================================================================
def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{v:.4g}"


def nice_time(secs):
    secs = int(secs)
    if secs >= 3600:
        return f"{secs / 3600:.1f} h"
    if secs >= 60:
        return f"{round(secs / 60)} min"
    return f"{secs} s"


def recent_run_time():
    """Median time of one run in the newest study with 4+ runs on this PC (None if there is none).
    Runs of a study share the CPU like the next study will, so this already includes the slow-down."""
    for name in study_folders()[:5]:
        try:
            m = pd.read_csv(os.path.join(runs_root(), name, "master.csv"))
            t = m.loc[m["status"].astype(str).str.startswith("ok"), "elapsed_s"].dropna()
            if len(t) >= 4:
                return float(t.median())
        except Exception:
            continue
    return None


def copies():
    """SCAPS copies at once: measured fastest on this PC (one-time speed check), else 2."""
    return speedcheck.best_copies(scaps_dir())


def estimate_seconds(n_runs, extras=0):
    """Wall time for n runs with copies() copies at once: from this PC's latest study if there is one,
    otherwise from the development PC (13 s per run alone, about 25 % slower per extra copy)."""
    n_cop = copies()
    waves = math.ceil(n_runs / max(1, n_cop))
    per_wave = recent_run_time() or SECONDS_PER_RUN * (1 + 0.25 * (n_cop - 1))
    return waves * per_wave * (1 + 0.6 * extras)


@st.cache_data(show_spinner=False)
def load_model_text(path, mtime):
    with open(path, "rb") as f:
        return f.read().decode("latin-1")


@st.cache_resource(show_spinner="Looking for SCAPS on this PC…")
def _found_scaps():
    return paths.find_scaps()


def scaps_dir():
    """The SCAPS folder: chosen in Settings (remembered), else found automatically."""
    if "scaps_dir" not in st.session_state:
        st.session_state["scaps_dir"] = _found_scaps() or paths.DEFAULT_SCAPS_DIR
    return st.session_state["scaps_dir"]


def runs_root():
    return paths.runs_root(scaps_dir())


def scaps_found():
    try:
        find_exe(scaps_dir())
        return True
    except Exception:
        return False


def model_files():
    """{label: path} of every model: SCAPS's def folder, then models uploaded in the app."""
    out = {}
    def_dir = os.path.join(scaps_dir(), "def")
    if os.path.isdir(def_dir):
        for f in sorted(os.listdir(def_dir), key=str.lower):
            if f.lower().endswith(".def"):
                out[f] = os.path.join(def_dir, f)
    up = paths.models_dir()
    for f in sorted(os.listdir(up), key=str.lower):
        if f.lower().endswith(".def"):
            out[f + UPLOADED] = os.path.join(up, f)
    return out


def store_uploaded_model(up):
    """Keep an uploaded .def in SCAPSync's models folder (never in SCAPS's folder). Returns its label
    or raises ValueError if the file is not a SCAPS model."""
    data = up.getvalue()
    try:
        m = defmodel.DefFile(data.decode("latin-1"), up.name)
        if not m.layer_names():
            raise ValueError("no layers found")
    except Exception as e:                      # noqa: BLE001
        raise ValueError(f"{up.name} is not a SCAPS model file that SCAPSync can read ({e})") from None
    name = os.path.basename(up.name)
    base, ext = os.path.splitext(name)
    target, i = os.path.join(paths.models_dir(), name), 1
    while os.path.exists(target) and open(target, "rb").read() != data:
        i += 1
        name = f"{base} ({i}){ext}"
        target = os.path.join(paths.models_dir(), name)
    with open(target, "wb") as f:
        f.write(data)
    return name + UPLOADED


def current_model():
    """The model chosen on the New study page (default: CdSeTe Model.def or the first .def)."""
    files = model_files()
    defs = list(files)
    if not defs:
        return defs, None, None, None
    name = st.session_state.get("def_name")
    if name not in defs:
        name = "CdSeTe Model.def" if "CdSeTe Model.def" in defs else defs[0]
        st.session_state["def_name"] = name
    path = files[name]
    model = defmodel.DefFile(load_model_text(path, os.path.getmtime(path)), path)
    return defs, name, path, model


def param_label(key, model, names):
    p = catalogue.resolve(key, model)
    if p.layer:
        where = f"L{p.layer} {names.get(p.layer, '')}" + (f", defect {p.defect}" if p.defect else "")
    elif p.interface:
        where = f"interface {p.interface}, defect {p.ifdefect}"
    elif p.contact:
        where = f"{p.contact} contact"
    else:
        where = "conditions"
    now = verify.read_back(model, p) if p.kind != "action" else None
    if p.kind == "contact" and p.script_name == "workfunction" and now is None:
        now = "flat band"
    return f"{where} · {p.label} ({p.unit}) · now {fmt(now)}"


def part_of(key, model, names):
    p = catalogue.resolve(key, model)
    if p.layer:
        return f"Layer {p.layer} · {names.get(p.layer, '')}"
    if p.interface:
        return f"Interface {p.interface} · {names.get(p.interface, '')} / {names.get(p.interface + 1, '')}"
    if p.contact:
        return "Contacts (left = back, right = front)"
    return "Temperature, light and resistances"


def short_label(key, labels):
    lab = labels[key].split(" · ", 1)[-1]
    if ".D" in key and not key.startswith("I"):
        lab = f"defect {key.split('.D')[1].split('.')[0]}: " + lab
    if "@B" in key:
        lab += "  [pure B side of the grading]"
    return lab


ENERGY_SPAN = 0.3
DEFAULT_NOW = {"WP.temperature": 300.0, "WP.voltage": 0.0, "WP.points": 5.0, "EXT.Rs": 1.0, "EXT.Rsh": 1e4}
LOG_UNITS = ("cm-3", "cm-2", "cm/s", "cm2", "cm3/s", "cm6/s")


def default_range(p, now):
    """Physically sensible start: +-0.3 eV for energies, x0.1-x10 (log) for densities, velocities and
    cross sections, x0.5-x2 otherwise."""
    if not isinstance(now, float):
        now = DEFAULT_NOW.get(p.key, 4.5 if p.unit == "eV" else 1.0)
    if p.unit == "eV":
        return round(now - ENERGY_SPAN, 6), round(now + ENERGY_SPAN, 6), False
    if p.unit in LOG_UNITS and now > 0:
        return now / 10, now * 10, True
    if p.unit == "K":
        return now - 50, now + 50, False
    if now == 0:
        return 0.0, 1.0, False
    return now / 2, now * 2, False


def param_picker(prefix, keys, model, names, labels, exclude=()):
    """Choose a part of the device, then its parameters; switch to another part to add more. Choices in
    every part are kept (each part has its own list), and the chosen ones are listed per part below."""
    groups = {}
    for k in keys:
        if k not in exclude:
            groups.setdefault(part_of(k, model, names), []).append(k)
    import zlib
    sig = zlib.crc32("|".join(groups).encode()) % 100000      # a new model gets fresh lists
    gkeys = {g: f"{prefix}_ms_{i}_{sig}" for i, g in enumerate(groups)}
    for g, wk in gkeys.items():
        if wk in st.session_state:
            st.session_state[wk] = [k for k in st.session_state[wk] if k in groups[g]]
    count = {g: len(st.session_state.get(wk, [])) for g, wk in gkeys.items()}
    c = st.columns([2, 3])
    part = c[0].selectbox("Layer / part", list(groups), key=f"{prefix}_part_{sig}",
                          format_func=lambda g: g + (f"  ({count[g]} chosen)" if count[g] else ""))
    c[1].multiselect("Parameters of this part", groups[part], format_func=lambda k: short_label(k, labels),
                     key=gkeys[part], placeholder="Choose one or more")
    chosen = [k for g, wk in gkeys.items() for k in st.session_state.get(wk, [])]
    if chosen:
        rows = []
        for g, wk in gkeys.items():
            ks = st.session_state.get(wk, [])
            if ks:
                rows.append(f"**{g}:** " + ", ".join(short_label(k, labels).split(" · now")[0] for k in ks))
        cc = st.columns([6, 1])
        cc[0].caption("Chosen — " + "  \n".join(rows) + "  \nTo add parameters of another layer, pick that layer above.")
        if cc[1].button("Clear", key=f"{prefix}_clear", help="Remove all chosen parameters"):
            for wk in gkeys.values():
                st.session_state.pop(wk, None)
            st.rerun()
    return chosen


def values_from_range(lo, hi, steps, log):
    if steps == 1:
        return [lo]
    if log:
        a, b = math.log10(lo), math.log10(hi)
        return [10 ** (a + i * (b - a) / (steps - 1)) for i in range(steps)]
    return [lo + i * (hi - lo) / (steps - 1) for i in range(steps)]


def parse_list(text):
    return [float(t.strip()) for t in text.replace(";", ",").split(",") if t.strip()]


def launch(job, args):
    os.makedirs(os.path.join(WORK_DIR, "logs"), exist_ok=True)
    logf = open(os.path.join(WORK_DIR, "logs", f"{job}.log"), "w", encoding="utf-8")
    flags = 0x08000000 if os.name == "nt" else 0          # no console window
    # the packaged SCAPSync.exe runs studies itself ("SCAPSync.exe run study.json")
    cmd = [sys.executable] if paths.frozen() else [sys.executable, "-u", "-m", "fermipilot"]
    st.session_state.setdefault("jobs", {})[job] = subprocess.Popen(
        cmd + args, cwd=WORK_DIR,
        stdout=logf, stderr=subprocess.STDOUT, creationflags=flags)


def job_running(job):
    p = st.session_state.get("jobs", {}).get(job)
    return p is not None and p.poll() is None


def read_log(job, lines=12):
    lf = os.path.join(WORK_DIR, "logs", f"{job}.log")
    if not os.path.exists(lf):
        return ""
    with open(lf, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-lines:])


def live_log(job, lines=12):
    """A background job's log, refreshed every 3 s; the page reruns once when the job ends."""
    def body():
        st.code(read_log(job, lines) or "starting…")
        flag = f"_was_running_{job}"
        if job_running(job):
            st.session_state[flag] = True
        elif st.session_state.pop(flag, False):
            st.rerun()
    st.fragment(run_every=3)(body)()


def save_study_file(sj):
    os.makedirs(os.path.join(WORK_DIR, "studies"), exist_ok=True)
    path = os.path.join(WORK_DIR, "studies", f"{sj['name']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sj, f, indent=2)
    return path


def study_folders():
    root = runs_root()
    if not os.path.isdir(root):
        return []
    # names starting with "_" are SCAPSync's own test and speed-check runs, not the user's studies
    return sorted((d for d in os.listdir(root) if not d.startswith("_")
                   and os.path.exists(os.path.join(root, d, "master.csv"))),
                  key=lambda d: os.path.getmtime(os.path.join(root, d, "master.csv")), reverse=True)


def read_progress(name):
    pf = os.path.join(runs_root(), name, "progress.json")
    if not os.path.exists(pf):
        return None
    try:
        with open(pf, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def is_running(name):
    """True while a study is running: started from this browser session, or (after a page refresh)
    its progress file was updated in the last 10 minutes and it has not finished or stopped."""
    if st.session_state.get("running_name") == name and "study" in st.session_state.get("jobs", {}):
        return job_running("study")
    pg = read_progress(name)
    return bool(pg) and pg.get("current") not in ("finished", "stopped") and time.time() - pg.get("time", 0) < 600


def active_study():
    """The study that is running now, if any."""
    name = st.session_state.get("running_name")
    if name and is_running(name):
        return name
    root = runs_root()
    recent = sorted((d for d in (os.listdir(root) if os.path.isdir(root) else [])
                     if os.path.exists(os.path.join(root, d, "progress.json"))),
                    key=lambda d: os.path.getmtime(os.path.join(root, d, "progress.json")), reverse=True)
    for n in recent[:5]:
        if is_running(n):
            st.session_state["running_name"] = n
            return n
    return None


def base_study(name, def_path, meas=None, light=True):
    return {"name": name, "scaps_dir": scaps_dir(), "base_def": def_path, "light": light,
            "measurements": meas or {"iv": dict(DEFAULT_IV)},
            "fixed": {}, "design": {"type": "list", "runs": []},
            "timeout_s": TIME_LIMIT, "retry": True, "parallel": "auto"}


# ======================================================================
# page: New study
# ======================================================================
DESIGNS = {
    "Sweep":            "Change each parameter on its own through a range; everything else stays as in the "
                        "model. Shows each parameter's effect. 2 parameters × 5 steps = 10 runs.",
    "Every combination": "Run every combination of the chosen values. Shows how parameters interact. "
                         "2 parameters × 5 steps = 25 runs; 3 × 5 = 125.",
    "Change together":  "Change several parameters at the same time, step by step (SCAPS batch "
                        "'simultaneous'), e.g. temperature together with band gap. All need the same number of steps.",
    "Random sampling":  "Spread a chosen number of runs evenly but randomly over all ranges at once. "
                        "Covers many parameters with few runs; the best input for Analyse.",
    "Single simulation": "One run with the exact values you type.",
    "My own list":      "Upload a SCAPS batch file (.sbf) from SCAPS's Batch set-up, or a CSV with the exact runs.",
}


DEFAULT_IV = {"start": 0.0, "stop": 1.0, "points": 41, "stop_after_voc": True}


def measurement_ui():
    """Default: I-V under light, 0 to 1.2 V in 0.025 V steps, ending just after Voc.
    Tested 24-10-2026 (12 runs): ending after Voc gives identical Voc, Jsc, FF, efficiency and MPP;
    the sweep can therefore reach 1.2 V (so Voc is always found) at no extra cost."""
    with st.expander("Change the measurement (optional)"):
        light = st.toggle("Light on (AM1.5G from the model); off = dark", value=True, key="m_light")
        meas = {}
        c = st.columns(4)
        if c[0].checkbox("I-V", value=True, key="m_iv"):
            meas["iv"] = {"start": c[0].number_input("from (V)", value=DEFAULT_IV["start"], key="iv_a"),
                          "stop": c[0].number_input("to (V)", value=DEFAULT_IV["stop"], key="iv_b"),
                          "points": int(c[0].number_input("points", 2, 201, DEFAULT_IV["points"], key="iv_n"))}
            if light:
                meas["iv"]["stop_after_voc"] = True
        if c[1].checkbox("C-V", key="m_cv"):
            meas["cv"] = {"start": c[1].number_input("from (V)", value=-0.8, key="cv_a"),
                          "stop": c[1].number_input("to (V)", value=0.8, key="cv_b"),
                          "points": int(c[1].number_input("points", 2, 201, 81, key="cv_n"))}
        if c[2].checkbox("C-f", key="m_cf"):
            meas["cf"] = {"start": c[2].number_input("from (Hz)", value=1e2, format="%.3g", key="cf_a"),
                          "stop": c[2].number_input("to (Hz)", value=1e6, format="%.3g", key="cf_b"),
                          "per_decade": int(c[2].number_input("points per decade", 1, 50, 5, key="cf_n"))}
        if c[3].checkbox("QE", key="m_qe"):
            meas["qe"] = {"start": c[3].number_input("from (nm)", value=300.0, key="qe_a"),
                          "stop": c[3].number_input("to (nm)", value=900.0, key="qe_b"),
                          "points": int(c[3].number_input("points", 2, 201, 61, key="qe_n"))}
        if st.checkbox("Band diagram at short circuit (0 V) for every run: adds band offsets, electric field "
                       "and the recombination loss of each layer and interface to the results table "
                       "(saves the .eb and .gen files; each run takes about 30 % longer)", key="m_eb"):
            meas["eb"] = True
            meas["genrec"] = True
        if light and "iv" in meas and meas["iv"]["stop"] < 1.0:
            st.warning("The I-V stops below 1.0 V: if Voc is higher, SCAPS cannot report Voc and FF.")
    if not meas:
        meas = {"iv": dict(DEFAULT_IV)}
        if not light:
            meas["iv"].pop("stop_after_voc")
    parts = []
    if "iv" in meas:
        parts.append(f"I-V {meas['iv']['start']:g} → {meas['iv']['stop']:g} V ({meas['iv']['points']} points"
                     + (", ends just after Voc)" if meas["iv"].get("stop_after_voc") else ")"))
    parts += [n for k, n in (("cv", "C-V"), ("cf", "C-f"), ("qe", "QE")) if k in meas]
    if "eb" in meas:
        parts.append("band diagram at 0 V")
    st.write(("**Light** · " if light else "**Dark** · ") + " · ".join(parts))
    return light, meas


def progress_panel(name):
    """Live progress of a running study, refreshed every 3 s, with a stop button."""
    def body():
        pg = read_progress(name) or {"total": 0, "done": 0, "current": "starting"}
        running = is_running(name)
        total, done = max(pg["total"], 1), pg["done"]
        bad = 0
        mfile = os.path.join(runs_root(), name, "master.csv")
        if os.path.exists(mfile):
            try:
                m = pd.read_csv(mfile)
                bad = int((~m["status"].astype(str).str.startswith("ok")).sum())
            except Exception:
                pass
        if running:
            left = estimate_seconds(pg["total"] - done)
            st.progress(done / total, text=f"**{name}** · {done} of {pg['total']} runs done · "
                                           f"about {nice_time(left)} left"
                                           + (f" · {bad} not ok" if bad else ""))
            if st.button("Stop after the runs in progress", key="stop_btn"):
                open(os.path.join(runs_root(), name, "STOP"), "w").close()
                st.toast("Stopping… start the same study again later to continue where it stopped.")
        else:
            if pg.get("current") == "stopped":
                st.warning(f"**{name}** stopped after {done} of {pg['total']} runs. "
                           "Press Start again (same settings) to continue.")
            else:
                st.success(f"**{name}** finished: {done} runs" + (f", {bad} not ok" if bad else ", all ok") + ".")
            if st.button("See the results", type="primary", key="see_res"):
                st.session_state["results_pick"] = name
                st.switch_page(PAGES["results"])
        with st.expander("Details (log)"):
            st.code(read_log("study", 10) or "starting…")
        flag = "_study_was_running"
        if running:
            st.session_state[flag] = True
        elif st.session_state.pop(flag, False):
            st.rerun()
    st.fragment(run_every=3)(body)()


PARAM_STATE = ("vary_", "fixed_", "lo_", "hi_", "n_", "lg_", "rlo_", "rhi_", "rlg_", "v_", "x_", "f_")


def forget_parameters():
    """A different model has different parameters and values: start the choices afresh."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(PARAM_STATE):
            del st.session_state[k]


def page_new_study():
    st.title("SCAPSync")
    st.caption("Run, check, graph and design SCAPS-1D simulations in one place · an independent tool for SCAPS-1D")
    st.header("New study")
    st.subheader("1 · Model")
    up = st.file_uploader("Upload a SCAPS model (.def) from anywhere on this PC, or pick one below",
                          type=["def"], key="def_upload")
    if up is not None and st.session_state.get("_def_upload_id") != up.file_id:
        st.session_state["_def_upload_id"] = up.file_id
        try:
            st.session_state["def_name"] = store_uploaded_model(up)
            forget_parameters()
        except ValueError as e:
            st.error(str(e))
    defs, def_name, def_path, model = current_model()
    if model is None:
        st.info(f"No models found in {os.path.join(scaps_dir(), 'def')}. Upload a .def file above.")
        return
    running = active_study()
    st.selectbox("SCAPS model", defs, key="def_name", on_change=forget_parameters,
                 help="Models in your SCAPS def folder, and models you uploaded here. Your files are never changed.")
    names = model.layer_names()
    layers, contacts, interfaces = defmodel.summary(model)
    st.caption(" → ".join(f"{l['name']} ({fmt(l['d (um)'])} µm)" for l in layers))
    with st.expander("Show all layer, contact and interface values"):
        st.dataframe(pd.DataFrame(layers).map(fmt), hide_index=True)
        c1, c2 = st.columns(2)
        c1.dataframe(pd.DataFrame(contacts), hide_index=True)
        if interfaces:
            c2.dataframe(pd.DataFrame(interfaces), hide_index=True)
        st.caption("Graded layers show their pure A (y = 0) values; the pure B side can be chosen as a parameter.")

    # ---- 2. type
    st.subheader("2 · Type of study")
    design = st.radio("What do you want to find out?", list(DESIGNS), captions=list(DESIGNS.values()),
                      key="design", label_visibility="collapsed")

    # ---- 3. parameters
    keys, _ = catalogue.catalogue(model)
    labels = {k: param_label(k, model, names) for k in keys}
    st.subheader("3 · Parameters")
    specs, problems, design_json = {}, [], None
    if design == "My own list":
        st.write("Either a **SCAPS batch file (.sbf)** saved from SCAPS's Batch set-up panel, or a **CSV** "
                 "with one column per parameter and one row per run. CSV column names are parameter codes, e.g. "
                 "`L2.mun` (layer 2 electron mobility) or `L2.D1.Ntotal` (layer 2, defect 1 density).")
        st.download_button("Download a CSV template with every parameter code of this model",
                           pd.DataFrame(columns=keys).to_csv(index=False), "fermipilot_template.csv", "text/csv")
        up = st.file_uploader("Your .sbf or .csv file", type=["sbf", "csv"])
        runs = []
        if up is not None and up.name.lower().endswith(".sbf"):
            try:
                sdesign, sreport = sbf.read(up.getvalue().decode("latin-1"))
                runs = sdesign["runs"]
                missing = sorted({k for r in runs for k in r} - set(keys) - set(catalogue.FILE_PARAMS))
                if missing:
                    problems.append(f"This batch uses parameters that this model does not have: {', '.join(missing)}")
                for fk, (flabel, sub) in catalogue.FILE_PARAMS.items():
                    names_ = sorted({r[fk] for r in runs if fk in r})
                    folder_ = os.path.dirname(def_path) if fk == "MODEL.file" else os.path.join(scaps_dir(), sub)
                    lost = [n for n in names_ if not os.path.exists(os.path.join(folder_, n))]
                    if lost:
                        problems.append(f"These files of the batch are not in {folder_}: {', '.join(lost)}")
                st.success(f"SCAPS batch read: {len(sreport)} batch parameter(s) → {len(runs)} runs.")
                st.dataframe(pd.DataFrame(sreport).drop(columns=["label seen before"]), hide_index=True)
                if not all(r["label seen before"] for r in sreport):
                    st.info("Some batch labels were mapped by name but not yet seen in a real SCAPS file; "
                            "every run's proof check will confirm SCAPS used the right parameter.")
            except ValueError as e:
                problems.append(f"Could not use this batch file: {e}")
        elif up is not None:
            df = pd.read_csv(up)
            ok_cols = set(keys) | set(catalogue.FILE_PARAMS)      # MODEL.file / LIGHT.spectrum hold file names
            bad = [c for c in df.columns if c not in ok_cols]
            if bad:
                problems.append(f"These columns are not parameter codes of this model: {', '.join(bad)}")
            try:
                runs = [{k: (str(v) if k in catalogue.FILE_PARAMS else float(v)) for k, v in r.items()
                         if pd.notna(v) and k in ok_cols} for r in df.to_dict("records")]
            except ValueError as e:
                problems.append(f"A value in the CSV is not a number: {e}")
                runs = []
        for k in sorted({k for r in runs for k in r}):
            if (k in catalogue.ACTION_PARAMS or k == "LIGHT.spectrum") and any(k not in r for r in runs):
                problems.append(f"{k} is given for some runs only. Give it for every run (or none): SCAPS keeps "
                                "this setting from one run to the next.")
        design_json = {"type": "list", "runs": runs}
        chosen = []
    else:
        chosen = param_picker("vary", keys, model, names, labels)
        for k in chosen:
            p = catalogue.resolve(k, model)
            now = verify.read_back(model, p) if p.kind != "action" else None
            lo0, hi0, log0 = default_range(p, now)
            guess = now if isinstance(now, float) else DEFAULT_NOW.get(k, (lo0 + hi0) / 2)
            lab = f"**{catalogue.readable(k, model)}**"
            if design == "Single simulation":
                c = st.columns([3, 2], vertical_alignment="bottom")
                c[0].markdown(lab)
                specs[k] = c[1].number_input("value", value=float(guess), format="%.6g", key=f"v_{k}")
                continue
            if design == "Random sampling":
                c = st.columns([3, 1.3, 1.3, 0.8], vertical_alignment="bottom")
                c[0].markdown(lab)
                lo = c[1].number_input("from", value=float(lo0), format="%.6g", key=f"rlo_{k}")
                hi = c[2].number_input("to", value=float(hi0), format="%.6g", key=f"rhi_{k}")
                lg = c[3].checkbox("log", value=log0, key=f"rlg_{k}", help="Space the values evenly on a log scale")
                if lg and (lo <= 0 or hi <= 0):
                    problems.append(f"{k}: log spacing needs positive values")
                specs[k] = [lo, hi, "log" if lg else "lin"]
                continue
            c = st.columns([3, 1.3, 1.3, 0.9, 0.9], vertical_alignment="bottom")
            c[0].markdown(lab)
            lo = c[1].number_input("from", value=float(lo0), format="%.6g", key=f"lo_{k}")
            hi = c[2].number_input("to", value=float(hi0), format="%.6g", key=f"hi_{k}")
            n = c[3].number_input("steps", 1, 200, 5, key=f"n_{k}")
            lg = c[4].checkbox("log", value=log0, key=f"lg_{k}", help="Space the values evenly on a log scale")
            exact = st.session_state.get(f"x_{k}", "").strip()
            if exact:
                try:
                    specs[k] = parse_list(exact)
                except ValueError:
                    problems.append(f"{k}: could not read the exact values")
            elif lg and (lo <= 0 or hi <= 0):
                problems.append(f"{k}: log spacing needs positive values")
            else:
                specs[k] = values_from_range(lo, hi, int(n), lg)
        if chosen and design in ("Sweep", "Every combination", "Change together"):
            with st.expander("Use exact values instead of a range (optional)"):
                st.caption("Type values separated by commas; a filled box replaces that parameter's range.")
                for k in chosen:
                    st.text_input(f"`{k}` · {short_label(k, labels)}", key=f"x_{k}", placeholder="e.g. 10, 25, 50, 100")
        if design == "Single simulation":
            design_json = {"type": "list", "runs": [specs]}
        elif design == "Sweep":
            design_json = {"type": "oat", "params": specs}
        elif design == "Every combination":
            design_json = {"type": "grid", "params": specs}
        elif design == "Change together":
            design_json = {"type": "linked", "params": specs}
        else:
            n_runs = st.number_input("Number of runs", 2, 5000, 40, key="lhs_n",
                                     help="40 runs is enough to rank 5 parameters; 200+ for inverse design.")
            design_json = {"type": "lhs", "n": int(n_runs), "seed": int(st.session_state.get("seed", 1)),
                           "params": specs}
        if not chosen:
            st.info("Choose a part of the device, then the parameter(s) you want to change.")

    # ---- 4. measurement
    st.subheader("4 · Measurement")
    light, meas = measurement_ui()

    # ---- optional
    if "study_name" not in st.session_state:
        st.session_state["study_name"] = f"study_{time.strftime('%Y%m%d_%H%M')}"
    with st.expander("More options (optional)"):
        st.text_input("Study name (also the name of its results folder)", key="study_name")
        st.caption("Keep the same name to continue a stopped study; use a new name for a new study.")
        st.markdown("**Change a value for every run of this study** (without editing your .def file)")
        fixed_keys = param_picker("fixed", keys, model, names, labels, exclude=chosen)
        fixed = {}
        for k in fixed_keys:
            p = catalogue.resolve(k, model)
            now = verify.read_back(model, p) if p.kind != "action" else None
            fixed[k] = st.number_input(f"`{k}` · {short_label(k, labels)}",
                                       value=float(now) if isinstance(now, float) else DEFAULT_NOW.get(k, 1.0),
                                       format="%.6g", key=f"f_{k}")
        if design == "Random sampling":
            st.number_input("Random seed (the same seed gives the same samples)", 0, 10 ** 6, 1, key="seed")

    # ---- 5. run
    st.subheader("5 · Run")
    name = st.session_state["study_name"].strip() or "study"
    planned = []
    try:
        planned = study.plan(design_json) if design_json else []
    except Exception as e:
        problems.append(str(e))
    if design != "My own list" and not chosen:
        planned = []
    for pr in problems:
        st.error(pr)
    if running:
        progress_panel(running)
        return
    if st.session_state.get("running_name") and not running:
        progress_panel(st.session_state["running_name"])
    if planned and not problems:
        extras = sum(k in meas for k in ("cv", "cf", "qe"))
        n_cop = copies()
        check = speedcheck.stored(scaps_dir()) is None and len(planned) >= speedcheck.MIN_STUDY
        st.info(f"**{len(planned)} run{'s' if len(planned) != 1 else ''}** · about "
                f"**{nice_time(estimate_seconds(len(planned), extras))}** "
                + ("plus a one-time speed check of this PC (about 3 min)" if check else
                   f"({n_cop} SCAPS cop{'ies' if n_cop > 1 else 'y'} working at once)"))
        with st.expander("See the list of runs"):
            st.dataframe(pd.DataFrame(planned[:500]))
        if st.button("Start study", type="primary"):
            sj = base_study(name, def_path, meas, light)
            sj.update(fixed=fixed, design=design_json)
            launch("study", ["run", save_study_file(sj)])
            st.session_state["running_name"] = name
            st.session_state.pop("study_name", None)        # next study gets a fresh name
            st.rerun()


# ======================================================================
# page: Results
# ======================================================================
def page_results():
    st.title("Results")
    sts = study_folders()
    if not sts:
        st.info("No results yet. Start one on the New study page.")
        return
    pick0 = st.session_state.pop("results_pick", None)
    pick = st.selectbox("Study (newest first)", sts, index=sts.index(pick0) if pick0 in sts else 0)
    folder = os.path.join(runs_root(), pick)
    df = pd.read_csv(os.path.join(folder, "master.csv"))
    ok = df["status"].astype(str).str.startswith("ok")
    m = st.columns(4)
    m[0].metric("Runs", len(df))
    m[1].metric("OK", int(ok.sum()))
    m[2].metric("Not OK", int((~ok).sum()))
    if ok.any() and "eta_pct" in df:
        best = df[ok].loc[df[ok]["eta_pct"].idxmax()]
        m[3].metric("Best η", f"{best['eta_pct']:.2f}%", help=f"Best efficiency: {best['eta_pct']:.4f} % ({best['run_id']})")
    if (~ok).any():
        with st.expander(f"Why {int((~ok).sum())} run(s) are not OK"):
            for stt in sorted(set(df.loc[~ok, "status"].astype(str))):
                st.markdown(f"- **{stt}** ({int((df['status'] == stt).sum())}): {study.STATUS_HELP.get(stt, '')}")
            st.caption("To find which parameter makes runs fail, and where, use **Analyse → Convergence limit**.")

    ins = [c for c in df.columns if "." in c]
    N = study_names(folder, ins)
    outs = [c for c in analysis.OUTPUTS if c in df] + [c for c in df.columns if c.startswith("eb_")]
    if ins and outs:
        import altair as alt
        c1, c2, c3 = st.columns([2, 2, 1])
        x = c1.selectbox("Horizontal axis", ins, format_func=lambda k: N.get(k, k))
        y = c2.selectbox("Vertical axis", outs, format_func=lambda o: OUT_NAMES.get(o, o))
        numeric = pd.api.types.is_numeric_dtype(df[x])     # file-name parameters (models, spectra) are text
        logx = numeric and c3.checkbox("log scale", value=df[x].dropna().gt(0).all() and
                                       (df[x].max() / max(df[x].min(), 1e-300) > 50 if df[x].notna().any() else False))
        plot = df[ok & df[x].notna()].rename(columns=lambda c: c.replace(".", "_"))
        xf, yf = x.replace(".", "_"), y.replace(".", "_")
        st.altair_chart(alt.Chart(plot).mark_circle(size=70, color=RED).encode(
            x=alt.X(f"{xf}:Q", title=N.get(x, x), scale=alt.Scale(type="log" if logx else "linear", zero=False))
            if numeric else alt.X(f"{xf}:N", title=N.get(x, x)),
            y=alt.Y(f"{yf}:Q", title=OUT_NAMES.get(y, y), scale=alt.Scale(zero=False)),
            tooltip=["run_id", alt.Tooltip(f"{xf}:{'Q' if numeric else 'N'}", title=N.get(x, x)),
                     alt.Tooltip(f"{yf}:Q", title=y)]))
    shown = df.drop(columns=["folder", "proof", "attempts"], errors="ignore").copy()
    for c in shown.columns:
        if pd.api.types.is_float_dtype(shown[c]):
            shown[c] = shown[c].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
    st.dataframe(shown.rename(columns={**N, **OUT_NAMES}), hide_index=True)
    c = st.columns([1, 3])
    c[0].download_button("Download table (CSV)", df.to_csv(index=False), f"{pick}.csv", "text/csv", type="primary")
    c[1].caption(f"All files of every run (.iv, .eb, .gen, the exact model used) are in `{folder}`. "
                 "Open them in the SCAPS-1D Data Visualizer for publication plots.")


# ======================================================================
# page: Analyse
# ======================================================================
OUTPUT_GUESS = {"voc": "Voc_V", "jsc": "Jsc_mAcm2", "ff": "FF_pct", "efficiency": "eta_pct", "eta": "eta_pct",
                "eta_pct": "eta_pct", "voc_v": "Voc_V", "jsc_macm2": "Jsc_mAcm2", "ff_pct": "FF_pct"}
TOOLS = {
    "Parameter ranking":   "Which parameters change the result most. Best on a Random sampling study.",
    "Convergence limit":   "Why runs failed to converge, and the exact value where SCAPS starts to fail.",
    "Loss analysis":       "Where the generated current is lost in one run: bulk, interfaces or contacts.",
    "Inverse design":      "Give a target (e.g. 24 % efficiency); SCAPSync finds designs that reach it, verified in SCAPS.",
}
# the small "?" next to each tool's title: what to give it, what it does, where to look
TOOL_HELP = {
    "Parameter ranking": (
        "**What you need:** a finished study where several parameters changed (Random sampling is best; "
        "at least ~5 more OK runs than parameters).\n\n**What it does:** measures how strongly each parameter "
        "moves the chosen result. Failed runs are left out.\n\n**How to read it:** longer bar = more influence."),
    "Convergence limit": (
        "**1 · Choose a study with failed runs.** SCAPSync compares the failed and OK runs and names the "
        "parameter that causes the failures (works best on a Random sampling study with 20+ runs).\n\n"
        "**2 · Find the exact limit.** SCAPS is run again and again at the halfway value until the value where it "
        "starts to fail is known to about 0.2 %. Usually 5 to 10 runs.\n\n"
        "**Not sure about the cause?** *Confirm with SCAPS* changes a failed run back to an OK run one parameter "
        "at a time; the change that makes it converge is the cause.\n\n"
        "No study yet? Choose *No study: pick the parameter myself*."),
    "Loss analysis": (
        "**What you need:** one OK run from any study.\n\n**What it does:** splits the lost current by place "
        "(bulk, interfaces, contacts) and, in the bulk, by mechanism (defects/SRH, radiative, Auger), using SCAPS's "
        "own recombination currents.\n\n**Use it to:** see what limits a design before deciding what to improve."),
    "Inverse design": (
        "**Give a target, get designs that reach it, verified in SCAPS.**\n\n"
        "**Let SCAPSync run SCAPS (recommended):** choose the parameters SCAPSync may change, their ranges "
        "and a target. SCAPSync runs a small spread-out start, learns, then picks each next round of runs where "
        "they help most (likely hits, and places it is unsure about), until enough different designs are "
        "verified in SCAPS. Typically 2-4x fewer runs than random sampling for hard targets.\n\n"
        "**From a finished study:** learns from runs you already have and proposes designs; check them in SCAPS."),
}
MANUAL = "No study: pick the parameter myself"
AL_MODE = "Let SCAPSync run SCAPS"
OUT_NAMES = {"eta_pct": "Efficiency η [%]", "Voc_V": "Voc [V]", "Jsc_mAcm2": "Jsc [mA/cm²]", "FF_pct": "Fill factor [%]",
             "Vmpp_V": "V at max power [V]", "Jmpp_mAcm2": "J at max power [mA/cm²]"}
oname = lambda o: OUT_NAMES.get(o, o)           # noqa: E731


@st.cache_data(show_spinner=False)
def _names_for(model_path, mtime, keys):
    try:
        m = defmodel.DefFile(load_model_text(model_path, mtime), model_path)
    except Exception:                           # noqa: BLE001
        m = None
    return {k: catalogue.readable(k, m) for k in keys}


def study_names(folder, keys):
    """{'L4.chi': 'FTO (layer 4): electron affinity [eV]', ...} using the model of that study."""
    try:
        with open(os.path.join(folder, "study.json"), encoding="utf-8") as f:
            path = json.load(f)["base_def"]
        if not os.path.exists(path):            # study made on another PC: same model name in this SCAPS
            path = os.path.join(scaps_dir(), "def", os.path.basename(path.replace("\\", "/")))
        return _names_for(path, os.path.getmtime(path), tuple(keys))
    except Exception:                           # noqa: BLE001
        return {k: catalogue.readable(k) for k in keys}


def numeric_inputs(df):
    return [c for c in df.columns if "." in c and pd.api.types.is_numeric_dtype(df[c])
            and df[c].notna().all() and df[c].nunique() > 1]


def dataset_picker(prefix, keys):
    """Newest SCAPSync study by default; several can be combined; or a CSV from elsewhere."""
    sts = study_folders()
    src = st.radio("Data", ["SCAPSync studies", "A CSV from elsewhere"], horizontal=True, key=f"{prefix}_src")
    if src == "SCAPSync studies":
        if not sts:
            st.info("No studies yet.")
            return None, [], None
        chosen = st.multiselect("Studies (several are combined)", sts, default=sts[:1], key=f"{prefix}_st")
        if not chosen:
            return None, [], None
        df = pd.concat([pd.read_csv(os.path.join(runs_root(), n, "master.csv")) for n in chosen], ignore_index=True)
        inputs = [c for c in df.columns if "." in c and pd.api.types.is_numeric_dtype(df[c])
                  and df[c].notna().all() and df[c].nunique() > 1]
        if not inputs:
            st.warning("These studies vary no parameter in every run. Use a Random sampling or "
                       "Every combination study.")
        st.session_state[f"{prefix}_names"] = study_names(os.path.join(runs_root(), chosen[0]),
                                                          [c for c in df.columns if "." in c])
        return df, inputs, os.path.join(runs_root(), chosen[0], "study.json")
    up = st.file_uploader("CSV with one row per run", type=["csv"], key=f"{prefix}_up")
    if up is None:
        return None, [], None
    raw = pd.read_csv(up)
    st.caption("Tell SCAPSync what each column is: a parameter code, a result, or ignore.")
    choices = ["ignore"] + keys + analysis.OUTPUTS + ["status"]
    mapping, cols_ui = {}, st.columns(3)
    for i, c in enumerate(raw.columns):
        guess = c if c in keys else OUTPUT_GUESS.get(c.lower(), "status" if c.lower() == "status" else "ignore")
        mapping[c] = cols_ui[i % 3].selectbox(c, choices, index=choices.index(guess), key=f"{prefix}_map_{c}")
    df = raw.rename(columns={c: m for c, m in mapping.items() if m != "ignore"})[
        [m for m in mapping.values() if m != "ignore"]]
    df["status"] = (df["status"].map(lambda v: "ok" if str(v).lower() in ("ok", "success") else str(v))
                    if "status" in df else "ok")
    return df, [m for m in mapping.values() if "." in m], None


def page_analyse():
    st.title("Analyse")
    defs, def_name, def_path, model = current_model()
    if model is None:
        st.error("No model found.")
        return
    names = model.layer_names()
    keys, _ = catalogue.catalogue(model)
    labels = {k: param_label(k, model, names) for k in keys}
    tool = st.radio("Tool", list(TOOLS), captions=list(TOOLS.values()), horizontal=True, key="tool",
                    label_visibility="collapsed")
    st.divider()
    st.subheader(tool, help=TOOL_HELP[tool])

    if tool == "Parameter ranking":
        df, inputs, _ = dataset_picker("sens", keys)
        if df is not None and inputs:
            out = st.selectbox("Result", [o for o in analysis.OUTPUTS if o in df], format_func=oname)
            try:
                res = analysis.sensitivity(df, inputs, out)
                N = st.session_state.get("sens_names") or {}
                res["parameter"] = res["parameter"].map(lambda k: N.get(k) or catalogue.readable(k, model))
                import altair as alt
                metric = "rf_importance" if "rf_importance" in res else "src"
                st.altair_chart(alt.Chart(res).mark_bar(color=RED).encode(
                    x=alt.X(f"{metric}:Q", title="importance (Random Forest)" if metric == "rf_importance"
                            else "standardized coefficient"),
                    y=alt.Y("parameter:N", sort="-x", title=None, axis=alt.Axis(labelLimit=420))))
                st.caption(f"Based on {res.attrs['n']} OK runs. In *Numbers*, + means raising the parameter "
                           f"raises {oname(out)}.")
                with st.expander("Numbers"):
                    st.dataframe(res, hide_index=True)
            except ValueError as e:
                st.warning(str(e))

    elif tool == "Loss analysis":
        sts = study_folders()
        if not sts:
            st.info("No studies yet.")
            return
        c = st.columns(3)
        pick = c[0].selectbox("Study", sts, key="loss_st")
        mdf = pd.read_csv(os.path.join(runs_root(), pick, "master.csv"))
        mdf = mdf[mdf["status"].astype(str).str.startswith("ok")]
        if mdf.empty:
            st.info("This study has no OK runs.")
            return
        run_id = c[1].selectbox("Run", list(mdf["run_id"]))
        at = c[2].radio("At", ["mpp", "jsc", "voc"], horizontal=True,
                        format_func={"mpp": "max power", "jsc": "short circuit", "voc": "open circuit"}.get)
        try:
            L = analysis.losses(os.path.join(runs_root(), pick, run_id, "result.iv"), at)
            m = st.columns(3)
            m[0].metric("Generated", f"{L['J_generated']:.3f} mA/cm²")
            m[1].metric("Lost", f"{L['J_recombined']:.3f} mA/cm²")
            m[2].metric("Lost share", f"{100 * L['J_recombined'] / L['J_generated']:.2f} %")
            st.dataframe(pd.DataFrame(L["mechanisms"]), hide_index=True)
            st.caption(f"At V = {L['V']} V. 'where' splits the loss by location; 'bulk by type' splits the "
                       "bulk part by mechanism.")
        except (ValueError, FileNotFoundError) as e:
            st.warning(str(e))

    elif tool == "Convergence limit":
        page_limit(keys, model, names, labels, def_name, def_path)

    else:
        how = st.radio("How", [AL_MODE, "From a finished study"], horizontal=True, key="inv_how",
                       captions=["SCAPSync chooses and runs the SCAPS simulations itself, learning after each "
                                 "round. Any model.", "Uses runs you already have (Random sampling study)."])
        if how == AL_MODE:
            inverse_active(keys, model, names, labels, def_name, def_path)
            return
        df, inputs, base = dataset_picker("inv", keys)
        if df is None or not inputs:
            return
        sig = abs(hash(tuple(st.session_state.get("inv_st", [])))) % 10 ** 6   # new data -> fresh defaults
        N = st.session_state.get("inv_names") or {}
        nm = lambda k: N.get(k) or catalogue.readable(k, model)   # noqa: E731
        inputs = st.multiselect("Parameters SCAPSync may change", inputs, default=inputs, key=f"inv_in_{sig}",
                                format_func=nm)
        if not inputs:
            st.info("Choose at least one parameter.")
            return
        outs = [o for o in analysis.OUTPUTS if o in df]
        targets = {}
        for i in range(3):
            c = st.columns(3)
            o = c[0].selectbox(f"Target {i + 1}", ["(none)"] + outs, index=1 if (i == 0 and outs) else 0,
                               key=f"t_o{i}", format_func=oname)
            if o != "(none)":
                v = c[1].number_input("value", value=float(df[o].median()), format="%.5g", key=f"t_v{i}_{sig}")
                tol = c[2].number_input("± allowed", value=float(max(df[o].std() / 4, 0.05) if o == "eta_pct"
                                                                 else (df[o].std() / 4 or 0.01)),
                                        format="%.4g", key=f"t_t{i}_{sig}")
                targets[o] = (v, tol)
        npick = st.slider("How many different designs", 1, 10, 5)
        if st.button("Propose designs", type="primary", key="propose") and targets and inputs:
            try:
                with st.spinner("Learning from your runs and searching 50,000 candidates…"):
                    picks, info = analysis.inverse_design(df, inputs, targets, n_pick=npick)
                info.setdefault("targets", dict(targets))
                st.session_state["inv"] = (picks, info, base)
            except ValueError as e:
                st.warning(str(e))
        if "inv" not in st.session_state:
            return
        picks, info, base = st.session_state["inv"]
        st.write(f"Learned from {info['trained_on']} runs (accuracy R² "
                 + ", ".join(f"{k} {v}" for k, v in info["cv_r2"].items())
                 + f"); {info['within_tolerance']} of {info['candidates']} candidates hit the target.")
        for w in info["warnings"]:
            st.warning(w)
        st.dataframe(picks.map(lambda v: f"{v:.5g}").rename(columns=lambda c: nm(c) if "." in c else c),
                     hide_index=True)
        if base is None:
            st.info("To check these in SCAPS, use SCAPSync studies as the data (their model and "
                    "measurement are reused).")
        elif st.button("Check these designs in SCAPS", disabled=job_running("verify")):
            with open(base, encoding="utf-8") as f:
                bj = json.load(f)
            vname = f"inverse_{time.strftime('%Y%m%d_%H%M%S')}"
            bj.update(name=vname, design={"type": "list", "runs": []}, parallel=1)
            vpath = save_study_file(bj)
            os.makedirs(os.path.join(runs_root(), vname), exist_ok=True)
            pcsv = os.path.join(runs_root(), vname, "picks.csv")
            picks.to_csv(pcsv, index=False)
            launch("verify", ["verify", vpath, pcsv])
            st.session_state["verify_dir"] = os.path.join(runs_root(), vname)
        if "verify" in st.session_state.get("jobs", {}):
            live_log("verify", 12)
            vcsv = os.path.join(st.session_state["verify_dir"], "verification.csv")
            if os.path.exists(vcsv) and not job_running("verify"):
                vdf = pd.read_csv(vcsv)
                st.subheader("Predicted vs real SCAPS")
                for o, (tv, tt) in info.get("targets", {}).items():
                    pc, ac = f"predicted {o}", f"SCAPS {o}"
                    if pc in vdf and ac in vdf:
                        okm = vdf[ac].notna()
                        err = (vdf.loc[okm, ac] - vdf.loc[okm, pc]).abs()
                        hit = int(((vdf.loc[okm, ac] - tv).abs() <= tt).sum())
                        st.write(f"**{o}**: target {tv:g} ± {tt:g} · average prediction error {err.mean():.4g} · "
                                 f"{hit} of {len(vdf)} designs inside the target in real SCAPS")
                st.dataframe(vdf.map(lambda v: f"{v:.5g}" if isinstance(v, float) else v), hide_index=True)


def _fail_count(name):
    try:
        d = pd.read_csv(os.path.join(runs_root(), name, "master.csv"), usecols=["status"])
        return int(d["status"].astype(str).isin(analysis.FAILED).sum())
    except Exception:                           # noqa: BLE001
        return 0


def page_limit(keys, model, names, labels, def_name, def_path):
    sts = study_folders()
    nfail = {n: _fail_count(n) for n in sts}
    opts = [n for n in sts if nfail[n]] + [MANUAL]
    src = st.selectbox("Study", opts, key="lim_src",
                       format_func=lambda n: n if n == MANUAL else f"{n}  ·  {nfail[n]} failed run(s)")
    if len(opts) == 1:
        st.caption("None of your studies has failed runs.")
    if src == MANUAL:
        limit_manual(keys, model, names, labels, def_name, def_path)
        return

    folder = os.path.join(runs_root(), src)
    df = pd.read_csv(os.path.join(folder, "master.csv"))
    inputs = numeric_inputs(df)
    if not inputs:
        st.info("This study changed no parameter, so there is nothing to compare. "
                f"Choose *{MANUAL}* above.")
        return
    N = study_names(folder, inputs)
    nm = lambda k: N.get(k, k)                  # noqa: E731
    stat = df["status"].astype(str)
    failed = df[stat.isin(analysis.FAILED)]

    # 1 · cause
    st.markdown("#### 1 · Cause")
    try:
        _, verdict = analysis.diagnose_failures(df, inputs, N)
    except ValueError as e:
        st.warning(str(e))
        return
    (st.success if verdict["kind"] == "single" else st.info if verdict["kind"] == "mostly" else st.warning)(
        verdict["text"])
    suspect = verdict["suspect"]
    if verdict["kind"] == "unclear" and verdict.get("tree"):
        st.markdown("Two parameters together may explain it:\n" + "\n".join(f"- {r}" for r in verdict["tree"]))
    import altair as alt
    pdf = df[stat.str.startswith("ok") | stat.isin(analysis.FAILED)].copy()
    pdf["run"] = pdf["status"].astype(str).map(lambda s: "failed" if s in analysis.FAILED else "OK")
    pdf["_x"] = pdf[suspect]
    st.altair_chart(alt.Chart(pdf).mark_tick(thickness=3, size=26).encode(
        x=alt.X("_x:Q", title=nm(suspect),
                scale=alt.Scale(type="log" if analysis.is_log(suspect, pdf[suspect]) else "linear", zero=False)),
        y=alt.Y("run:N", title=None),
        color=alt.Color("run:N", scale=alt.Scale(domain=["OK", "failed"], range=["#4c9f70", RED]), legend=None),
        tooltip=["run_id", "status", alt.Tooltip("_x:Q", title=nm(suspect))]).properties(height=110),
        width="stretch")

    # failed run used below: the one closest to the OK runs on the suspect parameter
    okv = df.loc[stat.str.startswith("ok"), suspect]
    gap = failed[suspect].map(lambda v: (okv - v).abs().min() if len(okv) else 0)
    start = failed.loc[gap.idxmin(), "run_id"]

    with st.expander("Not sure? Confirm the cause with SCAPS (optional)"):
        ok = analysis.nearest_ok(df, inputs, start)
        st.caption(f"Takes failed run {start} and changes its parameters, one at a time, to those of the most "
                   f"similar OK run ({ok['run_id']}), running SCAPS each time. About {len(inputs)} SCAPS run(s).")
        out = os.path.join(folder, f"whyfail_{start}.json")
        if st.button("Confirm with SCAPS", disabled=job_running("whyfail")):
            launch("whyfail", ["whyfail", folder, start])
        if job_running("whyfail") or "whyfail" in st.session_state.get("jobs", {}) and not os.path.exists(out):
            live_log("whyfail", 8)
        if os.path.exists(out) and not job_running("whyfail"):
            with open(out, encoding="utf-8") as f:
                res = json.load(f)
            (st.success if res["causes"] or res["pairs"] else st.warning)(res["text"])

    # 2 · exact limit
    st.markdown("#### 2 · Exact limit")
    key = st.selectbox("Parameter", inputs, index=inputs.index(suspect), format_func=nm, key=f"lim_key_{src}")
    frow = failed[failed["run_id"] == start].iloc[0]
    if key == suspect and verdict["kind"] != "unclear":
        good0, bad0 = float(verdict["last_ok"]), float(verdict["first_fail"])
    else:
        good0, bad0 = float(analysis.nearest_ok(df, inputs, start)[key]), float(frow[key])
    c = st.columns(2)
    good = c[0].number_input("Works at", value=good0, format="%.6g", key=f"lg_{src}_{key}")
    bad = c[1].number_input("Fails at", value=bad0, format="%.6g", key=f"lb_{src}_{key}")
    st.caption(f"Filled in from your runs. The other parameters stay as in failed run {start}.")
    if st.button("Find the limit", type="primary", disabled=job_running("boundary")):
        with open(os.path.join(folder, "study.json"), encoding="utf-8") as f:
            bj = json.load(f)
        bj.update(name=f"_limit_{src}_{key.replace('.', '_').replace('@', '_')}",
                  design={"type": "list", "runs": []}, parallel=1)
        hold = os.path.join(WORK_DIR, "limit_hold.json")
        with open(hold, "w", encoding="utf-8") as f:
            json.dump({k: float(frow[k]) for k in inputs}, f)
        launch("boundary", ["boundary", save_study_file(bj), key, str(good), str(bad), hold])
    if "boundary" in st.session_state.get("jobs", {}):
        live_log("boundary", 16)


def limit_manual(keys, model, names, labels, def_name, def_path):
    st.caption(f"Model: `{def_name}` (change it on New study). All other parameters stay as in this model.")
    groups = {}
    for k in keys:
        groups.setdefault(part_of(k, model, names), []).append(k)
    import zlib
    bsig = zlib.crc32("|".join(groups).encode()) % 100000
    c = st.columns([2, 3])
    part = c[0].selectbox("Part of the device", list(groups), key=f"b_part_{bsig}")
    bkey = c[1].selectbox("Parameter", groups[part], format_func=lambda k: short_label(k, labels),
                          key=f"b_key_{bsig}")
    p = catalogue.resolve(bkey, model)
    now = verify.read_back(model, p) if p.kind != "action" else None
    lo0, hi0, _ = default_range(p, now)
    c = st.columns(2)
    good = c[0].number_input("Works at", value=float(now) if isinstance(now, float) else lo0,
                             format="%.6g", key=f"bg_{bkey}")
    bad = c[1].number_input("Fails at", value=float(hi0), format="%.6g", key=f"bb_{bkey}")
    if st.button("Find the limit", type="primary", disabled=job_running("boundary")):
        bname = f"_limit_{bkey.replace('.', '_').replace('@', '_')}"
        bj = base_study(bname, def_path)
        bj["parallel"] = 1
        launch("boundary", ["boundary", save_study_file(bj), bkey, str(good), str(bad)])
    if "boundary" in st.session_state.get("jobs", {}):
        live_log("boundary", 16)


def inverse_active(keys, model, names, labels, def_name, def_path):
    """Active-learning inverse design (fermipilot.active): SCAPS runs chosen by ML until enough different
    designs are verified to hit the target."""
    st.caption(f"Model: `{def_name}` (change it on New study). Parameters not chosen stay as in this model.")
    chosen = param_picker("al", keys, model, names, labels)
    specs, problems = {}, []
    for k in chosen:
        p = catalogue.resolve(k, model)
        now = verify.read_back(model, p) if p.kind != "action" else None
        lo0, hi0, log0 = default_range(p, now)
        c = st.columns([3, 1.3, 1.3, 0.8], vertical_alignment="bottom")
        c[0].markdown(f"**{catalogue.readable(k, model)}**")
        lo = c[1].number_input("from", value=float(lo0), format="%.6g", key=f"al_lo_{k}")
        hi = c[2].number_input("to", value=float(hi0), format="%.6g", key=f"al_hi_{k}")
        lg = c[3].checkbox("log", value=log0, key=f"al_lg_{k}")
        if lo >= hi:
            problems.append(f"{catalogue.readable(k, model)}: 'from' must be below 'to'")
        if lg and lo <= 0:
            problems.append(f"{catalogue.readable(k, model)}: log spacing needs positive values")
        specs[k] = [lo, hi, "log" if lg else "lin"]
    if not chosen:
        st.info("Choose a part of the device, then the parameters SCAPSync may change.")
    c = st.columns([2, 1.4, 1.2, 1.2])
    out = c[0].selectbox("Target", analysis.OUTPUTS[:4], format_func=oname, key="al_out")
    kind = c[1].radio("Reach", ["at least", "about"], horizontal=True, key="al_kind")
    val = c[2].number_input("value", value={"eta_pct": 24.0, "Voc_V": 0.95, "Jsc_mAcm2": 29.0,
                                            "FF_pct": 80.0}[out], format="%.4g", key=f"al_val_{out}")
    tol = c[3].number_input("± allowed", value=0.1 if out in ("eta_pct", "FF_pct", "Jsc_mAcm2") else 0.005,
                            format="%.4g", key=f"al_tol_{out}", disabled=kind == "at least")
    c = st.columns(3)
    want = c[0].number_input("Different designs wanted", 1, 20, 5, key="al_want")
    budget = c[1].number_input("Most SCAPS runs allowed", 20, 2000, 120, step=10, key="al_budget",
                               help="SCAPSync stops earlier when enough designs are verified.")
    c[2].metric("Time at most", nice_time(estimate_seconds(int(budget))))
    for pr in problems:
        st.error(pr)
    if st.button("Find designs", type="primary", disabled=not chosen or bool(problems) or job_running("design")):
        name = f"design_{time.strftime('%Y%m%d_%H%M%S')}"
        spec = base_study(name, def_path)
        spec.update(params=specs, budget=int(budget), want=int(want), batch=max(4, copies() * 2),
                    target={"output": out, "min": val} if kind == "at least"
                    else {"output": out, "value": val, "tol": tol})
        launch("design", ["design", save_study_file(spec)])
        st.session_state["al_dir"] = os.path.join(runs_root(), name)
    if "design" in st.session_state.get("jobs", {}):
        live_log("design", 14)
    folder = st.session_state.get("al_dir")
    if folder and os.path.exists(os.path.join(folder, "result.json")) and not job_running("design"):
        with open(os.path.join(folder, "result.json"), encoding="utf-8") as f:
            res = json.load(f)
        hits = pd.read_csv(os.path.join(folder, "designs_on_target.csv"))
        (st.success if len(hits) else st.warning)(
            f"{len(hits)} different design(s) verified in SCAPS to reach the target, using {res['runs']} SCAPS runs.")
        if len(hits):
            show = hits[[k for k in hits.columns if "." in k] + [o for o in analysis.OUTPUTS if o in hits]]
            st.dataframe(show.rename(columns=lambda k: catalogue.readable(k, model) if "." in k else oname(k))
                         .map(lambda v: f"{v:.5g}" if isinstance(v, float) else v), hide_index=True)
        st.caption(f"Every run, failed ones included, is in `{folder}` (also listed on the Results page).")


# ======================================================================
# page: Graphs (the SCAPS-1D Data Visualizer, built in)
# ======================================================================
GRAPH_FILES = {"iv": "I-V (.iv)", "cv": "C-V (.cv)", "cf": "C-f (.cf)", "qe": "QE (.qe)",
               "eb": "Band diagram (.eb)", "gen": "Generation / recombination (.gen)"}


def short_name(k, model):
    """'L2: electron mobility [cm²/Vs]' (the Visualizer's own style); other parameters by their readable name."""
    p = catalogue.resolve(k, model)
    unit = catalogue._UNIT.get(p.unit, p.unit)
    if p.kind in ("layer", "defect"):
        what = (f"D{p.defect} " if p.kind == "defect" else "") + p.label.replace(" (pure B)", "")
        return f"L{p.layer}: {what}{' (B side)' if '@B' in k else ''}" + (f" [{unit}]" if unit else "")
    return catalogue.readable(k, model)


def page_graphs():
    from fermipilot import graphs
    st.title("Graphs")
    src = st.radio("Data", ["Runs of a SCAPSync study", "Upload a SCAPS file"], horizontal=True, key="g_src")
    if src == "Upload a SCAPS file":
        graphs.render(header=False)
        return
    sts = study_folders()
    if not sts:
        st.info("No studies yet. Run one on New study, or upload a SCAPS file.")
        return
    c = st.columns([2, 1])
    pick = c[0].selectbox("Study", sts, key="g_study")
    folder = os.path.join(runs_root(), pick)
    df = pd.read_csv(os.path.join(folder, "master.csv"))
    df = df[df["status"].astype(str).str.startswith("ok")]
    if df.empty:
        st.info("This study has no solved runs.")
        return
    run_dir = lambda r: r.get("folder") if isinstance(r.get("folder"), str) and os.path.isdir(  # noqa: E731
        str(r.get("folder"))) else os.path.join(folder, str(r["run_id"]))
    first = run_dir(df.iloc[0])
    kinds = [k for k in GRAPH_FILES if os.path.exists(os.path.join(first, f"result.{k}"))]
    if not kinds:
        st.info("The runs of this study have no result files to plot.")
        return
    kind = c[1].selectbox("File", kinds, format_func=GRAPH_FILES.get, key=f"g_kind_{pick}")
    varied = [k for k in df.columns if "." in k and k != "folder" and df[k].notna().all()
              and pd.api.types.is_numeric_dtype(df[k]) and df[k].nunique() > 1]
    defs, _, _, model = current_model()
    try:
        with open(os.path.join(folder, "study.json"), encoding="utf-8") as f:
            mp = json.load(f)["base_def"]
        if os.path.exists(mp):
            model = defmodel.DefFile(load_model_text(mp, os.path.getmtime(mp)), mp)
    except Exception:                           # noqa: BLE001
        pass
    names = {k: short_name(k, model) for k in varied} if model is not None else {k: k for k in varied}

    def lab(rid):
        r = df[df["run_id"] == rid].iloc[0]
        return rid + ("  ·  " + ", ".join(f"{names[k]} = {r[k]:g}" for k in varied) if varied else "")
    ids = list(df["run_id"])
    chosen = st.multiselect("Runs to plot (several are overlaid, like a SCAPS batch)", ids,
                            default=ids[:min(5, len(ids))], format_func=lab, key=f"g_runs_{pick}_{kind}")
    if not chosen:
        st.info("Choose at least one run.")
        return
    sims = []
    for i, rid in enumerate(chosen, 1):
        r = df[df["run_id"] == rid].iloc[0]
        path = os.path.join(run_dir(r), f"result.{kind}")
        if not os.path.exists(path):
            st.warning(f"{rid}: no result.{kind}")
            continue
        with open(path, "rb") as f:
            got = graphs.parse_uploaded_file(f"result.{kind}", f.read())
        if not got:
            st.warning(f"{rid}: no data in result.{kind}")
            continue
        s = got[-1]                              # the run's own (last) calculation
        s["sim_number"] = i
        s["batch_params"] = {names[k]: float(r[k]) for k in varied if pd.notna(r[k])}
        s["step_of"] = None
        sims.append(s)
    st.divider()
    if sims:
        graphs.render(source=(f"{pick}:{kind}:{','.join(chosen)}", sims), header=False)


# ======================================================================
# page: Help
# ======================================================================
def page_help():
    st.title("Help")
    COPIES = copies()
    measured = speedcheck.stored(scaps_dir())
    st.markdown(f"""
**SCAPSync runs SCAPS-1D for you.** Choose a model and the parameters to change, press Start, and
SCAPSync runs every simulation in the background, checks each one, and collects all results in one table.

**The four steps**
1. **New study** — model → type of study → parameters → measurement → Start. Progress shows on the same page.
2. **Results** — the table, a chart, and a CSV download. Every run's SCAPS files are kept.
3. **Analyse** *(optional)* — rank parameters, find which parameter makes runs fail and its exact limit, see losses, or design for a target. Each tool has a **?** with instructions.

**Types of study**
""" + "\n".join(f"- **{k}** — {v}" for k, v in DESIGNS.items()) + f"""

**What SCAPSync decides for you**
- **{COPIES} SCAPS cop{'ies run' if COPIES > 1 else 'y runs'} at once** on this PC ({'measured fastest by the one-time speed check' if measured else 'default; the first study of 30+ runs measures what is fastest on this PC, once, in about 3 minutes'}), and **each copy keeps SCAPS open for up to
  {study.CHUNK} runs in a row** instead of restarting it for every run (starting SCAPS costs about 4 s).
  Measured on a Ryzen 7 4700U laptop: 7.4 s per run restarting SCAPS every run; this way 4.3 s per run with
  1 copy, 3.0 s with 2, 2.7 s with 3 and 2.3 s with 4 copies (SCAPS's own batch: 8.6 s), with identical results.
- If one run stalls inside a session, it is moved to its own fresh SCAPS start automatically.
- **The I-V sweep (0 to 1.2 V) ends just after Voc** (SCAPS's own *stop after Voc* option). Tested on 12 runs:
  identical Voc, Jsc, FF, efficiency and maximum power point; the curve beyond Voc is not calculated.
- **Band diagram option:** SCAPS writes a band diagram at the last voltage it calculated, which differs from
  run to run. SCAPSync calculates it at a fixed 0 V instead, so the band offsets, fields and losses in the
  table are comparable between runs. Check: generation minus all recombination equals Jsc (29.0991 mA/cm²).
- **A run that is still busy after {TIME_LIMIT} s** ({TIME_LIMIT * (1 + 0.5 * (COPIES - 1)):.0f} s while {COPIES} copies
  share the processor) is stopped and marked *timeout*, so one stuck value never blocks a study.
- A run that does not converge is retried once with more working-point steps.
- Every run is checked: SCAPS's own saved settings are read back to prove it used your values.

**Run status**
""" + "\n".join(f"- **{k}** — {v}" for k, v in study.STATUS_HELP.items()) + f"""

**Where files are**
- Results: `{runs_root()}\\<study name>\\` (one folder per run plus `master.csv`).
- Your own `.def` files are never changed; each run uses its own copy.

**Version** {__version__} · By Azharul Islam, University of Chittagong. SCAPS-1D is by M. Burgelman et al.,
University of Gent; SCAPSync uses your own licensed installation.
""")
    with st.expander("Settings: SCAPS folder", expanded=not scaps_found()):
        st.caption("SCAPSync looks for SCAPS by itself. Change this only if it found the wrong folder.")
        new = st.text_input("Folder that contains scaps3309.exe (or another SCAPS version)", value=scaps_dir())
        if new != scaps_dir():
            if paths.scaps_exe(new):
                st.session_state["scaps_dir"] = new
                paths.save_setting("scaps_dir", new)        # remembered for next time
                st.rerun()
            else:
                st.error(f"No scaps*.exe in {new}.")
        st.caption(f"Results are saved in `{runs_root()}`.")


# ======================================================================
# navigation
# ======================================================================
PAGES = {
    "new": st.Page(page_new_study, title="New study", icon=":material/science:", default=True),
    "results": st.Page(page_results, title="Results", icon=":material/table_chart:"),
    "graphs": st.Page(page_graphs, title="Graphs", icon=":material/show_chart:"),
    "analyse": st.Page(page_analyse, title="Analyse", icon=":material/insights:"),
    "help": st.Page(page_help, title="Help", icon=":material/help:"),
}

with st.sidebar:
    st.markdown(f"### SCAPSync\nAutomation for SCAPS-1D · v{__version__}")
nav = st.navigation(list(PAGES.values()))
with st.sidebar:
    if not scaps_found():
        st.error("SCAPS not found. Open Help → Settings and choose your SCAPS folder.")
    else:
        st.caption(f"SCAPS: `{os.path.basename(find_exe(scaps_dir()))}` ✓")
    run_name = active_study()
    if run_name:
        def side_progress():
            pg = read_progress(run_name) or {"total": 0, "done": 0}
            if is_running(run_name):
                st.progress(pg["done"] / max(pg["total"], 1), text=f"Running {run_name}: {pg['done']}/{pg['total']}")
            else:
                st.caption(f"{run_name} finished.")
        st.fragment(run_every=5)(side_progress)()
if not scaps_found() and nav.title != "Help":
    st.title("SCAPS not found")
    st.write("SCAPSync looked in the usual places (Program Files, C:\\, D:\\, Desktop, Documents, Downloads) "
             "but did not find SCAPS. Open **Help → Settings** and enter the folder that contains scaps3309.exe. "
             "SCAPS-1D is free from Ghent University: https://scaps.elis.ugent.be/")
else:
    nav.run()
