"""Build the SCAPS script for one run.

Every job: errors to the log file (no popups), no screen updates, a fresh action
list, the parameter set commands, a saved copy of SCAPS's internal state (for the
proof), one clean simulation, the requested result files, then SCAPS quits itself.
"""

MAX_POINTS = 201   # SCAPS limit per curve


def _pts(n):
    n = int(n)
    if not 2 <= n <= MAX_POINTS:
        raise ValueError(f"number of points must be 2..{MAX_POINTS}, got {n}")
    return n


def measurement_lines(meas, light=True):
    lines = ["clear actions", "action light" if light else "action dark"]
    if "iv" in meas:
        m = meas["iv"]
        lines += [f"action iv.startv {m.get('start', 0.0):g}", f"action iv.stopv {m.get('stop', 1.0):g}",
                  f"action iv.points {_pts(m.get('points', 41))}", "action iv.doiv"]
        if "stop_after_voc" in m:      # SCAPS option: end the I-V sweep once the current changes sign
            lines.append("action iv.stopafterVoc" if m["stop_after_voc"] else "action iv.continueafterVoc")
    if "cv" in meas:
        m = meas["cv"]
        lines += [f"action cv.startv {m.get('start', -0.8):g}", f"action cv.stopv {m.get('stop', 0.8):g}",
                  f"action cv.points {_pts(m.get('points', 81))}", "action cv.docv"]
    if "cf" in meas:
        m = meas["cf"]
        lines += [f"action cf.startf {m.get('start', 1e2):g}", f"action cf.stopf {m.get('stop', 1e6):g}",
                  f"action cf.pointsperdecade {int(m.get('per_decade', 5))}", "action cf.docf"]
    if "qe" in meas:
        m = meas["qe"]
        lines += [f"action qe.startlambda {m.get('start', 300):g}", f"action qe.stoplambda {m.get('stop', 900):g}",
                  f"action qe.points {_pts(m.get('points', 61))}", "action qe.doqe"]
    return lines


def build_script(run_id, def_path, assignments, meas, out, light=True):
    """assignments: list of (Param, value) for script-route parameters.
    out: dict of output paths: state, iv, cv, cf, qe, eb, genrec.

    Band diagram (.eb) and generation-recombination (.gen) files: tested 24-10-2026, SCAPS writes
    them at the LAST bias it calculated (the end of the I-V sweep: 1.0 V, or 0.925 V with
    stop-after-Voc), so files of different runs would not be comparable. FermiPilot therefore makes
    one extra working-point calculation at a fixed bias (meas['eb_voltage'], default 0 V = short
    circuit), saves .eb/.gen, and only then runs the measurements. The I-V is saved last, which the
    session watchdog relies on (the .iv file is the sign that a run is complete)."""
    actions = [(p, v) for p, v in assignments if getattr(p, "kind", None) == "action"]
    others = [(p, v) for p, v in assignments if getattr(p, "kind", None) != "action"]
    lines = [f"// FermiPilot job {run_id}",
             "set errorhandling.overwritefile",
             "set script_display_mode.fully_suppressed",
             f"load definitionfile {def_path}"]
    for p, v in others:
        lines += p.script_lines(v)
    act_lines = [x for p, v in actions for x in p.script_lines(v)]
    lines.append(f"save settings.definitionfile {out['state']}")
    if meas.get("eb") or meas.get("genrec"):
        # 'clear actions' resets working-point actions, so the study's action values are applied again
        lines += ["clear actions", "action light" if light else "action dark",
                  f"action workingpoint.voltage {float(meas.get('eb_voltage', 0.0)):g}"] + act_lines
        lines += ["clear simulations", "calculate"]
        if meas.get("eb"):
            lines.append(f"save results.eb {out['eb']}")
        if meas.get("genrec"):
            lines.append(f"save results.genrec {out['genrec']}")
    lines += measurement_lines(meas, light) + act_lines
    lines += ["clear simulations", "calculate"]
    for kind in ("cv", "cf", "qe", "iv"):
        if kind in meas or (kind == "iv" and "cv" in meas):
            lines.append(f"save results.{kind} {out[kind]}")
    lines.append("set quitscript.quitSCAPS")
    return "\r\n".join(lines) + "\r\n"
