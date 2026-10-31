"""Apply .def-route edits, and prove after a run that SCAPS used the intended values.

Proof works by asking SCAPS to save its own internal state as a .def file
(`save settings.definitionfile`) right after all set commands, then reading
the value back from that file. A value SCAPS ignored cannot pass.
"""


def apply_def_edit(model, param, value):
    """Write a 'def'-route parameter (pure B of a graded layer) into a DefFile in place,
    the same way the SCAPS GUI does: pure-B column plus the side values it implies."""
    sec = model.find_section(param.def_section, layer=param.layer)
    cols = model.get_columns(sec, param.def_line)
    if cols is None:
        raise KeyError(f"{param.key}: line '{param.def_line}' not found")
    law = int(cols[7]) if len(cols) > 7 else 0
    if law != 1:
        raise ValueError(f"{param.key}: composition dependence is not linear (law {law}); "
                         "not supported yet")
    yl, yr = model.composition(param.layer)
    a, b = cols[0], value * param.factor
    model.set_columns(sec, param.def_line, {2: b, 6: a + (b - a) * yl, 7: a + (b - a) * yr})


def read_back(state, param):
    """Value SCAPS stored for `param`, converted back to GUI units (None if not stored)."""
    if param.kind == "action":
        return None
    sec = state.find_section(param.def_section, layer=param.layer, defect=param.defect,
                             interface=param.interface, ifdefect=param.ifdefect)
    if sec is None:
        return None
    cols = state.get_columns(sec, param.def_line)
    if not cols:
        return None
    return cols[param.column - 1] / param.factor


def close(a, b):
    """.def files print 4 significant digits (e-notation) or 3 decimals (eV values),
    so allow 0.1 % relative or 0.0006 absolute for small numbers like Et."""
    tol = 1e-3 * abs(b)
    if abs(b) < 10:
        tol = max(tol, 6e-4)
    return abs(a - b) <= tol


def prove(state, assignments):
    """assignments: list of (Param, value). Returns list of dicts, one per parameter."""
    out = []
    for p, v in assignments:
        got = read_back(state, p)
        if got is None:
            status = "not stored" if p.kind == "action" else "missing"
        else:
            status = "ok" if close(got, v) else "MISMATCH"
        if p.kind == "contact" and p.script_name == "workfunction":
            sec = state.find_section(p.def_section)
            i = state.find_line(sec, "flatband")
            if i is None or state.numbers(state.lines[i])[0] != 0:
                status = "MISMATCH (flat band still on)"
        out.append({"key": p.key, "wanted": v, "scaps": got, "status": status})
    return out
