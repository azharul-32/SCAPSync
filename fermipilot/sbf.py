"""Read a SCAPS batch settings file (.sbf) and turn it into a FermiPilot study design.

Format learned from files saved by the SCAPS 3.3.09 batch panel (23-10-2026):
    Batch parameter 0 :
    Label 1 : layer 2            | interface 3            | T
    Label 2 : mu n               | interface defect 1     | nihil
    Label 3 : nihil              | IF Nt total            | nihil
    Label 4 : nihil              | (pure B material for the B side of a graded layer)
    simultaneous :  0            (1 = varies together with the parameter above)
    values from a list :  0
    logarithmic variation :  0
    startvalue :   2.00000000e+01
    stopvalue :   1.00000000e+02
    number of steps :   16
A custom list replaces start/stop/steps with:
    values from a list :  1 / values are file names :  0 / number of parameters :  3
    parameter   0:   1.00000000e+00   (one line per value)
Values are in the GUI units of the batch panel (cm-based), the same units FermiPilot uses.
'number of steps' is the number of values: SCAPS's own singleshotbatch.sbf ("one single
calculation") has number of steps = 1.
Parameters not nested ('simultaneous') form groups that step together; groups are nested.
File lists (from SCAPS's own 'example batch setting CdTe.sbf'):
    variation of definition files :  1 / number of definition files : 3 /
    definition filename   0:CdTe-base.def ...                 -> MODEL.file (outer loop)
    Label 1 : illumination / Label 2 : spectrum file / values are file names : 1 /
    number of file names : 6 / file   0:400nm fixed JL.spe ... -> LIGHT.spectrum
"""
import itertools
import math
import re

# SCAPS batch label (Label 2 for a layer) -> FermiPilot layer parameter name.
# Seen in saved files: "mu n", "bandgap". The others follow the same panel naming and are
# marked as not yet seen; an unknown label is reported instead of guessed.
LAYER_LABELS = {"mu n": "mun", "mu p": "mup", "bandgap": "Eg"}
DEFECT_LABELS = {"Nt total": "Ntotal"}
IFDEFECT_LABELS = {"IF Nt total": "Ntotal"}
SEEN = {"mu n", "bandgap", "Nt total", "IF Nt total", "T", "spectrum file"}   # spectrum: SCAPS's example file


def _blocks(text):
    parts = re.split(r"^\s*Batch parameter\s+(\d+)\s*:\s*$", text, flags=re.M)
    out = []
    for i in range(1, len(parts), 2):
        body = parts[i + 1]
        rec = {"index": int(parts[i])}
        for line in body.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                rec[k.strip()] = v.strip()
        out.append(rec)
    return out


def _key(rec):
    """FermiPilot parameter key for one batch row, or raise ValueError with the reason."""
    l1, l2, l3, l4 = (rec.get(f"Label {i}", "nihil") for i in range(1, 5))
    if l1 == "T":
        return "WP.temperature"
    if l1 == "illumination":
        if l2 == "spectrum file":
            return "LIGHT.spectrum"
        raise ValueError(f"this row varies the illumination ({l2}); FermiPilot can vary the spectrum file, "
                         "not this yet")
    m = re.match(r"layer (\d)$", l1)
    if m:
        n = m.group(1)
        d = re.match(r"defect (\d)$", l2)
        if d:
            if l3 not in DEFECT_LABELS:
                raise ValueError(f"defect property '{l3}' is not mapped yet")
            return f"L{n}.D{d.group(1)}.{DEFECT_LABELS[l3]}"
        if l2 not in LAYER_LABELS:
            raise ValueError(f"layer property '{l2}' is not mapped yet")
        key = f"L{n}.{LAYER_LABELS[l2]}"
        return key + "@B" if l4 == "pure B material" else key
    m = re.match(r"interface (\d)$", l1)
    if m:
        d = re.match(r"interface defect (\d)$", l2)
        if not d or l3 not in IFDEFECT_LABELS:
            raise ValueError(f"interface property '{l2} / {l3}' is not mapped yet")
        return f"I{m.group(1)}.D{d.group(1)}.{IFDEFECT_LABELS[l3]}"
    raise ValueError(f"batch parameter '{l1} / {l2}' is not mapped yet")


def _values(rec):
    if rec.get("values from a list", "0").strip() != "0":
        if rec.get("values are file names", "0").strip() != "0":
            # saved as:  number of file names : 6 / file   0:400nm fixed JL.spe / ...
            names = [(int(re.search(r"\d+", k).group()), v.strip()) for k, v in rec.items()
                     if re.match(r"file\s+\d+$", k)]
            if not names:
                raise ValueError("the list of file names is empty")
            n = rec.get("number of file names")
            if n is not None and int(float(n)) != len(names):
                raise ValueError(f"the file list says {n} files but {len(names)} are listed")
            return [v for _, v in sorted(names)]
        # saved by the batch panel as:  number of parameters : 3 / parameter   0:   1.0e+00 / ...
        vals = [(int(re.search(r"\d+", k).group()), float(v)) for k, v in rec.items()
                if re.match(r"parameter\s+\d+$", k)]
        if not vals:
            raise ValueError("the custom value list is empty")
        n = rec.get("number of parameters")
        if n is not None and int(float(n)) != len(vals):
            raise ValueError(f"the custom list says {n} values but {len(vals)} are listed")
        return [v for _, v in sorted(vals)]
    a, b = float(rec["startvalue"]), float(rec["stopvalue"])
    n = int(float(rec["number of steps"]))
    if n <= 1:
        return [a]
    if rec.get("logarithmic variation", "0").strip() == "1":
        if a <= 0 or b <= 0:
            raise ValueError("logarithmic variation needs positive start and stop values")
        la, lb = math.log10(a), math.log10(b)
        return [10 ** (la + i * (lb - la) / (n - 1)) for i in range(n)]
    return [a + i * (b - a) / (n - 1) for i in range(n)]


def read(path_or_text):
    """Returns (design, report). design is a FermiPilot 'list' design; report lists each batch row."""
    text = path_or_text
    if "\n" not in path_or_text and path_or_text.lower().endswith(".sbf"):
        with open(path_or_text, "rb") as f:
            text = f.read().decode("latin-1")
    text = text.replace("\r", "")
    rows, report, groups = _blocks(text), [], []
    if re.search(r"variation of definition files\s*:\s*1", text):
        # header lines:  number of definition files : 3 / definition filename   0:CdTe-base.def / ...
        defs = [(int(i), v.strip()) for i, v in re.findall(r"^definition filename\s+(\d+)\s*:(.*)$", text, flags=re.M)]
        if not defs:
            raise ValueError("the batch varies definition files but lists none")
        m = re.search(r"number of definition files\s*:\s*(\d+)", text)
        if m and int(m.group(1)) != len(defs):
            raise ValueError(f"the batch says {m.group(1)} definition files but lists {len(defs)}")
        names = [v for _, v in sorted(defs)]
        groups.append([("MODEL.file", names)])        # whole models: the outer loop
        report.append({"batch row": "models", "parameter": "MODEL.file", "values": len(names),
                       "from": names[0], "to": names[-1], "with the row above": False, "label seen before": True})
    for rec in rows:
        try:
            key, vals = _key(rec), _values(rec)
        except (ValueError, KeyError) as e:
            raise ValueError(f"batch parameter {rec['index']}: {e}") from None
        together = rec.get("simultaneous", "0").strip() == "1" and groups
        if together:
            if len(vals) != len(groups[-1][0][1]):
                raise ValueError(f"batch parameter {rec['index']} is 'simultaneous' but has a different "
                                 "number of steps than the parameter above")
            groups[-1].append((key, vals))
        else:
            groups.append([(key, vals)])
        report.append({"batch row": rec["index"], "parameter": key, "values": len(vals),
                       "from": vals[0], "to": vals[-1], "with the row above": bool(together),
                       "label seen before": rec.get("Label 2") in SEEN or rec.get("Label 3") in SEEN
                       or rec.get("Label 1") in SEEN})
    runs = []
    for combo in itertools.product(*[list(range(len(g[0][1]))) for g in groups]):
        run = {}
        for g, i in zip(groups, combo):
            for key, vals in g:
                run[key] = vals[i]
        runs.append(run)
    return {"type": "list", "runs": runs}, report
