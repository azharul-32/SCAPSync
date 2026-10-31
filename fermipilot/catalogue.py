"""Every parameter FermiPilot can vary, with the route that reaches it.

Parameter keys (GUI units, as in the SCAPS panels):
    L2.mun          layer 2 electron mobility (pure A / uniform value)       route: script
    L3.Eg@B         layer 3 band gap of the pure B material (graded layer)  route: def edit
    L2.D1.Ntotal    layer 2, defect 1 total density                          route: script
    I3.D1.Ntotal    interface 3 (right of layer 3), interface defect 1      route: script
    left.workfunction, right.Sn ...   contacts                               route: script
    WP.temperature, WP.voltage, LIGHT.ND ...   working point                 route: script (action)

For each parameter we also know where SCAPS stores it in a saved .def (section,
line name, column and the GUI->SI factor), so every run can be proven.
Tested on SCAPS 3.3.09 (22 Oct 2026): mun, mup, NA, thickness, Eg(A), chi,
defect Ntotal and Et, contact work function, def-edit of Eg pure B. The rest are
documented commands; the proof-of-input check verifies them on first use.
"""
import re

# script name, .def line name, GUI->SI factor, GUI unit, label
LAYER_PARAMS = {
    "thickness":  ("thickness",  "d",         1e-6,  "um",        "thickness"),
    "Eg":         ("Eg",         "Eg",        1.0,   "eV",        "band gap"),
    "chi":        ("chi",        "chi",       1.0,   "eV",        "electron affinity"),
    "epsilon":    ("epsilon",    "eps",       1.0,   "-",         "dielectric permittivity"),
    "NC":         ("NC",         "Nc",        1e6,   "cm-3",      "CB effective density of states"),
    "NV":         ("NV",         "Nv",        1e6,   "cm-3",      "VB effective density of states"),
    "vthn":       ("vthn",       "v_th_n",    1e-2,  "cm/s",      "electron thermal velocity"),
    "vthp":       ("vthp",       "v_th_p",    1e-2,  "cm/s",      "hole thermal velocity"),
    "mun":        ("mun",        "mu_n",      1e-4,  "cm2/Vs",    "electron mobility"),
    "mup":        ("mup",        "mu_p",      1e-4,  "cm2/Vs",    "hole mobility"),
    "NA":         ("NA",         "Na(",       1e6,   "cm-3",      "shallow acceptor density"),
    "ND":         ("ND",         "Nd(",       1e6,   "cm-3",      "shallow donor density"),
    "radiative":  ("radiative",  "K_rad",     1e-6,  "cm3/s",     "radiative recombination coefficient"),
    "Augern":     ("Augern",     "c_n_auger", 1e-12, "cm6/s",     "Auger electron coefficient"),
    "Augerp":     ("Augerp",     "c_p_auger", 1e-12, "cm6/s",     "Auger hole coefficient"),
}
DEFECT_PARAMS = {
    "Ntotal":   ("Ntotal",                          "Nt(",     1e6,  "cm-3", "total defect density"),
    "Et":       ("Et",                              "Et",      1.0,  "eV",   "defect energy level"),
    "sigma_n":  ("capture_cross_section.electron",  "sigma_n", 1e-4, "cm2",  "electron capture cross section"),
    "sigma_p":  ("capture_cross_section.hole",      "sigma_p", 1e-4, "cm2",  "hole capture cross section"),
}
IFDEFECT_PARAMS = {
    "Ntotal":   ("Ntotal",                          "N",           1e4,  "cm-2", "interface defect density"),
    "Et":       ("Et",                              "Et",          1.0,  "eV",   "interface defect energy level"),
    "sigma_n":  ("capture_cross_section.electron",  "sigma_nleft", 1e-4, "cm2",  "electron capture cross section"),
    "sigma_p":  ("capture_cross_section.hole",      "sigma_pleft", 1e-4, "cm2",  "hole capture cross section"),
}
CONTACT_PARAMS = {
    "Sn":           ("Sn",           "Sn",   1e-2, "cm/s", "electron surface recombination velocity"),
    "Sp":           ("Sp",           "Sp",   1e-2, "cm/s", "hole surface recombination velocity"),
    "workfunction": ("workfunction", "Fi_m", 1.0,  "eV",   "metal work function (flat band switched off)"),
}
# working point and illumination: script 'action' commands; not stored in a .def
ACTION_PARAMS = {
    "WP.temperature": ("action workingpoint.temperature", "K",  "temperature"),
    "WP.voltage":     ("action workingpoint.voltage",     "V",  "working point voltage"),
    "WP.frequency":   ("action workingpoint.frequency",   "Hz", "working point frequency"),
    "WP.points":      ("action workingpoint.numberofpoints", "-", "working point steps"),
    "LIGHT.ND":       ("action intensity.ND",             "-",  "neutral density filter"),
    "LIGHT.T":        ("action intensity.T",              "%",  "filter transmission"),
    "EXT.Rs":         ("set external.Rs",                 "ohm.cm2", "series resistance"),
    "EXT.Rsh":        ("set external.Rsh",                "ohm.cm2", "shunt resistance"),
}

# file-valued parameters (the value is a file name, not a number), as in SCAPS's batch
# "values are file names" and "variation of definition files"
FILE_PARAMS = {
    "MODEL.file":     ("whole model (.def file in the def folder)", "def"),
    "LIGHT.spectrum": ("illumination spectrum file (in the spectrum folder)", "spectrum"),
}
# Tested 24-10-2026 (diag/test_spectrum.py): 'action spectrumfile <name in the spectrum folder>' works
# (AM0.spe -> 136.04 mW/cm2 in the .iv header); 'load spectrumfile' was silently ignored; a missing file
# is logged and replaced by the default spectrum, which the proof check (spectrum named in .iv) catches.
SPECTRUM_COMMAND = "action spectrumfile"

TESTED = {"L.mun", "L.mup", "L.NA", "L.thickness", "L.Eg", "L.chi",
          "D.Ntotal", "D.Et", "C.workfunction", "L.Eg@B"}

_KEY = re.compile(r"^(?:L(?P<layer>\d)(?:\.D(?P<defect>\d))?|I(?P<iface>\d)\.D(?P<ifd>\d)|(?P<contact>left|right))"
                  r"\.(?P<name>[A-Za-z_]+)(?:@(?P<side>[AB]))?$")


class Param:
    """A resolved parameter: how to set it and how to prove it was set."""

    def __init__(self, key, kind, script_name=None, def_section=None, def_line=None,
                 factor=1.0, unit="", label="", route="script", column=1,
                 layer=None, defect=None, interface=None, ifdefect=None, contact=None,
                 action=None):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    @property
    def tested(self):
        short = {"layer": "L", "defect": "D", "ifdefect": "IFD", "contact": "C"}.get(self.kind, "")
        name = self.key.split(".")[-1]
        return f"{short}.{name}" in TESTED

    def script_lines(self, value):
        """Script commands that set this parameter to `value` (GUI units)."""
        if self.route != "script":
            return []
        if self.kind == "action":
            return [f"{self.action} {value}" if isinstance(value, str) else f"{self.action} {value:.10g}"]
        if self.kind == "layer":
            return [f"set layer{self.layer}.{self.script_name} {value:.10g}"]
        if self.kind == "defect":
            return [f"set layer{self.layer}.defect{self.defect}.{self.script_name} {value:.10g}"]
        if self.kind == "ifdefect":
            return [f"set interface{self.interface}.IFdefect{self.ifdefect}.{self.script_name} {value:.10g}"]
        if self.kind == "contact":
            prefix = "contactleft" if self.contact == "left" else "contactright"
            lines = []
            if self.script_name == "workfunction":
                # SCAPS ignores a work function while flat band is imposed
                lines.append(f"set {prefix}.flatband.off")
            lines.append(f"set {prefix}.{self.script_name} {value:.10g}")
            return lines
        raise ValueError(self.key)

    def describe(self):
        return f"{self.key}: {self.label} [{self.unit}] via {self.route}" + ("" if self.tested else " (first-use check)")


def resolve(key, model=None):
    """Turn a key like 'L2.mun', 'L3.Eg@B', 'L2.D1.Et', 'I3.D1.Ntotal', 'left.workfunction'
    or 'WP.temperature' into a Param. `model` (a DefFile) enables structure checks."""
    if key in ACTION_PARAMS:
        cmd, unit, label = ACTION_PARAMS[key]
        return Param(key, "action", unit=unit, label=label, action=cmd)
    if key == "LIGHT.spectrum":
        # an action (re-applied after every 'clear actions'); its value is a file name
        return Param(key, "action", unit="file", label=FILE_PARAMS[key][0], action=SPECTRUM_COMMAND)
    if key == "MODEL.file":
        # each run starts from this .def instead of the study's model; nothing is 'set'
        return Param(key, "model", unit="file", label=FILE_PARAMS[key][0], route="model")
    m = _KEY.match(key)
    if not m:
        raise ValueError(f"unknown parameter key '{key}'")
    g = m.groupdict()
    name, side = g["name"], g["side"] or "A"
    if g["contact"]:
        if name not in CONTACT_PARAMS:
            raise ValueError(f"unknown contact parameter '{name}'")
        sn, dl, f, u, lab = CONTACT_PARAMS[name]
        sec = "back contact" if g["contact"] == "left" else "front contact"
        return Param(key, "contact", sn, sec, dl, f, u, lab, contact=g["contact"])
    if g["iface"]:
        if name not in IFDEFECT_PARAMS:
            raise ValueError(f"unknown interface-defect parameter '{name}'")
        sn, dl, f, u, lab = IFDEFECT_PARAMS[name]
        i, d = int(g["iface"]), int(g["ifd"])
        if model is not None and model.find_section("interface recombination", interface=i, ifdefect=d) is None:
            raise ValueError(f"interface {i} has no defect {d} in this model")
        return Param(key, "ifdefect", sn, "interface recombination", dl, f, u, lab,
                     interface=i, ifdefect=d)
    layer = int(g["layer"])
    if model is not None and model.find_section("layer", layer=layer) is None:
        raise ValueError(f"layer {layer} does not exist in this model")
    if g["defect"]:
        if name not in DEFECT_PARAMS:
            raise ValueError(f"unknown defect parameter '{name}'")
        sn, dl, f, u, lab = DEFECT_PARAMS[name]
        d = int(g["defect"])
        if model is not None and model.find_section("srhrecombination", layer=layer, defect=d) is None:
            raise ValueError(f"layer {layer} has no defect {d} in this model")
        return Param(key, "defect", sn, "srhrecombination", dl, f, u, lab, layer=layer, defect=d)
    if name not in LAYER_PARAMS:
        raise ValueError(f"unknown layer parameter '{name}'")
    sn, dl, f, u, lab = LAYER_PARAMS[name]
    if side == "B":
        if model is not None and not model.is_graded(layer):
            raise ValueError(f"layer {layer} is not graded: it has no separate pure B value")
        return Param(key, "layer", sn, "layer", dl, f, u, lab + " (pure B)", route="def",
                     column=2, layer=layer)
    return Param(key, "layer", sn, "layer", dl, f, u, lab, layer=layer)


def catalogue(model):
    """All parameter keys available for a given model, grouped for a user interface."""
    keys = []
    names = model.layer_names()
    for s in model.sections:
        if s.kind == "layer":
            graded = model.is_graded(s.layer)
            for p in LAYER_PARAMS:
                keys.append(f"L{s.layer}.{p}")
                if graded and p != "thickness":
                    keys.append(f"L{s.layer}.{p}@B")
        elif s.kind == "srhrecombination":
            keys += [f"L{s.layer}.D{s.defect}.{p}" for p in DEFECT_PARAMS]
        elif s.kind == "interface recombination":
            keys += [f"I{s.interface}.D{s.ifdefect}.{p}" for p in IFDEFECT_PARAMS]
    keys += [f"{c}.{p}" for c in ("left", "right") for p in CONTACT_PARAMS]
    keys += list(ACTION_PARAMS)
    return keys, names


# ---------------- names people can read ----------------
_UNIT = {"cm-3": "cm⁻³", "cm-2": "cm⁻²", "cm2/Vs": "cm²/Vs", "cm2": "cm²", "cm3/s": "cm³/s", "cm6/s": "cm⁶/s",
         "um": "µm", "ohm.cm2": "Ω·cm²", "-": "", "file": ""}


def readable(key, model=None):
    """'L4.chi' -> 'FTO (layer 4): electron affinity [eV]'. Falls back to the key itself."""
    try:
        p = resolve(key)
    except ValueError:
        return key
    names = model.layer_names() if model is not None else {}
    unit = _UNIT.get(p.unit, p.unit)
    u = f" [{unit}]" if unit else ""
    label = p.label.replace(" (pure B)", "")
    if p.kind in ("layer", "defect"):
        where = f"{names[p.layer]} (layer {p.layer})" if p.layer in names else f"Layer {p.layer}"
        what = f"defect {p.defect}, {label}" if p.kind == "defect" else label
        side = ", pure B side of the grading" if "@B" in key else ""
        return f"{where}: {what}{side}{u}"
    if p.kind == "ifdefect":
        i = p.interface
        pair = f" ({names[i]} / {names[i + 1]})" if i in names and i + 1 in names else ""
        return f"Interface {i}{pair}: defect {p.ifdefect}, {label}{u}"
    if p.kind == "contact":
        side = "Back contact (left)" if p.contact == "left" else "Front contact (right)"
        return f"{side}: {label}{u}"
    return label[:1].upper() + label[1:] + u
