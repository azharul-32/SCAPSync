"""Read and edit SCAPS problem-definition (.def) files.

A .def file is plain text with SI units. Sections start with a header line
("layer", "srhrecombination", "interface properties", ...). Parameter lines look like
    mu_n :\t 5.000000e-03\t 5.000000e-03\t ...\t 1\t 0\t[m^2/Vs]
For layer properties the numeric columns are (verified against GUI-made files):
    col1 = pure A (y=0), col2 = pure B (y=1), col6 = value at left side,
    col7 = value at right side, then grading-type flags and the unit.
Edits are done in place on the original text, keeping every other byte
(including CRLF line endings) unchanged.
"""
import re

HEADERS = {"convergence", "back contact", "layer", "interface properties",
           "interface recombination", "srhrecombination", "front contact",
           "working point"}

_num = r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?"


class Section:
    def __init__(self, kind, start):
        self.kind, self.start, self.end = kind, start, None
        self.layer = None        # 1-based layer number for layer / srh sections
        self.defect = None       # 1-based defect number inside its layer
        self.interface = None    # 1-based interface number (= layer on its left)
        self.ifdefect = None     # 1-based interface defect number

    def __repr__(self):
        return (f"<{self.kind} L{self.layer} D{self.defect} I{self.interface} "
                f"IFD{self.ifdefect} lines {self.start}-{self.end}>")


class DefFile:
    def __init__(self, text, path=None):
        self.path = path
        self.eol = "\r\n" if "\r\n" in text else "\n"
        self.lines = text.split(self.eol)
        self._scan()

    # ---------- loading / saving ----------
    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls(f.read().decode("latin-1"), path)

    def text(self):
        return self.eol.join(self.lines)

    def save(self, path):
        with open(path, "wb") as f:
            f.write(self.text().encode("latin-1"))

    # ---------- structure ----------
    def _scan(self):
        self.sections = []
        layer = interface = 0
        defect = ifdefect = 0
        cur = None
        for i, raw in enumerate(self.lines):
            s = raw.strip()
            if s.startswith(">") or s not in HEADERS:
                continue
            if cur is not None:
                cur.end = i
            cur = Section(s, i)
            if s == "layer":
                layer += 1
                defect = 0
                cur.layer = layer
            elif s == "srhrecombination":
                defect += 1
                cur.layer, cur.defect = layer, defect
            elif s == "interface properties":
                interface = layer          # interface N sits right of layer N
                ifdefect = 0
                cur.interface = interface
            elif s == "interface recombination":
                ifdefect += 1
                cur.interface, cur.ifdefect = interface, ifdefect
            self.sections.append(cur)
        if cur is not None:
            cur.end = len(self.lines)

    def find_section(self, kind, layer=None, defect=None, interface=None, ifdefect=None):
        for s in self.sections:
            if s.kind != kind:
                continue
            if layer is not None and s.layer != layer:
                continue
            if defect is not None and s.defect != defect:
                continue
            if interface is not None and s.interface != interface:
                continue
            if ifdefect is not None and s.ifdefect != ifdefect:
                continue
            return s
        return None

    def find_line(self, section, key):
        """Index of the line in `section` whose name part (before ' :') equals key
        or starts with key when key ends with '('  (e.g. 'Na(' matches 'Na(uniform)')."""
        for i in range(section.start + 1, section.end):
            raw = self.lines[i]
            if ":" not in raw or raw.lstrip().startswith(">"):
                continue
            name = raw.split(":", 1)[0].strip()
            if name == key or (key.endswith("(") and name.startswith(key)):
                return i
        return None

    # ---------- values ----------
    @staticmethod
    def numbers(line):
        """All numbers after the ' :' of a parameter line, in order."""
        rest = line.split(":", 1)[1]
        rest = rest.split("[", 1)[0]          # drop the unit
        return [float(x) for x in re.findall(_num, rest)]

    def get_columns(self, section, key):
        i = self.find_line(section, key)
        return None if i is None else self.numbers(self.lines[i])

    def set_columns(self, section, key, new_values):
        """Replace numeric columns of a line in place. new_values: {col_index_1based: value}.
        Each number keeps its original width and its fixed / exponential style."""
        i = self.find_line(section, key)
        if i is None:
            raise KeyError(f"line '{key}' not found in {section}")
        raw = self.lines[i]
        name, rest = raw.split(":", 1)
        unit = ""
        if "[" in rest:
            rest, unit = rest.split("[", 1)
            unit = "[" + unit
        out, pos, col = [], 0, 0
        for m in re.finditer(_num, rest):
            col += 1
            out.append(rest[pos:m.start()])
            tok = m.group(0)
            if col in new_values:
                tok = _format_like(tok, new_values[col])
            out.append(tok)
            pos = m.end()
        out.append(rest[pos:])
        self.lines[i] = name + ":" + "".join(out) + unit

    def layer_names(self):
        res = {}
        for s in self.sections:
            if s.kind == "layer":
                i = self.find_line(s, "name")
                res[s.layer] = self.lines[i].split(":", 1)[1].strip() if i else f"layer {s.layer}"
        return res

    def is_graded(self, layer):
        sec = self.find_section("layer", layer=layer)
        cols = self.get_columns(sec, "composition grading")
        # graded when y differs between the two sides, or a grading law is set
        return bool(cols) and (cols[0] != cols[1] or int(cols[7]) != 0)

    def composition(self, layer):
        """(y_left, y_right) of a layer."""
        sec = self.find_section("layer", layer=layer)
        c = self.get_columns(sec, "composition grading")
        return c[5], c[6]


def _format_like(old, value):
    """Format `value` in the same style and width as the token `old`."""
    if "e" in old.lower():
        mant = old.lower().split("e")[0]
        decimals = len(mant.split(".")[1]) if "." in mant else 0
        exp_digits = len(old.lower().split("e")[1].lstrip("+-"))
        s = f"{value:.{decimals}e}"
        m, e = s.split("e")
        sign = "-" if e.startswith("-") else "+"
        s = f"{m}e{sign}{abs(int(e)):0{exp_digits}d}"
    else:
        decimals = len(old.split(".")[1]) if "." in old else 0
        s = f"{value:.{decimals}f}"
    return s.rjust(len(old))


def summary(model):
    """Plain overview of a model in GUI units: one dict per layer, plus contacts and interfaces."""
    layers = []
    names = model.layer_names()
    for s in model.sections:
        if s.kind != "layer":
            continue
        g = lambda key, f=1.0, col=0: (model.get_columns(s, key) or [None])[col]
        val = lambda key, f, col=0: None if g(key) is None else (model.get_columns(s, key)[col] / f)
        na = val("Na(", 1e6); nd = val("Nd(", 1e6)
        ndef = sum(1 for d in model.sections if d.kind == "srhrecombination" and d.layer == s.layer)
        layers.append({"layer": f"L{s.layer}", "name": names[s.layer],
                       "d (um)": val("d", 1e-6), "Eg (eV)": val("Eg", 1), "chi (eV)": val("chi", 1),
                       "mu_n (cm2/Vs)": val("mu_n", 1e-4), "mu_p (cm2/Vs)": val("mu_p", 1e-4),
                       "NA (cm-3)": na, "ND (cm-3)": nd,
                       "type": "p" if (na or 0) > (nd or 0) else ("n" if (nd or 0) > (na or 0) else "i"),
                       "graded": "yes" if model.is_graded(s.layer) else "no", "defects": ndef})
    contacts = []
    for side, kind in (("left (back)", "back contact"), ("right (front)", "front contact")):
        s = model.find_section(kind)
        fb = model.get_columns(s, "flatband")
        wf = model.get_columns(s, "Fi_m")
        contacts.append({"contact": side,
                         "work function": "flat band" if fb and fb[0] == 1 else (f"{wf[0]:g} eV" if wf else "?"),
                         "Sn (cm/s)": model.get_columns(s, "Sn")[0] / 1e-2,
                         "Sp (cm/s)": model.get_columns(s, "Sp")[0] / 1e-2})
    interfaces = []
    for s in model.sections:
        if s.kind == "interface recombination":
            interfaces.append({"interface": f"I{s.interface}", "defect": f"D{s.ifdefect}",
                               "N (cm-2)": model.get_columns(s, "N")[0] / 1e4,
                               "Et (eV)": model.get_columns(s, "Et")[0]})
    return layers, contacts, interfaces
