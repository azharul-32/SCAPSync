"""Read SCAPS result files (.iv, .cv, .cf, .qe, .eb, .gen) - summary values and tables."""
import re

SUMMARY = {"Voc": "Voc_V", "Jsc": "Jsc_mAcm2", "FF": "FF_pct", "eta": "eta_pct",
           "V_MPP": "Vmpp_V", "J_MPP": "Jmpp_mAcm2"}


def read_text(path):
    with open(path, "rb") as f:
        data = f.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def iv_summary(path):
    """Solar cell parameters of the LAST simulation in an .iv file."""
    text = read_text(path)
    blocks = re.split(r"(?:Single shot|Batch) simulation", text)
    last = blocks[-1]
    out = {}
    for line in last.splitlines():
        m = re.match(r"\s*(Voc|Jsc|FF|eta|V_MPP|J_MPP)\s*=\s*([-+\d.eE]+)", line)
        if m:
            out[SUMMARY[m.group(1)]] = float(m.group(2))
    out["n_simulations"] = len(blocks) - 1
    return out


def iv_blocks(path):
    """Solar cell parameters of EVERY simulation in an .iv file (e.g. a SCAPS batch result),
    with the batch parameter values SCAPS printed in each block header, if any."""
    text = read_text(path)
    parts = re.split(r"(?:Single shot|Batch) simulation", text)[1:]
    out = []
    for blk in parts:
        rec = {}
        for line in blk.splitlines():
            m = re.match(r"\s*(Voc|Jsc|FF|eta|V_MPP|J_MPP)\s*=\s*([-+\d.eE]+)", line)
            if m:
                rec[SUMMARY[m.group(1)]] = float(m.group(2))
        bp = re.findall(r"^\s*(?:batch parameter|Batch parameter)[^\n]*$", blk, flags=re.M)
        if bp:
            rec["batch_header"] = " | ".join(x.strip() for x in bp)
        out.append(rec)
    return out


def table(path):
    """First data table of any SCAPS result file: (column names, rows of floats)."""
    cols, rows = None, []
    for line in read_text(path).splitlines():
        s = line.strip()
        if cols is None:
            if re.match(r"((v|f|lambda)\([^)]*\)|i\s+x\(um\)|x\s*\([^)]*\))\t", s):
                cols = [c.strip() for c in re.split(r"\t+", s) if c.strip()]
            continue
        if not s:
            if rows:
                break
            continue
        try:
            vals = [float(x) for x in re.split(r"\t+|\s{2,}", s) if x.strip()]
        except ValueError:
            break
        if len(vals) == len(cols):
            rows.append(vals)
    return cols, rows


def _eb_tables(text):
    """(bulk columns, bulk rows, interface columns, interface rows) of a .eb file."""
    bulk_cols = if_cols = None
    bulk, ifs, part = [], [], None
    for line in text.replace("\r", "").split("\n"):
        s = line.strip()
        if s == "bulk":
            part = "bulk"
            continue
        if s == "interface":
            part = "if"
            continue
        cells = [c.strip() for c in line.split("\t") if c.strip()]
        if not cells:
            continue
        if cells[0] in ("i", "IF"):
            if part == "bulk":
                bulk_cols = cells
            elif part == "if":
                if_cols = cells
            continue
        if not re.match(r"^\d+$", cells[0]):
            continue
        try:
            vals = [float(c) for c in cells]
        except ValueError:
            continue
        if part == "bulk" and bulk_cols and len(vals) == len(bulk_cols):
            bulk.append(vals)
        elif part == "if" and if_cols and len(vals) == len(if_cols):
            ifs.append(vals)
    return bulk_cols, bulk, if_cols, ifs


def eb_extras(path, interfaces_um=None):
    """Scalar quantities from a band diagram file (.eb), for the results table.
    Names start with 'eb_'. Layers and interfaces are numbered from the left (back) contact, as in
    SCAPS (interface i lies between layer i and layer i+1).
      eb_bias_V                  bias of this band diagram
      eb_I<i>_dEc_eV, _dEv_eV    conduction / valence band step at interface i: value just right of
                                 the interface minus value just left of it
      eb_I<i>_Jrec_mAcm2         recombination current at interface i (SCAPS 'js')
      eb_L<n>_Jrec_mAcm2         bulk recombination current in layer n (from SCAPS's cumulative
                                 recombination, L to R)
      eb_L<n>_Emax_Vcm           largest |electric field| in layer n
      eb_Jgen_mAcm2              total optical generation current
      eb_Jrec_total_mAcm2        total recombination current, bulk + interfaces
    Check at short circuit under light (CdSeTe model, 24-10-2026): Jgen 29.9905 - Jrec_total 0.8914
    = 29.0991 mA/cm2 = Jsc from the I-V; layer sums + interface js = Jrec_total.
    interfaces_um: positions of the interfaces from the left contact (from the layer thicknesses SCAPS
    used). SCAPS prints x with 6 decimals, so a very fine mesh (e.g. near a highly doped contact) also
    repeats x; only repeats at these positions are interfaces (seen 30-10-2026: 'layer 12' in a
    4-layer model). Without positions, repeats closer than 1 nm to the contact or to each other are ignored."""
    text = read_text(path)
    cols, rows, icols, irows = _eb_tables(text)
    if not cols or len(rows) < 2:
        return {}
    out = {}
    m = re.search(r"Band diagramme at actual bias voltage\s+([-+\d.eE]+)", text)
    if m:
        out["eb_bias_V"] = float(m.group(1))

    def col(name):
        for i, c in enumerate(cols):
            if c.lower().startswith(name.lower()):
                return i
        return None
    ix, iec, iev, ie = col("x("), col("Ec("), col("Ev("), col("E(V/cm)")
    igen, irec = col("cumulative generation"), col("cumulative recombination L to R")
    # SCAPS repeats the position at an interface: one mesh point for each side
    rep = [k for k in range(1, len(rows)) if abs(rows[k][ix] - rows[k - 1][ix]) < 1e-9]
    if interfaces_um:
        cuts = []
        for pos in interfaces_um:
            near = [k for k in rep if abs(rows[k][ix] - pos) <= 2e-6 + 1e-5 * abs(pos)]
            if near:
                cuts.append(min(near, key=lambda k: abs(rows[k][ix] - pos)))
        cuts = sorted(set(cuts))
    else:
        cuts, last = [], 0.0
        for k in rep:
            if rows[k][ix] - last > 1e-3:
                cuts.append(k)
                last = rows[k][ix]
    bounds = [0] + cuts + [len(rows)]
    for i, k in enumerate(cuts, 1):
        if iec is not None:
            out[f"eb_I{i}_dEc_eV"] = round(rows[k][iec] - rows[k - 1][iec], 6)
        if iev is not None:
            out[f"eb_I{i}_dEv_eV"] = round(rows[k][iev] - rows[k - 1][iev], 6)
    for n in range(1, len(bounds)):
        a, b = bounds[n - 1], bounds[n]
        seg = rows[a:b]
        if ie is not None:
            out[f"eb_L{n}_Emax_Vcm"] = max(abs(r[ie]) for r in seg)
        if irec is not None:
            out[f"eb_L{n}_Jrec_mAcm2"] = seg[-1][irec] - seg[0][irec]
    if igen is not None:      # accumulated from the illuminated side, so the total sits at one end
        out["eb_Jgen_mAcm2"] = max(abs(rows[0][igen]), abs(rows[-1][igen]))
    if irec is not None:      # includes the interface recombination (the steps at the interfaces)
        out["eb_Jrec_total_mAcm2"] = rows[-1][irec]
    if icols and irows:
        ijs = next((i for i, c in enumerate(icols) if c.lower().startswith("js(")), None)
        if ijs is not None:
            for r in irows:
                out[f"eb_I{int(r[0]) + 1}_Jrec_mAcm2"] = r[ijs]
    return out
