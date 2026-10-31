"""SCAPS-1D Data Visualizer by Azharul Islam (github.com/azharul-32/scaps-graph-visualizer, version of
16-10-2026), built into FermiPilot unchanged except: the interface is the function render(), which can be
given simulations from FermiPilot runs instead of an uploaded file. Standalone use still works:
    streamlit run fermipilot/graphs.py
"""
"""
SCAPS Output File Graph Generator - Fully Customizable Tool
============================================================
Upload a SCAPS output file (.iv, .cv, .cf or .qe) and build fully
customized, publication-style graphs: pick columns, colors, line styles, fonts, overlay multiple
simulations, add custom annotations, auto-generate a figure caption,
or batch-process many files into a grid of consistently styled plots.

HOW TO RUN:
  1. Install once:  pip install streamlit pandas matplotlib numpy openpyxl
  2. Run:            streamlit run iv_graph_tool.py
  3. Your browser opens automatically with the interface.
"""

import re
import io
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, LogLocator, AutoMinorLocator
import streamlit as st

if __name__ == "__main__":   # FermiPilot sets its own page config
    st.set_page_config(page_title="SCAPS-1D Data Visualizer", layout="wide")


# ============================================================
# PARSER - reads ALL columns directly from the file's own header
# ============================================================

LAYER_NAMES = {}   # 'L2' -> 'CdTe - bulk (L2)', filled while parsing


def clean_unit(u):
    """Repairs unit text whose special characters (µ, ², ³) were corrupted
    on the way into the file - they can turn into '?', a replacement
    character, or half-width Japanese letters depending on how the file
    was saved. Only repairs cases that can be deduced with certainty."""
    u = u.strip()
    odd = r"[^A-Za-z0-9/()\-\s.\^]{1,3}"   # 1-3 'unusual' characters
    if re.fullmatch(odd + "m", u):          # ?m, \uff75m, \ufffdm, Âµm -> micrometre
        return "\u00b5m"
    if re.fullmatch("cm" + odd + "/Vs", u):  # cm?/Vs                ->  mobility
        return "cm\u00b2/Vs"
    return u


def fix_column_name(col):
    """Repairs corrupted special characters in a column header. Only the
    cases that can be deduced with certainty are touched: W(?m) is the
    depletion width in micrometres and Z.phase(?) is the impedance phase
    in degrees."""
    m = re.match(r"^(W)\((.{1,3}m)\)$", col)
    if m:
        return "W(" + clean_unit(m.group(2)) + ")"
    if re.match(r"^Z\.phase\(.{1,3}\)$", col):
        return "Z.phase(\u00b0)"
    return col


def excel_to_pseudo_text(file_bytes):
    """Converts a .xlsx export of a SCAPS .iv file (same content, just
    saved as Excel with one value per cell) back into the same
    tab-separated line format the text parser already understands -
    this lets us reuse the exact same, already-tested parsing logic
    for both file types instead of maintaining two separate parsers."""
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(file_bytes), read_only=True)
    ws = wb.active
    lines = []
    for row in ws.iter_rows(values_only=True):
        cells = [c for c in row if c is not None]
        if not cells:
            lines.append("")
            continue
        lines.append("\t".join(str(c) for c in cells))
    text = "\n".join(lines)
    # SCAPS writes µ (byte 0xB5) and ² (byte 0xB2) in a Western
    # encoding, but Excel exports can reinterpret those same bytes as
    # half-width Japanese katakana (ｵ and ｲ). Map them back so the
    # unit labels read [µm] and [cm²/Vs], as in the original .iv file.
    for wrong, right in {"\uff75": "\u00b5", "\uff72": "\u00b2", "\uff73": "\u00b3",
                         "\uff70": "\u00b0"}.items():
        text = text.replace(wrong, right)
    return text


def parse_uploaded_file(filename, file_bytes):
    """Dispatches to the right reader based on file extension, then
    runs the shared parsing logic on the resulting text either way."""
    if filename.lower().endswith((".xlsx", ".xls")):
        text = excel_to_pseudo_text(file_bytes)
    else:
        # SCAPS files come in two encodings depending on how they were
        # saved: UTF-8 (µ = 2 bytes) or single-byte latin-1 (µ = 1 byte).
        # Try UTF-8 first - a latin-1 file is never valid UTF-8 as soon
        # as it contains µ or ², so this cannot misread it - then fall
        # back to latin-1, which accepts every byte.
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")
    return parse_iv_file(text)


KIND_NAMES = {"energy band diagrammes": "EB",
              "generation - recombination profiles": "GEN"}


def make_table(columns, rows):
    """DataFrame from parsed rows. Position column is normalised to 'x(\u00b5m)';
    the pure bookkeeping columns of the band-diagram table (index i, and the
    always-zero 1-D coordinate y) are dropped."""
    df = pd.DataFrame(rows, columns=columns)
    df = df.rename(columns={c: "x(\u00b5m)" for c in df.columns
                            if re.match(r"x\s*\([^)]*\)$", c)})
    return df.drop(columns=[c for c in ("i", "y") if c in df.columns])


def parse_iv_file(text):
    lines = text.splitlines()

    simulations = []
    current_data_rows = []
    current_columns = None
    current_sim_number = None
    current_batch_params = {}
    in_data = False
    in_batch_params = False
    summary = {}
    file_kind = "I-V"          # I-V, C-V, C-f, QE, EB or GEN - read from the file itself
    step_of = None             # (this step, total steps) of a batch run
    bias = None                # bias at which EB / GEN profiles were calculated
    dup_params = {}            # parameter listed twice in one batch block
    block_seen = {}
    iface_cols, iface_rows, in_iface = None, [], False

    def finish_sim():
        if current_columns is not None and current_data_rows:
            sim = {"sim_number": current_sim_number,
                   "data": make_table(current_columns, current_data_rows),
                   "summary": summary, "batch_params": current_batch_params,
                   "kind": file_kind, "step_of": step_of, "bias": bias,
                   "dup_params": dup_params}
            if iface_cols is not None and iface_rows:
                sim["interface"] = make_table(iface_cols, iface_rows)
            simulations.append(sim)

    for line in lines:
        stripped = line.strip()

        kind_m = re.match(r"Simulation of (.+?)(?: curves)?\s*$", stripped)
        if kind_m:
            raw_kind = kind_m.group(1).strip()
            file_kind = KIND_NAMES.get(raw_kind.lower(), raw_kind)
            continue

        if "Single shot simulation" in stripped or "Batch simulation" in stripped:
            finish_sim()
            current_data_rows = []
            current_columns = None
            current_batch_params = {}
            summary = {}
            step_of, bias, dup_params, block_seen = None, None, {}, {}
            iface_cols, iface_rows, in_iface = None, [], False
            m = re.search(r"#\s*(\d+)", stripped)
            step_m = re.search(r"step\s+(\d+)\s+of\s+(\d+)", stripped)
            if step_m:
                step_of = (int(step_m.group(1)), int(step_m.group(2)))
            step_m = re.search(r"step\s+(\d+)", stripped)
            if step_m:
                current_sim_number = int(step_m.group(1))
            elif m:
                current_sim_number = int(m.group(1))
            else:
                current_sim_number = len(simulations) + 1
            in_data = False
            in_batch_params = False
            continue

        # SCAPS writes a "**Batch parameters**" section listing exactly
        # which layer/parameter/unit/value was varied for this specific
        # step - this is the authoritative source for auto-labeling,
        # far more reliable than guessing from filenames or step order
        if "**Batch parameters**" in stripped:
            in_batch_params = True
            block_seen = {}            # SCAPS may repeat the whole block; only
            continue                   # a repeat INSIDE one block is suspicious
        if in_batch_params:
            if stripped == "":
                in_batch_params = False
                continue
            # Format: "CdTe - bulk (L2)>>thickness[µm]:\t 1.000e+00"
            bp_match = re.match(r"(.+?)>>(.+?)\[(.*?)\]:\s*(.+)", stripped)
            if bp_match:
                layer_name = bp_match.group(1).strip()
                param_name = bp_match.group(2).strip()
                unit = clean_unit(bp_match.group(3))
                try:
                    value = float(bp_match.group(4).strip())
                    # IMPORTANT: extract the short layer code (L1/L2/L3...)
                    # and keep it in the key. Two different layers can have
                    # a parameter with the same name (e.g. both L2 and L3
                    # have "thickness") - without the layer code, those
                    # would collide into one ambiguous label and silently
                    # overwrite each other / look like the same parameter
                    # with two contradictory values.
                    code_match = re.search(r"\((L\d+)\)", layer_name)
                    layer_code = code_match.group(1) if code_match else layer_name
                    LAYER_NAMES[layer_code] = layer_name
                    key = f"{layer_code}: {param_name} [{unit}]"
                    if key in block_seen and block_seen[key] != value:
                        lst = dup_params.setdefault(key, [block_seen[key]])
                        if lst[-1] != value:
                            lst.append(value)
                    block_seen[key] = value
                    current_batch_params[key] = value
                except ValueError:
                    pass
            continue

        bias_m = re.match(r".*at actual bias voltage\s+([-\d.eE+]+)", stripped)
        if bias_m:
            bias = float(bias_m.group(1))
            continue

        # Band-diagram files hold a second table (interfaces) after the bulk one
        if current_columns is not None and re.match(r"IF\t\s*x\(", stripped):
            iface_cols = [fix_column_name(c.strip())
                          for c in re.split(r"\t+", stripped) if c.strip()]
            in_iface, in_data = True, False
            continue
        if in_iface:
            if stripped == "":
                continue
            parts = [p for p in re.split(r"\t+|\s{2,}", stripped) if p.strip()]
            try:
                row = [float(p) for p in parts]
                if len(row) == len(iface_cols):
                    iface_rows.append(row)
            except ValueError:
                pass
            continue

        # The column header starts with the x-quantity of each file type:
        # v(V) for I-V and C-V, f(Hz) for C-f, lambda(nm) for QE,
        # i / x(um) for energy bands, x (um) for generation-recombination
        if current_columns is None and re.match(
                r"((v|f|lambda)\([^)]*\)|i\t\s*x\(um\)|x\s*\([^)]*\))\t", stripped):
            current_columns = [fix_column_name(c.strip())
                               for c in re.split(r"\t+", stripped) if c.strip()]
            in_data = True
            continue

        if in_data:
            if stripped == "":
                continue
            if stripped.startswith("solar"):
                in_data = False
                continue
            parts = re.split(r"\t+|\s{2,}", stripped)
            parts = [p for p in parts if p.strip() != ""]
            try:
                row = [float(p) for p in parts]
                if len(row) == len(current_columns):
                    current_data_rows.append(row)
            except ValueError:
                continue
            continue

        m = re.match(r"([A-Za-z_]+)\s*=\s*([\d.eE+-]+)", stripped)
        if m:
            summary[m.group(1)] = float(m.group(2))

    finish_sim()
    return simulations


FRIENDLY_NAMES = {
    "jtot(mA/cm2)": "Total Current Density",
    "j_total_rec(mA/cm2)": "Total Recombination Current",
    "j_total_gen(mA/cm2)": "Total Generation Current",
    "jbulk(mA/cm2)": "Bulk Recombination Current",
    "jifr(mA/cm2)": "Interface Recombination Current",
    "jminor_left(mA/cm2)": "Minority Carrier Recomb. (Left Contact)",
    "jminor_right(mA/cm2)": "Minority Carrier Recomb. (Right Contact)",
    "j_SRH(mA/cm2)": "SRH Recombination Current",
    "j_Radiative(mA/cm2)": "Radiative Recombination Current",
    "j_Auger(mA/cm2)": "Auger Recombination Current",
    "C(nF/cm2)": "Capacitance",
    "G(S/cm2)": "Conductance",
    "f(Hz)": "Frequency",
    "Z.re(ohm.cm2)": "Impedance (Real Part)",
    "Z.im(ohm.cm2)": "Impedance (Imaginary Part)",
    "Z.magn(ohm.cm2)": "Impedance Magnitude",
    "Z.phase(\u00b0)": "Impedance Phase",
    "W(\u00b5m)": "Depletion Width",
    "Napp(/cm3)": "Apparent Doping Density",
    "lambda(nm)": "Wavelength",
    "QE(%)": "Quantum Efficiency",
    "photon energy (eV)": "Photon Energy",
    # energy-band diagram (.eb)
    "x(\u00b5m)": "Position",
    "Ec(eV)": "Conduction Band Ec",
    "Ev(eV)": "Valence Band Ev",
    "Fn(eV)": "Electron Quasi-Fermi Level Fn",
    "Fp(eV)": "Hole Quasi-Fermi Level Fp",
    "n(/cm3)": "Electron Density n",
    "p(/cm3)": "Hole Density p",
    "rho(defect) (/cm3)": "Defect Charge Density",
    "net doping (/cm3)": "Net Doping",
    "rho(/cm3)": "Space Charge Density",
    "E(V/cm)": "Electric Field",
    "jn(mA/cm2)": "Electron Current Density",
    "jp(mA/cm2)": "Hole Current Density",
    "jn_tunnel(mA/cm2)": "Electron Tunnelling Current Density",
    "jp_tunnel(mA/cm2)": "Hole Tunnelling Current Density",
    "generation(#/cm3.s)": "Generation Rate",
    "recombination(#/cm3.s)": "Recombination Rate",
    "cumulative generation (mA/cm2)": "Cumulative Generation",
    "cumulative recombination L to R (mA/cm2)": "Cumulative Recombination (Left to Right)",
    "cumulative recombination R to L (mA/cm2)": "Cumulative Recombination (Right to Left)",
    # generation - recombination profiles (.gen)
    "Geh (#/cm3.s)": "Optical Generation Rate (Geh)",
    "Total recombination (#/cm3.s)": "Total Recombination Rate",
    "SRH recombination (#/cm3.s)": "SRH Recombination Rate",
    "Radiative recombination (#/cm3.s)": "Radiative Recombination Rate",
    "Auger recombination (#/cm3.s)": "Auger Recombination Rate",
}

GEN_TERMS = {"net Gn": "Net electron generation", "IPV Gn": "IPV electron generation",
             "Gnth": "Thermal electron generation", "Rnth": "Thermal electron recombination",
             "net Gp": "Net hole generation", "IPV Gp": "IPV hole generation",
             "Gpth": "Thermal hole generation", "Rpth": "Thermal hole recombination"}


def friendly(col):
    if col in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[col]
    # per-layer defect terms, e.g. 'net Gn2,1,1 (#/cm3.s)' = layer 2, defect 1, level 1
    m = re.match(r"(net Gn|IPV Gn|Gnth|Rnth|net Gp|IPV Gp|Gpth|Rpth)(\d+),(\d+),(\d+) \(", col)
    if m:
        return f"{GEN_TERMS[m.group(1)]}, L{m.group(2)} (defect {m.group(3)}.{m.group(4)})"
    return col


def col_unit(col):
    """Unit read straight from the column name, e.g. 'jtot(mA/cm2)' -> 'mA/cm2'."""
    m = re.search(r"\(([^()]*)\)\s*$", col)
    return m.group(1).strip() if m else ""


def pretty_unit(u):
    return (u.replace("ohm.cm2", "\u03a9\u00b7cm\u00b2").replace("#/cm3.s", "cm\u207b\u00b3 s\u207b\u00b9")
             .replace("cm2", "cm\u00b2").replace("cm3", "cm\u00b3"))


def is_voltage_col(col):
    return col.strip().startswith("v(")


def axis_label_for(cols, is_x=False):
    """Axis label that follows the chosen column(s): name + unit."""
    cols = list(cols)
    unit = pretty_unit(col_unit(cols[0]))
    if len(cols) == 1:
        if is_voltage_col(cols[0]):
            name = "Voltage"
        else:
            name = friendly(cols[0])
            if name == cols[0]:
                name = re.sub(r"\([^()]*\)\s*$", "", cols[0]).strip()
    else:
        names = [friendly(c) for c in cols]
        if all("Impedance" in n for n in names):
            name = "Impedance"
        elif all("Recombination" in n or "Recomb." in n for n in names):
            name = "Recombination Current"      # e.g. SRH + Radiative + Auger
        elif "mA/cm" in col_unit(cols[0]):
            name = "Current Density"            # mixed current types
        elif col_unit(cols[0]) == "eV":
            name = "Energy"                     # Ec, Ev, Fn, Fp
        elif "#/cm3.s" in col_unit(cols[0]):
            if all("Recombination" in n for n in names):
                name = "Recombination Rate"
            elif all("Generation" in n for n in names):
                name = "Generation Rate"
            else:
                name = "Generation / Recombination Rate"
        elif col_unit(cols[0]) == "/cm3" and all("Density" in n or "Doping" in n
                                                 for n in names):
            name = "Density"
        else:
            name = "Value"
    return f"{name} ({unit})" if unit else name


def find_voc(df, x_col):
    jtot_col = next((c for c in df.columns if "jtot" in c), None)
    if jtot_col is None:
        return None
    x_vals, j_vals = df[x_col].values, df[jtot_col].values
    sign_changes = np.where(np.diff(np.sign(j_vals)) != 0)[0]
    if len(sign_changes) == 0:
        return None
    idx = sign_changes[0]
    x0, x1 = x_vals[idx], x_vals[idx + 1]
    j0, j1 = j_vals[idx], j_vals[idx + 1]
    return x0 + (0 - j0) * (x1 - x0) / (j1 - j0)


LINESTYLE_MAP = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-dot": "-."}
BRIGHT_PALETTE = ['#e60000', '#009e3d', '#ff8c00', '#8b00ff', '#00b3b3', '#d4a017',
                  '#ff1493', '#1a75ff', '#7a4b00', '#6b8e00', '#444444', '#c000c0']


def apply_axis_styling(ax, x_is_voltage, font_size, show_ticks_and_grid, log_scale,
                        y_min_val, y_max_val, fixed_current_ticks, x_log=False):
    # X-axis: 0.1 V labeled ticks (0.01 V minor) only when x really is voltage;
    # any other column gets automatic tick spacing suited to its own values
    if x_log:
        pass                                   # log axis keeps its own decade ticks
    elif x_is_voltage:
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        if show_ticks_and_grid:
            ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    elif show_ticks_and_grid:
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis='x', which='major', length=5, width=0.7, direction='out')
    if show_ticks_and_grid:
        ax.tick_params(axis='x', which='minor', length=2.5, width=0.5, direction='out')

    if log_scale:
        if show_ticks_and_grid:
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1,
                                                    numticks=100))
    else:
        span = (y_max_val - y_min_val) if (y_min_val is not None and y_max_val is not None) else None
        if fixed_current_ticks and span is not None and span <= 250:
            # total current density only: labeled every 10, small tick every 1
            ax.yaxis.set_major_locator(MultipleLocator(10))
            if show_ticks_and_grid:
                ax.yaxis.set_minor_locator(MultipleLocator(1))
        elif show_ticks_and_grid:
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        if y_min_val is not None and y_max_val is not None:
            ax.set_ylim(y_min_val, y_max_val)

    ax.tick_params(axis='y', which='major', length=5, width=0.7, direction='out')
    if show_ticks_and_grid:
        ax.tick_params(axis='y', which='minor', length=2.5, width=0.5, direction='out')

    if show_ticks_and_grid:
        ax.grid(True, which='major', linestyle='-', linewidth=0.35, color='#e0e0e0')
        ax.grid(True, which='minor', linestyle='-', linewidth=0.2, color='#eeeeee')
    else:
        ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    ax.tick_params(axis='both', labelsize=font_size * 0.85)


def pretty_key(key):
    """'L2: electron mobility [cm2/Vs]' -> ('L2 electron mobility', 'cm²/Vs')"""
    m = re.match(r"(L\d+): (.+) \[(.*)\]$", key)
    if not m:
        return key, ""
    return f"{m.group(1)} {m.group(2)}", pretty_unit(m.group(3))


def param_text(key, v):
    name, unit = pretty_key(key)
    return f"{name} = {v:g} {unit}".strip()


def varied_and_fixed(sims):
    bps = [s.get("batch_params", {}) for s in sims]
    keys = []
    for bp in bps:
        for k in bp:
            if k not in keys:
                keys.append(k)
    varied = [k for k in keys if len({bp.get(k) for bp in bps}) > 1]
    fixed = ([(k, bps[0][k]) for k in keys if k not in varied and k in bps[0]]
             if len(sims) > 1 else [])
    return varied, fixed


def layer_boundaries(df, x_col):
    """Layer interfaces in a band-diagram / profile file: SCAPS writes the
    interface position twice in a row, so a repeated x value marks a boundary."""
    if not x_col.startswith("x("):
        return []
    xs = df[x_col].values
    return [float(x) for x, d in zip(xs[1:], np.diff(xs)) if d == 0]


def layer_slices(df, x_col):
    """Row ranges (start, end) of each layer, split at the repeated-x interface rows."""
    xs = df[x_col].values
    cuts = [i + 1 for i, d in enumerate(np.diff(xs)) if d == 0]
    starts = [0] + cuts
    ends = cuts + [len(xs)]
    return list(zip(starts, ends))


def defect_level_table(df, x_col, layer_idx, et, ref):
    """Defect energy level of one layer: Ev(x) + Et  ('above Ev')  or  Ec(x) - Et  ('below Ec').
    Et is the value typed in by the user from the SCAPS defect panel."""
    a, b = layer_slices(df, x_col)[layer_idx]
    seg = df.iloc[a:b]
    base = seg["Ev(eV)"] + et if ref == "above Ev" else seg["Ec(eV)"] - et
    return pd.DataFrame({x_col: seg[x_col].values, "Defect level Et": base.values})


def curve_style(i):
    """Colour i from the palette; after the palette is used up, repeat it dashed."""
    n = len(BRIGHT_PALETTE)
    return BRIGHT_PALETTE[i % n], ("-" if (i // n) % 2 == 0 else "--")


def build_legend(ax, fig, series_list, font_size, inside, anchor=None):
    """Same compact legend either way (one line per curve). Default: placed
    directly below the graph. Optional: inside, top-left under the summary box."""
    kw = dict(fontsize=font_size * 0.7, framealpha=0.95, edgecolor="gray",
              borderpad=0.4, labelspacing=0.3, handlelength=1.6, borderaxespad=0)
    hs = [Line2D([], [], color=s["color"], linestyle=s["linestyle"], linewidth=1.0)
          for s in series_list]
    labs = [s["label"] for s in series_list]
    if inside:
        return ax.legend(hs, labs, loc="upper left", bbox_to_anchor=anchor, **kw)
    fig.canvas.draw()
    xl = ax.xaxis.label.get_window_extent()
    y = ax.transAxes.inverted().transform((0, xl.y0))[1] - 0.04
    return ax.legend(hs, labs, loc="upper center", bbox_to_anchor=(0.5, y), **kw)


def draw_figure(series_list, x_col, y_cols, fixed, multi_sim, *, fig_width, font_size,
                log_scale, show_voc_line, show_ticks_and_grid, show_summary,
                y_min_val, y_max_val, only_jtot, xlabel, ylabel, annotations,
                legend_inside, x_log=False, layer_bounds=None):
    x_is_voltage = is_voltage_col(x_col)
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.72), dpi=110)

    for s in series_list:
        ax.plot(s["df"][x_col], s["df"][s["ycol"]], linewidth=0.9,
                color=s["color"], linestyle=s["linestyle"], zorder=4)

    voc_val = None
    if show_voc_line and x_is_voltage:
        voc_val = find_voc(series_list[0]["df"], x_col)
        if voc_val is not None:
            ax.axvline(x=voc_val, color="#0057ff", linestyle="--", linewidth=0.8,
                       zorder=3, alpha=0.85)

    if x_log:
        ax.set_xscale("log")
    apply_axis_styling(ax, x_is_voltage, font_size, show_ticks_and_grid, log_scale,
                       y_min_val, y_max_val, fixed_current_ticks=only_jtot, x_log=x_log)
    if log_scale:
        ax.set_yscale("log", nonpositive="mask")   # zeros are left out, not drawn as spikes
    if layer_bounds:
        xs_all = series_list[0]["df"][x_col].values
        edges = [float(xs_all.min())] + list(layer_bounds) + [float(xs_all.max())]
        for b in layer_bounds:
            ax.axvline(b, color="gray", linestyle=":", linewidth=0.7, zorder=2)
        if len(edges) - 1 <= 8:
            for k in range(len(edges) - 1):
                ax.text((edges[k] + edges[k + 1]) / 2, 0.99, f"L{k + 1}",
                        transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=font_size * 0.75, color="gray")
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)

    if voc_val is not None:
        ax.annotate(f"Voc = {voc_val:.4f} V", xy=(voc_val, ax.get_ylim()[1]),
                    xytext=(8, -8), textcoords="offset points",
                    fontsize=font_size * 0.85, color="#0057ff", ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="#0057ff", linewidth=0.5, alpha=0.95))

    for ann in annotations:
        ax.annotate(ann["text"], xy=(ann["x"], ann["y"]), fontsize=font_size * 0.8,
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow",
                              edgecolor="gray", linewidth=0.5, alpha=0.9))

    fig.tight_layout()

    # Voc/Jsc/FF/eta box (single simulation only), always top-left inside
    first_summary = series_list[0]["summary"]
    legend_anchor_y = 0.985
    if show_summary and first_summary and not multi_sim:
        lines = [f"{k} = {first_summary[k]:.4f}" for k in ["Voc", "Jsc", "FF", "eta"]
                 if k in first_summary]
        if lines:
            box = ax.text(0.015, 0.985, "\n".join(lines), transform=ax.transAxes,
                          fontsize=font_size * 0.7, va="top",
                          bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                                    edgecolor="gray", linewidth=0.6, alpha=0.95))
            fig.canvas.draw()
            bb = box.get_bbox_patch().get_window_extent()
            legend_anchor_y = ax.transAxes.inverted().transform((bb.x0, bb.y0))[1] - 0.006

    # A lone curve needs no legend under the graph (the axis label names it)
    if len(series_list) > 1 or legend_inside:
        build_legend(ax, fig, series_list, font_size, legend_inside,
                     (0.015, legend_anchor_y))
    return fig, first_summary


def render(source=None, header=True):
    """The whole SCAPS-1D Data Visualizer interface. source=None: upload a file (the original tool).
    source=(label, simulations): FermiPilot passes simulations read from its own runs."""
    # ============================================================
    # STREAMLIT INTERFACE
    # ============================================================

    if header:                  # standalone tool only; inside FermiPilot the page has its own title
        st.title("SCAPS-1D Data Visualizer")
        st.markdown(
            "<p style='font-size:16px; margin-bottom:0px;'>By Azharul Islam</p>"
            "<p style='font-size:13px; color:gray; margin-top:2px;'>University of Chittagong</p>",
            unsafe_allow_html=True
        )
    if source is None:
        st.caption("Upload a SCAPS output file (.iv, .cv, .cf, .qe, .eb or .gen) and build a fully customized graph. "
               "All values are read directly from the file - no approximation.")

    if "annotations" not in st.session_state:
        st.session_state.annotations = []

    if source is None:          # a guide for uploads only
        st.radio(
            "Mode",
            ["Single Simulation (upload a file from a single-shot run)",
             "Batch Sweep (upload a file from a batch run)"],
            horizontal=True,
            help="This is just a guide - whichever you pick, the tool reads your "
                 "file's actual content and gives you the right interface either way."
        )

    # ============================================================
    # UNIFIED FILE HANDLING - auto-detects single vs. batch content
    # regardless of which mode label the user clicked above, so an
    # accidental mismatch never blocks their workflow
    # ============================================================

    if True:
        if source is None:
            uploaded_file = st.file_uploader("Upload a SCAPS output file (.iv, .cv, .cf, .qe, .eb, .gen)",
                                              type=["iv", "cv", "cf", "qe", "eb", "gen", "txt",
                                                    "xlsx", "xls"])
            file_label = uploaded_file.name if uploaded_file is not None else None
            simulations = parse_uploaded_file(uploaded_file.name, uploaded_file.read()) \
                if uploaded_file is not None else None
        else:                       # FermiPilot: simulations already read from the chosen runs
            file_label, simulations = source

        if file_label is not None:

            if not simulations:
                st.error("No valid simulation data found in this file.")
            else:
                st.success(f"Found {len(simulations)} simulation(s) in this file.")

                # Some SCAPS file types (.eb, .gen) keep only the LAST step of a batch run
                so = simulations[0].get("step_of")
                if so and so[1] > len(simulations):
                    st.warning(f"This file holds {len(simulations)} of the {so[1]} batch steps "
                               f"(step {so[0]}). SCAPS stores only the last step for this file "
                               "type, so curves from different steps cannot be overlaid from "
                               "one file. Run each case separately to compare them.")
                for sim0 in simulations:
                    for k, vals in sim0.get("dup_params", {}).items():
                        shown = ", ".join(f"{v:g}" for v in vals)
                        st.warning(f"SCAPS lists '{k}' more than once in the batch parameters "
                                   f"({shown}). The tool uses the last value listed "
                                   f"({vals[-1]:g}). Check the batch setup in SCAPS.")

                # --- Feature: overlay multiple simulations, with automatic
                # labeling from SCAPS's own "**Batch parameters**" section
                # when available (shows exactly which parameter(s) were
                # varied, e.g. "thickness=1.0µm, mun=10"), falling back to
                # a plain step number if no batch parameters were found ---
                # In Single-Parameter Sweep mode this holds the one parameter
                # being compared, so legend entries can show just that value
                # (e.g. "L2: electron mobility [cm²/Vs] = 10") instead of
                # repeating every fixed parameter on every curve
                label_only_key = None

                def build_sim_label(s, only_key=None):
                    bp = s.get("batch_params", {})
                    if only_key and only_key in bp:
                        return f"{only_key} = {bp[only_key]:g}"
                    if bp:
                        # Keys are already short and distinct (e.g. "L2:
                        # thickness [µm]") thanks to the layer-code fix above
                        parts = [f"{k}={v:g}" for k, v in bp.items()]
                        return ", ".join(parts)
                    return f"Simulation #{s['sim_number']}"

                if len(simulations) > 1:
                    has_batch_params = any(s.get("batch_params") for s in simulations)

                    if has_batch_params:
                        st.info("Detected SCAPS batch parameters.")
                        selection_mode = st.radio(
                            "How do you want to select which steps to plot?",
                            ["Single-Parameter Sweep (vary one parameter, you choose the "
                             "fixed value for every other)",
                             "Custom Step Selection (freely pick any steps yourself)"],
                            index=0
                        )
                    else:
                        selection_mode = "Custom Step Selection (freely pick any steps yourself)"

                    if selection_mode.startswith("Single-Parameter"):
                        all_param_keys = sorted(set(
                            k for s in simulations for k in s.get("batch_params", {}).keys()
                        ))
                        chosen_param = st.selectbox(
                            "Which parameter do you want to compare across its different values?",
                            all_param_keys)

                        # For every OTHER varied parameter, let the user
                        # explicitly choose which of its actual tested
                        # values to hold fixed - via a real dropdown listing
                        # every distinct value that parameter was run at,
                        # not an automatic guess
                        other_keys = [k for k in all_param_keys if k != chosen_param]
                        reference_values = {}
                        if other_keys:
                            st.write(f"**Fix the other varied parameter(s):**")
                            fix_cols = st.columns(len(other_keys))
                            for col, k in zip(fix_cols, other_keys):
                                distinct_vals = sorted(set(
                                    s["batch_params"][k] for s in simulations
                                    if k in s.get("batch_params", {})
                                ))
                                with col:
                                    chosen_val = st.selectbox(
                                        k, distinct_vals,
                                        format_func=lambda v: f"{v:g}",
                                        key=f"fix_{k}")
                                    reference_values[k] = chosen_val

                        matching_sims = [
                            s for s in simulations
                            if chosen_param in s.get("batch_params", {})
                            and all(s["batch_params"].get(k) == v
                                    for k, v in reference_values.items())
                        ]
                        matching_sims.sort(key=lambda s: s["batch_params"][chosen_param])

                        if matching_sims:
                            found_vals = ", ".join(
                                f"{s['batch_params'][chosen_param]:g}" for s in matching_sims)
                            st.success(f"Found {len(matching_sims)} curve(s) for "
                                       f"{chosen_param} = [{found_vals}]")
                            selected_sims = matching_sims
                            label_only_key = chosen_param
                            if reference_values:
                                st.caption("Held fixed: " + ", ".join(
                                    f"{k} = {v:g}" for k, v in reference_values.items()))
                        else:
                            st.warning("No steps match this exact combination of fixed values.")
                            selected_sims = []
                    else:
                        sim_labels = [build_sim_label(s) for s in simulations]
                        if len(set(sim_labels)) != len(sim_labels):
                            sim_labels = [f"{lbl} (#{s['sim_number']})"
                                          for lbl, s in zip(sim_labels, simulations)]
                        selected_labels = st.multiselect(
                            "Select simulation(s) to plot (choose 2+ to overlay)",
                            sim_labels, default=[sim_labels[0]])
                        selected_sims = [simulations[sim_labels.index(lbl)] for lbl in selected_labels]
                else:
                    selected_sims = simulations

                if not selected_sims:
                    st.stop()

                if not selected_sims:
                    st.info("Select at least one simulation above.")
                    st.stop()

                all_columns = list(selected_sims[0]["data"].columns)
                file_kind = selected_sims[0].get("kind", "I-V")
                file_desc = {"EB": "energy band diagram", "GEN": "generation-recombination profile"}
                st.caption(f"Detected file type: {file_desc.get(file_kind, file_kind + ' curves')}")
                file_bias = selected_sims[0].get("bias")
                if file_kind in ("EB", "GEN") and file_bias is not None:
                    st.caption(f"Profile calculated at an applied bias of {file_bias:g} V.")

                # Sensible starting axes for each SCAPS file type
                #   I-V: Voltage vs Total Current Density      C-V: Voltage vs Capacitance
                #   C-f: Frequency vs Capacitance              QE : Wavelength vs Quantum Efficiency
                #   EB : Position vs Ec, Ev, Fn, Fp         GEN: Position vs Geh and total recombination
                x_prefix, y_prefix = {"I-V": ("v(", "jtot"), "C-V": ("v(", "C("),
                                      "C-f": ("f(", "C("), "QE": ("lambda(", "QE("),
                                      "EB": ("x(", "Ec("), "GEN": ("x(", "Geh")
                                      }.get(file_kind, ("v(", "jtot"))
                x_guess = next((c for c in all_columns if c.startswith(x_prefix)), all_columns[0])
                x_col = st.selectbox("X-axis parameter", all_columns,
                                      index=all_columns.index(x_guess), format_func=friendly)

                y_guess = next((c for c in all_columns if c != x_col and y_prefix in c), None)
                y_candidates = [c for c in all_columns if c != x_col]
                default_sel = [y_guess] if y_guess else y_candidates[:1]
                if file_kind == "EB":
                    default_sel = [c for c in ("Ec(eV)", "Ev(eV)", "Fn(eV)", "Fp(eV)")
                                   if c in y_candidates] or default_sel
                elif file_kind == "GEN":
                    default_sel = [c for c in ("Geh (#/cm3.s)", "Total recombination (#/cm3.s)")
                                   if c in y_candidates] or default_sel
                y_cols = st.multiselect("Y-axis parameter(s)", y_candidates,
                                         format_func=friendly, default=default_sel)

                if not y_cols:
                    st.info("Select at least one Y-axis parameter.")
                    st.stop()

                # Only parameters sharing the same unit may share one Y-axis
                units_found = {}
                for c in y_cols:
                    units_found.setdefault(pretty_unit(col_unit(c)), []).append(friendly(c))
                if len(units_found) > 1:
                    groups = "; ".join(f"{', '.join(names)} \u2014 unit: {u}"
                                       for u, names in units_found.items())
                    st.error(
                        "Graph cannot be generated: the selected Y-axis parameters have "
                        f"different units ({groups}). They cannot share one Y-axis. "
                        "Remove the parameters of one unit to continue."
                    )
                    st.stop()

                # Build the list of series: one per (simulation, column), each with
                # a single-line label built from the parameters that actually vary
                multi_sim = len(selected_sims) > 1
                fixed = None
                series_list = []
                for sim in selected_sims:
                    sim_label_short = build_sim_label(sim, only_key=label_only_key)
                    for ycol in y_cols:
                        if multi_sim:
                            label = (f"{sim_label_short} ({friendly(ycol)})" if len(y_cols) > 1
                                     else sim_label_short)
                        else:
                            label = friendly(ycol)
                        series_list.append({"df": sim["data"], "ycol": ycol, "label": label,
                                             "summary": sim["summary"]})

                # --- Optional: defect energy level (the green line in the SCAPS band diagram).
                # The .eb file does not contain Et, so the value is typed in from the
                # SCAPS defect panel; the line is then Ev + Et or Ec - Et, per layer ---
                if (file_kind == "EB" and x_col.startswith("x(") and not multi_sim
                        and "Ev(eV)" in all_columns and "Ec(eV)" in all_columns):
                    base_df = selected_sims[0]["data"]
                    n_layers = len(layer_slices(base_df, x_col))
                    with st.expander("Defect energy level (green line in SCAPS)"):
                        st.caption("Not stored in the .eb file. Enter Et and its reference exactly "
                                   "as in the SCAPS defect panel of each layer.")
                        for li in range(n_layers):
                            d1, d2, d3 = st.columns([1, 1, 1])
                            with d1:
                                use = st.checkbox(f"L{li + 1} defect level", value=False,
                                                  key=f"def_use_{li}")
                            with d2:
                                et = st.number_input(f"Et L{li + 1} (eV)", value=0.0, step=0.01,
                                                     format="%.4f", key=f"def_et_{li}")
                            with d3:
                                ref = st.selectbox(f"Reference L{li + 1}", ["above Ev", "below Ec"],
                                                   key=f"def_ref_{li}")
                            if use:
                                series_list.append({
                                    "df": defect_level_table(base_df, x_col, li, et, ref),
                                    "ycol": "Defect level Et", "label": f"Defect level Et, L{li + 1}",
                                    "summary": {}, "defect": True})

                with st.expander("Per-curve styling", expanded=not multi_sim):
                    for i, s in enumerate(series_list):
                        c1, c2, c3 = st.columns([2, 1, 1])
                        default_color, default_ls = curve_style(i)
                        if s["ycol"] in ("Fn(eV)", "Fp(eV)"):
                            default_ls = "--"           # quasi-Fermi levels dashed
                        if s.get("defect"):
                            default_color, default_ls = "#00a000", "-"
                        with c1:
                            st.write(s["label"])
                        with c2:
                            s["color"] = st.color_picker(f"Color##{i}", default_color,
                                                          label_visibility="collapsed")
                        with c3:
                            names = list(LINESTYLE_MAP.keys())
                            s["linestyle"] = LINESTYLE_MAP[st.selectbox(
                                f"Style##{i}", names,
                                index=0 if default_ls == "-" else 1,
                                label_visibility="collapsed")]

                # --- General options ---
                has_summary = any(s["summary"] for s in selected_sims)
                is_iv = file_kind == "I-V" and is_voltage_col(x_col)
                oc1, oc2 = st.columns(2)
                with oc1:
                    # generation/recombination rates span 1e9-1e22: only a log axis shows them
                    log_scale = st.checkbox("Log scale on Y-axis", value=(file_kind == "GEN"),
                                            key=f"ylog_{file_kind}")
                    # Frequency sweeps are conventionally shown on a log axis
                    x_log = st.checkbox("Log scale on X-axis", value=x_col.startswith("f("),
                                        key=f"xlog_{x_col}")
                    show_summary = (st.checkbox(
                        "Show Voc/Jsc/FF/Efficiency box (single curve only)", value=True)
                        if has_summary else False)
                with oc2:
                    show_voc_line = (st.checkbox("Show Voc marker (dashed line)", value=False)
                                     if is_iv else False)
                    legend_inside = st.checkbox("Place legend inside the graph", value=False)
                    show_ticks_and_grid = st.checkbox("Show minor tick marks and gridlines",
                                                       value=False)
                    show_layers = (st.checkbox("Show layer boundaries", value=False)
                                   if file_kind in ("EB", "GEN") and x_col.startswith("x(")
                                   else False)

                # Default Y range follows what is actually being plotted:
                # total current density keeps its dedicated -35 to 80 range;
                # every other parameter gets a range fitted to its own values
                only_jtot = file_kind == "I-V" and all("jtot" in c for c in y_cols)
                only_qe = all(c.startswith("QE(") for c in y_cols)
                y_all = np.concatenate([s["df"][s["ycol"]].values for s in series_list])
                data_min, data_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
                if only_jtot:
                    default_ymin, default_ymax = -35.0, 80.0
                elif only_qe:
                    default_ymin, default_ymax = 0.0, 100.0     # QE is a percentage
                else:
                    # padding: 5 % of the data span, but never less than 5 % of the
                    # largest value - so a nearly flat curve is not squeezed
                    # against the top edge of the graph
                    pad = max(0.05 * (data_max - data_min), abs(data_max) * 0.05, 1e-9)
                    default_ymin = 0.0 if data_min >= 0 else data_min - pad
                    default_ymax = data_max + pad
                    default_ymin = float(f"{default_ymin:.5g}")
                    default_ymax = float(f"{default_ymax:.5g}")
                range_step = (default_ymax - default_ymin) / 20 or 1.0
                range_key = (file_label + "|" + "|".join(y_cols) + "#" + ",".join(
                    str(s["sim_number"]) for s in selected_sims))

                y_min_val, y_max_val = None, None
                if not log_scale:
                    rcol1, rcol2 = st.columns(2)
                    with rcol1:
                        y_min_val = st.number_input("Y-axis minimum", value=default_ymin,
                                                    step=float(range_step), format="%.6g",
                                                    key=f"ymin_{range_key}")
                    with rcol2:
                        y_max_val = st.number_input("Y-axis maximum", value=default_ymax,
                                                    step=float(range_step), format="%.6g",
                                                    key=f"ymax_{range_key}")

                # --- Feature: custom axis labels (default follows the chosen columns) ---
                st.write("**Custom axis labels**")
                lc1, lc2 = st.columns(2)
                with lc1:
                    custom_xlabel = st.text_input("X-axis label",
                                                   value=axis_label_for([x_col], is_x=True),
                                                   key=f"xlabel_{x_col}")
                with lc2:
                    custom_ylabel = st.text_input("Y-axis label",
                                                   value=(pretty_unit(col_unit(y_cols[0]))
                                                          if file_kind == "GEN"
                                                          else axis_label_for(y_cols)),
                                                   key=f"ylabel_{'|'.join(y_cols)}")

                # --- Feature: font and graph size ---
                sc1, sc2 = st.columns(2)
                with sc1:
                    font_size = st.slider("Overall font size", min_value=6, max_value=16, value=9)
                with sc2:
                    fig_width = st.slider("Graph width (inches)", min_value=4.0, max_value=12.0,
                                          value=7.0, step=0.5)

                # --- Feature: annotation tool ---
                with st.expander("Add custom annotations"):
                    ac1, ac2, ac3, ac4 = st.columns([2, 1, 1, 1])
                    with ac1:
                        ann_text = st.text_input("Annotation text", key="ann_text")
                    with ac2:
                        ann_x = st.number_input("X position", value=0.5, key="ann_x")
                    with ac3:
                        ann_y = st.number_input("Y position", value=0.0, key="ann_y")
                    with ac4:
                        st.write("")
                        if st.button("Add"):
                            if ann_text:
                                st.session_state.annotations.append(
                                    {"text": ann_text, "x": ann_x, "y": ann_y})
                    if st.session_state.annotations:
                        st.write(f"{len(st.session_state.annotations)} annotation(s) added.")
                        if st.button("Clear all annotations"):
                            st.session_state.annotations = []

                if st.button("Generate Graph", type="primary"):
                    plt.rcParams["font.size"] = font_size
                    fig, first_summary = draw_figure(
                        series_list, x_col, y_cols, fixed, multi_sim,
                        fig_width=fig_width, font_size=font_size, log_scale=log_scale,
                        show_voc_line=show_voc_line, show_ticks_and_grid=show_ticks_and_grid,
                        show_summary=show_summary, y_min_val=y_min_val, y_max_val=y_max_val,
                        only_jtot=only_jtot, xlabel=custom_xlabel, ylabel=custom_ylabel,
                        annotations=st.session_state.annotations, legend_inside=legend_inside,
                        x_log=x_log,
                        layer_bounds=(layer_boundaries(selected_sims[0]["data"], x_col)
                                      if show_layers else None))
                    x_is_voltage = is_voltage_col(x_col)
                    st.pyplot(fig, use_container_width=False)

                    # --- Feature: auto-generated figure caption ---
                    col_names = ", ".join(dict.fromkeys(friendly(s["ycol"]) for s in series_list))
                    x_name = "Voltage" if x_is_voltage else friendly(x_col)
                    caption_parts = [f"Figure: {col_names} vs {x_name}"]
                    if first_summary:
                        stats = ", ".join(f"{k} = {first_summary[k]:.4f}" for k in
                                           ["Voc", "Jsc", "FF", "eta"] if k in first_summary)
                        if stats:
                            caption_parts.append(f"({stats})")
                    if file_kind in ("EB", "GEN") and file_bias is not None:
                        caption_parts.append(f"(at V = {file_bias:g} V)")
                    auto_caption = " ".join(caption_parts) + "."
                    st.text_area("Suggested figure caption (editable)", value=auto_caption,
                                  height=70)

                    png_buf = io.BytesIO()
                    fig.savefig(png_buf, format='png', dpi=300, bbox_inches='tight')
                    pdf_buf = io.BytesIO()
                    fig.savefig(pdf_buf, format='pdf', bbox_inches='tight')

                    dcol1, dcol2 = st.columns(2)
                    with dcol1:
                        st.download_button("Download PNG", png_buf.getvalue(),
                                            "scaps_graph.png", "image/png")
                    with dcol2:
                        st.download_button("Download PDF (publication)", pdf_buf.getvalue(),
                                            "scaps_graph.pdf", "application/pdf")

                with st.expander("View raw parsed data table"):
                    st.dataframe(selected_sims[0]["data"])

                iface_df = selected_sims[0].get("interface")
                if iface_df is not None:
                    with st.expander(f"Interface data ({len(iface_df)} interfaces)"):
                        st.dataframe(iface_df)


if __name__ == "__main__":
    render()
