# SCAPSync

*An independent automation tool for SCAPS-1D (formerly FermiPilot).*

No-code automation for SCAPS-1D: run large simulation studies on any parameter of a SCAPS model,
unattended, with proof that SCAPS used the values you asked for, and every result in one table.

Version 0.9.3, pre-release. Developed and verified on SCAPS 3.3.09 (Windows). SCAPS itself is not
included: it is free from Ghent University and must be installed separately.

## Download and start (3 steps)
1. Install **SCAPS-1D** (free from Ghent University) if you have not yet.
2. Download **`SCAPSync-<version>-windows.zip`** from
   **[Releases](https://github.com/azharul-32/SCAPSync/releases/latest)** and unzip it anywhere.
3. Double-click **`SCAPSync.exe`**. It opens in your web browser and finds SCAPS and your models by itself.

Nothing else to install: Python and all libraries are inside. Keep the black window open while you use it.

Pages: New study · Results · Graphs · Analyse · Help. Models (.def) can be uploaded in the app from any folder.
From the source code instead: `py -3 launcher.py` (needs Python with streamlit, pandas, scikit-learn,
altair, matplotlib).

## What it does
- **Studies:** sweep, every combination, change together (SCAPS "simultaneous"), random sampling
  (Latin hypercube), single run, or your own list: a SCAPS batch file (.sbf) or a CSV.
  SCAPS batch files may vary numbers, spectrum files and whole model (.def) files.
- **Speed:** each SCAPS copy does up to 8 runs per start, and several copies run side by side.
  A one-time speed check picks the fastest number of copies for the PC.
  Measured on a Ryzen 7 4700U laptop: 2.1 s per I-V run, against 7.4 s restarting SCAPS every run.
- **Safety:** a stuck run is stopped and retried on its own; a run that does not converge is retried
  with more working-point steps; every run's settings are read back from SCAPS as proof.
- **Extra results (optional):** band diagram at 0 V for every run, giving band offsets, the largest
  electric field and the recombination loss of each layer and interface in the results table.
- **Graphs:** the SCAPS-1D Data Visualizer (github.com/azharul-32/scaps-graph-visualizer) is built in: plot
  any study's runs (.iv, .cv, .cf, .qe, .eb, .gen) overlaid like a SCAPS batch, with all its styling options,
  or upload any SCAPS file.
- **Inverse design with active learning:** give a target (e.g. efficiency 24 %); SCAPSync runs a small
  spread-out start, learns (with an uncertainty for every prediction and a model of where SCAPS fails), then
  chooses each next round of SCAPS runs where it helps most, until several different designs are verified in
  SCAPS. For hard targets it needed 2-4x fewer runs than random sampling in a first test on a stand-in built
  from 1,192 real CdSeTe runs (real-SCAPS comparison pending).
  Also from the command line: `python -m fermipilot design design.json`.
- **Analyse (optional):** parameter ranking, convergence limit (which parameter makes runs fail, and the exact
  value where SCAPS starts to fail), loss analysis, inverse design from a finished study.
- **Checks on every result:** an I-V sweep that SCAPS stopped before Voc is marked "incomplete I-V" (SCAPS
  would report an extrapolated Voc, e.g. 8.6 V); interrupted studies resume without repeating finished runs or
  giving failed runs a second chance; SCAPS starts minimized (its calculation panels can still appear briefly:
  SCAPS opens them itself while it calculates).

## Command line
```
python -m fermipilot inspect  "C:\Program Files (x86)\Scaps3309\def\CdSeTe Model.def"
python -m fermipilot params   "C:\Program Files (x86)\Scaps3309\def\CdSeTe Model.def"
python -m fermipilot run      studies\acceptance.json
```

By Azharul Islam, University of Chittagong. SCAPS-1D is by M. Burgelman et al., University of Gent.
