# SCAPSync
**Run large SCAPS-1D simulation studies on any parameter of your model, unattended, with proof that SCAPS used every value you asked for and every result in one table.**

*(formerly FermiPilot)* · Version 0.9.4, pre-release · Windows · tested on SCAPS 3.3.09.
SCAPS itself is not included: it is free from Ghent University and must be installed separately.

## Download and start
1. Install SCAPS-1D 3.3.09 if you have not yet.
2. Download `SCAPSync-0.9.4-windows.zip` from [Releases](https://github.com/azharul-32/SCAPSync/releases) and unzip it anywhere.
3. Double-click `SCAPSync.exe`. It opens in your web browser and finds SCAPS and your models.
   Keep the black window open while you use it.

Nothing else to install, no coding. Your .def and result files are never changed.

## What it does for you
- **Thousands of runs without sitting at the PC.** Choose the parameters (any layer, interface or
  contact) and their values. A sweep, every combination, random sampling over many parameters at once,
  or your own SCAPS batch (.sbf) or CSV list. SCAPSync runs them all, several SCAPS windows side by
  side: about 2 s per I-V run on a laptop, against 7.4 s when SCAPS restarts for every run.
  A crash or power cut costs nothing: the study continues from where it stopped.
- **Results you can trust.** For every run SCAPSync reads back what SCAPS actually used, as proof
  that it took your values. It labels runs SCAPS could not solve (no convergence, stuck, I-V stopped
  before Voc) instead of letting bad numbers such as an 8.6 V Voc slip into your data.
  One table holds every run's inputs, Voc, Jsc, FF and efficiency, ready for Excel or Python.
- **Know exactly why runs fail to converge.** Instead of guessing, SCAPSync names the input behind
  the failures (e.g. "back contact work function, too low: 2 failed runs") and proves it in SCAPS:
  each failed run is run again with one parameter at a time set back to a normal value, and the one
  that makes it converge is the cause. It then finds the exact value where failures start.
  Only one failing model? Compare it with a similar one that works and SCAPSync tests every difference.
  SCAPS failures repeat exactly when re-run, so this answer is reliable.
- **Designs that reach your target, with fewer simulations.** Give a target such as 24 % efficiency.
  SCAPSync learns from each round of runs and chooses the next ones where they help most, until
  several different designs reach the target in SCAPS. In a test on a CdSeTe cell it needed
  60 SCAPS runs for 24 % and 44 for ≥ 27.5 %, where random sampling needed 87-264 and 77-152.
- **Publication-ready graphs.** The SCAPS-1D Data Visualizer is built in: I-V, C-V, C-f, QE, band
  diagrams and generation, many runs overlaid, styled and exported, or any SCAPS file you upload.
- **See what limits your cell.** Rank which inputs change the efficiency most, and see where the current
  is lost in a run: in the bulk, at an interface or at a contact.

## From the source code
`py -3 launcher.py` (needs Python with streamlit, pandas, scikit-learn, altair, matplotlib).

By Azharul Islam, University of Chittagong. SCAPS-1D is by M. Burgelman et al., Ghent University.
SCAPSync is not affiliated with Ghent University.
