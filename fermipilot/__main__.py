"""Command line:
    python -m fermipilot where                    where SCAPS and the results folder are
    python -m fermipilot inspect  <model.def>     show the model's structure
    python -m fermipilot params   <model.def>     list every parameter key it accepts
    python -m fermipilot run      <study.json>    run a study headless
    python -m fermipilot boundary <study.json> <key> <good> <bad> [hold.json]
                                                  bisection for the convergence limit of one parameter
    python -m fermipilot verify   <study.json> <picks.csv>
                                                  run inverse-design picks in SCAPS, compare with predictions
    python -m fermipilot design   <design.json>       active-learning inverse design: SCAPS runs chosen
                                                  by ML until enough designs hit the target
    python -m fermipilot whyfail  <study folder> <run_id>
                                                  one-at-a-time test: which parameter makes this failed run fail
"""
import json
import os
import sys

from . import catalogue, defmodel, study


def inspect(path):
    m = defmodel.DefFile.load(path)
    names = m.layer_names()
    for s in m.sections:
        if s.kind == "layer":
            sec = s
            d = m.get_columns(sec, "d")[0] * 1e6
            graded = " (graded)" if m.is_graded(s.layer) else ""
            print(f"L{s.layer}  {names[s.layer]:<16} d = {d:.4g} um{graded}")
        elif s.kind == "srhrecombination":
            print(f"    defect D{s.defect}")
        elif s.kind == "interface recombination":
            print(f"I{s.interface}  interface defect D{s.ifdefect}")


def main(argv):
    if argv[:1] == ["where"]:
        # where FermiPilot finds SCAPS and where it writes (for support questions)
        import tempfile
        from . import paths
        d = paths.find_scaps()
        print("SCAPS folder:", d)
        if d:
            try:
                fd, probe = tempfile.mkstemp(prefix=".fermipilot_write_test_", dir=d)
                os.close(fd)
                os.remove(probe)
                print("SCAPS folder writable: yes")
            except OSError as e:
                print("SCAPS folder writable: no -", e)
            print("results folder:", paths.runs_root(d))
        print("settings:", paths.load_settings())
        return 0
    if len(argv) < 2:
        print(__doc__)
        return 1
    cmd, arg = argv[0], argv[1]
    if cmd == "inspect":
        inspect(arg)
    elif cmd == "params":
        keys, _ = catalogue.catalogue(defmodel.DefFile.load(arg))
        print("\n".join(keys))
    elif cmd == "run":
        with open(arg, encoding="utf-8") as f:
            study.run_study(json.load(f))
    elif cmd == "boundary":
        from . import analysis
        with open(arg, encoding="utf-8") as f:
            st = json.load(f)
        key, good, bad = argv[2], float(argv[3]), float(argv[4])
        hold = {}
        if len(argv) > 5:                       # optional JSON file: other parameters held at these values
            with open(argv[5], encoding="utf-8") as f:
                hold = json.load(f)
        sr = study.StudyRunner(st)
        name = catalogue.readable(key, sr.base)
        print(f"Looking for the limit of {name} between {good:g} (works) and {bad:g} (fails)", flush=True)
        res = analysis.find_boundary(sr, key, good, bad, log=lambda m: print(m, flush=True), hold=hold, label=name)
        res["name"] = name
        with open(os.path.join(sr.out_root, "boundary.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(res["message"], flush=True)
    elif cmd == "whyfail":
        import pandas as pd
        from . import analysis
        folder, run_id = arg, argv[2]
        with open(os.path.join(folder, "study.json"), encoding="utf-8") as f:
            st = json.load(f)
        df = pd.read_csv(os.path.join(folder, "master.csv"))
        inputs = [c for c in df.columns if "." in c and pd.api.types.is_numeric_dtype(df[c])
                  and df[c].notna().all() and df[c].nunique() > 1]
        failed = df[df["run_id"] == run_id].iloc[0]
        ok = analysis.nearest_ok(df, inputs, run_id)
        st = dict(st, name=f"_whyfail/{st['name']}_{run_id}", design={"type": "list", "runs": []}, parallel=1)
        sr = study.StudyRunner(st, log=lambda m: print(m, flush=True))
        names = {k: catalogue.readable(k, sr.base) for k in inputs}
        res = analysis.swap_test(sr, failed, ok, inputs, log=lambda m: print(m, flush=True), names=names)
        with open(os.path.join(folder, f"whyfail_{run_id}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)
        print(res["text"], flush=True)
    elif cmd == "design":
        from . import active
        with open(arg, encoding="utf-8") as f:
            spec = json.load(f)
        active.run_design(spec, log=lambda m: print(m, flush=True))
    elif cmd == "verify":
        import pandas as pd
        from . import analysis
        with open(arg, encoding="utf-8") as f:
            st = json.load(f)
        picks = pd.read_csv(argv[2])
        inputs = [c for c in picks.columns if "." in c and not c.startswith("predicted")]
        sr = study.StudyRunner(st)
        print(f"Verifying {len(picks)} candidates in SCAPS", flush=True)
        res = analysis.verify_candidates(sr, picks, inputs, log=lambda m: print(m, flush=True))
        res.to_csv(os.path.join(sr.out_root, "verification.csv"), index=False)
        print("Done. verification.csv written", flush=True)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
