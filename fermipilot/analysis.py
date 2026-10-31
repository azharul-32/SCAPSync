"""Analysis on top of verified FermiPilot data.

  sensitivity(df, inputs, output)    which parameters matter most
  losses(iv_path)                     where photo-generated current is lost (recombination split)
  find_boundary(runner, key, ...)     where SCAPS stops converging along one parameter (bisection)
  inverse_design(df, inputs, target)  parameter sets predicted to hit a target, picked to be diverse;
                                      verify them with verify_candidates() in real SCAPS runs
Uses numpy and pandas (installed with Streamlit); scikit-learn's Random Forest when available.
"""
import math

import numpy as np
import pandas as pd

from . import results

LOG_HINTS = ("NA", "ND", "Ntotal", "Npeak", "Sn", "Sp", "NC", "NV", "radiative", "Auger")
OUTPUTS = ["eta_pct", "Voc_V", "Jsc_mAcm2", "FF_pct", "Vmpp_V", "Jmpp_mAcm2"]


def ok_rows(df):
    return df[df["status"].astype(str).str.startswith("ok")].copy()


def is_log(key, values=None):
    if values is not None:
        v = np.asarray(values, float)
        if np.all(v > 0) and v.max() / v.min() >= 100:
            return True
    return key.split(".")[-1].split("@")[0] in LOG_HINTS


def _features(df, inputs):
    X = np.column_stack([np.log10(df[k].astype(float)) if is_log(k, df[k]) else df[k].astype(float)
                         for k in inputs])
    return X


# ---------------- sensitivity ----------------
def sensitivity(df, inputs, output="eta_pct"):
    """Rank inputs by influence on one output. Returns a DataFrame with:
    spearman (monotonic effect, signed), src (standardized regression coefficient, signed,
    on log scale for wide-range inputs) and rf_importance (Random Forest, if available)."""
    d = ok_rows(df).dropna(subset=inputs + [output])
    if len(d) < max(5, len(inputs) + 2):
        raise ValueError(f"need at least {max(5, len(inputs) + 2)} ok runs, have {len(d)}")
    X = _features(d, inputs)
    y = d[output].astype(float).values
    Xs = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    ys = (y - y.mean()) / (y.std() or 1)
    src, *_ = np.linalg.lstsq(np.column_stack([Xs, np.ones(len(ys))]), ys, rcond=None)
    rank = lambda a: pd.Series(a).rank().values
    spear = [np.corrcoef(rank(X[:, i]), rank(y))[0, 1] if X[:, i].std() > 0 else 0.0
             for i in range(len(inputs))]
    out = pd.DataFrame({"parameter": inputs, "scale": ["log" if is_log(k, d[k]) else "lin" for k in inputs],
                        "spearman": np.round(spear, 3), "src": np.round(src[:-1], 3)})
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=300, random_state=1).fit(X, y)
        out["rf_importance"] = np.round(rf.feature_importances_, 3)
        out = out.sort_values("rf_importance", ascending=False)
    except ImportError:
        out = out.reindex(out["src"].abs().sort_values(ascending=False).index)
    r2 = 1 - np.sum((ys - np.column_stack([Xs, np.ones(len(ys))]) @ src) ** 2) / np.sum(ys ** 2)
    out.attrs["linear_r2"] = round(float(r2), 3)
    out.attrs["n"] = len(d)
    return out.reset_index(drop=True)


# ---------------- losses ----------------
LOSS_COLS = {"jbulk(mA/cm2)": "bulk (all layers)", "jifr(mA/cm2)": "interfaces",
             "jminor_left(mA/cm2)": "left (back) contact", "jminor_right(mA/cm2)": "right (front) contact"}
BULK_COLS = {"j_SRH(mA/cm2)": "SRH (defects)", "j_Radiative(mA/cm2)": "radiative", "j_Auger(mA/cm2)": "Auger"}


def losses(iv_path, at="mpp"):
    """Recombination split at short circuit, the maximum power point or open circuit.
    Returns dict: voltage, generated current, and per-mechanism current (mA/cm2) + share (%)."""
    cols, rows = results.table(iv_path)
    if not rows or "j_total_rec(mA/cm2)" not in cols:
        raise ValueError("this .iv file has no recombination columns")
    a = np.array(rows)
    v = a[:, cols.index("v(V)")]
    s = results.iv_summary(iv_path)
    target = {"jsc": 0.0, "mpp": s.get("Vmpp_V"), "voc": s.get("Voc_V")}[at]
    if target is None:
        raise ValueError(f"no {at} value in this file")
    pick = lambda c: float(np.interp(target, v, a[:, cols.index(c)]))
    total = pick("j_total_rec(mA/cm2)")
    out = {"at": at, "V": round(target, 4), "J_generated": round(pick("j_total_gen(mA/cm2)"), 4),
           "J_recombined": round(total, 4), "mechanisms": []}
    for c, name in list(LOSS_COLS.items()) + list(BULK_COLS.items()):
        if c in cols:
            j = pick(c)
            out["mechanisms"].append({"mechanism": name, "group": "where" if c in LOSS_COLS else "bulk by type",
                                      "J (mA/cm2)": round(j, 5), "share (%)": round(100 * j / total, 2) if total else None})
    return out


# ---------------- stability boundary ----------------
def find_boundary(sr, key, good, bad, scale="auto", rel_tol=None, max_runs=12, log=print, hold=None, label=None):
    """Bisection between a value that converges (`good`) and one that does not (`bad`).
    Each probe is a real verified SCAPS run through StudyRunner `sr`. Returns the
    tightest bracket found and every probe."""
    if scale == "auto":
        scale = "log" if (good > 0 and bad > 0 and max(good, bad) / min(good, bad) >= 10) else "lin"
    mid = (lambda a, b: math.sqrt(a * b)) if scale == "log" else (lambda a, b: (a + b) / 2)
    if rel_tol is None:
        rel_tol = 0.02 if scale == "log" else 0.002
    hold = {k: v for k, v in (hold or {}).items() if k != key}   # other parameters kept at these values
    probes = []
    for end, which in ((good, "good"), (bad, "bad")):
        row = sr.run({**hold, key: end}, run_id=f"bnd_{which}")
        probes.append((end, row["status"]))
        if row["status"].startswith("ok") != (which == "good"):
            log(f"  the {'working' if which == 'good' else 'failing'} value {end:g} gave '{row['status']}'")
            return {"key": key, "bracket": None, "probes": probes,
                    "message": (f"The value you gave as working ({end:g}) failed ('{row['status']}'). Choose a lower-risk working value."
                            if which == "good" else f"The value you gave as failing ({end:g}) converged here. "
                            "With the other parameters as set, there is no failure; the failure needs them too.")}
    for i in range(max_runs):
        if abs(bad - good) <= rel_tol * max(abs(good), abs(bad)):
            break
        m = mid(good, bad)
        row = sr.run({**hold, key: m}, run_id=f"bnd_{i + 1:02d}")
        ok = row["status"].startswith("ok")
        probes.append((m, row["status"]))
        log(f"  try {i + 1}: {label or key} = {m:.6g} -> {row['status']}")
        good, bad = (m, bad) if ok else (good, m)
    return {"key": key, "bracket": (good, bad), "probes": probes,
            "message": f"Limit found: works up to {good:.6g}, fails from {bad:.6g}."}


# ---------------- inverse design ----------------
def _surrogate(X, Y):
    try:
        from sklearn.ensemble import RandomForestRegressor
        one = Y.shape[1] == 1
        m = RandomForestRegressor(n_estimators=300, random_state=1, n_jobs=-1).fit(X, Y.ravel() if one else Y)
        pred = (lambda Q: m.predict(Q).reshape(-1, 1)) if one else m.predict
        return pred, "random forest"
    except ImportError:
        mu, sd = X.mean(0), np.where(X.std(0) == 0, 1, X.std(0))
        Xn = (X - mu) / sd

        def knn(Q, k=5):
            Qn = (Q - mu) / sd
            out = []
            for q in Qn:
                d = np.sqrt(((Xn - q) ** 2).sum(1)) + 1e-9
                idx = np.argsort(d)[:k]
                w = 1 / d[idx]
                out.append((Y[idx] * w[:, None]).sum(0) / w.sum())
            return np.array(out)
        return knn, "k-nearest neighbours"


def _cv_score(X, Y, folds=5):
    """5-fold cross-validated R2 per output: how well the surrogate predicts runs it never saw."""
    n = len(X)
    idx = np.random.RandomState(1).permutation(n)
    pred = np.zeros_like(Y)
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        p, _ = _surrogate(X[train], Y[train])
        pred[test] = p(X[test])
    ss = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ((Y - pred) ** 2).sum(0) / np.where(ss == 0, 1, ss)


def inverse_design(df, inputs, targets, n_candidates=50000, n_pick=5, seed=1, bounds=None):
    """targets: {"eta_pct": (value, tolerance), ...}. Returns (picks DataFrame, info dict).
    Candidates are sampled inside the training ranges (no extrapolation) unless `bounds` narrows them."""
    d = ok_rows(df).dropna(subset=inputs + list(targets))
    if len(d) < 20:
        raise ValueError(f"need at least 20 ok runs to train on, have {len(d)}")
    outs = list(targets)
    X, Y = _features(d, inputs), d[outs].astype(float).values
    predict, kind = _surrogate(X, Y)
    r2 = _cv_score(X, Y)
    rng = np.random.RandomState(seed)
    lo, hi = X.min(0), X.max(0)
    if bounds:
        for i, k in enumerate(inputs):
            if k in bounds:
                b = np.array(bounds[k], float)
                b = np.log10(b) if is_log(k, d[k]) else b
                lo[i], hi[i] = max(lo[i], b[0]), min(hi[i], b[1])
    C = lo + rng.rand(n_candidates, len(inputs)) * (hi - lo)
    P = predict(C)
    err = np.zeros(len(C))
    within = np.ones(len(C), bool)
    for j, o in enumerate(outs):
        val, tol = targets[o]
        e = np.abs(P[:, j] - val) / (tol or 1e-9)
        err += e ** 2
        within &= e <= 1
    order = np.argsort(err)
    pool = order[within[order]] if within.any() else order[: max(n_pick * 20, 100)]
    # diverse selection: best match first, then always the candidate farthest from those picked
    span = np.where(hi - lo == 0, 1, hi - lo)
    N = (C[pool] - lo) / span
    chosen = [0]
    dist = np.sqrt(((N - N[0]) ** 2).sum(1))
    while len(chosen) < min(n_pick, len(pool)):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.sqrt(((N - N[nxt]) ** 2).sum(1)))
    rows = []
    for c in chosen:
        x = C[pool[c]]
        r = {k: (10 ** x[i] if is_log(k, d[k]) else x[i]) for i, k in enumerate(inputs)}
        for j, o in enumerate(outs):
            r[f"predicted {o}"] = round(float(P[pool[c], j]), 4)
        rows.append(r)
    warnings = []
    for j, o in enumerate(outs):
        val, tol = targets[o]
        tmin, tmax = float(Y[:, j].min()), float(Y[:, j].max())
        if not (tmin - tol <= val <= tmax + tol):
            warnings.append(f"{o} target {val:g} is outside the training data ({tmin:.4g} to {tmax:.4g}); "
                            f"a surrogate cannot predict beyond what SCAPS produced, so widen the "
                            f"parameter ranges and run a new training study to reach it")
    if not within.any():
        warnings.append("no candidate is predicted within tolerance; the picks are the closest matches")
    info = {"surrogate": kind, "trained_on": len(d), "cv_r2": {o: float(v) for o, v in zip(outs, np.round(r2, 3))},
            "candidates": n_candidates, "within_tolerance": int(within.sum()),
            "training_range": {o: (float(Y[:, j].min()), float(Y[:, j].max())) for j, o in enumerate(outs)},
            "targets": {o: (float(v), float(t)) for o, (v, t) in targets.items()},
            "warnings": warnings}
    return pd.DataFrame(rows), info


def verify_candidates(sr, picks, inputs, log=print):
    """Run each picked parameter set in real SCAPS; add the actual outputs next to the predictions."""
    out = []
    for i, r in picks.reset_index(drop=True).iterrows():
        row = sr.run({k: float(r[k]) for k in inputs}, run_id=f"verify_{i + 1:02d}")
        rec = dict(r)
        rec["status"] = row["status"]
        for o in OUTPUTS:
            if o in row and row[o] not in ("", None):
                rec[f"SCAPS {o}"] = float(row[o])
        log(f"  candidate {i + 1}: {row['status']} eta={row.get('eta_pct', '')}")
        out.append(rec)
    return pd.DataFrame(out)


# ---------------- why did runs fail? ----------------
FAILED = ("no convergence", "timeout", "not calculated", "no output", "incomplete I-V")


def _x(df, k):
    v = df[k].astype(float).values
    return (np.log10(v), True) if is_log(k, df[k]) else (v, False)


def diagnose_failures(df, inputs, names=None):
    """Which parameter separates the failed runs from the OK runs?
    For each parameter: the best single cut (fail above / fail below a value), how many runs it
    classifies right, and a rank-based separation score (0 = no pattern, 1 = perfect).
    Returns (table DataFrame, verdict dict). Runs rejected by the proof check are left out."""
    st_ = df["status"].astype(str)
    d = df[st_.str.startswith("ok") | st_.isin(FAILED)].dropna(subset=inputs).copy()
    fail = d["status"].astype(str).isin(FAILED).values
    nf, n = int(fail.sum()), len(d)
    if nf == 0:
        raise ValueError("this study has no failed runs (no convergence / timeout / not calculated)")
    if nf == n:
        raise ValueError("every run failed, so there is nothing to compare with; widen or shift the ranges")
    rows = []
    for k in inputs:
        x, logx = _x(d, k)
        if np.unique(x).size < 2:
            continue
        # rank separation (Mann-Whitney AUC, folded): how often a failed run lies on one side of an OK run
        r = pd.Series(x).rank().values
        auc = (r[fail].sum() - nf * (nf + 1) / 2) / (nf * (n - nf))
        sep = abs(auc - 0.5) * 2
        # best single cut
        xs = np.unique(x)
        cuts = (xs[:-1] + xs[1:]) / 2
        best = (0, None, None)
        for c in cuts:
            for side in ("above", "below"):
                pred = x > c if side == "above" else x < c
                acc = (pred == fail).mean()
                if acc > best[0]:
                    best = (acc, c, side)
        acc, c, side = best
        on = (x > c) if side == "above" else (x < c)
        to_val = (lambda t: 10 ** t) if logx else (lambda t: t)
        # the nearest OK and failed values around the cut (the boundary lies between them)
        ok_side, f_side = x[~fail & ~on], x[fail & on]
        last_ok = (ok_side.max() if side == "above" else ok_side.min()) if ok_side.size else None
        first_fail = (f_side.min() if side == "above" else f_side.max()) if f_side.size else None
        rows.append({"parameter": k, "separation": round(float(sep), 3), "best cut accuracy": round(float(acc), 3),
                     "fails when": side, "cut": float(to_val(c)),
                     "last OK value": None if last_ok is None else float(to_val(last_ok)),
                     "first failing value": None if first_fail is None else float(to_val(first_fail)),
                     "failed runs explained": int((fail & on).sum()), "OK runs on the failing side": int((~fail & on).sum())})
    tab = pd.DataFrame(rows).sort_values(["best cut accuracy", "separation"], ascending=False).reset_index(drop=True)
    N = (names or {}).get
    top = tab.iloc[0].copy()
    top["parameter"] = N(top["parameter"], top["parameter"])
    second = float(tab.iloc[1]["best cut accuracy"]) if len(tab) > 1 else 0.0
    base = max(nf, n - nf) / n            # accuracy of always guessing the majority
    if top["best cut accuracy"] == 1.0 and second < 1.0:
        kind = "single"
        text = (f"The cause is **{top['parameter']}**. All {nf} failed runs have it {top['fails when']} "
                f"{top['cut']:.4g}, and no OK run does. Last OK value {top['last OK value']:.6g}, first failing "
                f"value {top['first failing value']:.6g}. The other parameters make no difference.")
    elif top["best cut accuracy"] >= 0.9 and top["best cut accuracy"] - second >= 0.1:
        kind = "mostly"
        text = (f"Most likely cause: **{top['parameter']}** ({int(top['failed runs explained'])} of {nf} failed runs "
                f"have it {top['fails when']} {top['cut']:.4g}, but {int(top['OK runs on the failing side'])} OK run(s) "
                f"too). Another parameter may also play a part; confirm with SCAPS below.")
    else:
        kind = "unclear"
        text = ("No single parameter separates the failed runs from the OK ones, so the failures probably come from a "
                "combination of values (see the rules below). Use the one-at-a-time test on a failed run to confirm "
                "which parameters matter.")
        if top["best cut accuracy"] <= base + 1e-9:
            text += " (With so few runs, add more runs to the study for a clearer picture.)"
    tree = _tree_rule(d, inputs, fail, names)
    return tab, {"kind": kind, "text": text, "failed": nf, "runs": n, "suspect": tab.iloc[0]["parameter"], "tree": tree,
                 "last_ok": tab.iloc[0]["last OK value"], "first_fail": tab.iloc[0]["first failing value"]}


def _tree_rule(d, inputs, fail, names=None):
    """A two-level decision tree as readable rules (shows combinations), or None without scikit-learn."""
    try:
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        return None
    X = np.column_stack([_x(d, k)[0] for k in inputs])
    t = DecisionTreeClassifier(max_depth=2, random_state=1).fit(X, fail)
    tr, logs = t.tree_, [_x(d, k)[1] for k in inputs]
    leaf = t.apply(X)
    lines = []

    def walk(node, cond):
        if tr.children_left[node] == -1:
            here = leaf == node
            lines.append(f"{' and '.join(cond) or 'all runs'}: {int(fail[here].sum())} failed, "
                         f"{int((~fail[here]).sum())} OK")
            return
        k = (names or {}).get(inputs[tr.feature[node]], inputs[tr.feature[node]])
        thr = 10 ** tr.threshold[node] if logs[tr.feature[node]] else tr.threshold[node]
        walk(tr.children_left[node], cond + [f"{k} ≤ {thr:.4g}"])
        walk(tr.children_right[node], cond + [f"{k} > {thr:.4g}"])
    walk(0, [])
    return lines


def nearest_ok(df, inputs, run_id):
    """The OK run most similar to `run_id` (distance over all inputs, each scaled to 0-1, log where wide)."""
    ok = df[df["status"].astype(str).str.startswith("ok")]
    if ok.empty:
        raise ValueError("no OK run to compare with")
    cols = []
    for k in inputs:
        x, _ = _x(df, k)
        span = (x.max() - x.min()) or 1.0
        cols.append((x - x.min()) / span)
    X = np.column_stack(cols)
    i = int(np.where(df["run_id"].values == run_id)[0][0])
    dist = np.sqrt(((X[df.index.get_indexer(ok.index)] - X[i]) ** 2).sum(1))
    return ok.iloc[int(np.argmin(dist))]


def swap_test(sr, failed, ok, inputs, log=print, max_pairs=10, names=None):
    """Start from a failed run; put its parameters back to the OK run's values one at a time (then in
    pairs if no single change helps) and rerun each in real SCAPS. A change that makes it converge is
    a cause. Returns dict with every trial and the verdict."""
    N = lambda k: (names or {}).get(k, k)
    base = {k: float(failed[k]) for k in inputs}
    diff = [k for k in inputs if not math.isclose(float(failed[k]), float(ok[k]), rel_tol=1e-9)]
    trials = []

    def trial(keys):
        vals = dict(base, **{k: float(ok[k]) for k in keys})
        rid = "swap_" + "+".join(k.replace(".", "_").replace("@", "_") for k in keys)
        row = sr.run(vals, run_id=rid)
        good = str(row["status"]).startswith("ok")
        trials.append({"changed back": " + ".join(N(k) for k in keys), "to": ", ".join(f"{float(ok[k]):.6g}" for k in keys),
                       "status": row["status"], "eta_pct": row.get("eta_pct", ""), "fixes it": good})
        log(f"  {' + '.join(N(k) for k in keys)} back to {', '.join(f'{float(ok[k]):.6g}' for k in keys)} -> {row['status']}")
        return good
    log(f"Failed run {failed['run_id']} vs nearest OK run {ok['run_id']}; {len(diff)} parameter(s) differ.")
    singles = [k for k in diff if trial([k])]
    pairs = []
    if not singles and len(diff) > 1:
        import itertools
        for a, b in list(itertools.combinations(diff, 2))[:max_pairs]:
            if trial([a, b]):
                pairs.append((a, b))
    if len(singles) == 1:
        others = [k for k in diff if k not in singles]
        text = (f"Confirmed: the cause is {N(singles[0])}. Changing only this back to the OK value makes the run "
                f"converge" + (f"; changing {', '.join(N(k) for k in others)} does not." if others else "."))
    elif singles:
        text = (f"The failure needs {' and '.join(N(k) for k in singles)} together: changing back any one of them "
                f"makes the run converge. The cause is this combination, not one parameter.")
    elif pairs:
        text = ("No single parameter is the cause; it is the combination: " +
                "; ".join(f"{N(a)} together with {N(b)}" for a, b in pairs) + ".")
    else:
        text = ("No single parameter or pair fixes it on its own. The failure needs several values together, "
                "or SCAPS is unstable near this point; try a finer study around these values.")
    return {"failed_run": failed["run_id"], "ok_run": ok["run_id"], "differs": diff, "trials": trials,
            "causes": singles, "pairs": [list(p) for p in pairs], "text": text}
