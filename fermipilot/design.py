"""Inverse design on verified FermiPilot data (works for any SCAPS model).

Forward simulation answers "this design -> this efficiency". Inverse design answers "this target -> which
designs", and the answer is a REGION, not one point. This module maps that region and picks designs from it:

  Surrogate(df, params, scales)       ML model of the SCAPS results, with an uncertainty for every prediction
                                      (bootstrap ensemble of gradient-boosted trees) and a model of where SCAPS
                                      fails to converge (classifier on all runs, failed ones included).
  requirement_map(s, ranges, target)  the feasible region: candidates that reach the target with confidence
                                      and are predicted to converge; per-parameter necessary bounds; 2-D maps.
  minimum_change(s, baseline, ...)    the smallest change of 1, 2 or 3 parameters from a baseline that
                                      reaches the target (how to get there from a real starting point).
  robustness(s, x, ...)               chance a design stays above target under process variation.
  boundary_points(s, ...)             where the surrogate is least sure about target/not target: the most
                                      useful next SCAPS runs (active learning).
Every design proposed here must be checked in SCAPS (FermiPilot study, design type "list") before use.
"""
import itertools
import math

import numpy as np
import pandas as pd

OK = ("ok", "ok with warnings")


def to_unit(X, ranges, scales):
    """Parameter values -> [0, 1] per parameter (log10 for log-scaled ones). The common scale for
    distances ('how big a change')."""
    out = np.empty_like(X, dtype=float)
    for j, (lo, hi) in enumerate(ranges):
        if scales[j] == "log":
            out[:, j] = (np.log10(X[:, j]) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        else:
            out[:, j] = (X[:, j] - lo) / (hi - lo)
    return out


def from_unit(U, ranges, scales):
    out = np.empty_like(U, dtype=float)
    for j, (lo, hi) in enumerate(ranges):
        if scales[j] == "log":
            out[:, j] = 10 ** (math.log10(lo) + U[:, j] * (math.log10(hi) - math.log10(lo)))
        else:
            out[:, j] = lo + U[:, j] * (hi - lo)
    return out


class Surrogate:
    """Predicts SCAPS outputs (mean and spread) and the chance that SCAPS converges."""

    def __init__(self, df, params, ranges, scales, outputs=("eta_pct", "Voc_V", "Jsc_mAcm2", "FF_pct"),
                 n_models=8, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        self.params, self.ranges, self.scales, self.outputs = list(params), list(ranges), list(scales), list(outputs)
        status = df["status"].astype(str)
        X_all = to_unit(df[self.params].to_numpy(float), self.ranges, self.scales)
        solved = status.isin(OK).to_numpy()
        self.n_runs, self.n_solved = len(df), int(solved.sum())
        self.clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=seed)
        self.clf.fit(X_all, solved)
        rng = np.random.RandomState(seed)
        self.models = {}
        for o in self.outputs:
            m = solved & df[o].notna().to_numpy()
            X, y = X_all[m], df.loc[m, o].to_numpy(float)
            ens = []
            for i in range(n_models):
                idx = rng.randint(0, len(X), len(X))           # bootstrap: the spread of the ensemble = uncertainty
                r = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                                  l2_regularization=1.0, random_state=seed + i)
                ens.append(r.fit(X[idx], y[idx]))
            self.models[o] = ens

    def predict_unit(self, U, output="eta_pct"):
        P = np.stack([m.predict(U) for m in self.models[output]])
        return P.mean(0), P.std(0)

    def p_converge_unit(self, U):
        return self.clf.predict_proba(U)[:, 1]

    def predict(self, X, output="eta_pct"):
        return self.predict_unit(to_unit(np.asarray(X, float), self.ranges, self.scales), output)


def cv_report(df, params, ranges, scales, outputs=("eta_pct", "Voc_V", "Jsc_mAcm2", "FF_pct"), folds=5, seed=0):
    """Honest accuracy: 5-fold cross-validation (each run predicted by a model that never saw it).
    Returns {output: {"R2", "MAE", "coverage_2sigma"}} and the failure-classifier ROC-AUC."""
    from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
    rng = np.random.RandomState(seed)
    fold = rng.randint(0, folds, len(df))
    rep, pc = {}, np.zeros(len(df))
    preds = {o: (np.full(len(df), np.nan), np.full(len(df), np.nan)) for o in outputs}
    for f in range(folds):
        tr, te = fold != f, fold == f
        s = Surrogate(df[tr], params, ranges, scales, outputs, n_models=4, seed=seed + f)
        U = to_unit(df.loc[te, params].to_numpy(float), ranges, scales)
        pc[te] = s.p_converge_unit(U)
        for o in outputs:
            mu, sd = s.predict_unit(U, o)
            preds[o][0][te], preds[o][1][te] = mu, sd
    solved = df["status"].astype(str).isin(OK).to_numpy()
    for o in outputs:
        m = solved & df[o].notna().to_numpy()
        y, mu, sd = df.loc[m, o].to_numpy(float), preds[o][0][m], preds[o][1][m]
        rep[o] = {"R2": round(float(r2_score(y, mu)), 4), "MAE": round(float(mean_absolute_error(y, mu)), 5),
                  "within_2_sigma_pct": round(float(np.mean(np.abs(y - mu) <= 2 * sd + 1e-12) * 100), 1)}
    rep["converge_ROC_AUC"] = round(float(roc_auc_score(solved, pc)), 4)
    return rep


def sample_unit(n, d, seed=0):
    return np.random.RandomState(seed).rand(n, d)


def requirement_map(s, target, n=200_000, k_sigma=1.0, p_conv=0.9, seed=1, grid=20, pairs=None):
    """The feasible region for 'output >= target'.
    feasible = predicted mean - k_sigma * spread >= target AND predicted chance to converge >= p_conv.
    Returns dict with the feasible candidates (unit and real values), the share of the design space that is
    feasible, per-parameter necessary ranges (the 1 % and 99 % quantiles of feasible values; outside them
    practically nothing reaches the target) and, for each pair of parameters, a grid of the feasible share."""
    d = len(s.params)
    U = sample_unit(n, d, seed)
    mu, sd = s.predict_unit(U)
    pc = s.p_converge_unit(U)
    feas = (mu - k_sigma * sd >= target) & (pc >= p_conv)
    out = {"target": target, "n_candidates": n, "feasible_share_pct": round(float(feas.mean() * 100), 3),
           "n_feasible": int(feas.sum()), "U": U[feas], "mu": mu[feas], "sd": sd[feas]}
    X = from_unit(U, s.ranges, s.scales)
    nec = []
    for j, k in enumerate(s.params):
        v = X[feas, j]
        if len(v) == 0:
            nec.append({"parameter": k, "range_low": s.ranges[j][0], "range_high": s.ranges[j][1],
                        "feasible_from": None, "feasible_to": None, "restricts": True})
            continue
        lo, hi = np.quantile(v, [0.01, 0.99])
        width = float(np.quantile(U[feas, j], 0.99) - np.quantile(U[feas, j], 0.01))
        nec.append({"parameter": k, "range_low": s.ranges[j][0], "range_high": s.ranges[j][1],
                    "feasible_from": float(lo), "feasible_to": float(hi),
                    "share_of_range_allowed_pct": round(width * 100, 1), "restricts": width < 0.9})
    out["necessary"] = pd.DataFrame(nec)
    out["X"] = X[feas]
    if pairs is None:
        pairs = list(itertools.combinations(range(d), 2))
    maps = {}
    edges = np.linspace(0, 1, grid + 1)
    for a, b in pairs:
        ia = np.clip(np.digitize(U[:, a], edges) - 1, 0, grid - 1)
        ib = np.clip(np.digitize(U[:, b], edges) - 1, 0, grid - 1)
        tot = np.zeros((grid, grid))
        hit = np.zeros((grid, grid))
        np.add.at(tot, (ia, ib), 1)
        np.add.at(hit, (ia, ib), feas)
        maps[(s.params[a], s.params[b])] = np.where(tot > 0, hit / np.maximum(tot, 1), np.nan)
    out["pair_maps"] = maps
    return out


def minimum_change(s, baseline, target, max_k=3, n_per_subset=3000, k_sigma=1.0, p_conv=0.9, seed=2,
                   fixed=()):
    """Smallest changes from `baseline` (dict of parameter values) that reach the target.
    For every subset of 1..max_k parameters (others stay at the baseline), random values of only those
    parameters are tried; a candidate counts if mean - k_sigma*spread >= target and it converges.
    'Size' of a change = sum of the absolute moves on the unit scale (1 = the whole sampled range).
    Returns a DataFrame: one row per subset that can reach the target, best (smallest) change first."""
    d = len(s.params)
    b = to_unit(np.array([[baseline[k] for k in s.params]], float), s.ranges, s.scales)[0]
    b = np.clip(b, 0, 1)
    rng = np.random.RandomState(seed)
    free = [j for j, k in enumerate(s.params) if k not in fixed]
    rows = []
    mu0, sd0 = s.predict_unit(b[None, :])
    for k in range(1, max_k + 1):
        for sub in itertools.combinations(free, k):
            U = np.repeat(b[None, :], n_per_subset, 0)
            U[:, list(sub)] = rng.rand(n_per_subset, k)
            mu, sd = s.predict_unit(U)
            ok = (mu - k_sigma * sd >= target) & (s.p_converge_unit(U) >= p_conv)
            if not ok.any():
                continue
            size = np.abs(U[:, list(sub)] - b[list(sub)]).sum(1)
            i = int(np.where(ok)[0][np.argmin(size[ok])])
            x = from_unit(U[i:i + 1], s.ranges, s.scales)[0]
            rows.append({"n_changed": k, "changed": " + ".join(s.params[j] for j in sub), "change_size": float(size[i]),
                         "predicted": float(mu[i]), "spread": float(sd[i]),
                         **{s.params[j]: float(x[j]) for j in range(d)}})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["n_changed", "change_size"]).reset_index(drop=True)
    out.attrs["baseline_predicted"] = (float(mu0[0]), float(sd0[0]))
    return out


def robustness(s, X, target, rel=None, n=400, seed=3):
    """Chance that each design (rows of X, real values) stays >= target when every parameter varies at random
    as in production. rel: {parameter: relative spread (uniform +-)}; default +-10 % on everything
    (+-0.02 eV on energies)."""
    rng = np.random.RandomState(seed)
    X = np.atleast_2d(np.asarray(X, float))
    res = []
    for x in X:
        P = np.repeat(x[None, :], n, 0)
        for j, k in enumerate(s.params):
            r = (rel or {}).get(k)
            if k.endswith(("chi", "Eg", "Eg@B", "workfunction")):
                P[:, j] = x[j] + rng.uniform(-1, 1, n) * (r if r is not None else 0.02)
            else:
                P[:, j] = x[j] * (1 + rng.uniform(-1, 1, n) * (r if r is not None else 0.10))
        U = np.clip(to_unit(P, s.ranges, s.scales), 0, 1)
        mu, _ = s.predict_unit(U)
        res.append(float(np.mean(mu >= target)))
    return np.array(res)


def diverse_pick(U, score, n, min_dist=0.15):
    """Up to n rows of U, best score first, each at least min_dist (unit scale, Euclidean / sqrt(d)) from the
    ones already picked."""
    order = np.argsort(-score)
    picked = []
    d = U.shape[1]
    for i in order:
        if all(np.linalg.norm(U[i] - U[j]) / math.sqrt(d) >= min_dist for j in picked):
            picked.append(i)
        if len(picked) >= n:
            break
    return picked


def boundary_points(s, target, n=40, pool=50_000, seed=4):
    """Active learning: the candidates whose 'reaches target or not' is most uncertain (mean near the target,
    large spread), spread out over the space. Running these in SCAPS sharpens the requirement map most."""
    U = sample_unit(pool, len(s.params), seed)
    mu, sd = s.predict_unit(U)
    pc = s.p_converge_unit(U)
    score = -np.abs(mu - target) / np.maximum(sd, 1e-6) + np.log(np.maximum(pc, 1e-6))
    idx = diverse_pick(U, score, n, min_dist=0.12)
    return from_unit(U[idx], s.ranges, s.scales), mu[idx], sd[idx]
