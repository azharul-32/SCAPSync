"""Active-learning inverse design: reach a target with as few SCAPS runs as possible (any SCAPS model).

The loop
  1. start: a small Latin-hypercube set (default 3 x number of parameters, at least 16 runs)
  2. learn: a surrogate of the results with an uncertainty (Gaussian process while data are few,
     an ensemble of boosted trees once there are many) and a model of where SCAPS fails to converge
  3. choose the next batch: half 'exploit' (most likely to hit the target, spread out), half 'explore'
     (where the surrogate is least sure whether the target is hit)
  4. run the batch in SCAPS (verified, failures recorded) and go to 2
  Stop when enough DIFFERENT designs have been verified in SCAPS to hit the target, or the run budget is used.

Only SCAPS results count as hits: the surrogate only decides what to run next.
`evaluate(X)` does the runs; FermiPilot passes a function that runs a study, tests pass a stand-in.
"""
import math

import numpy as np
import pandas as pd

from .design import OK, diverse_pick, from_unit, to_unit


def _norm_cdf(z):
    return 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))


class SmallSurrogate:
    """Mean and uncertainty of one output on the unit cube, and the chance SCAPS converges.
    Few data (< 250 solved runs): Gaussian process with one length scale per parameter.
    More data: bootstrap ensemble of gradient-boosted trees (fast, handles thousands of runs)."""

    def __init__(self, U, y, solved, seed=0):
        self.U_all, self.solved = U, solved.astype(bool)
        Us, ys = U[self.solved], y[self.solved]
        self.kind = "gp" if len(ys) < 250 else "trees"
        if self.kind == "gp":
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
            d = U.shape[1]
            k = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.full(d, 0.5), length_scale_bounds=(0.03, 30),
                                                           nu=2.5) + WhiteKernel(1e-3, (1e-6, 1e-1))
            self.mu_y, self.sd_y = ys.mean(), ys.std() or 1.0
            self.gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=2, random_state=seed)
            self.gp.fit(Us, (ys - self.mu_y) / self.sd_y)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            rng = np.random.RandomState(seed)
            self.ens = []
            for i in range(8):
                idx = rng.randint(0, len(ys), len(ys))
                self.ens.append(HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                                              random_state=seed + i).fit(Us[idx], ys[idx]))

    def predict(self, U):
        if self.kind == "gp":
            m, s = self.gp.predict(U, return_std=True)
            return m * self.sd_y + self.mu_y, s * self.sd_y
        P = np.stack([e.predict(U) for e in self.ens])
        return P.mean(0), P.std(0)

    def p_converge(self, U, k=7):
        """Share of converged runs among the k nearest runs done so far (distance-weighted). Simple and
        honest with few data; a region with failures nearby gets a low value."""
        if self.solved.all():
            return np.ones(len(U))
        out = np.empty(len(U))
        for i0 in range(0, len(U), 4000):
            B = U[i0:i0 + 4000]
            d = np.sqrt(((B[:, None, :] - self.U_all[None, :, :]) ** 2).sum(-1))
            idx = np.argsort(d, 1)[:, :k]
            w = 1 / (np.take_along_axis(d, idx, 1) + 0.02)
            out[i0:i0 + 4000] = (w * self.solved[idx]).sum(1) / w.sum(1)
        return out


def p_hit(mu, sd, target, tol=None):
    """Chance that the true value hits the target: >= target (tol None) or within target +- tol."""
    sd = np.maximum(sd, 1e-9)
    if tol is None:
        return _norm_cdf((mu - target) / sd)
    return _norm_cdf((target + tol - mu) / sd) - _norm_cdf((target - tol - mu) / sd)


def is_hit(y, target, tol=None):
    y = np.asarray(y, float)
    return (y >= target) if tol is None else (np.abs(y - target) <= tol)


def propose(s, n, target, tol=None, pool=30_000, seed=0, done_U=None, min_dist=0.08):
    """Next batch: half exploitation (highest chance to hit x chance to converge), half exploration (most
    uncertain about hit / no hit, i.e. chance near 50 %, weighted by convergence), all spread out and away from
    runs already done."""
    d = s.U_all.shape[1]
    U = np.random.RandomState(seed).rand(pool, d)
    mu, sd = s.predict(U)
    pc = s.p_converge(U)
    ph = p_hit(mu, sd, target, tol)
    if done_U is not None and len(done_U):
        dist = np.sqrt(((U[:, None, :] - done_U[None, :, :]) ** 2).sum(-1)).min(1) / math.sqrt(d)
        far = dist >= min_dist / 2
    else:
        far = np.ones(pool, bool)
    n_exploit = (n + 1) // 2
    score_a = np.where(far, ph * pc, -1)
    a = diverse_pick(U, score_a, n_exploit, min_dist=min_dist)
    score_b = np.where(far, (1 - np.abs(2 * ph - 1)) * pc * sd / (sd.max() or 1), -1)
    score_b[a] = -1
    b = diverse_pick(U, score_b, n - len(a), min_dist=min_dist)
    idx = list(a) + list(b)
    return U[idx], mu[idx], sd[idx], ph[idx], ["exploit"] * len(a) + ["explore"] * len(b)


def run(params, ranges, scales, evaluate, target, output="eta_pct", tol=None, n_start=None, batch=8,
        budget=200, want_hits=5, distinct=0.1, seed=1, log=print):
    """The active-learning loop. evaluate(X real values, round) -> DataFrame with 'status' and the outputs, one
    row per row of X (failed runs included). Returns (history DataFrame, all runs DataFrame, hits DataFrame)."""
    d = len(params)
    n_start = n_start or max(16, 3 * d)
    from .study import latin_hypercube
    plan = latin_hypercube({k: [r[0], r[1], sc] for k, r, sc in zip(params, ranges, scales)}, n_start, seed)
    X0 = np.array([[row[k] for k in params] for row in plan], float)
    log(f"Round 0: {len(X0)} starting runs spread over the ranges")
    runs = evaluate(X0, 0).assign(round=0, why="start", predicted=np.nan, predicted_spread=np.nan)
    hist = []
    rnd = 0
    while True:
        solved = runs["status"].astype(str).isin(OK).to_numpy() & runs[output].notna().to_numpy()
        U = to_unit(runs[params].to_numpy(float), ranges, scales)
        hits = _distinct_hits(runs, params, ranges, scales, output, target, tol, distinct)
        best = runs.loc[solved, output].max() if solved.any() else float("nan")
        hist.append({"round": rnd, "runs": len(runs), "solved": int(solved.sum()), "best": best,
                     "verified_hits": len(hits)})
        log(f"  after round {rnd}: {len(runs)} runs, {int(solved.sum())} solved, best {output} {best:.4g}, "
            f"{len(hits)} different verified design(s) on target")
        if len(hits) >= want_hits:
            log(f"Done: {len(hits)} different designs verified in SCAPS to hit the target.")
            break
        if len(runs) >= budget:
            log(f"Run budget ({budget}) used: {len(hits)} verified design(s) on target.")
            break
        if solved.sum() < 3:
            log("Too few solved runs to learn from; widen the ranges or check the model.")
            break
        rnd += 1
        y = np.where(solved, runs[output].to_numpy(float), np.nan)
        s = SmallSurrogate(U, np.nan_to_num(y), solved, seed=seed + rnd)
        n = int(min(batch, budget - len(runs)))
        Un, mu, sd, ph, why = propose(s, n, target, tol, seed=seed + rnd, done_U=U)
        Xn = from_unit(Un, ranges, scales)
        log(f"Round {rnd}: {n} runs chosen by the {s.kind} surrogate "
            f"(chance to hit, best of batch {ph.max() * 100:.0f} %)")
        new = evaluate(Xn, rnd).assign(round=rnd, why=why, predicted=mu, predicted_spread=sd)
        runs = pd.concat([runs, new], ignore_index=True)
    hits = _distinct_hits(runs, params, ranges, scales, output, target, tol, distinct)
    return pd.DataFrame(hist), runs, hits


def _distinct_hits(runs, params, ranges, scales, output, target, tol, distinct):
    """Verified hits that differ from each other by at least `distinct` (unit scale, normalised distance)."""
    ok = runs["status"].astype(str).isin(OK) & runs[output].notna()
    h = runs[ok & is_hit(runs[output].fillna(-1e9), target, tol)]
    if h.empty:
        return h
    U = to_unit(h[params].to_numpy(float), ranges, scales)
    score = -np.abs(h[output].to_numpy(float) - target) if tol is not None else h[output].to_numpy(float)
    idx = diverse_pick(U, score, len(h), min_dist=distinct)
    return h.iloc[idx]


# ---------------- FermiPilot study integration ----------------
def run_design(spec, log=print):
    """Active-learning inverse design with real SCAPS runs.
    spec (JSON): name, scaps_dir, base_def, light, measurements, timeout_s, parallel,
      params: {key: [low, high, "lin"|"log"]},
      target: {"output": "eta_pct", "min": 24}  or  {"output": "eta_pct", "value": 24, "tol": 0.1},
      budget (max SCAPS runs, default 120), batch (runs per round, default 8), want (different verified designs
      on target, default 5), n_start (default 3 x parameters, at least 16), seed.
    Each round is a FermiPilot study  <runs>/<name>/round_NN  (verified, failures kept); the combined table is
    <runs>/<name>/master.csv. Running the same spec again resumes: finished rounds are not run again."""
    import json
    import os

    from . import paths, study
    params = list(spec["params"])
    ranges = [tuple(map(float, spec["params"][k][:2])) for k in params]
    scales = [spec["params"][k][2] if len(spec["params"][k]) > 2 else "lin" for k in params]
    tgt = spec["target"]
    output = tgt.get("output", "eta_pct")
    target = float(tgt["min"] if "min" in tgt else tgt["value"])
    tol = None if "min" in tgt else float(tgt.get("tol", 0.1))
    root = os.path.join(paths.runs_root(spec["scaps_dir"]), spec["name"])
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "design.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    base = {k: spec[k] for k in ("scaps_dir", "base_def", "light", "measurements", "timeout_s", "parallel", "fixed")
            if k in spec}
    base.setdefault("retry", True)
    base["keep_failed"] = True
    combined = []

    def evaluate(X, rnd):
        st = dict(base, name=f"{spec['name']}/round_{rnd:02d}",
                  design={"type": "list", "runs": [{k: float(v) for k, v in zip(params, x)} for x in X]})
        study.run_study(st, log=lambda m: log(m) if m.startswith(("  run_", "Using", "Speed")) else None)
        df = pd.read_csv(os.path.join(root, f"round_{rnd:02d}", "master.csv"))
        df = df.sort_values("run_id").reset_index(drop=True)
        if len(df) != len(X):
            raise RuntimeError(f"round {rnd}: {len(df)} results for {len(X)} runs (study stopped?). "
                               "Run the same design again to resume.")
        df["run_id"] = [f"r{rnd:02d}_{r}" for r in df["run_id"]]
        combined.append(df.assign(round=rnd))
        pd.concat(combined, ignore_index=True).to_csv(os.path.join(root, "master.csv"), index=False)
        return df

    hist, runs, hits = run(params, ranges, scales, evaluate, target, output=output, tol=tol,
                           n_start=spec.get("n_start"), batch=int(spec.get("batch", 8)),
                           budget=int(spec.get("budget", 120)), want_hits=int(spec.get("want", 5)),
                           seed=int(spec.get("seed", 1)), log=log)
    runs.to_csv(os.path.join(root, "master.csv"), index=False)
    hist.to_csv(os.path.join(root, "history.csv"), index=False)
    hits.to_csv(os.path.join(root, "designs_on_target.csv"), index=False)
    with open(os.path.join(root, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"target": tgt, "runs": len(runs), "verified_designs": len(hits),
                   "best": float(runs[output].max()) if runs[output].notna().any() else None}, f, indent=1)
    return hist, runs, hits
