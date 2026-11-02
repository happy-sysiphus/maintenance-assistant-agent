"""
A-5 (2026-11-01): are the failures tied to operating transitions? A-4 read the failure moments by eye - E01 / E03 at
the morning power-on, E04 in the power-off state, E02 around cap dressing / change. This tests those readings against
the same gun's other days (case-crossover) and its other transitions, with the dependence A-4 found (duplicate
physical guns, simultaneous failures) in the resampling unit.

Inputs, no raw CSV pass: A-4 event cache (results/A-4/cache: runs / c1 / welds / static), B-17 per-minute cache
(results/B-17/cache: min_<file>.parquet row counts = logging coverage, meta.csv fault times), A-4 failure_context.csv
(simultaneous groups). Kept files only: 72 failures (64 train + 8 test).

PRE-REGISTERED before computing any result below (2026-11-01):

Events per file (naive UTC; runs are A-4's - consecutive rows, a logging gap does not break a run)
    power_on      end of a c10-off run lasting >= 10 min (the next row is on)
    power_off     start of a c10-off run lasting >= 10 min
    short_cycle   start of a c10-off run lasting 5 s - 10 min
    cap_change    c1 step <= -0.5 mm (A-4 used <= -1.0; -0.5..-1.0 steps follow program c16 = 32 in 84% of cases,
                  so they are cap changes made at a low offset), steps within 120 s merged
    dressing      c1 step +0.05..+0.6 mm (A-4 section 3), steps within 120 s merged
    cap_maint     cap_change or dressing
    stop_code     an E003 / E011 episode (a state: counts if it overlaps the window)
    other_code    an episode of any other code except the file's own terminal code (state)
    production    a weld (c11 advanced)

Part 1 - case-crossover (one stratum per failure)
    case window     [T - 30 min, T)   T = fault onset (B-17 meta: first own terminal code in the last 15 min of the file)
    controls        the same clock window k = 1..6 days earlier (same gun, same file; files start at T - 7 days)
    also reported   [T - 10 min, T) and the early-warning zone [T - 60, T - 10 min] (lead >= 10 min, B-18)
    usable window   logging coverage >= 50% of its seconds (B-17 per-minute row counts); a stratum needs a usable case
                    window and >= 2 usable controls
    effect          Mantel-Haenszel matched odds ratio (1 case : M controls). Case share = recall of an "alert for the
                    window length after every such event" rule, control share = its same-clock chance (B-18's terms).
                    Cost = events per gun-day; precision = events followed by the failure inside the window / all events
                    before T (point events only)
    uncertainty     cluster = connected component of (same physical gun (c7, c8, c9)) or (simultaneous failure group,
                    A-4 section 7). 95% CI from 2000 cluster bootstraps; p from 2000 cluster permutations (each cluster
                    draws one day offset 0..6 as the pseudo-case, shared by its strata), two-sided on |log OR|
    primary (Holm over 3)
        H1  E01 + E03  x power_on    in [T - 30 min, T)
        H2  E04        x power_off   in [T - 30 min, T)
        H3  E02        x cap_maint   in [T - 30 min, T)
    supported if    Holm-adjusted p < 0.05 AND the CI's lower bound > 1. Everything else is exploratory (raw p, no claim)
    method placebo  the same machinery with the anchor moved to T - 1 day (controls T - 2..6 days): the primary ORs must
                    have CIs containing 1, else the design is biased here and the Part 1 claims are void

Part 2 - the failing transition vs the same gun's other transitions
    lags            for every failure: minutes since the last power_on, power_off, weld, cap change, dressing, stop code
    power-on-linked failures (a power_on in [T - 30 min, T)): that power-on vs the gun's other power-ons before T - 60 min
        off_min     length of the preceding off run (a run touching the file start is censored -> dropped)
        idle_h      hours since the last weld (no earlier weld in the file -> dropped)
        exploratory stop code in the 10 min before; |clock - the gun's mean power-on clock|; c1 at the power-on;
                    hours since the last cap change
    power-off-linked failures (a power_off in [T - 30 min, T)): that power-off vs the gun's other power-offs before T - 60 min
        clock_dev   |clock - nearest power-off clock of the same gun on another day| (an unscheduled stop is far from all)
        exploratory welds in the 10 min before (abrupt stop); hours since the previous power_on
    statistic       percentile of the failing event among the gun's other events ((#below + 0.5 #ties) / #others);
                    H0: mean 0.5. p from 10000 permutations (a random event of each stratum plays the failing one),
                    two-sided; 95% CI from 2000 cluster bootstraps
    primary (Holm over 3): P1 off_min, P2 idle_h (power-on-linked), P3 clock_dev (power-off-linked)
    supported if    Holm-adjusted p < 0.05 AND the CI of the mean percentile excludes 0.5

    python results/A-5/context.py crossover      -> crossover.json (+ .log via shell)
    python results/A-5/context.py transitions    -> transitions.json, transitions_events.csv
    python results/A-5/context.py figures        -> fig_crossover.png, fig_lags.png

Never writes outside results/A-5/.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
A4 = os.path.join(ROOT, "results", "A-4", "cache")
B17 = os.path.join(ROOT, "results", "B-17", "cache")
TERM = {"E01": "E012", "E02": "E016", "E03": "E028", "E04": "E029"}
STOP = ("E003", "E011")
KST = pd.Timedelta(hours=9)
DAY = np.timedelta64(86400, "s")
SEC = np.timedelta64(1, "s")
WINDOWS = {"30m": (30, 0), "10m": (10, 0), "zone": (60, 10)}     # (start, end) minutes before the anchor
K = 7                                                               # offset 0 = case, 1..6 = same-clock controls
COVER_MIN = 0.5
LONG_OFF_S, SHORT_OFF_S = 600, 5
N_BOOT, N_PERM, N_PERM2 = 2000, 2000, 10000
POINT = ("power_on", "power_off", "short_cycle", "cap_change", "dressing", "cap_maint")
EXPOSURES = POINT + ("stop_code", "other_code", "production")
GROUPS = {"all": ("E01", "E02", "E03", "E04"), "E01+E03": ("E01", "E03"), "E01": ("E01",), "E02": ("E02",),
          "E03": ("E03",), "E04": ("E04",)}
PRIMARY = (("H1", "E01+E03", "power_on"), ("H2", "E04", "power_off"), ("H3", "E02", "cap_maint"))
PRIMARY2 = (("P1", "on", "off_min"), ("P2", "on", "idle_h"), ("P3", "off", "clock_dev"))
COLOR = {"E01": "#2a78d6", "E02": "#eb6834", "E03": "#1baf7a", "E04": "#eda100"}


def log(msg):
    print(msg, flush=True)


def save(name, out):
    dst = os.path.join(HERE, f"{name}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False,
                  default=lambda o: None if isinstance(o, float) and o != o else str(o))
    log(f"saved {dst}")


def load_meta():
    meta = pd.read_csv(os.path.join(B17, "meta.csv"), parse_dates=["fault_time", "start", "end"])
    meta = meta[~meta["dropped"]].reset_index(drop=True)
    st = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(A4, "static_*.parquet"))])
    phys = dict(zip(st["file"], st[["c7", "c8", "c9"]].round(4).astype(str).agg("|".join, axis=1)))
    ctx = pd.read_csv(os.path.join(ROOT, "results", "A-4", "failure_context.csv"))
    sim = ctx[ctx["simultaneous"]].groupby("group")["file"].apply(list).tolist()
    # cluster = connected component of (same physical gun) or (same simultaneous group)
    parent = {f: f for f in meta["file"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    links = [[f for f in meta["file"] if phys[f] == p] for p in set(phys[f] for f in meta["file"])] + sim
    for grp in links:
        grp = [f for f in grp if f in parent]
        for f in grp[1:]:
            parent[find(f)] = find(grp[0])
    meta["phys"] = meta["file"].map(phys)
    meta["cluster"] = pd.factorize(meta["file"].map(find))[0]
    meta["c10_at"] = meta["file"].map(dict(zip(ctx["file"], ctx["c10_at"])))
    return meta


def merged(t, gap_s=120):
    t = np.sort(np.asarray(t, "datetime64[ns]"))
    if not len(t):
        return t
    keep = np.r_[True, np.diff(t) > np.timedelta64(gap_s, "s")]
    return t[keep]


def file_events(f, cls, start):
    r = pd.read_parquet(os.path.join(A4, f"runs_{f}.parquet"))
    off = r[r["state"] == "c10_off"]
    lo = off[off["dur_s"] >= LONG_OFF_S]
    sh = off[(off["dur_s"] >= SHORT_OFF_S) & (off["dur_s"] < LONG_OFF_S)]
    c1 = pd.read_parquet(os.path.join(A4, f"c1_{f}.parquet"), columns=["time", "old", "new"])
    d = (c1["new"] - c1["old"]).round(2)
    err = r[r["state"] == "error"]
    stop = err[err["code"].isin(STOP)]
    other = err[~err["code"].isin(STOP + (TERM[cls],))]
    ev = {"power_on": np.sort(lo["end"].to_numpy()), "power_off": np.sort(lo["start"].to_numpy()),
          "short_cycle": np.sort(sh["start"].to_numpy()), "cap_change": merged(c1.loc[d <= -0.5, "time"]),
          "dressing": merged(c1.loc[d.between(0.05, 0.6), "time"]),
          "production": np.sort(pd.read_parquet(os.path.join(A4, f"welds_{f}.parquet"), columns=["time"])["time"].to_numpy())}
    ev["cap_maint"] = np.sort(np.r_[ev["cap_change"], ev["dressing"]])
    states = {"stop_code": (stop["start"].to_numpy(), stop["end"].to_numpy()),
              "other_code": (other["start"].to_numpy(), other["end"].to_numpy())}
    lo = lo.sort_values("start")
    runs = {"on_dur_s": lo.set_index("end")["dur_s"], "on_censored": lo.set_index("end")["start"] <= start + pd.Timedelta(seconds=1),
            "all_off": off.loc[off["dur_s"] >= SHORT_OFF_S, ["start", "end", "dur_s"]].sort_values("start")}
    m = pd.read_parquet(os.path.join(B17, f"min_{f}.parquet"), columns=["minute", "n", "c1"])
    return ev, states, runs, m


def coverage(m, a, b):
    t = m["minute"].to_numpy()
    cum = np.r_[0, np.cumsum(m["n"].to_numpy())]
    i, j = np.searchsorted(t, a), np.searchsorted(t, b)
    return (cum[j] - cum[i]) / ((b - a) / SEC)


def any_in(t, a, b):
    return np.searchsorted(t, b) > np.searchsorted(t, a)


def count_in(t, a, b):
    return int(np.searchsorted(t, b) - np.searchsorted(t, a))


def overlaps(se, a, b):
    s, e = se
    return bool(((s < b) & (e >= a)).any())


def build_strata(meta, cache):
    """Per window type: exposure E[n, K, exposure], usable U[n, K], plus counts for the cost / precision columns."""
    out = {}
    for wname, (lo, hi) in WINDOWS.items():
        E = np.zeros((len(meta), K, len(EXPOSURES)), bool)
        U = np.zeros((len(meta), K), bool)
        hits = np.zeros((len(meta), len(EXPOSURES)), int)
        for i, r in meta.iterrows():
            ev, states, _, m = cache[r.file]
            T = np.datetime64(r.fault_time, "ns")
            for k in range(K):
                anchor = T - k * DAY
                a, b = anchor - np.timedelta64(lo, "m"), anchor - np.timedelta64(hi, "m")
                U[i, k] = coverage(m, a, b) >= COVER_MIN
                for j, x in enumerate(EXPOSURES):
                    E[i, k, j] = overlaps(states[x], a, b) if x in states else any_in(ev[x], a, b)
                if k == 0:
                    hits[i] = [count_in(ev[x], a, b) if x in ev else 0 for x in EXPOSURES]
        out[wname] = (E, U, hits)
    return out


def mh(a, c, M):
    N = M + 1
    return (a * (M - c) / N), ((1 - a) * c / N)


def crossover_stats(E, U, cl, rng, shift=0):
    """E, U: [n, K'] for one exposure; column `shift` is the case, later columns the controls."""
    E, U = E[:, shift:], U[:, shift:]
    inc = U[:, 0] & (U[:, 1:].sum(1) >= 2)
    E, U, cl = E[inc], U[inc], pd.factorize(cl[inc])[0]
    n, Kc = E.shape
    if n == 0:
        return {"strata": 0}
    a = E[:, 0].astype(float)
    c = (E[:, 1:] & U[:, 1:]).sum(1).astype(float)
    M = U[:, 1:].sum(1).astype(float)
    num, den = mh(a, c, M)
    orr = num.sum() / den.sum() if den.sum() > 0 else (np.inf if num.sum() > 0 else np.nan)
    ncl = cl.max() + 1
    W = rng.multinomial(ncl, np.full(ncl, 1 / ncl), size=N_BOOT)[:, cl]
    bn, bd = W @ num, W @ den
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = np.where(bd > 0, bn / bd, np.where(bn > 0, np.inf, np.nan))
    boot = boot[~np.isnan(boot)]
    # cluster permutation: one shared offset per cluster; an unusable pick falls back to a random usable window
    Jc = rng.integers(0, Kc, size=(N_PERM, ncl))[:, cl]
    rnd = np.where(U[None], rng.random((N_PERM, n, Kc)), -1).argmax(2)
    J = np.where(np.take_along_axis(np.broadcast_to(U, (N_PERM, n, Kc)), Jc[..., None], 2)[..., 0], Jc, rnd)
    tot = (E & U).sum(1).astype(float)
    pa = np.take_along_axis(np.broadcast_to(E, (N_PERM, n, Kc)), J[..., None], 2)[..., 0].astype(float)
    pc = tot[None] - pa
    pn, pd_ = mh(pa, pc, (U.sum(1) - 1).astype(float)[None])
    pn, pd_ = pn.sum(1), pd_.sum(1)

    def lor(x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.abs(np.log(np.where(y > 0, x / np.where(y > 0, y, 1), np.where(x > 0, np.inf, np.nan))))

    obs = lor(np.array([num.sum()]), np.array([den.sum()]))[0]
    perm = lor(pn, pd_)
    perm = np.where(np.isnan(perm), 0.0, perm)                     # 0/0: no information, never as extreme as obs
    p = float((1 + np.sum(perm >= obs)) / (N_PERM + 1)) if not np.isnan(obs) else np.nan
    return {"strata": int(n), "clusters": int(ncl), "case_share": round(float(a.mean()), 3),
            "control_share": round(float((c / M).mean()), 3), "or": round(float(orr), 2),
            "or_ci": [round(float(np.percentile(boot, 2.5, method="lower")), 2),
                      round(float(np.percentile(boot, 97.5, method="higher")), 2)]
            if len(boot) else None, "p": round(p, 4) if p == p else None,
            "discordant": {"case_only": round(float(num.sum()), 2), "control_only": round(float(den.sum()), 2)}}


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def load_all():
    meta = load_meta()
    cache = {r.file: file_events(r.file, r._2, r.start) for r in meta.itertuples()}
    return meta, cache


def crossover():
    meta, cache = load_all()
    log(f"{len(meta)} failures, {meta['phys'].nunique()} physical guns, {meta['cluster'].nunique()} clusters")
    strata = build_strata(meta, cache)
    rng = np.random.default_rng(0)
    gun_days = float(((meta["fault_time"] - meta["start"]).dt.total_seconds() / 86400).sum())
    res = {"failures": len(meta), "physical_guns": int(meta["phys"].nunique()), "clusters": int(meta["cluster"].nunique()),
           "windows": {}, "placebo": {}, "cost": {}}
    # cost: events per gun-day before T (point events), episode starts for the states
    for x in EXPOSURES:
        if x == "production":
            continue
        n = sum(count_in(cache[r.file][0][x], np.datetime64(r.start, "ns"), np.datetime64(r.fault_time, "ns"))
                if x in POINT else int((cache[r.file][1][x][0] < np.datetime64(r.fault_time, "ns")).sum())
                for r in meta.itertuples())
        res["cost"][x] = {"events": n, "per_gun_day": round(n / gun_days, 2)}
    for wname, (E, U, hits) in strata.items():
        res["windows"][wname] = {}
        for g, classes in GROUPS.items():
            sel = meta["class"].isin(classes).to_numpy()
            row = {}
            for j, x in enumerate(EXPOSURES):
                s = crossover_stats(E[sel, :, j], U[sel], meta.loc[sel, "cluster"].to_numpy(), rng)
                if x in POINT and g == "all":
                    tot = res["cost"][x]["events"]
                    s["precision"] = round(int(hits[:, j].sum()) / tot, 4) if tot else None
                row[x] = s
            res["windows"][wname][g] = row
        log(f"\n[{wname}] case share / same-clock share  OR [95% CI]  p")
        for g in GROUPS:
            log(f"  {g}")
            for x in EXPOSURES:
                s = res["windows"][wname][g][x]
                if s.get("strata"):
                    log(f"    {x:12s} n={s['strata']:2d}  {s['case_share']:.2f} / {s['control_share']:.2f}  "
                        f"OR {s['or']} {s['or_ci']}  p {s['p']}" + (f"  prec {s['precision']}" if 'precision' in s else ""))
    # method placebo: anchor at T - 1 day
    E, U, _ = strata["30m"]
    for h, g, x in PRIMARY:
        sel = meta["class"].isin(GROUPS[g]).to_numpy()
        res["placebo"][h] = crossover_stats(E[sel, :, EXPOSURES.index(x)], U[sel], meta.loc[sel, "cluster"].to_numpy(),
                                            rng, shift=1)
    prim = [res["windows"]["30m"][g][x] for _, g, x in PRIMARY]
    adj = holm(np.array([s["p"] for s in prim]))
    res["primary"] = {h: {"group": g, "exposure": x, **s, "p_holm": round(float(q), 4),
                          "supported": bool(q < 0.05 and s["or_ci"] and s["or_ci"][0] > 1),
                          "placebo_ci_contains_1": bool(res["placebo"][h]["or_ci"] and
                                                        res["placebo"][h]["or_ci"][0] <= 1 <= res["placebo"][h]["or_ci"][1])}
                      for (h, g, x), s, q in zip(PRIMARY, prim, adj)}
    log("\nprimary (30 min window, Holm over 3):")
    for h, s in res["primary"].items():
        pl = res["placebo"][h]
        log(f"  {h} {s['group']} x {s['exposure']}: {s['case_share']:.2f} vs {s['control_share']:.2f}, OR {s['or']} "
            f"{s['or_ci']}, p {s['p']} -> Holm {s['p_holm']}, supported {s['supported']} | placebo OR {pl['or']} "
            f"{pl['or_ci']} (contains 1: {s['placebo_ci_contains_1']})")
    log(f"\ncost (events / gun-day over {gun_days:.0f} gun-days): " + json.dumps(res["cost"]))
    # exploratory, added after seeing the results: the early-warning zone is the only window free of the failure's
    # own power drop (an off run that starts seconds before T and spans it); method placebo and per-split view there
    E, U, _ = strata["zone"]
    res["exploratory_zone"] = {}
    for x in ("power_on", "short_cycle"):
        j = EXPOSURES.index(x)
        row = {"placebo_all": crossover_stats(E[:, :, j], U, meta["cluster"].to_numpy(), rng, shift=1)}
        for split in ("train", "test"):
            sel = (meta["split"] == split).to_numpy()
            row[split] = crossover_stats(E[sel, :, j], U[sel], meta.loc[sel, "cluster"].to_numpy(), rng)
        res["exploratory_zone"][x] = row
        log(f"exploratory zone {x}: placebo {row['placebo_all']['case_share']} / {row['placebo_all']['control_share']} "
            f"OR {row['placebo_all']['or']} {row['placebo_all']['or_ci']} | train {row['train']['case_share']} / "
            f"{row['train']['control_share']} | test {row['test']['case_share']} / {row['test']['control_share']}")
    save("crossover", res)


def circ_dist(a, b):
    d = np.abs(a - b) % 1440
    return np.minimum(d, 1440 - d)


def clock_min(t):
    k = pd.DatetimeIndex(t) + KST
    return (k.hour * 60 + k.minute + k.second / 60).to_numpy(float)


def last_before(t, x):
    i = np.searchsorted(t, x) - 1
    return t[i] if i >= 0 else None


def on_features(f, ev, states, runs, m, start):
    """One row per power_on (end of a >= 10 min off run) of the file."""
    rows = []
    t_on = ev["power_on"]
    clk = clock_min(t_on)
    mean_clk = (np.degrees(np.angle(np.exp(1j * np.radians(clk / 4)).mean())) * 4) % 1440 if len(clk) else np.nan
    c1 = m.dropna(subset=["c1"])
    for t, ck in zip(t_on, clk):
        lw = last_before(ev["production"], t)
        lc = last_before(ev["cap_change"], t)
        i = np.searchsorted(c1["minute"].to_numpy(), t) - 1
        rows.append({"file": f, "time": t, "kind": "on",
                     "off_min": np.nan if runs["on_censored"].loc[t] else runs["on_dur_s"].loc[t] / 60,
                     "idle_h": np.nan if lw is None else (t - lw) / SEC / 3600,
                     "stop10": float(overlaps(states["stop_code"], t - np.timedelta64(10, "m"), t + SEC)),
                     "clock_dev": float(circ_dist(ck, mean_clk)),
                     "c1_at": float(c1["c1"].iloc[i]) if i >= 0 else np.nan,
                     "since_cap_h": np.nan if lc is None else (t - lc) / SEC / 3600})
    return rows


def off_features(f, ev):
    rows = []
    t_off = ev["power_off"]
    clk = clock_min(t_off)
    for idx, (t, ck) in enumerate(zip(t_off, clk)):
        other = np.abs((t_off - t) / SEC) > 12 * 3600
        lo = last_before(ev["power_on"], t)
        rows.append({"file": f, "time": t, "kind": "off",
                     "clock_dev": float(circ_dist(ck, clk[other]).min()) if other.any() else np.nan,
                     "welds10": float(count_in(ev["production"], t - np.timedelta64(10, "m"), t)),
                     "on_h": np.nan if lo is None else (t - lo) / SEC / 3600})
    return rows


def percentile_test(strata, cl, rng):
    """strata: list of (failing value, other values). Mean percentile of the failing value among the others."""
    pairs = [(x, o[~np.isnan(o)], c) for (x, o), c in zip(strata, cl) if not np.isnan(x) and np.sum(~np.isnan(o)) > 0]
    if not pairs:
        return {"strata": 0}

    def pct(x, o):
        return (np.sum(o < x) + 0.5 * np.sum(o == x)) / len(o)

    obs = np.array([pct(x, o) for x, o, _ in pairs])
    perm = np.zeros(N_PERM2)
    for i, (x, o, _) in enumerate(pairs):
        v = np.r_[x, o]
        pv = np.array([pct(v[j], np.delete(v, j)) for j in range(len(v))])
        perm += pv[rng.integers(0, len(v), N_PERM2)]
    perm /= len(pairs)
    p = float((1 + np.sum(np.abs(perm - 0.5) >= abs(obs.mean() - 0.5) - 1e-12)) / (N_PERM2 + 1))
    cl = pd.factorize(np.array([c for _, _, c in pairs]))[0]
    ncl = cl.max() + 1
    W = rng.multinomial(ncl, np.full(ncl, 1 / ncl), size=N_BOOT)[:, cl]
    boot = (W @ obs) / W.sum(1)
    return {"strata": len(pairs), "clusters": int(ncl), "mean_percentile": round(float(obs.mean()), 3),
            "ci": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
            "p": round(p, 4), "others_median": int(np.median([len(o) for _, o, _ in pairs])),
            "failing_median": round(float(np.median([x for x, _, _ in pairs])), 2),
            "others_median_value": round(float(np.median(np.concatenate([o for _, o, _ in pairs]))), 2)}


def transitions():
    meta, cache = load_all()
    rng = np.random.default_rng(1)
    lags, rows, linked = [], [], {"on": [], "off": []}
    for r in meta.itertuples():
        ev, states, runs, m = cache[r.file]
        T = np.datetime64(r.fault_time, "ns")
        lag = {"file": r.file, "class": r._2, "time_kst": r.fault_time + KST, "c10_at": r.c10_at, "cluster": r.cluster}
        for x in ("power_on", "power_off", "production", "cap_change", "dressing"):
            lb = last_before(ev[x], T)
            lag[f"min_since_{x}"] = np.nan if lb is None else round((T - lb) / SEC / 60, 2)
        s, e = states["stop_code"]
        e = e[s < T]
        lag["min_since_stop_code"] = round(max(0.0, (T - e.max()) / SEC / 60), 2) if len(e) else np.nan
        # any c10-off run (>= 5 s): is the gun inside one at T, and when did the last one start / end
        offs = runs["all_off"]
        inside = offs[(offs["start"] <= T) & (offs["end"] >= T - SEC)]
        lag["off_run_spans_T"] = bool(len(inside))
        lag["off_run_spans_T_min"] = round(float(inside["dur_s"].iloc[0]) / 60, 1) if len(inside) else np.nan
        st = offs.loc[offs["start"] < T, "start"]
        lag["min_since_any_off_start"] = round((T - st.max()) / SEC / 60, 2) if len(st) else np.nan
        en = offs.loc[offs["end"] < T, "end"]
        lag["min_since_any_on"] = round((T - en.max()) / SEC / 60, 2) if len(en) else np.nan
        lags.append(lag)
        on = pd.DataFrame(on_features(r.file, ev, states, runs, m, r.start))
        off = pd.DataFrame(off_features(r.file, ev))
        for kind, d in (("on", on), ("off", off)):
            if not len(d):
                continue
            t = d["time"].to_numpy()
            fail = (t >= T - np.timedelta64(30, "m")) & (t < T)
            d["failing"] = False
            if fail.any():
                d.loc[np.flatnonzero(fail)[-1], "failing"] = True
            d["before_T_60"] = t < T - np.timedelta64(60, "m")
            d["class"], d["cluster"] = r._2, r.cluster
            rows.append(d)
            if fail.any():
                linked[kind].append(r.file)
    ev_df = pd.concat(rows, ignore_index=True)
    lag_df = pd.DataFrame(lags)
    ev_df.assign(time_kst=ev_df["time"] + KST).to_csv(os.path.join(HERE, "transitions_events.csv"), index=False)
    lag_df.to_csv(os.path.join(HERE, "failure_lags.csv"), index=False)
    cls_of = dict(zip(meta["file"], meta["class"]))
    out = {"lags": {}, "linked": {k: {"n": len(v), "by_class": {c: int(n) for c, n in
                                                                pd.Series([cls_of[f] for f in v]).value_counts().items()}}
                                  for k, v in linked.items()}, "tests": {}}
    for cls, g in lag_df.groupby("class"):
        out["lags"][cls] = {c: {"median": round(float(g[c].median()), 1), "le_5": int((g[c] <= 5).sum()),
                                "le_30": int((g[c] <= 30).sum()), "n": int(g[c].notna().sum())}
                            for c in lag_df.columns if c.startswith("min_since")}
        out["lags"][cls]["c10_off_at_T"] = int((g["c10_at"] == "off").sum())
        out["lags"][cls]["off_run_spans_T"] = int(g["off_run_spans_T"].sum())
        out["lags"][cls]["off_started_le_1min_before_T"] = int((g["min_since_any_off_start"] <= 1).sum())
    feats = {"on": ("off_min", "idle_h", "stop10", "clock_dev", "c1_at", "since_cap_h"),
             "off": ("clock_dev", "welds10", "on_h")}
    for kind, cols in feats.items():
        d = ev_df[ev_df["kind"] == kind]
        out["tests"][kind] = {}
        for c in cols:
            st, cl = [], []
            for f, g in d.groupby("file"):
                if not g["failing"].any():
                    continue
                st.append((float(g.loc[g["failing"], c].iloc[0]), g.loc[g["before_T_60"] & ~g["failing"], c].to_numpy(float)))
                cl.append(g["cluster"].iloc[0])
            out["tests"][kind][c] = percentile_test(st, cl, rng)
    prim = [out["tests"][k][c] for _, k, c in PRIMARY2]
    adj = holm(np.array([s["p"] for s in prim]))
    out["primary"] = {h: {"kind": k, "feature": c, **s, "p_holm": round(float(q), 4),
                          "supported": bool(q < 0.05 and not (s["ci"][0] <= 0.5 <= s["ci"][1]))}
                      for (h, k, c), s, q in zip(PRIMARY2, prim, adj)}
    # descriptive: what happened between the linked power-on and T
    desc = []
    for f in linked["on"]:
        r = meta.set_index("file").loc[f]
        ev, states, runs, m = cache[f]
        T = np.datetime64(r.fault_time, "ns")
        t = ev["power_on"][(ev["power_on"] >= T - np.timedelta64(30, "m")) & (ev["power_on"] < T)][-1]
        g = ev_df[(ev_df["file"] == f) & ev_df["failing"] & (ev_df["kind"] == "on")].iloc[0]
        desc.append({"file": f, "class": r["class"], "power_on_kst": str(pd.Timestamp(t) + KST),
                     "s_to_failure": int((T - t) / SEC), "welds_between": count_in(ev["production"], t, T),
                     "off_min": round(float(g["off_min"]), 1), "idle_h": round(float(g["idle_h"]), 1),
                     "stop_code_10min_before": bool(g["stop10"])})
    out["linked_on_detail"] = desc
    log(f"linked failures: {json.dumps(out['linked'])}")
    log("lags by class (median min; n <= 5 min / <= 30 min):")
    for cls, v in out["lags"].items():
        log(f"  {cls}: " + ", ".join(f"{k[10:]} {s['median']} ({s['le_5']}/{s['le_30']} of {s['n']})"
                                     for k, s in v.items() if isinstance(s, dict))
            + f", c10 off at T {v['c10_off_at_T']}, inside an off run {v['off_run_spans_T']}, "
              f"off started <= 1 min before {v['off_started_le_1min_before_T']}")
    log("percentile tests (failing transition among the gun's others):")
    for kind, d in out["tests"].items():
        for c, s in d.items():
            log(f"  {kind:3s} {c:12s} {json.dumps(s)}")
    log("primary (Holm over 3): " + json.dumps({h: {k: s[k] for k in ("mean_percentile", "ci", "p", "p_holm", "supported")}
                                                for h, s in out["primary"].items()}))
    log("linked power-on detail:")
    for x in desc:
        log(f"  {json.dumps(x, ensure_ascii=False)}")
    save("transitions", out)


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#9a9a96", "xtick.color": "#55554f", "ytick.color": "#55554f",
                         "axes.labelcolor": "#33332f", "axes.titlesize": 11, "axes.titleweight": "bold"})
    res = json.load(open(os.path.join(HERE, "crossover.json"), encoding="utf-8"))["windows"]["30m"]
    label = {"power_on": "전원 켜짐(10분+ 꺼짐 뒤)", "power_off": "전원 꺼짐(10분+)", "short_cycle": "짧은 전원 깜빡임",
             "cap_change": "캡 교체", "dressing": "캡 드레싱", "cap_maint": "캡 교체·드레싱", "stop_code": "정지 코드 E003·E011",
             "other_code": "기타 에러 코드", "production": "용접 중"}
    panels = ("E01", "E03", "E04", "E02")
    fig, ax = plt.subplots(1, 4, figsize=(12, 4.6), sharey=True)
    y = np.arange(len(EXPOSURES))[::-1]
    for a, g in zip(ax, panels):
        r = res[g]
        cs = [r[x]["case_share"] * 100 for x in EXPOSURES]
        ct = [r[x]["control_share"] * 100 for x in EXPOSURES]
        for yi, c0, c1 in zip(y, ct, cs):
            a.plot([c0, c1], [yi, yi], color="#c9c9c4", linewidth=2, zorder=1)
        a.scatter(ct, y, s=48, color="#9a9a96", zorder=2, edgecolor="white", linewidth=1.2,
                  label="같은 건·같은 시각의 다른 날 (회색)" if g == panels[0] else None)
        a.scatter(cs, y, s=58, color=COLOR[g], zorder=3, edgecolor="white", linewidth=1.2,
                  label="고장 직전 30분 (색)" if g == panels[0] else None)
        a.set_title(f"{g} ({TERM[g]}, {r['power_on']['strata']}건)", loc="left")
        a.set_xlim(-3, 103)
        a.set_xlabel("창에 그 사건이 있었던 비율 (%)")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([label[x] for x in EXPOSURES])
    fig.suptitle("고장 직전 30분 vs 같은 건의 다른 날 같은 시각 (사례-교차)", x=0.01, ha="left", fontsize=12)
    fig.legend(loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(HERE, "fig_crossover.png"), dpi=130)
    plt.close(fig)
    lag = pd.read_csv(os.path.join(HERE, "failure_lags.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    order = ("E01", "E03", "E04", "E02")
    for a, col, title in zip(ax, ("min_since_power_on", "min_since_any_off_start"),
                             ("마지막 기동(10분+ 꺼짐 뒤 켜짐)부터 고장까지 (분)", "마지막 전원 꺼짐(5초+) 시작부터 고장까지 (분)")):
        for yi, cls in enumerate(order[::-1]):
            v = lag.loc[lag["class"] == cls, col].dropna().clip(lower=0.1)
            jit = np.random.default_rng(yi).uniform(-0.18, 0.18, len(v))
            a.scatter(v, yi + jit, s=30, color=COLOR[cls], edgecolor="white", linewidth=1, zorder=3)
        a.set_xscale("log")
        a.set_xlim(0.08, 2e4)
        for x in (5, 30):
            a.axvline(x, color="#9a9a96", linewidth=0.8, linestyle="--")
        a.set_xticks([0.1, 1, 5, 30, 60 * 6, 60 * 24, 60 * 24 * 7])
        a.set_xticklabels(["0.1", "1", "5", "30", "6h", "1d", "7d"])
        a.set_title(title, loc="left")
    ax[0].set_yticks(range(len(order)))
    ax[0].set_yticklabels([f"{c} ({TERM[c]})" for c in order[::-1]])
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_lags.png"), dpi=130)
    plt.close(fig)
    log("saved fig_crossover.png, fig_lags.png")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "crossover":
        crossover()
    elif cmd == "transitions":
        transitions()
    elif cmd == "figures":
        figures()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
