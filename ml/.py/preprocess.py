"""
RSW train-set preprocessing for anomaly-detection model training.

Every rule below comes from eda_out/ (72 files) and the paper's pre-processing section:

  gaps        1 Hz grid; whole rows drop out. 99 % of gaps are <= 35 s, so gaps up to
              --gap-fill-limit are forward-filled; longer ones break the series into
              segments (rows dropped) so windows never straddle a hole.
  discard     files with missing rate > --max-missing-rate (paper: 40 %). That is the 8
              E02 files at 64-84 %, including E02_2 (10 min "demo" file).
  outliers    c16 (sheet-thickness setpoint) <= 0 = cap dressing / changing, not welding.
              Sensor values in those rows are blanked and forward-filled (paper), and the rows
              are flagged `non_welding` so the model can drop or keep them. The paper/benchmark
              also treat c16 > 5 (6) as non-welding; that branch is OFF by default since
              2026-09-23: in the train set it fires on < 1.3 % of rows, but two test guns run at
              a setpoint of 8.7 for 85-92 % of the time while welding, and the rule turned
              nearly all their data into carried-forward constants. --outlier-hi 5 restores it.
  counters    c11 / c12 are cumulative and gun-specific -> replaced by per-second deltas and
              a 10-min rolling weld count (welding-behaviour pattern, paper Fig. 8).
  c10         on/off -> 1/0.
  static      c7, c8, c9 are constant within a file (gun attributes). Kept as-is; they act as
              gun identity after scaling. Drop them with --drop-static if the detector should
              not be able to tell guns apart.
  error       a *state* column: codes persist for hours. Kept as `error_code`, plus
              `error_active` (any non-zero code), a 10-min rolling share (readme: minor-error
              frequency as precursor feature) and `terminal_code` = the code that marks the
              failure for this class (E01 E012, E02 E016, E03 E028, E04 E029 - 18/18, 17/18,
              18/18, 18/18 files).
  label       `ttf_s` seconds-to-failure and `label` = 1 inside the last --pre-failure-window
              seconds. For an unsupervised detector train on label == 0 rows (and optionally
              error_active == 0); the scaler is fitted on those rows only.
  scaling     z-score (paper) fitted on normal rows, global across files (--scale global) or
              per file (--scale per-file, removes gun-level offsets). Parameters are written
              to scaler.json so test files get the identical transform.
  test split  --split test runs the SAME pipeline on test/test_N.csv: every option the user does
              not pass explicitly is taken from the train run's preprocess_config.json, the
              global scaler is loaded from scaler.json instead of being fitted, and the output
              goes to <train out-dir>/test/. Test files carry no class in the name, so the class
              is inferred from the terminal code seen in the last 10 min (E012 E01, E016 E02,
              E028 E03, E029 E04 - near-deterministic in the train set); the manifest records
              the source (`class_source`). ttf_s / label are computed the same way (each test
              file ends at its failure, like the train files).

Outputs (in --out-dir):
  <file>.parquet (or .csv)   one row per second (or per --resample bin), float32
  manifest.csv               per-file rows / segments / drop reason
  scaler.json                mean/std per scaled column
  preprocess_config.json     the arguments used

Usage:
    python preprocess.py                         # ./train -> ./preprocessed
    python preprocess.py --split test            # ./test  -> ./preprocessed/test (train config + scaler)
    python preprocess.py --resample 10s          # 10-second bins (mean of sensors, max of flags)
    python preprocess.py --scale per-file --drop-static
    python preprocess.py --pattern "E04_*" --max-files 3     # quick check
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .py/ -> repo root

SENSOR_COLS = [f"c{i}" for i in range(1, 20)]
COUNTER_COLS = ["c11", "c12"]
STATIC_COLS = ["c7", "c8", "c9"]
BINARY_COL = "c10"
# continuous sensors that get blanked in non-welding rows and z-scored
VALUE_COLS = [c for c in SENSOR_COLS if c not in COUNTER_COLS + [BINARY_COL]]
CLASSES = ["E01", "E02", "E03", "E04"]
TERMINAL_CODE = {"E01": "E012", "E02": "E016", "E03": "E028", "E04": "E029"}
TERMINAL_TO_CLASS = {v: k for k, v in TERMINAL_CODE.items()}
ROLL = "600s"  # 10-minute rolling window for activity / error-share features

FLAG_COLS = ["error_active", "terminal_code", "non_welding", "label"]
# derived continuous columns that get z-scored; the 10-min shares are already in [0, 1] and stay raw
DERIVED_CONT = ["welds_delta", "welds_10min", "pos_delta"]
# options the test split inherits from the train run unless passed explicitly
INHERITED_OPTIONS = ["max_missing_rate", "gap_fill_limit", "outlier_hi", "outlier_lo", "pre_failure_window",
                     "resample", "scale", "drop_static", "format"]


# ------------------------------------------------------------------- loading
def load_file(path):
    """CSV -> 1 Hz grid DataFrame (NaN rows where the panel is missing), present mask."""
    df = pd.read_csv(path, low_memory=False, dtype={"c10": "string", "error": "string"})
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df = df.drop_duplicates("time").set_index("time").sort_index()
    num = [c for c in SENSOR_COLS if c != BINARY_COL]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce").astype("float32")
    raw_index = df.index
    df = df.resample("s").asfreq()
    present = df.index.isin(raw_index)
    return df, present


def fill_gaps(x, present, limit):
    """Gap handling shared with main.py. A gap = run of missing 1 Hz rows. Gaps up to `limit` seconds
    are forward-filled; the rows of longer gaps are dropped, which starts a new segment downstream.
    (ffill + bfill with the same limit - the old code - silently bridged gaps up to 2 * limit.)
    Rows that still hold a NaN sensor value afterwards are dropped as well."""
    present = pd.Series(np.asarray(present, dtype=bool), index=x.index)
    run = (present != present.shift()).cumsum()
    gap_len = (~present).groupby(run).transform("size").where(~present, 0)
    x = x[gap_len <= limit].ffill(limit=limit)
    return x[x.notna().all(axis=1)]


# -------------------------------------------------------------- per file
def infer_class(code, t_end, tail_s=600):
    """test_N files carry no class in the name. The class-specific terminal codes show up inside the
    last 10 min before failure in 71/72 train files (MEMORY.md 2), so the last one seen there names
    the class. Returns (class, source) with class == "unknown" when none is present."""
    tail = code[code.index > t_end - pd.Timedelta(seconds=tail_s)]
    hits = tail[tail.isin(TERMINAL_TO_CLASS)]
    if hits.empty:
        return "unknown", "no terminal code in the last 10 min"
    last = hits.iloc[-1]
    return TERMINAL_TO_CLASS[last], f"terminal code {last}"


def preprocess_file(path, args):
    name = os.path.splitext(os.path.basename(path))[0]
    prefix, gun = name.split("_")
    df, present = load_file(path)
    missing_rate = 1 - present.mean()
    t_end = df.index[-1]

    # --- error state: carry the code across gaps, then flags
    code = df["error"].where(present).ffill().fillna("0").astype(str)
    if prefix in CLASSES:  # E0x_N (train)
        cls, class_source = prefix, "filename"
    else:  # test_N
        cls, class_source = infer_class(code, t_end)
    base = {"file": name, "class": cls, "class_source": class_source, "gun": int(gun),
            "rows": 0, "segments": 0, "missing_rate": missing_rate}
    if missing_rate > args.max_missing_rate:
        return None, {**base, "dropped": f"missing rate {missing_rate:.0%}"}

    out = pd.DataFrame(index=df.index)
    out["error_code"] = code
    out["error_active"] = (code != "0").astype("float32")
    terminal = [TERMINAL_CODE[cls]] if cls in TERMINAL_CODE else list(TERMINAL_CODE.values())
    out["terminal_code"] = code.isin(terminal).astype("float32")

    # --- sensors: c10 -> 0/1, non-welding rows blanked (paper outlier rule), then gap fill
    x = df[SENSOR_COLS].copy()
    x[BINARY_COL] = (x[BINARY_COL] == "on").astype("float32").where(present)
    # 1) gaps: fill up to the limit, drop the rows of longer gaps (long block-outs)
    x = fill_gaps(x, present, args.gap_fill_limit)
    out = out.loc[x.index]
    if len(x) == 0:
        return None, {**base, "dropped": "no rows left after gap filtering"}
    # 2) non-welding rows (cap dressing / changing): blank the sensors and carry the last
    #    welding value across, however long the block is (paper). Rows stay, flagged.
    non_welding = x["c16"] <= args.outlier_lo
    if args.outlier_hi is not None:
        non_welding |= x["c16"] > args.outlier_hi
    out["non_welding"] = non_welding.astype("float32")
    # duty cycle from the MEASURED force, before the carry: during cap dressing the carried value would count as welding
    pressing = (x["c2"] > 0).astype("float32")
    x.loc[non_welding, VALUE_COLS + [BINARY_COL]] = np.nan  # counters keep counting during cap dressing
    x = x.ffill().bfill()

    # --- contiguous segments (a break wherever consecutive kept rows are > 1 s apart)
    step = x.index.to_series().diff().dt.total_seconds().fillna(1)
    out["segment_id"] = (step > 1).cumsum().astype("int32")

    # --- counters -> deltas (per segment, non-negative), rolling activity
    for c, new in [("c11", "welds_delta"), ("c12", "pos_delta")]:
        d = x[c].groupby(out["segment_id"]).diff().fillna(0).clip(lower=0)
        out[new] = d.astype("float32")
    out["welds_10min"] = out["welds_delta"].rolling(ROLL, min_periods=1).sum().astype("float32")
    out["weld_duty_10min"] = pressing.rolling(ROLL, min_periods=1).mean().astype("float32")
    out["error_share_10min"] = out["error_active"].rolling(ROLL, min_periods=1).mean().astype("float32")

    # --- sensor columns
    sensor_keep = [c for c in VALUE_COLS if not (args.drop_static and c in STATIC_COLS)] + [BINARY_COL]
    for c in sensor_keep:
        out[c] = x[c].astype("float32")

    # --- time-of-day (shift pattern, paper Fig. 8) and time-to-failure label
    hour = out.index.hour + out.index.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    out["dow"] = out.index.dayofweek.astype("int8")
    out["ttf_s"] = (t_end - out.index).total_seconds().astype("int64")
    out["label"] = (out["ttf_s"] <= args.pre_failure_window).astype("float32")

    if args.resample:
        out = resample(out, args.resample)

    out.insert(0, "gun", np.int16(gun))
    out.insert(0, "class", cls)
    out.insert(0, "file", name)
    info = {**base, "rows": len(out), "segments": int(out["segment_id"].nunique()),
            "non_welding_share": float(out["non_welding"].mean()), "label_rows": int(out["label"].sum()),
            "start": out.index[0], "end": out.index[-1], "dropped": ""}
    return out, info


def resample(out, rule):
    """Aggregate 1 Hz rows into bins: mean for continuous, max for flags, sum for deltas, last for time fields."""
    agg = {c: "mean" for c in out.columns}
    agg.update({c: "max" for c in FLAG_COLS})
    agg.update({"welds_delta": "sum", "pos_delta": "sum", "segment_id": "min", "dow": "last",
                "ttf_s": "min", "error_code": "last"})
    grouped = out.groupby([out["segment_id"], pd.Grouper(freq=rule)]).agg(agg)
    grouped = grouped.droplevel(0)
    grouped = grouped[grouped["ttf_s"].notna()]
    for c in grouped.columns:
        if c not in ("error_code", "segment_id", "dow", "ttf_s"):
            grouped[c] = grouped[c].astype("float32")
    grouped["segment_id"] = grouped["segment_id"].astype("int32")
    grouped["dow"] = grouped["dow"].astype("int8")
    grouped["ttf_s"] = grouped["ttf_s"].astype("int64")
    return grouped


# -------------------------------------------------------------- scaling
class StreamingStats:
    """Running mean/variance of the columns to scale, fitted on normal rows only."""

    def __init__(self, cols):
        self.cols, self.n, self.s, self.ss = cols, 0, np.zeros(len(cols)), np.zeros(len(cols))

    def add(self, df):
        v = df[self.cols].to_numpy(dtype=np.float64)
        self.n += len(v)
        self.s += v.sum(axis=0)
        self.ss += (v ** 2).sum(axis=0)

    def params(self):
        mean = self.s / max(self.n, 1)
        std = np.sqrt(np.maximum(self.ss / max(self.n, 1) - mean ** 2, 0))
        std = np.where(std < 1e-6, 1.0, std)  # constant columns stay as (x - mean)
        return {c: {"mean": float(m), "std": float(s)} for c, m, s in zip(self.cols, mean, std)}


def normal_rows(df):
    return df[(df["label"] == 0) & (df["error_active"] == 0)]


def apply_scaler(df, params):
    for c, p in params.items():
        if c in df.columns:
            df[c] = ((df[c] - p["mean"]) / p["std"]).astype("float32")
    return df


# -------------------------------------------------------------- I/O
def writer(fmt):
    if fmt == "parquet":
        try:
            import pyarrow  # noqa: F401
            return "parquet", lambda df, p: df.to_parquet(p + ".parquet")
        except ImportError:
            print("pyarrow not installed -> writing CSV")
    return "csv", lambda df, p: df.to_csv(p + ".csv")


def reader(fmt):
    if fmt == "parquet":
        return lambda p: pd.read_parquet(p + ".parquet")
    return lambda p: pd.read_csv(p + ".csv", index_col=0, parse_dates=True)


def optional_float(v):
    """argparse type: 'none' / 'off' -> None, otherwise float."""
    return None if str(v).lower() in ("none", "off", "null") else float(v)


def explicit_options(ap, argv=None):
    """dests of the options given on the command line (an explicit value equal to the default counts too)."""
    argv = sys.argv[1:] if argv is None else argv
    return {a.dest for a in ap._actions
            if any(arg == s or arg.startswith(s + "=") for s in a.option_strings for arg in argv)}


def inherit_train_config(ap, args):
    """Test files must get exactly the train transform: every inherited option the user did not
    pass explicitly is read from the train run's preprocess_config.json."""
    p = os.path.join(args.train_out_dir, "preprocess_config.json")
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found - run the train split first (python preprocess.py)")
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    for k in INHERITED_OPTIONS:
        if k in cfg and k not in explicit_options(ap):
            setattr(args, k, cfg[k])
    return p


def load_train_scaler(args):
    p = os.path.join(args.train_out_dir, "scaler.json")
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found - run the train split first (python preprocess.py)")
    with open(p, encoding="utf-8") as f:
        sc = json.load(f)
    if sc["scale"] != "global":
        raise SystemExit(f"train scaler is '{sc['scale']}', cannot apply it to the test split - pass --scale {sc['scale']}")
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="test: ./test -> <train-out-dir>/test with the train config and scaler")
    ap.add_argument("--input-dir", "--train-dir", dest="input_dir", default=None,
                    help="CSV folder; default ./train or ./test by --split")
    ap.add_argument("--out-dir", default=None, help="default ./preprocessed (train) or ./preprocessed/test")
    ap.add_argument("--train-out-dir", default=os.path.join(PROJECT_ROOT, "preprocessed"),
                    help="test split: where the train run wrote scaler.json / preprocess_config.json")
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-missing-rate", type=float, default=0.40, help="paper: discard guns above this")
    ap.add_argument("--gap-fill-limit", type=int, default=60, help="seconds; longer gaps split segments")
    ap.add_argument("--outlier-hi", type=optional_float, default=None,
                    help="c16 above this = non-welding; default none (see docstring), paper rule: 5")
    ap.add_argument("--outlier-lo", type=float, default=0.0, help="c16 at/below this = non-welding")
    ap.add_argument("--pre-failure-window", type=int, default=3600, help="seconds before failure labelled 1")
    ap.add_argument("--resample", default=None, help="pandas offset, e.g. 10s or 1min; default keeps 1 Hz")
    ap.add_argument("--scale", choices=["global", "per-file", "none"], default="global")
    ap.add_argument("--drop-static", action="store_true", help="drop c7/c8/c9 (constant per gun)")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = ap.parse_args()
    is_test = args.split == "test"
    config_source, train_scaler = None, None
    if is_test:
        config_source = inherit_train_config(ap, args)
        if args.scale == "global":
            train_scaler = load_train_scaler(args)
    args.input_dir = args.input_dir or os.path.join(PROJECT_ROOT, args.split)
    args.out_dir = args.out_dir or (os.path.join(args.train_out_dir, "test") if is_test else args.train_out_dir)

    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))[: args.max_files]
    if not paths:
        raise SystemExit(f"no files matching {args.pattern!r} in {os.path.abspath(args.input_dir)} - pass --input-dir")
    os.makedirs(args.out_dir, exist_ok=True)
    fmt, write = writer(args.format)
    read = reader(fmt)

    scale_cols = [c for c in VALUE_COLS if not (args.drop_static and c in STATIC_COLS)] + DERIVED_CONT
    if train_scaler and train_scaler["columns"] != scale_cols:
        raise SystemExit(f"scaler columns {train_scaler['columns']} != {scale_cols} - check --drop-static")
    stats = StreamingStats(scale_cols)
    manifest, t0 = [], time.time()
    if is_test:
        print(f"test split: options from {config_source}, scale={args.scale}"
              + (f", scaler from {os.path.join(args.train_out_dir, 'scaler.json')}" if train_scaler else ""))

    # pass 1: clean, derive, write; accumulate global scaler stats on normal rows
    for i, p in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {os.path.basename(p)}", flush=True)
        out, info = preprocess_file(p, args)
        manifest.append(info)
        if out is None:
            print("  dropped:", info["dropped"])
            continue
        if info.get("class_source") != "filename":
            print(f"  class {info['class']} ({info['class_source']})")
        if args.scale == "per-file":
            local = StreamingStats(scale_cols)
            local.add(normal_rows(out))
            out = apply_scaler(out, local.params())
        elif args.scale == "global" and is_test:
            out = apply_scaler(out, train_scaler["params"])  # never refit on test data
        elif args.scale == "global":
            stats.add(normal_rows(out))
        write(out, os.path.join(args.out_dir, info["file"]))

    # pass 2: apply the global scaler (train split only; test files were scaled on the way in)
    params = {}
    if args.scale == "global" and not is_test and stats.n > 0:
        params = stats.params()
        for info in manifest:
            if info["dropped"]:
                continue
            p = os.path.join(args.out_dir, info["file"])
            write(apply_scaler(read(p), params), p)
    elif args.scale == "per-file":
        params = {"note": "per-file z-score; fit the same way on each test file (normal rows)"}

    pd.DataFrame(manifest).to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)
    if not is_test:  # the scaler belongs to the train run; the test split only reads it
        with open(os.path.join(args.out_dir, "scaler.json"), "w", encoding="utf-8") as f:
            json.dump({"scale": args.scale, "columns": scale_cols, "params": params, "n_fit_rows": stats.n}, f, indent=2)
    cfg = vars(args)
    if is_test:
        cfg = {**cfg, "config_source": config_source,
               "scaler_source": os.path.join(args.train_out_dir, "scaler.json") if train_scaler else None}
    with open(os.path.join(args.out_dir, "preprocess_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    m = pd.DataFrame(manifest)
    kept = m[m["dropped"] == ""]
    print(f"\ndone in {time.time() - t0:.0f}s -> {args.out_dir} ({fmt})")
    print(f"files kept {len(kept)}/{len(m)}, rows {int(kept['rows'].sum()):,}, "
          f"label=1 rows {int(kept['label_rows'].sum()):,}")
    if (m["dropped"] != "").any():
        print("dropped:", ", ".join(f"{r.file} ({r.dropped})" for r in m[m["dropped"] != ""].itertuples()))


if __name__ == "__main__":
    main()
