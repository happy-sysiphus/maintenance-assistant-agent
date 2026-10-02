"""
RSW welding-gun fault prediction benchmark — Stage 1 (target-series forecasting).

Bug-fixed rewrite of Benchmark.py. Same constants, same 10 models, same outputs
(one PNG per model x test file, one metric CSV per test file). See the paper
(Wang et al., Sci Data 2024) for the setup this reproduces.

Fixes vs. the original:
  * every plot drew `predl_NBEATS` (undefined until the last block) -> NameError
  * forecast was made past the END of the test series, so metrics had no
    overlap with actuals -> now the last N_PRED seconds are held out
  * PNG names used file_name[-4:] (".csv") -> all test files overwrote each other
  * accelerator="gpu" hard-coded -> "auto" (this machine has CPU-only torch)
  * TFT needs add_relative_index=True when no future covariates are given
  * darts.mape raises on zeros in the actual series (c2/c4 are ~50-85 % zero)
  * pandas 2.x removals: resample('S'), fillna(method=...), to_numeric(errors='ignore')
  * leading gap left NaN after ffill -> bfill for the leading edge only
  * outlier rows blanked AFTER one-hot encoding, so c10 dummies became all-zero
    instead of being forward-filled -> mask, fill, then encode
  * dead `switch` resume logic removed
  * paper: guns with > 40 % missing data are discarded (E02_0/4/6 are ~80 %)
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, mape, marre, mse
from darts.models import (
    BlockRNNModel,
    LightGBMModel,
    LinearRegressionModel,
    NBEATSModel,
    RandomForest,
    RegressionModel,
    TCNModel,
    TFTModel,
)
from sklearn.linear_model import BayesianRidge

LEN = 1200  # max training samples taken from each time series
HIS = 120  # look-back length (input chunk / lags)
AHEAD = 60  # forecast length per chunk (paper: lead time N = 60)
N_PRED = 120  # horizon forecast at test time (original used 120; > AHEAD -> auto-regressive)
MAX_MISSING_RATE = 0.4  # paper: "abandon the data whose missing rate is larger than 40 percent"
TARGET_COLS = ["c1", "c2", "c3", "c4", "c5"]  # first 5 sensors, as in the original script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .py/ -> repo root
TRAIN_DIR = os.path.join(PROJECT_ROOT, "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "test")
OUT_DIR = os.path.join(PROJECT_ROOT, "results")
TRAINER = {"accelerator": "auto"}


# ---------------------------------------------------------------- preprocessing
def load_csv(path):
    """Read one gun CSV -> (1 Hz DataFrame with c10 one-hot, missing rate)."""
    df = pd.read_csv(path, index_col="time", low_memory=False)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.resample("s").asfreq()  # 1 Hz grid; gaps become NaN rows
    missing_rate = df["c16"].isna().mean()  # whole rows drop out ("block out"), any column works

    df = df.drop(columns=["error"])
    num_cols = [c for c in df.columns if c != "c10"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # sheet-thickness setpoint out of welding range -> cap dressing/changing, not welding
    outlier = (df["c16"] > 5) | (df["c16"] <= 0)
    df.loc[outlier, :] = np.nan

    df = df.ffill().bfill()  # "fill the blank with the most recent correctly collected point"
    df = pd.get_dummies(df, columns=["c10"], dtype=np.float32)
    return df, missing_rate


def split_target_cov(df, cov_columns=None):
    tar = df[TARGET_COLS]
    cov = df.drop(columns=TARGET_COLS)
    if cov_columns is not None:  # align c10 dummies to the training schema
        cov = cov.reindex(columns=cov_columns, fill_value=0.0)
    return tar, cov


def to_series(df):
    return TimeSeries.from_dataframe(df, freq="s").astype(np.float32)


def build_training_set(train_dir):
    tar_list, cov_df_list = [], []
    for path in sorted(glob.glob(os.path.join(train_dir, "*.csv"))):
        name = os.path.basename(path)
        print("Start processing file " + name)
        df, missing_rate = load_csv(path)
        if missing_rate > MAX_MISSING_RATE:
            print(f"  skipped: missing rate {missing_rate:.1%} > {MAX_MISSING_RATE:.0%}")
            continue
        tar, cov = split_target_cov(df)
        tar_list.append(to_series(tar))
        cov_df_list.append(cov)  # dummy columns may differ per file; unify below
        print("Finish processing file " + name)

    cov_columns = sorted(set().union(*[d.columns for d in cov_df_list]))
    cov_list = [to_series(d.reindex(columns=cov_columns, fill_value=0.0)) for d in cov_df_list]
    return tar_list, cov_list, cov_columns


# ---------------------------------------------------------------------- models
def build_models():
    reg = dict(lags=HIS, lags_past_covariates=HIS, output_chunk_length=AHEAD)
    nn = dict(input_chunk_length=HIS, output_chunk_length=AHEAD, pl_trainer_kwargs=TRAINER)
    rnn = dict(hidden_dim=64, n_rnn_layers=2, dropout=0.2, random_state=42, **nn)
    return {
        "BayesianRidge": RegressionModel(model=BayesianRidge(), **reg),
        "RandomForest": RandomForest(n_estimators=100, max_depth=None, **reg),
        "LinearRegression": LinearRegressionModel(random_state=43, **reg),
        "LightGB": LightGBMModel(random_state=43, **reg),
        "RNN": BlockRNNModel(model="RNN", **rnn),
        "GRU": BlockRNNModel(model="GRU", **rnn),
        "LSTM": BlockRNNModel(model="LSTM", **rnn),
        "TCN": TCNModel(kernel_size=3, num_filters=3, dilation_base=2, weight_norm=False, dropout=0.2, **nn),
        "TFT": TFTModel(add_relative_index=True, **nn),
        "NBEATS": NBEATSModel(random_state=42, **nn),
    }


def is_torch(model):
    return hasattr(model, "pl_trainer_kwargs")


def train_models(models, tar_list, cov_list, epochs=100):
    for name, model in models.items():
        print("Training " + name)
        kw = dict(epochs=epochs, verbose=True) if is_torch(model) else {}
        model.fit(tar_list, past_covariates=cov_list, max_samples_per_ts=LEN, **kw)


# --------------------------------------------------------------------- testing
def safe_mape(actual, pred):
    try:
        return mape(actual, pred)
    except ValueError:  # actual contains zeros
        return np.nan


def evaluate(models, test_dir, out_dir, cov_columns):
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(test_dir, "*.csv"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        print("Start processing test file " + stem)
        df, _ = load_csv(path)
        tar, cov = split_target_cov(df, cov_columns)
        series_tar, series_cov = to_series(tar), to_series(cov)
        history, actual = series_tar[:-N_PRED], series_tar[-N_PRED:]  # hold out the last N_PRED s

        rows = []
        for name, model in models.items():
            pred = model.predict(n=N_PRED, series=history, past_covariates=series_cov)

            plt.figure(figsize=(35, 10))
            series_tar[-(HIS + N_PRED):].plot(label="actual")
            pred.plot(label="forecast")
            plt.legend()
            plt.savefig(os.path.join(out_dir, f"{name}_{stem}.png"), bbox_inches="tight", pad_inches=0.1, dpi=500)
            plt.close()

            rows.append(
                {
                    "model": name,
                    "mae": mae(actual, pred),
                    "mape": safe_mape(actual, pred),
                    "marre": marre(actual, pred),
                    "mse": mse(actual, pred),
                }
            )
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"metric_{stem}.csv"), index=False)
        print("Finish testing file " + stem)


def main(train_dir=TRAIN_DIR, test_dir=TEST_DIR, out_dir=OUT_DIR, epochs=100):
    tar_list, cov_list, cov_columns = build_training_set(train_dir)
    models = build_models()
    train_models(models, tar_list, cov_list, epochs=epochs)
    evaluate(models, test_dir, out_dir, cov_columns)


if __name__ == "__main__":
    main()
