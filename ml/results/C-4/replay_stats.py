"""C-4 acceptance (history.md §19): replay each real test file through the API in-process, then compare the gun's
/stats normal alarm rate with the offline evaluate() of the same bundle (models/baseline_<model>_test_metrics.json)
and with score_frame on the same population (windows without an error code - serving does not know the label).

    python results/C-4/replay_stats.py                 # 8 test files, serving default model
    python results/C-4/replay_stats.py test_4 test_5   # a subset (appends to the CSV)

One file per run of the loop, results appended as they come (a run cut short keeps what it finished).
"""
import json
import os
import sys
import tempfile
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "replay_stats.csv")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, ".py"))
os.environ["RSW_STATE_DIR"] = tempfile.mkdtemp(prefix="rsw_ml5_")  # never the real state/ folder
os.environ.pop("RSW_RAG_URL", None)
import main  # noqa: E402
import train as T  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from replay import replay_file  # noqa: E402

files = sys.argv[1:] or [f"test_{i}" for i in range(8)]
with TestClient(main.app) as c:
    d = main.app.state.detector
    model = d.info.type
    off = json.load(open(os.path.join(ROOT, "models", f"{d.info.name}_test_metrics.json"), encoding="utf-8"))["test"]["per_file"]
    print(f"model {d.info.name} ({model}), state {os.environ['RSW_STATE_DIR']}", flush=True)
    for f in files:
        t0 = time.time()
        replay_file(c, os.path.join(ROOT, "test", f"{f}.csv"), f, chunk_s=1200)
        st = c.get(f"/guns/{f}/stats", params={"days": 31}).json()
        tot = st["total"]
        sf = T.score_frame(d.bundle, pd.read_parquet(os.path.join(ROOT, "preprocessed", "test", f"{f}.parquet")))
        ok = sf["error_active"] == 0
        row = {"file": f, "model": d.info.name, "online": tot["normal_alarm_rate"],
               "offline_evaluate": off[f]["alarm_rate_normal"], "offline_same_population": float(sf.loc[ok, "alarm"].mean()),
               "windows_online": tot["windows"], "windows_offline": len(sf), "alarms_online": tot["alarms"],
               "alarms_offline": int(sf["alarm"].sum()), "rule": tot["rule_triggers"], "repeats": tot["rule_repeats"],
               "critical_events": tot["critical_events"], "sustained": tot["sustained_episodes"],
               "drift": "; ".join(st["drift_warning"]), "seconds": round(time.time() - t0)}
        row["diff_pp"] = (row["online"] - row["offline_evaluate"]) * 100
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        c.delete(f"/guns/{f}")  # free the gun's buffer / trace before the next file
