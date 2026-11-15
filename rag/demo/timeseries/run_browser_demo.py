"""Start the isolated ML/API/UI demo, capture the browser, and remove its Aura records."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from dotenv import load_dotenv

from rag.config import get_database, graph_driver, load_environment
from rag.warning_memory import WarningMemory

ROOT = Path(__file__).resolve().parents[3]


def ready(url, servers):
    for _ in range(120):
        if any(process.poll() is not None for process in servers):
            raise RuntimeError("server exited; inspect the demo logs")
        try:
            if httpx.get(url, timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.4)
    raise RuntimeError(f"server did not become ready: {url}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser", default="msedge")
    args = parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    load_environment()
    uri = urlsplit(os.getenv("NEO4J_URI", ""))
    if uri.scheme != "neo4j+s" or not (uri.hostname or "").endswith(".databases.neo4j.io"):
        parser.error("configure the encrypted Aura endpoint")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    namespace = f"timeseries-ui-demo-{uuid4()}"
    env = {**os.environ, "RAG_WARNING_TIMESERIES": "1", "RAG_WARNING_FEEDBACK": "1", "RAG_WARNING_VECTORS": "1",
           "RAG_WARNING_NAMESPACE": namespace, "RAG_WARNING_EQUIPMENT_SCOPE": "cpx-weld-2g-sa",
           "API_DB": str(output / f"{namespace}.db"), "ML_URL": "http://127.0.0.1:19111",
           "RSW_DEMO_REPLAY": "1", "RSW_RAG_URL": "http://127.0.0.1:19112/api/ml/handoffs",
           "RSW_OUTBOX_POLL_S": "0.2", "VITE_USE_MOCK": "false",
           "VITE_API_PROXY_TARGET": "http://127.0.0.1:19112", "VITE_API_BASE_URL": "/api"}
    public_env = {key: value for key, value in env.items()
                  if key not in {"NEO4J_PASSWORD", "NEO4J_URI", "NEO4J_USERNAME", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}}
    servers = []

    def start(name, command, process_env, cwd=ROOT):
        with (output / f"{name}.log").open("w", encoding="utf-8") as stream:
            servers.append(subprocess.Popen(command, cwd=cwd, env=process_env, stdout=stream, stderr=stream,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))

    def app(folder, port):
        return [sys.executable, "-B", "-m", "uvicorn", "main:app", "--app-dir", folder,
                "--host", "127.0.0.1", "--port", port]

    with graph_driver() as driver:
        database = get_database()
        WarningMemory(driver, database=database, namespace=namespace).initialize()
        try:
            start("ml", app("ml", "19111"), public_env)
            start("api", app("api", "19112"), env)
            start("ui", ["node", "node_modules/vite/bin/vite.js", "--host", "127.0.0.1",
                         "--port", "19113", "--strictPort"], public_env, ROOT / "ui")
            for endpoint in ("http://127.0.0.1:19111/health", "http://127.0.0.1:19112/api/health",
                             "http://127.0.0.1:19113/"):
                ready(endpoint, servers)
            print("ML, API, UI ready", flush=True)
            subprocess.run([sys.executable, "-B", "-m", "rag.demo.timeseries.browser_demo",
                            "--output", str(output), "--browser", args.browser], cwd=ROOT, env=public_env, check=True)
            with driver.session(database=database, default_access_mode="READ") as session:
                count = session.run("""MATCH (e:WarningEvent {namespace:$ns})-[:HAS_SENSOR_WINDOW]->(:WarningSensorWindow)
                    MATCH (e)-[:HAS_WARNING_RESOLUTION]->(r:WarningResolution)-[:USED_WARNING_CASE]->(:WarningManualCase)
                    RETURN count(r) AS n""", ns=namespace).single()["n"]
            print("Aura linked repairs:", count, flush=True)
            if count != 2:
                raise RuntimeError("expected two linked demo repairs")
        finally:
            for process in servers:
                if process.poll() is None:
                    process.terminate()
            try:
                for process in servers:
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            finally:
                with driver.session(database=database) as session:
                    session.run("""MATCH (n {namespace:$ns})
                        WHERE n:WarningEvent OR n:WarningSensorWindow OR n:WarningResolution OR n:WarningManualCase
                        DETACH DELETE n""", ns=namespace).consume()
                    remaining = session.run("MATCH (n {namespace:$ns}) RETURN count(n) AS n",
                                            ns=namespace).single()["n"]
                print("Demo nodes remaining:", remaining, flush=True)


if __name__ == "__main__":
    main()
