"""Run with python -m rag from the directory containing the rag package."""

import importlib
import sys


COMMANDS = {
    "parse": "rag.pipeline.parse",
    "chunk": "rag.pipeline.chunk",
    "ingest": "rag.pipeline.ingest",
    "build": "rag.pipeline.build",
    "ask": "rag.service",
}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python -m rag {parse,chunk,ingest,build,ask} [options]")
        print("명령별 옵션: python -m rag <command> --help")
        return 0
    command = sys.argv.pop(1)
    if command not in COMMANDS:
        print(f"지원하지 않는 명령: {command}", file=sys.stderr)
        return 2
    sys.argv[0] = f"python -m rag {command}"
    module = importlib.import_module(COMMANDS[command])
    return module.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
