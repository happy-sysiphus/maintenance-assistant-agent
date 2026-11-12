"""
Model registry of the D-11 model loop (docs/superpowers/specs/2026-11-11-model-loop-design.md, 8.1).

One folder:

    registry.json     versions, champion (the serving model), challenger (trained, not serving yet),
                      decisions (every A/B judgement with its numbers), history (champion changes), ticks
    v001/, v002/ ...  one folder per version with its bundle (written by train.py)

model_loop.py writes the registry; main.py reads it when RSW_MODEL_REGISTRY names the folder. Only the standard
library is used here because main.py imports this module.

The functions below the class change the registry dict in memory; Registry.save writes it in one atomic step.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

SCHEMA = 1
STATUSES = ("champion", "challenger", "retired", "rejected")
FILE, LOCK = "registry.json", "registry.lock"


class RegistryError(Exception):
    """The registry cannot be used as it is (unreadable file, another config, locked). Nothing was changed."""


class Registry:
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.path = os.path.join(self.root, FILE)

    def load(self) -> dict[str, Any] | None:
        """The registry dict, or None when there is no registry.json yet."""
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as e:
            raise RegistryError(f"cannot read {self.path}: {e}") from e
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise RegistryError(f"{self.path} is not a schema-{SCHEMA} model registry")
        return data

    def init(self, config: dict[str, Any]) -> dict[str, Any]:
        """The stored registry, created empty on first use. A registry made with other settings is refused:
        decisions taken under different rules do not belong in one history."""
        data = self.load()
        if data is None:
            data = empty(config)
            self.save(data)
            return data
        if data["config"] != config:
            keys = sorted(k for k in set(data["config"]) | set(config) if data["config"].get(k) != config.get(k))
            raise RegistryError(f"{self.path} was created with other settings (differs in: {', '.join(keys)}); "
                                "pass the same settings or use a new --registry folder")
        return data

    def save(self, data: dict[str, Any]) -> None:
        """Temp file + rename: a crash mid-write leaves the previous registry, never half a file."""
        os.makedirs(self.root, exist_ok=True)
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=1)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        for attempt in range(5):  # Windows: a reader holding registry.json makes the rename fail briefly
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02 * (attempt + 1))

    def version_dir(self, version: str) -> str:
        return os.path.join(self.root, version)

    def bundle_path(self, data: dict[str, Any], version: str) -> str:
        return os.path.join(self.root, *data["versions"][version]["bundle"].split("/"))

    @contextmanager
    def lock(self) -> Iterator[None]:
        """One model_loop run at a time: registry.lock exists while a run works on this registry."""
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, LOCK)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RegistryError(f"{path} exists: another model_loop run is using this registry "
                                "(if none is running, delete the file)") from None
        try:
            os.write(fd, f"pid {os.getpid()}\n".encode())
            os.close(fd)
            yield
        finally:  # also on an exception or Ctrl+C
            if os.path.exists(path):
                os.remove(path)


def empty(config: dict[str, Any]) -> dict[str, Any]:
    return {"schema": SCHEMA, "config": config, "last_tick": None, "champion": None, "challenger": None,
            "versions": {}, "decisions": [], "history": [], "ticks": []}


def _number(version: str) -> int:
    return int(version[1:])


def latest_version(data: dict[str, Any]) -> str | None:
    """The version trained last (highest number), whatever became of it."""
    return max(data["versions"], key=_number, default=None)


def next_version(data: dict[str, Any]) -> str:
    last = latest_version(data)
    return f"v{(_number(last) if last else 0) + 1:03d}"


def add_version(data: dict[str, Any], version: str, *, cutoff: str, train_files: list[str], seed: int, created: str,
                bundle: str, status: str) -> None:
    """cutoff: the tick the version was trained at; created: the bundle's own timestamp (what the server reports);
    bundle: path below the registry folder, with forward slashes."""
    if version in data["versions"]:
        raise ValueError(f"version {version} is already registered")
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    data["versions"][version] = {"cutoff": cutoff, "train_files": list(train_files), "seed": seed, "created": created,
                                 "bundle": bundle, "status": status}


def set_champion(data: dict[str, Any], version: str, *, at: str, by: str) -> None:
    """by: "first" | "decision" | "rollback". The champion before it becomes "retired"."""
    if version not in data["versions"]:
        raise ValueError(f"unknown version {version}")
    old = data["champion"]
    if old and old != version:
        data["versions"][old]["status"] = "retired"
    data["versions"][version]["status"] = "champion"
    data["champion"] = version
    if data["challenger"] == version:
        data["challenger"] = None
    data["history"].append({"at": at, "champion": version, "by": by})


def set_challenger(data: dict[str, Any], version: str | None) -> None:
    if version is not None:
        data["versions"][version]["status"] = "challenger"
    data["challenger"] = version


def reject(data: dict[str, Any], version: str) -> None:
    data["versions"][version]["status"] = "rejected"
    if data["challenger"] == version:
        data["challenger"] = None


def record_decision(data: dict[str, Any], decision: dict[str, Any]) -> None:
    data["decisions"].append(decision)


def record_tick(data: dict[str, Any], at: str, actions: list[str]) -> None:
    data["ticks"].append({"at": at, "actions": list(actions)})
    data["last_tick"] = at


def previous_champion(data: dict[str, Any]) -> str | None:
    """The champion before the current one (what a rollback without --to goes back to)."""
    h = data["history"]
    return h[-2]["champion"] if len(h) >= 2 else None


def _brief(data: dict[str, Any], version: str | None) -> dict[str, Any] | None:
    if version is None:
        return None
    v = data["versions"][version]
    return {"version": version, "cutoff": v["cutoff"], "train_files": len(v["train_files"])}


def summary(data: dict[str, Any], limit: int = 10) -> dict[str, Any]:
    """What GET /model/loop shows of the registry: champion, challenger, last tick, the latest decisions."""
    return {"champion": _brief(data, data["champion"]), "challenger": _brief(data, data["challenger"]),
            "last_tick": data["last_tick"], "decisions": data["decisions"][-limit:]}
