"""Deterministic stand-in for the shared graph in offline API and UI tests."""
from copy import deepcopy


class TestWarningMemory:
    __test__ = False
    database = "test"

    def __init__(self):
        self.rows = {}

    def lookup(self, pattern_key):
        return [deepcopy(row) for row in list(self.rows.values()) if row["pattern_key"] == pattern_key]

    def catalog(self):
        return [deepcopy(row["manual_case"]) for row in list(self.rows.values()) if row["manual_case"]]

    def resolution(self, event_id):
        return deepcopy(self.rows.get(event_id))

    def record(self, resolution, case=None):
        event_id = resolution["event_id"]
        previous = self.resolution(event_id)
        if previous is not None:
            if any(previous[key] != value for key, value in resolution.items()):
                raise ValueError("event_id already has a different resolution")
            return previous
        self.rows[event_id] = {**deepcopy(resolution), "manual_case": deepcopy(case),
                               "counterexample_reviewed": False}
        return self.resolution(event_id)

    def approve(self, event_id, case):
        self.rows[event_id]["manual_case"] = deepcopy(case)
        return self.resolution(event_id)
