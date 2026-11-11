"""Persistent, namespaced warning history, independent of the rebuildable :Rag graph."""
import hashlib
import json

from .warning_input import required_text


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_key(*parts):
    return hashlib.sha256(canonical(parts).encode("utf-8")).hexdigest()


class WarningMemory:
    def __init__(self, driver, *, database, namespace):
        self.driver = driver
        self.database = required_text(database, "database")
        self.namespace = required_text(namespace, "namespace")
        self.bookmarks = driver.execute_query_bookmark_manager

    def _session(self):
        return self.driver.session(database=self.database, bookmark_manager=self.bookmarks)

    def initialize(self):
        """Run explicitly on the intended DB, before the application starts."""
        with self._session() as session:
            for label in ("WarningPattern", "WarningResolution", "WarningManualCase"):
                session.run(f"CREATE CONSTRAINT warning_{label.lower()}_id IF NOT EXISTS "
                            f"FOR (n:{label}) REQUIRE n.id IS UNIQUE").consume()
            session.run("CREATE INDEX warning_case_namespace IF NOT EXISTS "
                        "FOR (c:WarningManualCase) ON (c.namespace)").consume()

    def _id(self, key):
        return stable_key(self.namespace, key)

    @staticmethod
    def _attach_case(tx, record_id, namespace, case):
        refs = [case["source"]]
        for step in case["checks"] + case["actions"]:
            refs.append(step["cite"])
            if step.get("criterion"):
                refs.append(step["criterion"]["cite"])
        revision = sorted({(ref["doc"], ref["sha256"]) for ref in refs})
        case_id = stable_key(namespace, case["manual_case_id"], revision)
        row = tx.run(
            """MERGE (c:WarningManualCase {id:$id})
            ON CREATE SET c.namespace=$namespace, c.payload=$payload
            RETURN c.payload AS payload""",
            id=case_id, namespace=namespace, payload=canonical(case),
        ).single(strict=True)
        if row["payload"] != canonical(case):
            raise ValueError("manual_case_id already exists with different content")
        tx.run(
            """MATCH (r:WarningResolution {id:$record_id}), (c:WarningManualCase {id:$case_id})
            MERGE (r)-[:USED_WARNING_CASE]->(c)""", record_id=record_id, case_id=case_id,
        ).consume()

    def record(self, resolution, case=None):
        """A retry can repeat an identical close, but cannot change its evidence."""
        record_id = self._id(resolution["event_id"])
        payload = canonical(resolution)
        pattern = resolution["pattern_key"]

        def write(tx):
            row = tx.run(
                """MERGE (r:WarningResolution {id:$id})
                ON CREATE SET r.namespace=$namespace, r.payload=$payload
                RETURN r.payload AS payload""",
                id=record_id, namespace=self.namespace, payload=payload,
            ).single(strict=True)
            if row["payload"] != payload:
                # gun_id is optional trace metadata, including on records from before case-based counting.
                previous = json.loads(row["payload"])
                previous.pop("gun_id", None)
                if previous != {k: v for k, v in resolution.items() if k != "gun_id"}:
                    raise ValueError("event_id already has a different resolution")
            if pattern:
                tx.run(
                    """MATCH (r:WarningResolution {id:$id})
                    MERGE (p:WarningPattern {id:$pattern_id})
                    ON CREATE SET p.namespace=$namespace, p.pattern_key=$pattern
                    MERGE (r)-[:OBSERVED_WARNING_PATTERN]->(p)""",
                    id=record_id, pattern_id=self._id(pattern),
                    namespace=self.namespace, pattern=pattern,
                ).consume()
            if case:
                self._attach_case(tx, record_id, self.namespace, case)

        with self._session() as session:
            session.execute_write(write)
        return self.resolution(resolution["event_id"])

    def approve(self, event_id, case):
        record_id = self._id(event_id)

        def write(tx):
            row = tx.run(
                """MATCH (r:WarningResolution {id:$id})
                SET r.review_lock=true
                WITH r OPTIONAL MATCH (r)-[:USED_WARNING_CASE]->(c:WarningManualCase)
                RETURN r.payload AS payload, c.payload AS manual_case""", id=record_id,
            ).single()
            if row is None:
                raise ValueError("resolution does not exist")
            record = json.loads(row["payload"])
            if record["manual_case_id"] is not None or not record.get("manual_source"):
                raise ValueError("resolution is not a manual review submission")
            if row["manual_case"] and row["manual_case"] != canonical(case):
                raise ValueError("manual resolution already reviewed with a different case")
            self._attach_case(tx, record_id, self.namespace, case)

        with self._session() as session:
            session.execute_write(write)
        return self.resolution(event_id)

    def acknowledge_counterexample(self, event_id):
        def write(tx):
            row = tx.run(
                """MATCH (r:WarningResolution {id:$id})-[:USED_WARNING_CASE]->(c:WarningManualCase)
                RETURN r.payload AS payload, c.payload AS manual_case""", id=self._id(event_id),
            ).single()
            if not row:
                raise ValueError("a reviewed counterexample is required")
            record, case = json.loads(row["payload"]), json.loads(row["manual_case"])
            if (not record.get("pattern_key") or not record.get("rejected_auto_case_id")
                    or record["rejected_auto_case_id"] == case["manual_case_id"]):
                raise ValueError("record does not contradict an automatic card")
            tx.run("MATCH (r:WarningResolution {id:$id}) SET r.counterexample_reviewed=true",
                   id=self._id(event_id)).consume()

        with self._session() as session:
            session.execute_write(write)

    @staticmethod
    def _decode(row):
        return {**json.loads(row["payload"]),
                "manual_case": json.loads(row["manual_case"]) if row["manual_case"] else None,
                "counterexample_reviewed": bool(row["counterexample_reviewed"])}

    def resolution(self, event_id):
        with self._session() as session:
            row = session.execute_read(lambda tx: tx.run(
                """MATCH (r:WarningResolution {id:$id})
                OPTIONAL MATCH (r)-[:USED_WARNING_CASE]->(c:WarningManualCase)
                RETURN r.payload AS payload, c.payload AS manual_case,
                       r.counterexample_reviewed AS counterexample_reviewed""",
                id=self._id(event_id),
            ).single())
        return self._decode(row) if row else None

    def lookup(self, pattern_key):
        if not pattern_key:
            return []
        with self._session() as session:
            rows = session.execute_read(lambda tx: tx.run(
                """MATCH (:WarningPattern {id:$id})<-[:OBSERVED_WARNING_PATTERN]-(r:WarningResolution)
                OPTIONAL MATCH (r)-[:USED_WARNING_CASE]->(c:WarningManualCase)
                RETURN r.payload AS payload, c.payload AS manual_case,
                       r.counterexample_reviewed AS counterexample_reviewed""",
                id=self._id(pattern_key),
            ).data())
        return [self._decode(row) for row in rows]

    def catalog(self):
        with self._session() as session:
            rows = session.execute_read(lambda tx: tx.run(
                "MATCH (c:WarningManualCase {namespace:$namespace}) RETURN c.payload AS payload",
                namespace=self.namespace,
            ).data())
        return [json.loads(row["payload"]) for row in rows]

    def manual_chunk(self, chunk_id):
        """Resolve a server-issued evidence ID against the current manual graph."""
        with self._session() as session:
            row = session.execute_read(lambda tx: tx.run(
                """MATCH (c:Rag:Chunk {id:$id}) WHERE c.searchable=true
                RETURN c.id AS id, c.doc AS doc, c.page_number AS page, c.text AS text,
                       c.quote AS quote, c.review_needed AS review_needed, c.source_json AS source_json""",
                id=chunk_id,
            ).single())
        return row.data() if row else None
