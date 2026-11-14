"""Durable call budgets, candidate lineage, and the measured archive."""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .core import candidate_id


class BudgetExhausted(RuntimeError):
    pass


class TaskBudgetExhausted(BudgetExhausted):
    """The active task used its share. Reserve the rest for later tasks."""


def auxiliary_role(role):
    """Keep provider metadata traffic separate from generation attempts."""
    return role.endswith((":count_tokens", ":models.get"))


def operation_correct(result):
    """Separate numerical correctness from the speed acceptance gate."""
    if not isinstance(result, dict) or result.get("error"):
        return False
    cases = result.get("cases")
    if result.get("performance_scope") == "cached_decode":
        timed = [i for i, c in enumerate(cases or []) if c.get("generation_step", {}).get("phase") == "decode"]
        stages = result.get("stages", {})
        initial = stages.get("initial_case_indices", [])
        return bool(timed and timed == result.get("timed_case_indices") and stages.get("outcome") == "confirmed"
                    and initial and len(initial) == stages.get("initial_correctness_calls")
                    and all(type(i) is int and 0 <= i < len(cases) and cases[i].get("correct") is True
                            and cases[i].get("inputs_unchanged") is True for i in [*initial, *timed]))
    if cases:
        return all(case.get("correct") is True for case in cases)
    return result.get("correct") is True or result.get("decision", {}).get("accepted") is True


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY, role TEXT, reserved INTEGER, actual INTEGER,
                    status TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY, task TEXT, island TEXT, generation INTEGER,
                    parent TEXT, source TEXT, result TEXT, score REAL, accepted INTEGER);
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS call_tasks (call_id INTEGER PRIMARY KEY, task_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS candidate_islands (
                    candidate TEXT, island TEXT, generation INTEGER, PRIMARY KEY(candidate,island));
                CREATE TABLE IF NOT EXISTS emissions (
                    generation INTEGER, writer INTEGER, candidate TEXT, island TEXT,
                    duplicate INTEGER, selection TEXT, PRIMARY KEY(generation,writer));
                CREATE TABLE IF NOT EXISTS candidate_parents (
                    candidate TEXT, parent TEXT, PRIMARY KEY(candidate,parent));
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, role, tokens, config):
        try:
            return self._reserve(role, tokens, config)
        except TaskBudgetExhausted:
            # A rejected request still reveals its exact input + output headroom.
            # Save it after the failed reservation transaction has rolled back.
            category = ('coordinator' if role == 'coordinator' else
                        'writer' if role.startswith('writer_') else None)
            if category and tokens > 0:
                with self.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    row = db.execute("SELECT value FROM state WHERE key='request_reservation_floors'").fetchone()
                    floors = json.loads(row[0]) if row else {}
                    floors[category] = max(tokens, floors.get(category, 0))
                    db.execute("INSERT OR REPLACE INTO state VALUES('request_reservation_floors',?)",
                               (json.dumps(floors),))
            raise

    def _reserve(self, role, tokens, config):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            used = db.execute("SELECT COUNT(*) n, COALESCE(SUM(COALESCE(actual,reserved)),0) tokens FROM calls").fetchone()
            cap = config.get("max_calls")
            generation_calls = db.execute("SELECT COUNT(*) FROM calls WHERE role NOT LIKE '%:count_tokens' AND role NOT LIKE '%:models.get'").fetchone()[0]
            if ((not auxiliary_role(role) and cap is not None and generation_calls >= cap)
                    or used["tokens"] + tokens > config["max_total_tokens"]):
                raise BudgetExhausted("The API call or token budget is exhausted.")
            shield = db.execute("SELECT value FROM state WHERE key='budget_reserve'").fetchone()
            shield = json.loads(shield[0]) if shield else None
            if shield and not auxiliary_role(role) and role not in shield.get("allowed_roles", []):
                protected = max(0, int(shield.get("role_protected_tokens", {}).get(
                    role, shield.get("protected_tokens", 0))))
                if used["tokens"] + tokens + protected > config["max_total_tokens"]:
                    raise BudgetExhausted("The remaining tokens are reserved for review and repair.")
            phase = db.execute("SELECT value FROM state WHERE key='phase_budget'").fetchone()
            limits = None
            if phase:
                limits = json.loads(phase[0])
                if limits and ((limits.get("call_ceiling") is not None and used["n"] >= limits["call_ceiling"])
                               or (limits.get("token_ceiling") is not None and used["tokens"] + tokens > limits["token_ceiling"])):
                    raise TaskBudgetExhausted("The active task used its reserved budget share.")
            cursor = db.execute("INSERT INTO calls(role,reserved,status,created) VALUES(?,?,?,?)",
                                (role, tokens, "reserved", time.time()))
            if limits and limits.get("task_id"):
                db.execute("INSERT INTO call_tasks VALUES(?,?)", (cursor.lastrowid, limits["task_id"]))
            return cursor.lastrowid

    def task_token_usage(self):
        with self.connect() as db:
            return dict(db.execute("SELECT t.task_id,SUM(COALESCE(c.actual,c.reserved)) FROM call_tasks t "
                                   "JOIN calls c ON c.id=t.call_id GROUP BY t.task_id"))

    def finish_call(self, call, actual=None, status="complete"):
        with self.connect() as db:
            db.execute("UPDATE calls SET actual=?,status=? WHERE id=?", (actual, status, call))

    def add(self, candidate, island, generation, parent):
        key = candidate_id(candidate)
        with self.connect() as db:
            inserted = db.execute("INSERT OR IGNORE INTO candidates VALUES(?,?,?,?,?,?,NULL,NULL,0)",
                (key, candidate["task_id"], island, generation, parent, json.dumps(candidate))).rowcount
            db.execute("INSERT OR IGNORE INTO candidate_islands VALUES(?,?,?)", (key, island, generation))
        return key, bool(inserted)

    def record_emission(self, generation, writer, key, island, duplicate, selection):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO emissions VALUES(?,?,?,?,?,?)",
                       (generation, writer, key, island, int(duplicate), json.dumps(selection)))
            for parent in selection.get("provided_parent_ids", selection.get("parent_ids", [selection.get("candidate_id")])):
                if parent and parent != key:
                    db.execute("INSERT OR IGNORE INTO candidate_parents VALUES(?,?)", (key, parent))

    def emissions(self):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM emissions ORDER BY generation,writer").fetchall()
        return [{**dict(r), "selection": json.loads(r["selection"])} for r in rows]

    def record(self, key, result):
        decision = result.get("decision", {})
        with self.connect() as db:
            db.execute("UPDATE candidates SET result=?,score=?,accepted=? WHERE id=?",
                       (json.dumps(result), result.get("search_score", decision.get("speedup")),
                        int(decision.get("accepted", False)), key))

    def candidates(self, task=None):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM candidates" + (" WHERE task=?" if task else "") +
                              " ORDER BY COALESCE(score,0) DESC", (task,) if task else ()).fetchall()
        return [self._candidate(row) for row in rows]

    @staticmethod
    def _candidate(row):
        result = json.loads(row["result"]) if row["result"] else None
        return {**dict(row), "source": json.loads(row["source"]),
                "result": result, "correct": operation_correct(result)}

    def candidate(self, identity):
        """Read one candidate without loading every source and timing record."""
        with self.connect() as db:
            row = db.execute("SELECT * FROM candidates WHERE id=?", (identity,)).fetchone()
        return self._candidate(row) if row else None

    def candidate_summaries(self):
        """Status needs no kernel source or raw timing arrays."""
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT id,task,island,score,accepted FROM candidates ORDER BY rowid")]

    def set(self, key, value):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (key, json.dumps(value)))

    def set_many(self, values):
        with self.connect() as db:
            db.executemany("INSERT OR REPLACE INTO state VALUES(?,?)",
                           [(key, json.dumps(value)) for key, value in values.items()])

    def get(self, key, default=None):
        with self.connect() as db:
            row = db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def usage(self):
        with self.connect() as db:
            return dict(db.execute("SELECT COUNT(*) calls, COALESCE(SUM(COALESCE(actual,reserved)),0) tokens FROM calls").fetchone())

    def call_stats(self):
        """Report paid use, unresolved outcomes, and request classes separately."""
        with self.connect() as db:
            rows = db.execute("SELECT role,reserved,actual,status FROM calls").fetchall()
        result = {"generation_calls": 0, "auxiliary_calls": 0, "reported_tokens": 0,
                  "reserved_tokens": 0, "unresolved_tokens": 0, "inflight_tokens": 0, "per_role": {}}
        for row in rows:
            result["auxiliary_calls" if auxiliary_role(row["role"]) else "generation_calls"] += 1
            reserved, actual = row["reserved"] or 0, row["actual"]
            result["reserved_tokens"] += reserved
            if actual is None:
                result["unresolved_tokens"] += reserved
                if row["status"] == "reserved":
                    result["inflight_tokens"] += reserved
            else:
                result["reported_tokens"] += actual
            role = result["per_role"].setdefault(row["role"],
                {"calls": 0, "tokens": 0, "reserved": 0, "unknown": 0, "pending": 0})
            role["calls"] += 1
            role["tokens"] += actual if actual is not None else reserved
            role["reserved"] += reserved
            role["unknown"] += int(actual is None)
            role["pending"] += int(row["status"] == "reserved")
        return result

    def budget_snapshot(self, config):
        usage = self.usage()
        return {**usage, **self.call_stats(), "max_total_tokens": config["max_total_tokens"],
                "remaining_tokens": max(0, config["max_total_tokens"] - usage["tokens"])}
