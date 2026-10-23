"""Durable call budgets, candidate lineage, and the measured archive."""

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .core import candidate_id


class BudgetExhausted(RuntimeError):
    pass


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
                CREATE TABLE IF NOT EXISTS candidate_islands (
                    candidate TEXT, island TEXT, generation INTEGER, PRIMARY KEY(candidate,island));
                CREATE TABLE IF NOT EXISTS emissions (
                    generation INTEGER, writer INTEGER, candidate TEXT, island TEXT,
                    duplicate INTEGER, selection TEXT, PRIMARY KEY(generation,writer));
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
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            used = db.execute("SELECT COUNT(*) n, COALESCE(SUM(COALESCE(actual,reserved)),0) tokens FROM calls").fetchone()
            if used["n"] >= config["max_calls"] or used["tokens"] + tokens > config["max_total_tokens"]:
                raise BudgetExhausted("The API call or token budget is exhausted.")
            cursor = db.execute("INSERT INTO calls(role,reserved,status,created) VALUES(?,?,?,?)",
                                (role, tokens, "reserved", time.time()))
            return cursor.lastrowid

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

    def has_island(self, candidate, island, before_generation=None):
        with self.connect() as db:
            query = "SELECT 1 FROM candidate_islands WHERE candidate=? AND island=?"
            params = [candidate, island]
            if before_generation is not None:
                query += " AND generation<?"
                params.append(before_generation)
            return db.execute(query, params).fetchone() is not None

    def record_emission(self, generation, writer, key, island, duplicate, selection):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO emissions VALUES(?,?,?,?,?,?)",
                       (generation, writer, key, island, int(duplicate), json.dumps(selection)))

    def emissions(self):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM emissions ORDER BY generation,writer").fetchall()
        return [{**dict(r), "selection": json.loads(r["selection"])} for r in rows]

    def record(self, key, result):
        decision = result.get("decision", {})
        with self.connect() as db:
            db.execute("UPDATE candidates SET result=?,score=?,accepted=? WHERE id=?",
                       (json.dumps(result), decision.get("speedup"), int(decision.get("accepted", False)), key))

    def candidates(self, task=None):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM candidates" + (" WHERE task=?" if task else "") +
                              " ORDER BY COALESCE(score,0) DESC", (task,) if task else ()).fetchall()
        return [{**dict(r), "source": json.loads(r["source"]),
                 "result": json.loads(r["result"]) if r["result"] else None} for r in rows]

    def set(self, key, value):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (key, json.dumps(value)))

    def get(self, key, default=None):
        with self.connect() as db:
            row = db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def usage(self):
        with self.connect() as db:
            return dict(db.execute("SELECT COUNT(*) calls, COALESCE(SUM(COALESCE(actual,reserved)),0) tokens FROM calls").fetchone())
