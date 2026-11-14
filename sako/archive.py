"""Keep several correct designs available as shared search parents."""

from __future__ import annotations

import math

from .core import digest
from .lessons import lesson_context, hypothesis_summary


def summarize_parent_use(assignments):
    """Count source choices. Keep repair parents separate from refinements."""
    if isinstance(assignments, dict):
        assignments = assignments.values()
    result = {"fresh_starts": 0, "archive_refinements": 0, "archive_merges": 0, "repairs": 0,
              "corrections_without_parent": 0, "by_parent": {}}
    for assignment in assignments:
        parent = assignment.get("parent_id")
        reason = assignment.get("selection", {}).get("reason")
        correction = reason == "repair" or bool(assignment.get("repair"))
        kind = ("repairs" if parent else "corrections_without_parent") if correction else (
            "archive_refinements" if parent else "fresh_starts")
        result[kind] += 1
        if reason == "archive_merge":
            result["archive_merges"] += 1
        parents = (assignment.get("selection", {}).get("provided_parent_ids") or [parent]) if parent else []
        for parent in parents:
            counts = result["by_parent"].setdefault(parent, {"archive_refinements": 0, "repairs": 0,
                                                            "active_refinements": 0, "completed_refinements": 0})
            counts[kind] += 1
            if kind == "archive_refinements":
                if assignment.get("status") == "completed":
                    counts["completed_refinements"] += 1
                elif assignment.get("status") != "failed":
                    counts["active_refinements"] += 1
    return result


def _resource_signature(row):
    reports = []
    for case in (row.get("result") or {}).get("cases", []):
        build = case.get("build") or {}
        reports.extend(build.get("resource_reports", []))
    # Coarse buckets keep minor compiler noise from creating fake design diversity.
    registers = [r.get("registers_per_thread") for r in reports
                 if type(r.get("registers_per_thread")) is int]
    shared = [r.get("static_shared_memory_bytes") for r in reports
              if type(r.get("static_shared_memory_bytes")) is int]
    return {"register_bucket": max(registers) // 16 if registers else None,
            "shared_memory_bucket": max(shared) // 4096 if shared else None}


class CandidateArchive:
    def __init__(self, store, limit=4, *, allow_history=False):
        if type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError("Use one to 32 active archive candidates.")
        self.store, self.limit = store, limit
        self.allow_history = allow_history

    def _bind(self, task, hardware):
        context, scope_key = lesson_context(task, hardware)
        key = "swarm_archive_context:" + task["id"]
        previous = self.store.get(key)
        if previous is not None and previous != context:
            raise ValueError("Archive context changed. Use a new run for different hardware or shapes.")
        self.store.set(key, context)
        return context, scope_key

    def refresh(self, task, hardware):
        context, scope_key = self._bind(task, hardware)
        blocked = self.store.get("blocked_model_candidates", {})
        rejected = self.store.get("rejected_model_candidates", {})
        best = self.store.get("best_model_measurement") or {}
        accepted_models = {entry["candidate_id"] for entry in best.get("manifest", {}).get("entries", [])}
        hypotheses = {}
        for emission in self.store.emissions():
            selection = emission.get("selection", {})
            hypothesis = selection.get("hypothesis")
            if hypothesis:
                design = selection.get("writer_design")
                hypotheses[emission["candidate"]] = ({**hypothesis, **design, "design_owner": "writer"}
                    if design else hypothesis)
        eligible, excluded = [], []
        for row in self.store.candidates(task["id"]):
            reason = ("model_incorrect_repair_only" if row["id"] in blocked else
                      "operation_not_correct" if not row["correct"] else
                      "no_finite_timing" if type(row["score"]) not in (int, float)
                      or not math.isfinite(row["score"]) or row["score"] <= 0 else None)
            if reason:
                excluded.append({"id": row["id"], "reason": reason})
                continue
            hypothesis = hypothesis_summary(hypotheses.get(row["id"], {"specialty": row["island"]}))
            diversity = {"specialty": hypothesis.get("specialty", row["island"]),
                         "change": hypothesis.get("change"), "resources": _resource_signature(row)}
            from .trial_evidence import model_saving
            from .designs import describe
            code = describe(row["source"]["source"]) if row.get("source", {}).get("source") else None
            saving = model_saving(task, row.get("result") or {})
            eligible.append({**(saving or {}), "id": row["id"], "score": row["score"], "correct": True,
                "accepted": bool(row["accepted"]), "parent_id": row.get("parent"),
                "model_status": "accepted" if row["id"] in accepted_models else
                                "rejected" if row["id"] in rejected else "untested",
                "hypothesis": hypothesis, "diversity": diversity, "code_design": code,
                "selection_reason": None})
        def rank(row):
            return row.get("estimated_model_saving_ms", row["score"])
        eligible.sort(key=lambda row: (-rank(row), row["id"]))
        selected, seen_families, seen_resources, seen_code = [], set(), set(), set()
        if eligible:
            best = eligible.pop(0)
            selected.append({**best, "selection_reason": "largest_runtime_saving"
                             if "estimated_model_saving_ms" in best else "fastest_correct"})
        while eligible and len(selected) < self.limit:
            for row in selected:
                seen_families.add((row["diversity"]["specialty"], row["diversity"]["change"]))
                seen_resources.add(digest(row["diversity"]["resources"]))
                seen_code.add((row.get("code_design") or {}).get("group"))
            def novelty(row):
                family = (row["diversity"]["specialty"], row["diversity"]["change"])
                resource = digest(row["diversity"]["resources"])
                code = row.get("code_design") or {}
                competitive = row["score"] >= selected[0]["score"] / 1.2
                return (bool(code) and competitive and code["group"] not in seen_code,
                        family not in seen_families, resource not in seen_resources, rank(row))
            row = max(eligible, key=novelty)
            reason = ("different_code_structure" if novelty(row)[0] else "different_hypothesis" if novelty(row)[1]
                      else "different_resources" if novelty(row)[2] else "next_fastest")
            selected.append({**row, "selection_reason": reason})
            eligible.remove(row)
        snapshot = {"schema_version": 1, "context": context, "scope_key": scope_key,
                    "limit": self.limit, "candidates": selected, "excluded": excluded,
                    "eligible_not_selected": [row["id"] for row in eligible],
                    "note": "Archive membership permits exploration. It does not accept a complete model gain."}
        self.store.set("swarm_archive:" + scope_key, snapshot)
        return selected

    def retrieve(self, task, hardware):
        return self.refresh(task, hardware)

    def parent(self, candidate_id, task, hardware, repair=False):
        self._bind(task, hardware)
        row = self.store.candidate(candidate_id)
        if row is None or row["task"] != task["id"]:
            raise ValueError("The requested parent is not a candidate for this task.")
        if repair or (self.allow_history and row["result"] is not None):
            return row
        eligible = {entry["id"] for entry in self.refresh(task, hardware)}
        if candidate_id not in eligible:
            raise ValueError("The requested parent is outside the active archive. Use explicit repair for failed code.")
        return row
