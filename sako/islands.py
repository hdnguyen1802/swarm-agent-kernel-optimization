"""Explicit parent selection and scheduled transfers between search islands."""

import math

from .core import write_json


class ParentSelectionError(ValueError):
    pass


STRATEGIES = {
    "memory": {
        "goal": "Reduce data movement and improve memory access.",
        "experiments": ["Compare contiguous and strided access. Preserve supported strides.",
                        "Try aligned vector loads with correct handling of remaining elements.",
                        "Reuse data in registers or shared memory. Avoid unnecessary copies."],
        "report": "Explain which data moves less, or which access pattern improves, and why."
    },
    "parallel": {
        "goal": "Improve how threads and blocks share the work.",
        "experiments": ["Compare block sizes, tile sizes, and work per thread.",
                        "Balance register use, shared memory, and active blocks.",
                        "Reduce idle threads and synchronization. Handle all supplied shapes."],
        "report": "State the launch and tile changes. Explain the expected effect on parallel execution."
    },
    "compute": {
        "goal": "Improve arithmetic execution while preserving numerical behavior.",
        "experiments": ["Try sm_70 Tensor Core configurations where the operation permits them.",
                        "Compare arithmetic and reduction schedules with FP32 accumulation where required.",
                        "Preserve FP16 rounding boundaries. Inspect the reported numerical errors first."],
        "report": "State the arithmetic change and how it preserves the output contract."
    },
    "fusion": {
        "goal": "Reduce launches and intermediate tensors by combining operations.",
        "experiments": ["Look for an epilogue or adjacent operations that can share one launch.",
                        "Avoid materializing intermediate data where it is safe.",
                        "Preserve required FP16 rounding and synchronization between dependent operations."],
        "report": "State which operations combine, which intermediate data disappears, and the numerical risks."
    },
}


class ParentPolicy:
    def __init__(self, store, limits, directory, task_ids):
        self.store, self.limits, self.directory = store, limits, directory
        self.task_ids = set(task_ids)

    def island(self, slot):
        if type(slot) is not int or not 0 <= slot < 8:
            raise ParentSelectionError("Invalid writer slot.")
        return self.limits["islands"][slot // 2]

    def row(self, key):
        rows = [r for r in self.store.candidates() if r["id"] == key]
        if not rows:
            raise ParentSelectionError("Parent candidate is missing.")
        return rows[0]

    def eligible(self, row, generation):
        return (row["accepted"] and row["task"] in self.task_ids
                and row["generation"] < generation and row["score"] is not None
                and math.isfinite(row["score"]) and row["score"] > 0
                and row["id"] not in self.store.get("blocked_model_candidates", {}))

    def event_id(self, completed_round):
        return f"after_round_{completed_round}"

    def plan_migration(self, completed_round, pending):
        if completed_round <= 0 or completed_round % self.limits["migration_interval"]:
            return None
        key = "migration:" + self.event_id(completed_round)
        saved = self.store.get(key)
        if saved is not None:
            return saved
        transfers = []
        rows = sorted(self.store.candidates(), key=lambda r: (-(r["score"] or 0), r["id"]))
        for index, island in enumerate(self.limits["islands"]):
            slot = index * 2 + 1
            transfer = {"slot": slot, "receiving_island": island, "candidate_id": None,
                        "source_island": None, "task_id": None}
            if any(r["slot"] == slot for r in pending.values()):
                transfer.update(status="skipped", reason="repair_priority")
            else:
                candidates = [r for r in rows if self.eligible(r, completed_round)
                              and not self.store.has_island(r["id"], island, completed_round)]
                if candidates:
                    row = candidates[0]
                    transfer.update(status="planned", reason="scheduled_migration", candidate_id=row["id"],
                                    source_island=row["island"], task_id=row["task"])
                else:
                    transfer.update(status="skipped", reason="no_eligible_candidate")
            transfers.append(transfer)
        event = {"id": self.event_id(completed_round), "completed_round": completed_round,
                 "target_generation": completed_round, "transfers": transfers}
        self.store.set(key, event)
        write_json(self.directory / "migrations" / (event["id"] + ".json"), event)
        return event

    def select(self, slot, generation, task_id, repair):
        island = self.island(slot)
        parent = None
        reason = "fresh_start"
        event_id = None
        skipped = None
        event = self.store.get("migration:" + self.event_id(generation))
        if repair:
            task_id, parent, reason = repair["task_id"], self.row(repair["candidate_id"]), "repair"
            if event:
                skipped = "repair_priority"
        else:
            transfer = next((t for t in event["transfers"] if t["slot"] == slot), None) if event else None
            if transfer and transfer["status"] == "planned":
                migrant = self.row(transfer["candidate_id"])
                if self.eligible(migrant, generation):
                    parent, task_id, reason, event_id = migrant, migrant["task"], "migration", event["id"]
                else:
                    skipped = "migrant_ineligible"
            elif transfer:
                skipped = transfer["reason"]
            if parent is None:
                own = [r for r in self.store.candidates(task_id) if self.eligible(r, generation)
                       and self.store.has_island(r["id"], island, generation)]
                parent = own[0] if own else None
                reason = "own_island" if parent else "fresh_start"
        source_island = (parent["island"] if reason == "migration" else island) if parent else None
        selection = {"round": generation + 1, "generation": generation, "slot": slot,
                     "receiving_island": island, "task_id": task_id,
                     "candidate_id": parent["id"] if parent else None,
                     "source_island": source_island, "origin_island": parent["island"] if parent else None,
                     "reason": reason, "migration_event": event_id, "migration_skipped": skipped}
        assignment = {"slot": slot, "island": island, "task_id": task_id,
                      "parent_id": selection["candidate_id"], "selection": selection, "repair": repair,
                      "repairs_used": repair["repairs_used"] + 1 if repair else 0}
        self.validate(assignment, generation)
        return assignment

    def validate(self, assignment, generation, check_eligibility=True):
        def require(condition, message):
            if not condition:
                raise ParentSelectionError(message)
        slot, task_id, parent_id = assignment["slot"], assignment["task_id"], assignment["parent_id"]
        island = self.island(slot)
        s = assignment.get("selection")
        require(isinstance(s, dict), "Parent selection record is missing.")
        require(assignment["island"] == island and task_id in self.task_ids, "Invalid island or task.")
        require(all(s.get(k) == v for k, v in {"round": generation + 1, "generation": generation,
                "slot": slot, "receiving_island": island, "task_id": task_id,
                "candidate_id": parent_id}.items()), "Assignment and parent record differ.")
        reason = s.get("reason")
        require(reason in {"fresh_start", "own_island", "migration", "repair"}, "Unknown parent selection reason.")
        if reason == "fresh_start":
            require(parent_id is None and s.get("source_island") is None
                    and s.get("origin_island") is None and assignment["repair"] is None,
                    "A fresh start must have no parent.")
        else:
            row = self.row(parent_id)
            require(row["task"] == task_id and s.get("origin_island") == row["island"], "Parent task or origin differs.")
            if reason in {"own_island", "repair"}:
                require(self.store.has_island(parent_id, island, generation) and s.get("source_island") == island,
                        "Foreign parent has no migration authorization.")
            if reason == "repair":
                r = assignment["repair"]
                require(isinstance(r, dict) and r["candidate_id"] == parent_id and r["task_id"] == task_id
                        and r["slot"] == slot and assignment["repairs_used"] == r["repairs_used"] + 1,
                        "Repair must keep its exact parent, task, and writer.")
            else:
                require(assignment["repair"] is None and (not check_eligibility or self.eligible(row, generation)),
                        "Parent is not eligible.")
            if reason == "migration":
                require(generation > 0 and generation % self.limits["migration_interval"] == 0,
                        "Migration is outside its schedule.")
                event = self.store.get("migration:" + str(s.get("migration_event")))
                require(event is not None and event["id"] == self.event_id(generation)
                        and event["completed_round"] == generation and event["target_generation"] == generation,
                        "Migration event is missing or invalid.")
                t = next((t for t in event["transfers"] if t["slot"] == slot), {})
                require(slot % 2 == 1 and t.get("status") == "planned" and t.get("candidate_id") == parent_id
                        and t.get("task_id") == task_id and t.get("receiving_island") == island
                        and t.get("source_island") == s.get("source_island") == row["island"] != island,
                        "Foreign parent does not match the migration event.")
        if reason != "migration":
            require(s.get("migration_event") is None, "Only a migration can use a migration authorization.")

    def save(self, assignments, generation):
        completed = self.store.get(f"submissions:{generation}", {})
        for a in assignments:
            self.validate(a, generation, check_eligibility=str(a["slot"]) not in completed)
        write_json(self.directory / "selections" / f"round-{generation + 1:04d}.json",
                   {"round": generation + 1, "selections": [a["selection"] for a in assignments]})
