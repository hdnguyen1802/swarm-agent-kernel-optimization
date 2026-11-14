"""Show saved search events in the order the controller recorded them."""

from datetime import datetime, timezone

from rich.table import Table
from rich.text import Text
from textual.widgets import RichLog

from .runs import decision_outcome, speed_ratio
from .ratios import speed_text, timing_text

LIME = "#b3ec65"
MUTED = "#96a18f"
AMBER = "#efb678"
INK = "#e6ebe2"


def brief(value, limit=160):
    if isinstance(value, list):
        value = " · ".join(str(item) for item in value)
    words = " ".join(str(value or "").split())
    line = "".join(char for char in words if char.isprintable())
    return line if len(line) <= limit else line[:limit - 1].rstrip() + "…"


def clock_label(utc):
    try:
        return datetime.fromtimestamp(utc, timezone.utc).strftime("%H:%M:%SZ")
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def baseline_label(baseline):
    baseline = baseline or {}
    if baseline.get("kind") == "torch_compile":
        return "torch.compile"
    if baseline.get("kind") == "kept_kernel":
        return "kept kernel"
    if baseline.get("kind") == "per_shape":
        shapes = list((baseline.get("shapes") or {}).values())
        if shapes and not any(shapes):
            return "torch.compile"
        return "shape baselines"
    return "trial baseline"


def kernel_result(event, writer):
    if event.get("stages", {}).get("outcome") == "screened_slow":
        return [("×", "Speed screen rejected the kernel on every tested input", AMBER)]
    decision = event.get("decision") or {}
    passed = decision.get("accepted")
    partial = passed is False and bool(event.get("shapes_passed"))
    state = (f"partial {event['shapes_passed']}/{event['shapes_tested']}" if partial else
             {True: "passed", False: "failed"}.get(passed, "no decision"))
    identity = (event.get("candidate_id") or "")[:12]
    title = f"Kernel test {state}  ·  {identity}"
    ratio = speed_ratio(decision.get("speedup"))
    shape_results = event.get("shape_results", [])
    scoped = any(row["shape_id"] != "all" for row in shape_results)
    if not scoped:
        if ratio != "—":
            title += f"  ·  {ratio} vs {baseline_label(event.get('baseline'))}"
        else:
            title += "  ·  no speed ratio"
    rows = [({True: "✓", False: "×"}.get(passed, "•"), title,
             {True: LIME, False: AMBER}.get(passed, INK))]
    detail = [writer] if writer else []
    if partial:
        detail.append("only passing shapes can be used")
    if event.get("correct") is True:
        detail.append("output correct")
    reasons = decision.get("reasons") or []
    reason = event.get("error") or (reasons[0] if reasons else "")
    if reason:
        detail.append(brief(reason))
    if detail:
        rows.append(("└─", "  ·  ".join(detail), MUTED))
    if scoped:
        for shape in shape_results:
            passed = shape.get("accepted")
            label = f"{shape['shape']}  ·  {speed_text(shape.get('speedup'))} vs {shape['baseline']}"
            label += "  ·  " + ({True: "passed", False: "failed"}.get(passed, "no decision"))
            timing = timing_text(shape)
            if timing != "—":
                label += f"  ·  {timing} µs"
            rows.append(("└─", label, LIME if passed else AMBER))
            if shape.get("reason"):
                rows.append(("", brief(shape["reason"]), MUTED))
    return rows


class Timeline(RichLog):
    def __init__(self):
        super().__init__(id="timeline", min_width=1, wrap=True, auto_scroll=False)
        self.event_offset = 0
        self.strategies = {}
        self.writers = {}
        self.history = []
        self.preview = []
        self.last_width = 0

    def plan_rows(self, event):
        strategies = event.get("strategies") or []
        count = event.get("hypothesis_count", len(strategies))
        if count:
            label = "strategy" if count == 1 else "strategies"
            rows = [("◆", f"Coordinator proposed {count} {label}", LIME)]
        else:
            action = brief(event.get("action")).replace("_", " ")
            rows = [("◆", f"Coordinator decision  ·  {action}", INK)]
        for index, item in enumerate(strategies):
            name = brief(item.get("algorithm") or item.get("change") or item.get("claim"))
            label = "  ·  ".join(filter(None, (brief(item.get("specialty")), name)))
            if item.get("id"):
                self.strategies[item["id"]] = label
            estimate = speed_ratio(item.get("expected_operation_speedup"))
            if estimate != "—":
                label += f"  ·  estimate {estimate}"
            rows.append(("└─" if index == len(strategies) - 1 else "├─", label, INK))
        if not strategies and event.get("reason"):
            rows.append(("└─", brief(event["reason"]), MUTED))
        return rows

    def event_rows(self, event):
        kind = event.get("kind")
        task = brief(event.get("task_id"))
        region = brief(event.get("region_id"))
        reason = brief(event.get("reason"))
        assignment = event.get("assignment_id")
        slot = event.get("slot")
        writer = self.writers.get(assignment, f"Writer {slot}" if slot is not None else brief(assignment))
        if kind == "task_started":
            rows = [("●", f"Task  {task}", LIME)] + ([("└─", f"Region  {region}", MUTED)] if region else [])
            if event.get("allocated_tokens") is not None:
                rows.append(("└─", f"Token allocation  {event['allocated_tokens']:,}", MUTED))
            return rows
        if kind == "operation_search_split":
            return [("●", f"Separate searches for {task}", LIME),
                    ("└─", f"Measured ratio gap  {event['ratio_gap']:.2f}×", MUTED)]
        if kind == "task_revisited":
            return [("●", f"Revisit {event['count']}  ·  {task}", LIME),
                    ("└─", f"{event['remaining_tokens']:,} tokens available", MUTED)]
        if kind == "coordinator_decision":
            return self.plan_rows(event)
        if kind == "writer_started":
            strategy = self.strategies.get(event.get("hypothesis_id"), "")
            label = f"Writer {slot}" if slot is not None else brief(assignment) or "Writer"
            self.writers[assignment] = "  ·  ".join(filter(None, (label, strategy)))
            rows = [("›", f"{label} started" + (f"  ·  {strategy}" if strategy else ""), INK)]
            if event.get("merge_parent_ids"):
                rows.append(("└─", "Combining ideas from " + " + ".join(p[:12] for p in event["merge_parent_ids"]), LIME))
            return rows
        if kind == "writer_finished":
            status = brief(event.get("status"))
            action = "generated a kernel" if status == "generated" else status or "finished"
            label = f"Writer {slot}" if slot is not None else writer
            return [("›", f"{label} {action}", AMBER if status == "failed" else MUTED)]
        if kind in {"writer_rejected", "writer_paused", "coordinator_plan_rejected", "hypothesis_rejected"}:
            label = {"writer_rejected": "Writer rejected", "writer_paused": "Writer paused",
                     "coordinator_plan_rejected": "Plan rejected", "hypothesis_rejected": "Strategy rejected"}[kind]
            return [("×", "  ·  ".join(filter(None, (label, writer, reason))), AMBER)]
        if kind == "candidate_evaluated":
            return kernel_result(event, writer)
        if kind == "kernel_kept":
            label = f"Kept kernel  {(event.get('candidate_id') or '')[:12]}"
            if event.get("acceptance_scope") == "measured_shapes_only":
                label += "  ·  measured shapes"
            return [("★", label, LIME)]
        if kind == "target_model_selection":
            winner = event.get("winner")
            count = len(event.get("trials", []))
            return [("★", f"{task}  ·  {count} model trials  ·  " +
                     (f"kept {winner[:10]}" if winner else "kept previous model"), LIME if winner else MUTED)]
        if kind in {"model_checked", "model_repair_checked", "region_alternative", "region_integration"}:
            decision = event.get("decision") or {}
            outcome = decision_outcome(decision, event.get("manifest_entries"))
            style = LIME if decision.get("accepted") is True and event.get("manifest_entries") else MUTED
            rows = [("●", f"Full model vs torch.compile  ·  {outcome}", style)]
            detail = brief((decision.get("reasons") or [""])[0])
            if region or detail:
                rows.append(("└─", "  ·  ".join(filter(None, (region, detail))), MUTED))
            return rows
        if kind == "hypothesis_outcome" and event.get("status") == "correction_pending":
            return [("↻", f"Correction pending  ·  {writer}", AMBER)]
        if kind in {"profile_result", "profile_skipped"}:
            status = event.get("status") or "skipped"
            target = brief(event.get("candidate_id"), 12) or "baseline"
            return [("›", "  ·  ".join(filter(None, (f"Kernel profile {status}", target, reason))), MUTED)]
        if kind in {"task_finished", "stop_decision", "region_completed", "integration_repair",
                    "integration_tasks_reopened", "coordinator_deferred"}:
            label = {"stop_decision": "Search stopped"}.get(kind, kind.replace("_", " ").capitalize())
            detail = event.get("code") or task or region
            rows = [("●", "  ·  ".join(filter(None, (label, detail))), INK)]
            if reason:
                rows.append(("└─", reason, MUTED))
            return rows
        return []

    def write_rows(self, rows, follow):
        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(width=9, style=MUTED, no_wrap=True)
        table.add_column(width=2, no_wrap=True)
        table.add_column(ratio=1, overflow="fold")
        for stamp, mark, label, style in rows:
            table.add_row(Text(stamp), Text(mark, style=style), Text(label, style=style))
        self.write(table, expand=True, scroll_end=follow)

    def show_snapshot(self, data):
        from ..operation_names import operation_name
        names = data.get("operation_names", {})
        batch = data.get("events")
        follow = self.is_vertical_scroll_end
        added = []
        if batch is not None:
            start, count, events = batch
            if start < self.event_offset:
                self.clear()
                self.history.clear()
                self.preview.clear()
                self.strategies.clear()
                self.writers.clear()
            for event in events:
                event = {**event, **{k:operation_name(event[k], names) for k in ("task_id", "region_id") if event.get(k)}}
                stamp = clock_label(event.get("utc"))
                for index, row in enumerate(self.event_rows(event)):
                    added.append((stamp if index == 0 else "", *row))
            self.event_offset = count
        if added:
            if self.preview:
                self.clear()
                self.preview.clear()
            self.history.extend(added)
            self.write_rows(added, follow)
        elif not self.history:
            rows = []
            for identity, task, writer, outcome, ratio, reason in reversed(data["candidates"]):
                label = "  ·  ".join(filter(None, (operation_name(task, names), identity[:12], writer, outcome, ratio if ratio != "—" else None)))
                rows.append(("", "›", label, {"passed": LIME, "failed": AMBER}.get(outcome, INK)))
                if reason:
                    rows.append(("", "└─", brief(reason), MUTED))
            if not rows:
                rows = [("", "›", "No search events recorded yet.", MUTED)]
            if rows != self.preview:
                self.clear()
                self.preview = rows
                self.write_rows(rows, follow)

    def on_resize(self, event):
        super().on_resize(event)
        width = event.size.width
        rows = self.history or self.preview
        if width and self.last_width and width != self.last_width and rows:
            follow = self.is_vertical_scroll_end
            position = self.scroll_y
            self.clear()
            self.write_rows(rows, follow)
            if not follow:
                self.scroll_to(y=position, animate=False, immediate=True)
        if width:
            self.last_width = width
