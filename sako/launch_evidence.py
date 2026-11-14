"""Identify launches made by a warmed evaluator outside the timing interval."""

import re
import tempfile
from contextlib import nullcontext
from pathlib import Path

from .core import read_json


def dimensions(value):
    if isinstance(value, str):
        if not re.fullmatch(r"\s*\(?\s*\d+(?:\s*,\s*\d+){0,2}\s*\)?\s*", value):
            return None
        value = [int(v) for v in re.findall(r"\d+", value)]
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 3:
        return None
    if any(type(v) is not int or v <= 0 for v in value):
        return None
    return list(value) + [1] * (3 - len(value))


def _kernel_events(trace):
    return [e for e in trace.get("traceEvents", []) if e.get("cat") == "kernel" and e.get("ph") == "X"]


def _launch_records(events):
    # Kineto correlation IDs follow the host launch order, even on several streams.
    events.sort(key=lambda e: (e.get("args", {}).get("correlation", e.get("ts", 0)), e.get("ts", 0)))
    return [{"name": e["name"], "grid": dimensions(e.get("args", {}).get("grid")),
             "block": dimensions(e.get("args", {}).get("block"))} for e in events]


def trace_launches(trace):
    return _launch_records(_kernel_events(trace))


def tuning_capture(runner):
    observe = getattr(runner, "_sako_observe_tuning", None)
    return observe() if observe else nullcontext([])


def one_invocation(launches, repeats):
    size, remainder = divmod(len(launches), repeats)
    if not size or remainder or launches[:size] * repeats != launches:
        raise ValueError("The measured calls have no consistent sequence of GPU launches.")
    return launches[:size]


def graph_execution(trace, repeats):
    """Bind GPU activities to the host API that submitted the measured graph."""
    calls = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X" and
             e.get("cat") in ("cuda_runtime", "cuda_driver")]
    host_ids, graph_ids = set(), set()
    for call in calls:
        correlation = call.get("args", {}).get("correlation")
        if type(correlation) is int and correlation > 0:
            host_ids.add(correlation)
            if re.fullmatch(r"(?:cuda|cu)GraphLaunch(?:_ptsz)?(?:_v\d+)?", call.get("name", "")):
                graph_ids.add(correlation)
    if len(graph_ids) != 1:
        raise ValueError("The trace does not identify one CUDA graph submission.")
    graph, additional = [], []
    for event in _kernel_events(trace):
        correlation = event.get("args", {}).get("correlation")
        if correlation not in host_ids:
            raise ValueError("A GPU launch has no matching host call in the graph replay trace.")
        if correlation in graph_ids:
            graph.append(event)
        else:
            additional.append(event)
    return {"launches": one_invocation(_launch_records(graph), repeats), "graph_replay": {
        "operations_per_replay": repeats, "graph_node_launches": len(graph),
        "additional_timed_launches_per_replay": _launch_records(additional)}}


def observe_execution(invoke, runner=None, *, captured_tuners=None, repeats=1):
    import torch
    try:
        with tempfile.TemporaryDirectory() as folder:
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                with tuning_capture(runner) if captured_tuners is None else nullcontext(captured_tuners) as tuners:
                    invoke()
                torch.cuda.synchronize()
            path = Path(folder) / "launches.json"
            profiler.export_chrome_trace(str(path))
            trace = read_json(path)
            observed = (graph_execution(trace, repeats) if captured_tuners is not None else
                        {"launches": one_invocation(trace_launches(trace), repeats)})
        return {"status": "observed", **observed, "tuners": tuners}
    except Exception as error:
        # A diagnostic failure must leave the independent timing result intact.
        return {"status": "unavailable", "reason": str(error)}
