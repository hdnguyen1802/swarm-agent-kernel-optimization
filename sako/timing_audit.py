"""Compare the same kernel through direct and wrapped paths before search."""

import math
import statistics
from pathlib import Path

from .core import summary


def control_check(result):
    left, right = (summary(result[name]["samples_ms"]) for name in ("baseline", "candidate"))
    ratio = left["median_ms"] / right["median_ms"]
    errors = [statistics.stdev(v["samples_ms"]) / math.sqrt(len(v["samples_ms"])) / v["median_ms"]
              for v in (left, right)]
    uncertainty = 3 * math.sqrt(sum(v * v for v in errors))
    # Calibration reports its own uncertainty. It cannot relax a speed gate.
    return {"ratio": ratio, "estimated_noise_fraction": uncertainty,
            "passed": uncertainty <= .02 and abs(math.log(ratio)) <= max(.01, uncertainty)}


def audit(candidate, task, fixture_path, config, *, identity_control=False, compiler_visible=False):
    import torch
    from .gpu import compile_candidate, custom_op, model_operation, paired_timings, check_output
    from .shape_selection import groups
    fixture = torch.load(Path(fixture_path), map_location="cpu", weights_only=True)
    direct = compile_candidate(candidate)
    wrapped = custom_op(candidate, "silu" if identity_control else task["kind"])
    # Short controls need enough captured work for the GPU event clock.
    # This diagnostic count never changes an acceptance measurement.
    options = {**config, "repeats": max(100, config["repeats"])}
    results = []
    with torch.inference_mode():
        for group in groups(task).values():
            index = group["indices"][0]
            case = fixture["cases"][index]
            tensors = [t.cuda() for t in case["tensors"]]
            saved = [t.clone() for t in tensors]
            scalars = case["scalars"]
            raw, adapter = lambda: direct(tensors, scalars), lambda: wrapped(tensors, scalars)
            expected = tensors[0] if identity_control else case["eager_output"].cuda()
            checks = [check_output(fn(), expected, config) for fn in (raw, adapter)]
            if not all(c["passed"] for c in checks) or not all(torch.equal(a, b) for a, b in zip(tensors, saved)):
                raise ValueError("The timing control kernel failed accuracy or changed its inputs.")
            event_control = paired_timings(raw, raw, options)
            graph_control = paired_timings(raw, raw, options, mode="cuda_graph")
            event_wrapper = paired_timings(raw, adapter, options)
            graph_wrapper = paired_timings(raw, adapter, options, mode="cuda_graph")
            compiler = {}
            if compiler_visible:
                visible = model_operation(candidate, task["kind"], {**config, "triton_integration": "direct"})
                compiled = torch.compile(lambda: visible(tensors, scalars), fullgraph=True,
                                         dynamic=False, mode=config["compile_mode"])
                checked = check_output(compiled(), expected, config)
                if not checked["passed"]:
                    raise ValueError("The compiler-visible control failed accuracy.")
                compiler = {"compiled_accuracy": checked,
                    "event_compiler": paired_timings(raw, compiled, options),
                    "graph_compiler": paired_timings(raw, compiled, options, mode="cuda_graph")}
            if not all(torch.equal(a, b) for a, b in zip(tensors, saved)):
                raise ValueError("The timing control changed its inputs during measurement.")
            results.append({"fixture_index": index, "signature": group["signature"], "accuracy": checks,
                "event_control": event_control, "graph_control": graph_control,
                "event_wrapper": event_wrapper, "graph_wrapper": graph_wrapper,
                "graph_control_check": control_check(graph_control),
                "graph_wrapper_check": control_check(graph_wrapper)})
            results[-1].update(compiler)
    passed = bool(results) and all(r["graph_control_check"]["passed"] and r["graph_wrapper_check"]["passed"] for r in results)
    return {"scope": "timing_calibration", "status": "passed" if passed else "unstable",
            "control_kind": "identity_copy" if identity_control else "operation_seed",
            "calibration_repeats": options["repeats"],
            "acceptance_evidence": False, "cases": results,
            "note": "Same source and inputs on both sides. Full-model timing retains the deployment path. No model gain is claimed."}
