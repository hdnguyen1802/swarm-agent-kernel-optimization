"""Profile the accepted model state after a bottleneck search finishes."""

import copy
from pathlib import Path

from .compiled_ranking import profile_compiled
from .core import write_json


def reprofile(config, catalog, manifest, candidate_root, output_dir, *, session=None):
    from .model import load_model, clone_structure, apply_manifest
    from .hierarchy import enabled
    if not enabled(config):
        raise ValueError("Reprofiling requires a bottleneck tree.")
    from .layer_checks import replacement_paths
    replacement_paths(manifest, catalog)
    config = {**config, "revision": catalog["config"]["revision"]}
    model = load_model(config) if session is None else session.get_model(config)
    branch = clone_structure(model)
    apply_manifest(branch, manifest, catalog, candidate_root, config=config)
    tasks = copy.deepcopy(catalog["tasks"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    from .measurement import enabled as adaptive
    ranking, compiled = profile_compiled(branch, config, tasks, output,
        **({"starting": catalog.get("starting_baseline"), "hardware": catalog["hardware"]} if adaptive(config) else {}))
    write_json(output / "operation_profile.json", ranking)
    write_json(output / "compiled_profile.json", compiled)
    result = {"model_state": "provided_replacements" if manifest["entries"] else "original_baseline",
        "manifest": manifest, "task_priorities": {t["id"]: {"priority_ms": t["priority_ms"],
            "workload_medians_ms": t.get("workload_medians_ms", []),
            **({"compiled_evidence": {"generation_phase_ms": t["compiled_evidence"]["generation_phase_ms"]}}
               if "generation_phase_ms" in t.get("compiled_evidence", {}) else {}),
            "excluded_reason": t.get("ranking_excluded_reason")} for t in tasks},
        "workload_times": [{**{k: w[k] for k in ("workload", "median_ms")},
                            "timing_reference": "saved_original_start" if adaptive(config) else "profiled_model"}
                           for w in compiled.get("workloads", [])],
        "mapped_gpu_ms": ranking["mapped_gpu_ms"], "total_gpu_ms": ranking["total_gpu_ms"],
        "report": str(output / "operation_profile.json")}
    write_json(output / "result.json", result)
    return result
