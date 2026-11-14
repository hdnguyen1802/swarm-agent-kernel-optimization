"""One serialized worker per GPU. Agent code never sets benchmark results."""

import argparse
import contextlib
import os
import traceback
import time
from pathlib import Path

from .core import read_json, write_json, load_config, candidate_id, score_cases, digest, implementation_hash, validate_candidate


class InfrastructureError(RuntimeError):
    """The job environment differs from its prepared contract."""


@contextlib.contextmanager
def gpu_lock():
    import fcntl
    path = Path.home() / "sako/locks" / ("gpu-" + os.environ.get("CUDA_VISIBLE_DEVICES", "0") + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def execute(job, *, session=None):
    from .gpu import doctor
    from .model import profile, measure_model
    if job.get("implementation_hash") and job["implementation_hash"] != implementation_hash():
        raise InfrastructureError("Controller and worker source differ. Sync the current source before running jobs.")
    config = load_config(job["config"])
    action = job["action"]
    if action == "doctor":
        return doctor(config)
    if action == "profile":
        return profile(config, job["profile_dir"], session=session) if session is not None else profile(config, job["profile_dir"])
    catalog = read_json(Path(job["profile_dir"]) / "tasks.json")
    from .preparation import validate_catalog
    from .hardware import discover_hardware
    try:
        hardware = discover_hardware(config)
        validate_catalog(config, catalog, hardware)
    except (ValueError, RuntimeError) as error:
        raise InfrastructureError(str(error)) from error
    if action == "baseline":
        from .measurement import enabled as adaptive, cached_baseline_report
        if not adaptive(config):
            raise ValueError("The saved baseline action requires adaptive measurement.")
        return cached_baseline_report(config, catalog, hardware)
    if action == "timing_audit":
        from .timing_audit import audit
        candidate = read_json(job["candidate"])
        validate_candidate(candidate, config.get("kernel_backend", "triton"))
        task = next(t for t in catalog["tasks"] if t["id"] == candidate["task_id"])
        identity_control = job.get("identity_control", False)
        if type(identity_control) is not bool:
            raise InfrastructureError("Invalid timing control selection.")
        if identity_control:
            from .timing_controls import control_seed
            if candidate != control_seed(task):
                raise InfrastructureError("Only the trusted copy control can use the identity reference.")
        return audit(candidate, task, Path(job["profile_dir"]) / (task["id"] + ".pt"), config,
                     identity_control=identity_control,
                     **({"compiler_visible": True} if job.get("compiler_visible") is True else {}))
    if action == "rotating_audit":
        from .deployment_diagnostics import rotating_audit
        candidate = read_json(job["candidate"])
        validate_candidate(candidate, config.get("kernel_backend", "triton"))
        task = next(t for t in catalog["tasks"] if t["id"] == candidate["task_id"])
        return rotating_audit(candidate, task, job["profile_dir"], catalog, config, session=session)
    if action == "reprofile":
        manifest = read_json(job["manifest"])
        for entry in manifest["entries"]:
            validate_candidate(read_json(Path(job["candidate_root"]) / (entry["candidate_id"] + ".json")),
                               config.get("kernel_backend", "triton"))
        from .reprofile import reprofile
        return reprofile(config, catalog, manifest, job["candidate_root"], job["output_dir"], session=session)
    if action == "diagnose_model":
        if session is not None:
            session.clear()
        from .diagnostics import diagnose_model
        return diagnose_model(config, read_json(job["manifest"]), catalog,
                              job["candidate_root"], job["output_dir"])
    if action == "evaluate":
        return evaluate_operation(job, config, catalog, session=session, hardware=hardware)
    if action in {"model", "deployment_model"}:
        if action == "deployment_model":
            mode = job.get("integration_mode")
            if mode not in ("direct", "opaque"):
                raise InfrastructureError("Select direct or opaque integration for the deployment control.")
            config = {**config, "triton_integration": mode}
        for entry in read_json(job["manifest"])["entries"]:
            validate_candidate(read_json(Path(job["candidate_root"]) / (entry["candidate_id"] + ".json")),
                               config.get("kernel_backend", "triton"))
        incumbent = read_json(job["incumbent_manifest"]) if job.get("incumbent_manifest") else None
        for entry in (incumbent or {}).get("entries", []):
            validate_candidate(read_json(Path(job["candidate_root"]) / (entry["candidate_id"] + ".json")),
                               config.get("kernel_backend", "triton"))
        args = (config, read_json(job["manifest"]), catalog, job["candidate_root"])
        from .supplemental_correctness import capture_required_failures
        with capture_required_failures(job["profile_dir"], config, catalog) as capture:
            options = {"session": session} if session is not None else {}
            if job.get("correctness_only") is True:
                options["correctness_only"] = True
            from .measurement import saved_reference
            if saved_reference(config):
                options["reference_directory"] = Path(job["profile_dir"]) / "model-timings"
            if (config.get("kernel_search", {}).get("evaluation") == "staged"
                    and config["workloads"][0].get("mode") == "cached_generation"):
                from .staged_evaluation import prepared_generation
                options["prepared_generation"] = prepared_generation(job["profile_dir"], catalog)
            if incumbent is not None:
                options["incumbent_manifest"] = incumbent
                if job.get("incumbent_reference"):
                    options["incumbent_reference"] = read_json(job["incumbent_reference"])
            if job.get("trace_dir"):
                options["trace_dir"] = job["trace_dir"]
            if job.get("feedback_trace_dir") and config.get("diagnostics", {}).get("enabled"):
                options["feedback_trace_dir"] = job["feedback_trace_dir"]
            if action == "deployment_model" and job.get("reference_control") is True:
                options["reference_control"] = True
            result = measure_model(*args, **options)
            if "prepared_generation" in options and not result.get("decision", {}).get("accepted"):
                failures = [check for case in result.get("cases", []) for check in case.get("checks", [])
                            if check.get("scope") == "cached_generation" and not check.get("passed")]
                if failures:
                    path = Path(job["profile_dir"]) / "failed-inputs" / "generation" / (digest(args[1]) + ".json")
                    write_json(path, {"manifest": args[1], "input": catalog["compiled_profile"]["workload_times"][0]["input"],
                        "reference_output": catalog["starting_baseline"]["workloads"][0]["reference_output"], "checks": failures})
                    result["failing_generation_input"] = str(path)
        if capture is not None:
            result["supplemental_failure_artifacts"] = capture.artifacts
            result["supplemental_failure_capture"] = capture.failure_summary
        return result
    raise ValueError(f"Unknown worker action: {action}")


def evaluate_operation(job, config, catalog, *, session=None, fixture_only=False, hardware=None):
    from .gpu import evaluate_fixture
    candidate = read_json(job["candidate"])
    validate_candidate(candidate, config.get("kernel_backend", "triton"))
    task = next(t for t in catalog["tasks"] if t["id"] == candidate["task_id"])
    stage_times = {}
    from .kernel_selection import kernel_first, kernel_config
    baseline = read_json(job["baseline_candidate"]) if job.get("baseline_candidate") else None
    if baseline is not None:
        validate_candidate(baseline, config.get("kernel_backend", "triton"))
    if baseline is not None and (not kernel_first(config) or baseline["task_id"] != candidate["task_id"]):
        raise InfrastructureError("The kept baseline must belong to this operation in kernel-first mode.")
    options = {"baseline_candidate": baseline} if baseline is not None else {}
    if "observe_fixture" in job:
        options["observe_fixture"] = job["observe_fixture"]
    from .shape_selection import enabled as shape_enabled, groups, decide
    shape_baselines = {}
    if shape_enabled(config):
        if baseline is not None:
            raise InfrastructureError("Use per-shape baselines for shape selection.")
        allowed = groups(task)
        files = {}
        for shape, path in job.get("baseline_shapes", {}).items():
            if path not in files:
                files[path] = validate_candidate(read_json(path), config.get("kernel_backend", "triton"))
            value = files[path]
            if shape not in allowed or value["task_id"] != task["id"]:
                raise InfrastructureError("The kept shape baseline differs from this task.")
            shape_baselines[shape] = value
        options["baseline_candidates"] = shape_baselines
    from .staged_evaluation import enabled as staged, evaluate as evaluate_stages
    if staged(config):
        if not task.get("separate_operations"):
            raise InfrastructureError("The staged evaluator requires per-module inputs from a new profile.")
        if session is not None and session.model is not None:
            return session.evaluate_fixtures(job, config)
        from .hardware import discover_hardware, require_hardware
        hardware = require_hardware(hardware or discover_hardware(config))
        return evaluate_stages(candidate, task, job["profile_dir"], config, hardware, shape_baselines,
                               job={**job, "_catalog": catalog})
    from .supplemental_correctness import enabled as supplemental_enabled, capture_needed, ensure_fixture, load_fixture
    if session is not None:
        capture_s = 0.
        if supplemental_enabled(config) and capture_needed(job["profile_dir"], task["id"]):
            started = time.perf_counter()
            ensure_fixture(job["profile_dir"], config, catalog, task, session=session)
            capture_s = time.perf_counter() - started
        if session.model is not None:
            result = session.evaluate_fixtures(job, config)
            if capture_s:
                result["stage_times"]["supplemental_capture_s"] += capture_s
            return result
        fixture_only = True
    from .automatic_tuning import declared, policy as tuning_policy
    tuning_declaration = declared(candidate) if tuning_policy(config)["enabled"] else None
    if tuning_declaration and tuning_declaration.get("settings") != tuning_policy(config):
        raise InfrastructureError("The candidate tuning policy differs from the saved run settings.")
    can_tune = tuning_declaration and any(kernel.get("kind") for kernel in tuning_declaration.get("kernels", []))
    if (can_tune
            and not job.get("tuning_ready") and not job.get("tuning_check_only")):
        from .gpu import compile_candidate
        from .store import operation_correct
        started = time.perf_counter()
        with compile_candidate(candidate).base_configurations():
            initial = evaluate_operation({**job, "tuning_check_only": True}, config, catalog, fixture_only=fixture_only)
        initial_seconds = time.perf_counter() - started
        complete = len(initial.get("cases", [])) == len(task["cases"]) and operation_correct(initial)
        if not complete:
            initial["automatic_tuning"] = {"status": "blocked_by_initial_correctness", "initial_check_s": initial_seconds}
            return initial
        result = evaluate_operation({**job, "tuning_ready": True}, config, catalog, fixture_only=fixture_only)
        result["automatic_tuning"] = {"status": "checked_and_tuned", "initial_correctness": initial,
                                      "initial_check_s": initial_seconds,
                                      "note": "Tuning samples select settings. Independent pairs decide acceptance."}
        return result
    supplemental = None
    if supplemental_enabled(config):
        from .gpu import evaluate_supplemental
        started = time.perf_counter()
        if fixture_only:
            fixture, descriptor = load_fixture(job["profile_dir"], config, catalog, task)
            if not fixture or not fixture.get("coverage_complete"):
                raise InfrastructureError("The fixture worker needs all required original outputs.")
        else:
            fixture, descriptor = ensure_fixture(job["profile_dir"], config, catalog, task)
        stage_times["supplemental_capture_s"] = time.perf_counter() - started
        started = time.perf_counter()
        supplemental = evaluate_supplemental(candidate, task, fixture, config)
        supplemental["artifact"] = descriptor
        stage_times["supplemental_checks_s"] = time.perf_counter() - started
    if supplemental is not None and not supplemental["passed"]:
        cases = [case for case in supplemental["cases"] if not case["correct"]]
    else:
        if job.get("archive_on_peer") and config["worker"].get("archive_gpu") is not None:
            options["measure_archive"] = False
        if job.get("tuning_check_only"):
            options["numerical_only"] = True
        cases = evaluate_fixture(candidate, task, Path(job["profile_dir"]) / (task["id"] + ".pt"), config,
                                 stage_times=stage_times, **options)
    result = {"scope": "operation", "candidate_id": candidate_id(candidate), "cases": cases,
              "decision": score_cases(cases, kernel_config(config) if kernel_first(config) else config),
              "stage_times": stage_times}
    if tuning_declaration and not can_tune:
        result["automatic_tuning"] = {"status": "no_eligible_launches", "kernels": tuning_declaration["kernels"]}
    if supplemental is not None:
        result["supplemental_correctness"] = supplemental
        if not supplemental["passed"]:
            result["decision"]["accepted"] = False
            result["decision"]["reasons"].append("Required supplemental correctness failed; operation timing was skipped.")
    if kernel_first(config):
        result.update(timing_baseline={"kind": "kept_kernel" if baseline is not None else "torch_compile",
            "candidate_id": candidate_id(baseline) if baseline is not None else None},
            kernel_min_speedup=kernel_config(config)["min_speedup"], acceptance_scope="kernel_only")
    if len(cases) != len(task["cases"]):
        result["decision"]["accepted"] = False
        result["decision"]["reasons"].append("Not all fixture cases passed.")
    if shape_enabled(config):
        result["timing_baseline"] = {"kind": "per_shape", "shapes": {
            key: candidate_id(shape_baselines[key]) if key in shape_baselines else None for key in groups(task)}}
        result["shape_decisions"] = decide(result, task, kernel_config(config))
        result["acceptance_scope"] = "measured_shapes_only"
    return result


def run_payload(job, *, session=None):
    """Use the same evaluator in one-shot and resident processes."""
    start = time.perf_counter()
    result = {"job_hash": digest(job), "ok": False,
              "provenance": {"implementation_hash": implementation_hash()}}
    try:
        if session is not None:
            session.begin()
        value = execute(job, session=session) if session is not None else execute(job)
        result.update(result=value, ok=True)
        if session is not None:
            intact = session.verify_after_job(job["action"])
            if not intact or isinstance(value, dict) and value.get("reference_state_intact") is False:
                session.retire = True
                value["reference_state_intact"] = False
                for case in value.get("cases", []):
                    case["correct"] = False
                value.setdefault("decision", {}).update(accepted=False)
                value["decision"].setdefault("reasons", []).append("The candidate changed resident reference state.")
            value["resident_cache"] = dict(session.job_stats)
        import torch
        result["provenance"] = {"implementation_hash": implementation_hash(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    except Exception as error:
        if session is not None:
            session.retire = True
        result.update(ok=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                      failure_kind="infrastructure" if isinstance(error, InfrastructureError) else "worker")
        build_log = getattr(error, "build_log", None)
        if build_log is not None:
            log_path = Path(build_log).resolve()
            build_root = Path(__file__).resolve().parents[1] / "runs" / "build"
            try:
                log_path.relative_to(build_root.resolve())
                result["compiler_log"] = {"path": str(log_path),
                    "text": log_path.read_text(encoding="utf-8", errors="replace")}
            except (OSError, ValueError) as log_error:
                result["compiler_log_error"] = str(log_error)
        tuning_log = getattr(error, "tuning_log", None)
        if tuning_log is not None and Path(tuning_log).is_file():
            try:
                path = Path(tuning_log).resolve()
                path.relative_to((Path(__file__).resolve().parents[1] / "runs/build").resolve())
                result["tuning_trials"] = {"path": str(path), "report": read_json(path)}
            except (OSError, ValueError) as log_error:
                result["tuning_log_error"] = str(log_error)
    result["worker_total_s"] = time.perf_counter() - start
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    job = read_json(args.job)
    with gpu_lock():
        result = run_payload(job)
    write_json(args.output, result)
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
