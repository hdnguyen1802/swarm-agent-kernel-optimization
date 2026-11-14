"""Bind the task catalog to measured hardware and the test settings."""

from pathlib import Path

from .core import correctness_settings, digest, implementation_hash, read_json, require_broad_selection, write_json


PROFILE_FIELDS = ("model_id", "revision", "dtype", "attention", "workloads", "warmup",
                  "trials", "repeats", "correctness_seeds", "atol", "rtol",
                  "min_speedup", "max_p95_ratio", "compile_mode", "fullgraph",
                  "model_reference", "correctness_scope", "correctness_reference", "max_fixture_cases",
                  "containing_region_policy", "model_speed_scope", "supplemental_correctness", "measurement", "tuning",
                  "generation_objective", "report_eager_baseline")


def profile_settings(config):
    return {**{key: config.get(key) for key in PROFILE_FIELDS}, **correctness_settings(config),
            "operation_evaluation": config.get("kernel_search", {}).get("evaluation", "standard"),
            "kernel_backend": config.get("kernel_backend", "triton"),
            "ranking_mode": config.get("ranking_mode", "eager_regions"),
            "target_selection": config["agent"].get("target_selection", "flat"),
            "search_parent_regions": config["agent"].get("search_parent_regions", False),
            "top_k_operations": (None if config["agent"].get("target_selection") == "bottleneck_tree"
                                 else config["agent"].get("top_k_operations", 4))}


def validate_catalog(config, catalog, hardware):
    """Reject reused measurements when the target or evaluation contract changes."""
    expected = catalog.get("hardware", {}).get("fingerprint")
    actual = hardware.get("fingerprint")
    if not expected or not actual or expected != actual:
        raise ValueError("Profile hardware or software changed. Create a new profile and run directory.")
    previous = catalog.get("config", {})
    if config.get("kernel_search", {}).get("evaluation", "standard") != previous.get("kernel_search", {}).get("evaluation", "standard"):
        raise ValueError("Operation evaluation changed. Create a new profile.")
    if config["agent"].get("search_parent_regions", False) != previous.get("agent", {}).get("search_parent_regions", False):
        raise ValueError("Profile parent search changed. Create a new profile and run directory.")
    if config["agent"].get("target_selection", "flat") != previous.get("agent", {}).get("target_selection", "flat"):
        raise ValueError("Profile target selection changed. Create a new profile and run directory.")
    if config.get("kernel_backend", "triton") != previous.get("kernel_backend", "triton"):
        raise ValueError("Profile kernel backend changed. Create a new profile and run directory.")
    current_correctness, previous_correctness = correctness_settings(config), correctness_settings(previous)
    if config.get("ranking_mode", "eager_regions") != previous.get("ranking_mode", "eager_regions"):
        raise ValueError("Profile ranking mode changed. Create a new profile and run directory.")
    require_broad_selection(previous)
    for key in PROFILE_FIELDS:
        if key == "revision":
            # A named revision is resolved and pinned by profiling. Explicit commit IDs must agree.
            requested = str(config.get(key, ""))
            if len(requested) == 40 and requested != previous.get(key):
                raise ValueError("Profile and requested model revisions differ.")
        elif key in current_correctness:
            if current_correctness[key] != previous_correctness[key]:
                raise ValueError(f"Profile and evaluation settings differ: {key}.")
        elif config.get(key) != previous.get(key):
            raise ValueError(f"Profile and evaluation settings differ: {key}.")


def prepare(config, run_dir, profile_dir, worker, *, allow_empty=False):
    """Check the device, then collect a profile if one does not exist."""
    run_dir, profile_dir = Path(run_dir), Path(profile_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    doctor = worker.run("doctor", run_dir)
    write_json(run_dir / "doctor.json", doctor)
    hardware = doctor.get("hardware")
    if not isinstance(hardware, dict) or not hardware.get("fingerprint"):
        raise ValueError("Worker did not return a hardware fingerprint. Sync the current source first.")
    write_json(run_dir / "hardware.json", hardware)
    if hardware.get("support", {}).get("supported") is False:
        raise RuntimeError("Unsupported worker target: " + "; ".join(hardware["support"].get("reasons", [])))
    stamp_path = profile_dir / "preparation.json"
    catalog_path = profile_dir / "tasks.json"
    settings_hash = digest(profile_settings(config))
    code_hash = implementation_hash()
    if catalog_path.exists():
        catalog = read_json(catalog_path)
        validate_catalog(config, catalog, hardware)
        if not stamp_path.exists():
            raise ValueError("Profile has no preparation record. Use a new profile directory.")
        stamp = read_json(stamp_path)
        if (stamp.get("settings_hash") != settings_hash or stamp.get("implementation_hash") != code_hash
                or stamp.get("catalog_hash") != digest(catalog)):
            raise ValueError("Profile source, settings, or catalog changed. Use a new profile and run directory.")
    if not catalog_path.exists():
        relative = profile_dir.resolve().relative_to(worker.root.resolve()).as_posix()
        catalog = worker.run("profile", run_dir, profile_dir=relative)
        validate_catalog(config, catalog, hardware)
        if not catalog.get("selected_task_ids") and not allow_empty:
            raise ValueError("The profile contains no eligible operation regions.")
        write_json(catalog_path, catalog)
        write_json(stamp_path, {"settings_hash": settings_hash, "implementation_hash": code_hash,
                                "hardware_fingerprint": hardware["fingerprint"],
                                "catalog_hash": digest(catalog)})
    return catalog
