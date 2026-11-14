"""Time archive entries on a second SSH GPU. Keep each timing pair on one device."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from .core import candidate_id, read_json
from .preparation import prepare, profile_settings
from .store import operation_correct
from .transport import TransferError, WorkerJobError


def check_catalogs(primary, archive, task_id=None):
    left, right = primary["hardware"], archive["hardware"]
    identity = "uuid" if left["facts"].get("uuid") and right["facts"].get("uuid") else "pci_bus_id"
    if (not left["facts"].get(identity) or not right["facts"].get(identity)
            or left["facts"][identity] == right["facts"][identity]
            or left["target"] != right["target"] or left["facts"]["name"] != right["facts"]["name"]
            or left["software"] != right["software"]):
        raise TransferError("Archive ranking needs a distinct GPU of the same type with matching software.")
    if profile_settings(primary["config"]) != profile_settings(archive["config"]):
        raise TransferError("The two GPU profiles have different model or test settings.")
    if task_id is None:
        return
    tasks = [next((t for t in catalog["tasks"] if t["id"] == task_id), None) for catalog in (primary, archive)]
    if not all(tasks) or any(tasks[0].get(k) != tasks[1].get(k) for k in ("kind", "paths", "scalars", "cases")):
        raise TransferError("The archive GPU has a different fixture contract for this task.")


def run_paired(primary, peer, action, directory, fields):
    peer_dir = directory / ("archive-gpu-" + str(peer.settings["gpu"]))
    profile = peer_dir / "profile"
    relative = profile.relative_to(primary.root).as_posix()
    if action == "profile":
        if primary.config.get("report_eager_baseline", False):
            # Eager generation also measures CPU dispatch. A profile on the other
            # card can delay it. Finish each reference/profile job before the next.
            catalog = primary._run_action(action, directory, **fields)
            archive = prepare(peer.config, peer_dir, profile, peer, allow_empty=True)
            check_catalogs(catalog, archive)
            return catalog
        with ThreadPoolExecutor(max_workers=1) as pool:
            other = pool.submit(prepare, peer.config, peer_dir, profile, peer, allow_empty=True)
            catalog = primary._run_action(action, directory, **fields)
            check_catalogs(catalog, other.result())
        return catalog

    archive = (read_json(profile / "tasks.json") if (profile / "preparation.json").is_file()
               else prepare(peer.config, peer_dir, profile, peer, allow_empty=True))
    catalog = read_json(primary.root / fields["profile_dir"] / "tasks.json")
    path = Path(fields["candidate"])
    candidate = read_json(primary.root / path)
    check_catalogs(catalog, archive, candidate["task_id"])
    # Each SSH stream owns its staged files. Each card owns its fixtures, checks and timings.
    peer.put(primary.root / path, path.as_posix())
    with ThreadPoolExecutor(max_workers=1) as pool:
        started = time.perf_counter()
        other = pool.submit(peer.run, action, peer_dir, profile_dir=relative, candidate=path.as_posix(),
                            **({"focus_paths": fields["focus_paths"]} if fields.get("focus_paths") else {}))
        result = primary._run_action(action, directory, archive_on_peer=True, **fields)
        try:
            rank = other.result()
        except WorkerJobError as error:
            result["archive_measurement"] = {"gpu": peer.settings["gpu"], "error": str(error),
                                             "artifacts": error.artifacts}
            return result
    if rank.get("candidate_id") != candidate_id(candidate) or result.get("candidate_id") != rank["candidate_id"]:
        raise TransferError("The archive GPU returned a different candidate.")
    for worker, measured in ((primary, result), (peer, rank)):
        if measured.get("worker_provenance", {}).get("visible_devices") != str(worker.settings["gpu"]):
            raise TransferError("The worker ran on a different GPU than the one assigned.")
    result["archive_measurement"] = {
        "gpu": peer.settings["gpu"], "hardware_fingerprint": archive["hardware"]["fingerprint"],
        "worker_provenance": rank["worker_provenance"], "worker_total_s": rank["worker_total_s"],
        "parallel_wall_s": time.perf_counter() - started, "correct": operation_correct(rank),
        "jobs": (peer_dir / "jobs").relative_to(primary.root).as_posix()}
    if not (operation_correct(result) and operation_correct(rank)) or rank.get("reference_state_intact") is False:
        return result
    if len(result["cases"]) != len(rank["cases"]):
        raise TransferError("The archive GPU returned different fixture counts.")
    for case, other_case in zip(result["cases"], rank["cases"]):
        if any(case.get(k) != other_case.get(k) for k in ("case_index", "seed", "workload", "shapes", "source_path", "generation_step")):
            raise TransferError("The archive GPU returned different fixture cases.")
        case["search_reference"] = {**other_case["search_reference"],
            "hardware_fingerprint": archive["hardware"]["fingerprint"], "gpu": peer.settings["gpu"]}
    return result
