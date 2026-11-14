"""Give each decode candidate one GPU and that GPU's compiled reference."""

from dataclasses import dataclass
from pathlib import Path

from .core import read_json
from .archive_gpu import check_catalogs
from .transport import TransferError


@dataclass
class DecodeDevice:
    gpu: int
    worker: object
    directory: Path
    profile: Path
    fingerprint: str

    def verify(self, result):
        provenance = result.get("worker_provenance", {})
        if provenance.get("visible_devices") != str(self.gpu):
            raise TransferError(f"The decode job did not return measurements from GPU {self.gpu}.")
        result["evaluation_device"] = {"gpu": self.gpu, "hardware_fingerprint": self.fingerprint}


def decode_devices(worker, directory, profile, catalog):
    primary = DecodeDevice(worker.settings["gpu"], worker, directory, profile, catalog["hardware"]["fingerprint"])
    if worker.settings.get("archive_gpu") is None:
        return [primary]
    peer = worker.archive_peer()
    peer_dir = directory / ("archive-gpu-" + str(peer.settings["gpu"]))
    peer_profile = peer_dir / "profile"
    other = read_json(peer_profile / "tasks.json")
    check_catalogs(catalog, other)
    by_id = {task["id"]: task for task in other["tasks"]}
    for task in catalog["tasks"]:
        match = by_id.get(task["id"], {})
        if any(task.get(key) != match.get(key) for key in ("kind", "paths", "scalars", "cases")):
            raise TransferError("The GPUs have different inputs for " + task["id"])
    return [primary, DecodeDevice(peer.settings["gpu"], peer, peer_dir, peer_profile, other["hardware"]["fingerprint"])]
