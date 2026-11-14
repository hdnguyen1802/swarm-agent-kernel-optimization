"""Read the selected GPU. Keep hardware facts separate from search advice."""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.metadata
import math
import os
import platform
import re
import shutil
import subprocess
import uuid

from .core import digest

DEVICE_SOURCE = "https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html"
FEATURE_SOURCE = "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html"

# CUdevice_attribute values are part of the public CUDA driver ABI.
ATTRIBUTES = {
    "max_threads_per_block": 1, "max_block_dim_x": 2, "max_block_dim_y": 3,
    "max_block_dim_z": 4, "shared_memory_per_block_bytes": 8, "warp_size": 10,
    "registers_per_block": 12, "sm_count": 16, "memory_clock_khz": 36,
    "memory_bus_width_bits": 37, "l2_cache_bytes": 38,
    "max_threads_per_sm": 39, "shared_memory_per_sm_bytes": 81,
    "registers_per_sm": 82, "shared_memory_optin_per_block_bytes": 97,
    "max_blocks_per_sm": 106,
}


def query_driver_attributes():
    """Read the active context so CUDA_VISIBLE_DEVICES keeps its meaning."""
    facts, errors = {}, {}
    try:
        library = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL(
            ctypes.util.find_library("cuda") or "libcuda.so.1")
        device = ctypes.c_int()
        status = library.cuCtxGetDevice(ctypes.byref(device))
        if status:
            raise RuntimeError(f"cuCtxGetDevice returned {status}.")
        for name, attribute in ATTRIBUTES.items():
            value = ctypes.c_int()
            status = library.cuDeviceGetAttribute(ctypes.byref(value), attribute, device)
            if status == 0:
                facts[name] = value.value
            else:
                facts[name] = None
                errors[name] = f"CUDA driver status {status}"
        identifier = (ctypes.c_ubyte * 16)()
        if library.cuDeviceGetUuid(ctypes.byref(identifier), device) == 0:
            facts["uuid"] = "GPU-" + str(uuid.UUID(bytes=bytes(identifier)))
        pci = ctypes.create_string_buffer(32)
        if library.cuDeviceGetPCIBusId(pci, len(pci), device) == 0:
            facts["pci_bus_id"] = pci.value.decode("ascii")
        driver = ctypes.c_int()
        if library.cuDriverGetVersion(ctypes.byref(driver)) == 0:
            facts["driver_api_version"] = driver.value
    except (OSError, AttributeError, RuntimeError) as error:
        errors["driver_query"] = str(error)
    return facts, errors


def _command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30,
                                encoding="utf-8", errors="replace", check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def capability_features(capability):
    """Conservative baseline-target rules. Special instruction targets need a new profile."""
    number = capability[0] * 10 + capability[1]
    return {
        "fp16_tensor_cores": number >= 70,
        "hardware_async_copy_global_to_shared": number >= 80,
        "architecture_specific_instructions": False,
        "precision_policy": "Use the task dtype and rounding contract. Hardware support does not permit changing precision.",
        "source": FEATURE_SOURCE,
        "rule_version": 1,
    }


def hardware_fingerprint(profile):
    """Use stable identity and limits. Exclude free memory and current load."""
    return digest({key: profile[key] for key in ("target", "capability", "facts", "software", "features")})




def discover_hardware(config=None):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on the selected worker.")
    torch.cuda.init()
    # Ensure a current context exists before the driver attribute query.
    torch.empty(1, device="cuda")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    capability = [int(props.major), int(props.minor)]
    target = "sm_" + "".join(str(part) for part in capability)
    queried, unavailable = query_driver_attributes()
    facts = {name: queried.get(name) for name in ATTRIBUTES}
    facts.update(name=props.name, total_memory_bytes=int(props.total_memory),
                 sm_count=int(props.multi_processor_count),
                 uuid=queried.get("uuid") or (str(props.uuid) if getattr(props, "uuid", None) else None),
                 pci_bus_id=queried.get("pci_bus_id"))
    compiler = shutil.which("nvcc")
    compiler_version = _command([compiler, "--version"]) if compiler else None
    code_list = _command([compiler, "--list-gpu-code"]) if compiler else None
    compiler_targets = sorted(set(re.findall(r"\bsm_[0-9]+[af]?\b", code_list or "")))
    driver_versions = _command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    driver_version = sorted(set(driver_versions.splitlines())) if driver_versions else None
    software = {"python": platform.python_version(), "platform": platform.platform(), "host": platform.node(),
                "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
                "transformers": _version("transformers"), "triton": _version("triton"),
                "compiler": compiler, "compiler_version": compiler_version,
                "compiler_targets": compiler_targets, "driver_versions": driver_version,
                "driver_api_version": queried.get("driver_api_version")}
    reasons = []
    if software["triton"] is None:
        reasons.append("Triton is not installed on this worker.")
    if torch.version.cuda is None:
        reasons.append("The PyTorch build does not report a CUDA runtime.")
    if facts["uuid"] is None:
        unavailable["uuid"] = "The device UUID could not be read."
        if facts["pci_bus_id"] is None:
            reasons.append("The device has no verified UUID or PCI bus identity. Measurement reuse is disabled.")
    features = capability_features(capability)
    estimates = {}
    if facts["memory_clock_khz"] and facts["memory_bus_width_bits"]:
        estimates["peak_memory_bandwidth_gb_s"] = {
            "value": 2 * facts["memory_clock_khz"] * 1000 * facts["memory_bus_width_bits"] / 8 / 1e9,
            "method": "DDR clock times bus width. This is a theoretical estimate, not a measurement.",
            "source": DEVICE_SOURCE}
    guidance = [
        {"text": f"Compile for {target}. Use the queried block and SM limits.", "source": DEVICE_SOURCE},
        {"text": "Shared memory per block, per SM, and opt-in limits are distinct. Check each allocation.", "source": DEVICE_SOURCE},
        {"text": "Use explicit warp and block synchronization. Preserve the operation's numerical contract.", "source": FEATURE_SOURCE},
        {"text": "Baseline targets exclude architecture-specific instructions. Do not infer support from a newer GPU name.", "source": FEATURE_SOURCE},
    ]
    if not features["hardware_async_copy_global_to_shared"]:
        guidance.append({"text": "This target has no Ampere hardware async-copy path. Use compatible loads and stores.", "source": FEATURE_SOURCE})
    profile = {"schema_version": 1, "target": target, "capability": capability,
               "facts": facts, "software": software, "features": features,
               "support": {"supported": not reasons, "reasons": reasons,
                           "runtime_probe": "not_run", "compile_probe": "not_run"},
               "guidance": guidance, "estimates": estimates, "unavailable": unavailable,
               "fact_source": DEVICE_SOURCE}
    profile["fingerprint"] = hardware_fingerprint(profile)
    return profile


def require_hardware(profile):
    if not profile["support"]["supported"]:
        raise RuntimeError("Unsupported worker target: " + "; ".join(profile["support"]["reasons"]))
    return profile


def select_top_tasks(tasks, limit=4):
    """Choose disjoint replacement regions. Never add parent and child timings."""
    if type(limit) is not int or limit < 1:
        raise ValueError("The selected task limit must be positive.")
    selected, excluded, paths = [], [], []
    def measured_time(task):
        value = task.get("priority_ms")
        return value if type(value) in (int, float) and math.isfinite(value) and value > 0 else 0.
    ranking = sorted(tasks, key=lambda task: (-measured_time(task), task["id"]))
    for rank, task in enumerate(ranking, 1):
        overlap = any(a == b or a.startswith(b + ".") or b.startswith(a + ".")
                      for a in task["paths"] for b in paths)
        reason = ("contract_failed" if not task.get("contract", {}).get("validated") else
                  task["ranking_excluded_reason"] if task.get("ranking_excluded_reason") else
                  "no_measured_time" if not measured_time(task) else
                  "overlapping_region" if overlap else "top_k_limit" if len(selected) >= limit else None)
        if reason:
            excluded.append({"task_id": task["id"], "rank": rank, "reason": reason})
        else:
            selected.append(task["id"])
            paths.extend(task["paths"])
    return {"selected_task_ids": selected, "requested_count": limit,
            "available_count": len(selected), "excluded": excluded,
            "ranking": [{"rank": rank, "task_id": task["id"], "priority_ms": measured_time(task),
                         "method": task.get("priority_method")} for rank, task in enumerate(ranking, 1)],
            "policy": "Descending measured region time, equal workload weight, no overlapping paths."}
