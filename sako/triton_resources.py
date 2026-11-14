"""Read resources from the compiled kernels used by one untimed call."""

from contextlib import contextmanager


def kernel_objects(module):
    from triton.runtime.jit import JITFunction, KernelInterface
    found = set()
    for name, value in vars(module).items():
        while isinstance(value, KernelInterface):
            if isinstance(value, JITFunction) and id(value) not in found:
                found.add(id(value))
                yield name, value
            value = getattr(value, "fn", None)


def kernel_resources(name, compiled):
    """These are compiler/driver values, not runtime memory traffic counters."""
    metadata = getattr(compiled, "metadata", None)
    integer = lambda value: value if type(value) is int and value >= 0 else None
    return {"kernel": name, "compiled_name": str(getattr(compiled, "name", name)),
            "compiled_hash": str(getattr(compiled, "hash", "")),
            "registers_per_thread": integer(getattr(compiled, "n_regs", None)),
            "static_shared_memory_bytes": integer(getattr(metadata, "shared", None)),
            "reported_spills": integer(getattr(compiled, "n_spills", None)),
            "num_warps": integer(getattr(metadata, "num_warps", None)),
            "num_stages": integer(getattr(metadata, "num_stages", None)),
            "note": "Reported spills are driver metadata. They do not measure executed spill traffic."}


@contextmanager
def observe_resources(module):
    """Wrap JIT launches only during this diagnostic; restore every method."""
    saved, reports = [], {}
    inherited = object()
    try:
        for name, kernel in kernel_objects(module):
            previous = vars(kernel).get("run", inherited)
            original = kernel.run

            def observed(*args, _original=original, _name=name, **kwargs):
                compiled = _original(*args, **kwargs)
                try:
                    report = kernel_resources(_name, compiled)
                except Exception as error:
                    report = {"kernel": _name, "status": "unavailable", "error": str(error)}
                reports[(_name, report.get("compiled_hash", ""))] = report
                return compiled

            saved.append((kernel, previous))
            kernel.run = observed
        records = []
        yield records
        records.extend(reports.values())
    finally:
        for kernel, previous in reversed(saved):
            if previous is inherited:
                vars(kernel).pop("run", None)
            else:
                kernel.run = previous
