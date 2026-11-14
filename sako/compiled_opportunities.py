"""Label launches in compiled profiler evidence."""


def launch_kind(name):
    """Use the compiler launch name, never an agent's strategy description."""
    if name.startswith("triton_"):
        return "generated_triton"
    if name in {"aten::mm", "aten::addmm", "aten::bmm"}:
        return "vendor_gemm"
    if name.startswith("aten::"):
        return "framework_external"
    return "unknown"
