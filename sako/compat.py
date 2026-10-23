"""Fixed compiler settings shared by all evaluation paths."""


def configure_compiler(config):
    import torch
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.cache_size_limit = 256
