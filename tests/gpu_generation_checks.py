"""Exercise cached generation with small, locally initialized causal models on the validation host."""

import tempfile
from pathlib import Path

from sako.core import load_config
from sako.generation import GreedyRequest, check_generation, input_record
from sako.model import clone_structure


def check_architectures(device="cuda"):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (AutoModelForCausalLM, GPT2Config, OPTConfig, LlamaConfig,
                              PreTrainedTokenizerFast)

    cfg = load_config("configs/generation.json")
    workload = {**cfg["workloads"][0], "prompt": "one two three", "new_tokens": 8}
    common = {"vocab_size": 32, "bos_token_id": 1, "eos_token_id": None, "pad_token_id": 0}
    architectures = [GPT2Config(n_embd=32, n_layer=2, n_head=4, n_positions=64, **common),
                     OPTConfig(hidden_size=32, word_embed_proj_dim=32, ffn_dim=48, num_hidden_layers=2,
                               num_attention_heads=4, max_position_embeddings=64, **common),
                     LlamaConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, **common)]
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as root, torch.inference_mode():
        for idx, architecture in enumerate(architectures):
            name = architecture.model_type
            directory = Path(root) / name
            vocabulary = ["[UNK]", "p1", "p2", "p3", "p4", "p5", "p6", "two", "three"]
            vocabulary[3 + idx] = "one"
            tokens = Tokenizer(WordLevel({word: i for i, word in enumerate(vocabulary)}, unk_token="[UNK]"))
            tokens.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=tokens, unk_token="[UNK]").save_pretrained(directory)
            config = {**cfg, "model_id": str(directory), "revision": "local-test"}
            assert input_record(config, workload)["prompt_token_ids"] == [3+idx, 7, 8]
            model = AutoModelForCausalLM.from_config(architecture).eval().to(device)
            from transformers.pytorch_utils import Conv1D
            from sako.gpu import reference
            from sako.model import arguments, module_kind
            for projection in (m for m in model.modules() if isinstance(m, Conv1D)):
                x = torch.randn(1, 3, projection.weight.size(0), device=device, dtype=model.dtype)
                assert module_kind(projection) == "linear"
                tensors, scalars = arguments("linear", projection, x)
                torch.testing.assert_close(reference("linear", tensors, scalars), projection(x))
            request = GreedyRequest(model, config, workload)
            baseline_ids = model.generate(request.prompt, attention_mask=torch.ones_like(request.prompt),
                do_sample=False, use_cache=True, max_new_tokens=workload["new_tokens"])
            steps = []
            first_ids, _ = request.run(observe=lambda i, scores, token, cache: steps.append(i))
            assert steps == list(range(workload["new_tokens"])) and torch.equal(first_ids, baseline_ids), name
            assert torch.equal(first_ids, request.run()[0]), name
            other = GreedyRequest(clone_structure(model), config, workload)
            assert check_generation(request, other, config)["passed"], name
            original = other.call
            def wrong_step(**kwargs):
                out = original(**kwargs)
                out.logits = -out.logits
                return out
            other.call = wrong_step
            assert not check_generation(request, other, config)["passed"], name
            steps_seen = 0
            def wrong_cache(**kwargs):
                nonlocal steps_seen
                out = original(**kwargs)
                steps_seen += 1
                if steps_seen == workload["new_tokens"]:
                    from sako.generation import cache_layers
                    cache_layers(out.past_key_values)[0][0].add_(1.)
                return out
            other.call = wrong_cache
            corrupted = check_generation(request, other, config)
            assert corrupted["greedy_tokens_match"] and not corrupted["passed"], name
            print(name, device, "matched Transformers greedy generation; rejected altered scores and cache values", flush=True)
            if device == "cuda":
                from sako.generation import compare_models
                check, metrics = compare_models(model, clone_structure(model), config, workload)
                assert check["passed"] and check["steps_checked"] == workload["new_tokens"], check
                assert metrics["timing_scope"] == "complete_cached_generation"
                assert all(len(m["total_ms"]["samples_ms"]) == config["trials"]
                           for m in metrics["generation_metrics"].values())
                print(name, "compiled all cache/logit checks passed; total ms",
                      metrics["baseline"]["median_ms"], flush=True)


def check_operation_profile():
    """Run capture, rank and complete-model checks on a locally initialized OPT checkpoint."""
    import gzip
    import json
    from types import SimpleNamespace
    from unittest.mock import patch
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, OPTConfig, PreTrainedTokenizerFast
    from sako.model import profile, measure_model
    from sako.supplemental_correctness import ensure_fixture

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        model_path = root / "opt"
        tokens = Tokenizer(WordLevel({"[UNK]": 0, "Example": 1, "prompt": 2}, unk_token="[UNK]"))
        tokens.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tokens, unk_token="[UNK]").save_pretrained(model_path)
        arch = OPTConfig(vocab_size=128, hidden_size=128, word_embed_proj_dim=128, ffn_dim=256,
                         num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=256)
        torch.manual_seed(1729)
        AutoModelForCausalLM.from_config(arch).save_pretrained(model_path)
        cfg = load_config("configs/generation.json")
        workload = {**cfg["workloads"][0], "prompt": "Example prompt", "new_tokens": 5}
        cfg.update(model_id=str(model_path), revision="test", warmup=1, trials=5, workloads=[workload])
        directory = root / "profile"
        with patch("huggingface_hub.HfApi.model_info", return_value=SimpleNamespace(sha="test")):
            catalog = profile(cfg, directory)
        assert catalog["selected_task_ids"], "The compiled graph must have at least one safe target"
        task = next(t for t in catalog["tasks"] if t["id"] in catalog["selected_task_ids"])
        cases = task["cases"]
        assert {c["generation_step"]["cache_length_before"] for c in cases} == {0, 2, 5}, cases
        assert {c["shapes"][0][-2] for c in cases} == {1, 2}, cases
        phases = task["compiled_evidence"]["generation_phase_ms"]
        assert phases["prefill"][0] > 0 and phases["decode"][0] > 0, phases
        with gzip.open(directory / "generation-0-gpu-samples.jsonl.gz", "rt") as saved:
            assert {(r["trial"], r["step"]) for r in map(json.loads, saved)} == {
                (trial, step) for trial in range(5) for step in range(5)}
        fixture, _ = ensure_fixture(directory, cfg, catalog, task)
        assert fixture["coverage_complete"]
        from sako.core import candidate_id, write_json
        from sako.gpu import evaluate_fixture, evaluate_supplemental
        from sako.shape_selection import case_signature
        from triton_fixtures import LINEAR_SOURCE
        control = {"name": "generation_check", "task_id": task["id"], "backend": "triton", "references": [], "strategy": "Test one decode projection.", "source": LINEAR_SOURCE}
        key = candidate_id(control)
        write_json(root / (key + ".json"), control)
        assert evaluate_supplemental(control, task, fixture, cfg)["passed"]
        measured = evaluate_fixture(control, task, directory / (task["id"] + ".pt"), cfg)
        assert all(c["correct"] for c in measured) and {c["generation_step"]["step"] for c in measured} == {0, 1, 4}
        decode = next(c for c in cases if c["generation_step"]["phase"] == "decode")
        manifest = {"entries": [{"task_id": task["id"], "paths": [task["paths"][0]], "candidate_id": key,
                                 "signatures": [case_signature(decode, task["scalars"])]}]}
        result = measure_model(cfg, manifest, catalog, root)
        assert result["cases"][0]["correct"] and result["cases"][0]["checks"][-1]["steps_checked"] == 5
        print("OPT capture, rank, fixtures and Triton decode replacement passed", task["kind"], phases, flush=True)


if __name__ == "__main__":
    check_architectures()
    check_operation_profile()
