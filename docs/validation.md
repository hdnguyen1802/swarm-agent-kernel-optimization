# Validation

The test suite checks the controller, token ledger, candidate policy, evaluation,
saved timing identities, model acceptance, transport, and terminal interface.
Run it from the project folder:

```bash
python -m unittest discover -s tests -v
```

Tests create their local working folders. GitHub CI runs the suite on Python 3.11.
Native GPU checks remain separate scripts under `tests/gpu_*_checks.py`.

On November 15, 2026, a clean source copy passed all 576 CPU tests in 106.6 seconds.
That copy excluded private settings, old reports, generated files, and run data.
Its `runs/` folder did not exist before the tests started.

A later cleanup removed unused evidence readers, a separate layer-summary helper,
and a duplicate hardware helper. It also removed twelve tests that only exercised
those helpers. The current layer evaluation, model acceptance, and profile checks
retain their tests. A fresh source copy passed all 564 remaining tests in 96.586
seconds. The package build and an isolated package import also passed. The package
contains the Triton prompt and terminal styles, and excludes local run data.

The next recheck passed all 564 tests again in 144.887 seconds. Configuration,
local imports, compilation, CLI, package build, and isolated package checks also
passed. The tested runtime identity was
`9616a8ad3eabe54ab104c1955e2502e70390e386d9fac00c536585fda424d070`.

Both V100 cards also passed `tests/gpu_triton_checks.py` and
`tests/gpu_staged_checks.py` from a separate copy of that source. These checks
covered FP16 linear and RMSNorm kernels, input integrity, fullgraph compiled
callers, CUDA Graph reuse, tuning reuse, correct output, wrong output, and slow
candidate rejection. Ninja imports and executable calls were disabled for the
test processes. Both cards passed without Ninja.

The first GPU attempt could not write to the host's default compiler cache. A
fresh cache inside the test folder resolved that environment error. Both attempts
remain saved. These were small kernel checks, not new complete-model benchmarks.
They used no model API calls.

## Agent provider checks

The provider update passed 584 tests in 147.567 seconds on a fixed source copy.
This includes twenty DeepSeek tests for provider selection, JSON requests,
schema instructions, input reservations, reported usage, retries, credential
errors, response caching, and API checks. The tests also confirm that the
DeepSeek template retains the GPU workload and numerical settings.

The package build passed. Its DeepSeek adapter loaded with Python site packages
disabled, without importing Google packages. The existing Vertex tests also
passed. DeepSeek HTTP responses were mocked for these checks. No DeepSeek key was
configured, so this update does not claim a successful live DeepSeek request or
a new model speed result. Set `DEEPSEEK_API_KEY` on the controller before running
`api-check` with a private copy of `configs/generation-deepseek.json`.

## Recorded GPU results

These complete-model tests ran on November 14 and 15, 2026. They used two
Tesla V100-PCIE GPUs with 32 GB each. Each candidate ran on one GPU.

The workload used FP16, batch size 1, a fixed prompt, greedy generation, and an
enabled KV cache. Each generation produced 128 new tokens. The decode clock
excluded prompt processing and the first output token. It measured the remaining
127 tokens. Each new model version received three warmup runs and three timed
runs. Matching original and accepted reference times were reused on the same GPU.
The complete model passed the required token, score, and cache checks.

| Model | PyTorch ms/token | torch.compile ms/token | Optimized ms/token | Speedup vs PyTorch |
|---|---:|---:|---:|---:|
| Mistral-7B-Instruct-v0.3 | 29.914 | 26.667 | 21.548 | 1.388x |
| Qwen2.5-7B-Instruct | 28.691 | 25.621 | 20.915 | 1.372x |
| Llama-3.1-8B | 30.632 | 27.589 | 22.837 | 1.341x |

The GPU evaluator source identity for Qwen and Llama was
`405d7c2c682da8230a290c56ae2531f68aa0419700c3481c75e8a7809e8ab2c4`.
Mistral used the saved evaluator identity
`49b859139b7c356aac158b3a230fa23ea4fd5b3347e15f85e71c23e01eee0b19`.
Mistral reused its saved search and received additional complete-model checks
under the positive-gain rule. These source identities describe saved evaluator
files, not Git commit IDs.

The results apply to this workload and hardware. They do not establish gains on
other prompts, batch sizes, output lengths, models, or GPUs. The current controller
has later fixes for budget scheduling. CPU tests cover those fixes. A documentation
or configuration cleanup does not count as another GPU validation run.

Full candidate code, failures, and raw samples remain in local `runs/` folders.
Old development reports remain under local `.local/` archives. Git excludes both
locations so machine details and generated evidence do not enter the source tree.
