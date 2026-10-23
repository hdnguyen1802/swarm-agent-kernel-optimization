# Swarm Agent Kernel Optimization

SAKO uses a group of agents to write CUDA kernels. It tests each kernel on a GPU.
It then tests the selected kernels in a complete Hugging Face model.
The main baseline is `torch.compile` with `fullgraph=True`.

The first model adapter supports **Qwen/Qwen2.5-7B-Instruct** in FP16.
The target device is a **V100 with 32 GB**. CUDA 12.x builds code for `sm_70`.

## What the harness does

1. Load the model and resolve its exact Hugging Face revision.
2. Capture real inputs for linear layers, embeddings, normalization, MLPs, and attention.
3. Give eight writers tasks and relevant CUTLASS source excerpts.
4. Compile each candidate in a separate worker process.
5. Check output shape, dtype, values, and input changes.
6. Compare the operation with its compiled PyTorch reference.
7. Select compatible replacements from the measured archive.
8. Compile both complete model paths. Test them with the same weights and inputs.
9. Save raw samples, errors, source, lineage, plans, API use, and decisions.

SAKO does not use an LLM score to accept a kernel. Compile results, numerical tests,
and measured latency decide acceptance. An optional reviewer can explain failures.

The normal team has eleven logical roles: eight writers, one coordinator,
one performance analyst, and one integration agent. All roles use the configured
Gemini model. The four writer groups explore memory access, parallel execution,
compute, and fusion. Each group has two writers. Good candidates move between groups.
The controller permits two API calls at a time. Eight writers do not require eight
simultaneous API requests. Quota and temporary service errors use bounded retries.

## Layout

```text
sako/                  Controller, model adapter, evaluator, and SSH worker
sako/prompts/          Fixed writer contract
configs/project.json   Workloads, API limits, model, and server settings
references/            Pinned source catalog; generated index stays local
third_party/cutlass/   Git submodule at an exact commit
scripts/               Setup and validation helpers
tests/                 Tests that do not need a GPU
runs/                  Local results; excluded from Git
```

## Setup

Clone with submodules:

```bash
git clone --recurse-submodules <repository-url>
cd swarm-agent-kernel-optimization
python -m venv .venv
```

On the Windows controller:

```powershell
.venv\Scripts\python -m pip install -e '.[agent]'
$env:GOOGLE_APPLICATION_CREDENTIALS = 'C:\path\to\service-account.json'
.venv\Scripts\python -m sako references
.venv\Scripts\python -m sako api-check --run runs/validation
```

Set `worker.host`, `worker.identity`, and `worker.root` in the config.
The SSH host key must already be trusted. The controller uses batch SSH.
The default GPU is device 0. One worker holds a lock for each GPU.
The lock coordinates SAKO jobs under this account. It cannot stop another user's jobs.

Run `scripts/setup_gpu.sh` from the server checkout to create its Python environment.
Use Python 3.11, PyTorch 2.5.1 with CUDA 12.1 wheels, Triton 3.1.0,
Transformers 4.48.3, and a CUDA 12.x compiler. The server has CUDA 12.3.
The compiler and PyTorch can report a minor version difference.
Run the doctor command to test the actual device and compiler path.

```powershell
.venv\Scripts\python -m sako sync
.venv\Scripts\python -m sako doctor --run runs/validation
.venv\Scripts\python -m sako profile --run runs/validation
.venv\Scripts\python -m sako baseline --run runs/validation
.venv\Scripts\python -m sako search --run runs/search-001
.venv\Scripts\python -m sako report --run runs/search-001
```

`sync` sends an explicit file list. It excludes credentials, Git data, model weights,
and run files. The server obtains the pinned CUTLASS source directly from GitHub.
Profile fixtures stay on the server. The controller stores the task descriptions.

## Speed measurements

The test input is a tensor of token IDs. The LLM prompt asks the coding agents to
write a kernel. Its response time is not the model inference time.

The default workloads use `[batch, sequence]` sizes `[1,128]`, `[1,512]`, and `[2,128]`.
Each call runs the complete model and computes logits for every input token.
Weights, input IDs, FP16 precision, and attention settings stay fixed.
This is a model forward test. It does not measure a serving system or text generation loop.

The evaluator warms each path before measurement. It alternates test order.
It records CUDA event time and synchronized wall time. Compile time is excluded.
The ratio is `compiled_baseline_ms / candidate_ms`. Values above 1 mean faster execution.
Every workload must pass correctness, the minimum speed gate, and the p95 gate.
The defaults require a 1.02x gain and at most a 1.05x p95 latency ratio.

The empty-manifest baseline command compares two compiled model paths with shared,
unchanged weights. It is a control test. It does not claim an optimization gain.
SAKO never changes to eager mode after a compile failure.
The complete model reference is the original model under `torch.compile` with its
default precision behavior. The candidate must match those logits at the declared limits.
The evaluator also records the difference between compiled and eager logits as a
diagnostic. FP16 fusion can change these values. This diagnostic does not accept or
reject a candidate. Operation tests still compare candidates with the original eager
operation. No compiler patch or relaxed tolerance is used in the final setup.
The numerical limits are declared test limits, not a proof for every possible input.
Tests use several fixed seeds. Add more workloads and independent input sets before
you make a broad performance claim.

## CUTLASS references

CUTLASS is pinned to `f7b19de32c5d1f3cedfc735c2849f12b537522ee` (3.5.1).
`references/catalog.json` lists sources that apply to `sm_70`.
The index divides these files into excerpts. The controller retrieves excerpts for
each task and places them in the writer request. The full library stays outside
the model context. The compiler can also use its headers.

Candidates declare CUTLASS use and source paths. Keep copied license notices.
The submodule retains the upstream BSD-3-Clause license.

## Limits and recovery

The default run permits at most 120 API calls and 1,500,000 accounted tokens.
Concurrent calls reserve tokens before they start. Unknown request outcomes keep
their reservation. These are use limits, not a price guarantee.
Each retry consumes a call. A known quota rejection consumes zero accounted tokens.
The controller disables hidden SDK retries so it can count each attempt.
The search stops after ten generations or four unchanged generations.
An optional failure review uses the same limits.

Create `runs/search-001/STOP` to stop at the next generation boundary. Remove it and
repeat the same search command to resume. The SQLite archive survives a restart.
Keep the config and profile fixed for a run. A changed config needs a new run folder.
GPU jobs have a remote timeout. A disconnected SSH client cannot leave an unlimited job.

Generated native code runs under the GPU account. A process boundary contains many
crashes and timeouts, but it is not a security sandbox. Static source checks catch
common policy errors. They do not prove that code is safe or free from benchmark abuse.
Use a dedicated GPU account or a restricted container for untrusted public submissions.
Do not place cloud credentials in worker jobs or candidate files.

The integration agent supplies notes. Deterministic code prevents overlapping parent
and child replacements. A measured complete model gain is required before `export`
will produce an accepted bundle. A search can end with no accepted candidate.

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q sako
```

To test the CUDA ABI with a CUTLASS projection from the real model:

```bash
python -m scripts.make_seed
python -m sako evaluate --candidate runs/validation/seed.json --run runs/validation
```

The seed is a test candidate. It has no assumed speed advantage.

## Design sources

- [Original CUDA swarm discussion](https://www.reddit.com/r/CUDA/comments/1q792ap/i_made_64_swarm_agents_compete_to_write_gpu/)
- [AutoKernel](https://github.com/RightNow-AI/autokernel)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [Harness design study](https://arxiv.org/abs/2609.20804)

The design uses measured feedback, several search groups, a persistent plan,
and limited context. These sources do not prove that eleven roles are optimal.
SAKO must measure agent count, API cost, and model speed in separate experiments.
