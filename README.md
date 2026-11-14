# Swarm Agent Kernel Optimization

The candidate backend is Triton (`kernel_backend="triton"`). Writers
return Python source with `backend="triton"`, one or more
`@triton.jit` kernels, and `sako_run(tensors, scalars)`. The wrapper allocates output
tensors and launches kernels. Host compute, external compute libraries, file access,
benchmark access, precision changes, and suppressed errors are rejected by the
source policy. This policy check is not a security sandbox.

The GPU evaluator loads the saved Python source and lets Triton compile each shape
before timed trials. Definition load time is not JIT compile time. First launch
records and per-case setup times retain compilation work outside measured samples.
The default `triton_integration="direct"` exposes the validated Triton entry to
`torch.compile`. Eager accuracy checks keep the logged execution path. Legacy
configs can use the opaque Triton custom operation.
Build keys include backend policy, source, and the GPU/software fingerprint, which
includes the installed Triton version. Register and spill reports remain unknown
when no stable Triton metadata is available. Full errors remain in build and job logs.

The default run targets **batch 1, sequence 128 only**. Profiling, writer tasks,
kernel tests, supplemental layer checks, and full model tests use this one declared
workload. Select a different target by editing the single entry in `workloads` in
`configs/project.json`. Use a new run and prepared profile after changing it.
Results apply only to the declared workload, GPU, precision, and forward mode.
They do not establish performance on other input shapes or during cached decoding.
Historical runs keep their original scope and decisions.

The kernel gate requires a 1% gain. The default model gate accepts any measured
positive gain on the declared target. A config with multiple workloads also requires no median regression
on its non-target workloads. The eager accuracy
reference stays fixed. The default search budget is one million tokens. Use a new run and prepared profile.
New candidates must declare `backend="triton"`. Other candidate backends are rejected.
Saved reports and raw trials remain available for inspection.
The V100 probe passed copy and FP16 matrix multiplication with Triton 3.1.0.

SAKO uses one coordinator and up to five specialist writers to improve Triton kernels.
The writers work in parallel. A separate GPU worker compiles, checks, and measures
candidates. Only the evaluator can accept a measured gain.
Default acceptance combines eager layer accuracy with complete compiled model speed.

Set `model_id` and `revision` to select a decoder-only Hugging Face checkpoint.
The loader uses `AutoModelForCausalLM` in FP16. It checks each supported operation
against the actual model. Other operations stay in the model. The complete
model baseline uses `torch.compile` with `fullgraph=True`. The current GPU worker
has a **Tesla V100 with 32 GB**. Device discovery selects the CUDA target. A different
GPU needs compatible driver, PyTorch, and Triton builds.

## Configure a private run

Files under `configs/` are templates. They use example server and Google project
values. Keep your working settings in `.local/`, which Git ignores.

From the project folder, create a private copy if one does not exist:

```powershell
New-Item -ItemType Directory -Force .local | Out-Null
if (!(Test-Path .local/my-config.json)) {
    Copy-Item -LiteralPath configs/generation.json -Destination .local/my-config.json
}
```

Set the model and revision in that copy. Set `worker.host`, `worker.identity`,
`worker.python`, and `worker.cuda_home` for your setup. For Vertex, also set
`agent.project` and `agent.location`.
For two matching GPUs, set `worker.gpu` to `0` and `worker.archive_gpu` to `1`.
Each model must fit on one GPU. Each GPU tests a separate candidate.

The default `agent.provider="vertex"` uses Google application credentials on the controller. For a credential
file, set its path before starting the terminal:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\vertex-credentials.json"
```

Keep credentials and model weights outside Git. Restricted checkpoints also need
Hugging Face access on the GPU server. Keep that server and the controller online.
Pass `--config .local/my-config.json` to commands that need your working settings.

Current validation results are in [docs/validation.md](docs/validation.md).
Full run data and old development reports stay in local folders outside Git.

## Select the agent API

The coordinator and writers support two providers: `vertex` and `deepseek`.
Select one provider for a run. Changing the provider, agent model, or request
settings requires a new run. This selection is separate from the Hugging Face
model that the harness optimizes.

For DeepSeek, copy `configs/generation-deepseek.json` into `.local/deepseek.json`
and set the worker fields. The template uses these agent settings:

```json
{
  "provider": "deepseek",
  "model": "deepseek-flash",
  "api_key_env": "DEEPSEEK_API_KEY",
  "thinking": "enabled",
  "reasoning_effort": "low"
}
```

This is the agent section only. The full template also supplies the model
context and output limits, search budget, workload, and evaluator settings.
DeepSeek needs no Google project or Google credentials. Its adapter uses Python's
HTTPS client and does not need another SDK. Keep its API key in the controller's
environment. Do not put the key in JSON, source files, or chat.

In PowerShell, enter the key without placing it in command history:

```powershell
$key = Read-Host "DeepSeek API key" -AsSecureString
$env:DEEPSEEK_API_KEY = [System.Net.NetworkCredential]::new("", $key).Password
.venv\Scripts\python -m sako api-check --config .local/deepseek.json --run runs/deepseek-api-check
.venv\Scripts\python -m sako ui --config .local/deepseek.json
```

The terminal review and overview show the selected provider and agent model.
Both providers use the same token ledger, request limits, retry rules, and GPU
acceptance checks. DeepSeek API keys and error response bodies are not saved in call logs.

DeepSeek uses [Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)
at `https://api.deepseek.com`. The adapter requests JSON output and adds the
coordinator response structure to the system message. The controller still
validates the returned plan and candidate code. The API does not enforce our
coordinator schema. See [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/).

DeepSeek does not expose a public input-token counting endpoint. Before a call,
the adapter reserves a conservative UTF-8 byte bound for the full messages,
including JSON instructions and framing headroom, plus the output cap. It labels
this as a local bound in the saved metadata. It replaces the reservation with
the returned `usage.total_tokens` after the call. Reasoning tokens are not added
again. A timeout or missing usage retains the reservation. A conservative bound
can defer a request even when its exact token count might fit.

Set verified `model_context_tokens` and `model_output_tokens` for the chosen
DeepSeek model. The template uses a conservative 1,000,000-token context limit
and the documented 393,216-token output ceiling. Its requested output remains
8,192 tokens. Check [DeepSeek model limits](https://api-docs.deepseek.com/quick_start/pricing/)
when changing agent models. The GPU workload and numerical limits do not change
when selecting this template.

## Time answer generation

Use `configs/generation.json` as the template for cached generation. Set `model_id` and `revision`
in your private copy to select another checkpoint. The model must support cached
causal inference and full-graph compilation on the selected GPU. The checkpoint
must fit on one device. The harness uses the tokenizer from the same revision.
It records the actual input IDs. Token counts can differ between models.

```powershell
.venv\Scripts\python -m sako ui --config .local/my-config.json
```

Start a new run. The example uses a fixed prompt, batch 1, and 128 new tokens.
Each trial starts with an empty KV cache. The model processes the prompt once.
It selects the first token from the prediction scores. It then makes 127 calls
with one input token each. Greedy selection continues through the requested length.
The prompt comes from the [public generation benchmark](https://github.com/RightNow-AI/qwen3.5-triton/blob/dd9d56644999849992286761ee0ca4737dcebd80/benchmarks/batch_benchmark.py#L36).

The profiler measures GPU work for every token. The evaluator saves operation
inputs at prefill, first decode, and final decode. The search uses these shapes
and the measured costs for this model. A model trial checks the selected tokens,
all next-token scores, and the keys and values in each cache layer.

The generation config selects `generation_objective="decode"`. The clock starts
when the GPU finishes the first output token. It stops after the last output token.
The report gives milliseconds per decode token, tokens per second, and speedup.
Each GPU times the original compiled model and its candidate on the same inputs.
The run files retain the raw samples.

Search uses each target's share of profiled decode time. It screens candidate
speed on saved decode inputs. A promising kernel gets one initial correctness
call for each required input layout. A typical projection has two layouts, one
for the prompt and one for decode. The harness keeps all candidate code and results.
For each target, the harness saves its four fastest eligible kernels. It tests
ranks 1 and 2 in the complete model, using one GPU per candidate when two GPUs
are available. Both use the same accepted model as their starting point. If
both fail, the harness tests ranks 3 and 4. The fastest passing version in
the tested group becomes the base for the next target. If all fail, the current
model stays in use. Skipped kernels remain available for refinement.
After search, the harness runs a fresh correctness and
timing test on the combined model. The main table compares normal PyTorch,
`torch.compile`, and the selected model. Its speed ratios use normal PyTorch.
The report also keeps the compiled reference and the previous accepted comparison.

## Model measurement policy

The search groups candidates by code structure as well as measured resources.
It keeps competitive groups within 20% of the fastest local time available as
parents. Finalists still rank by measured time saved. Writer goals cover two
different refinements of the best design, another design, a fresh approach,
and a merge or repair. The writer selects the implementation.

The default task budget reserves up to 20% for corrections after model tests.
Small shares leave room for a writer and coordinator review before reserving
repair tokens. Each revisit starts with its own share. A visit with no new
kernel measurement skips repeat model tests.
If every finalist fails, correctness and compilation failures can use this reserve. The harness
checks the changed candidate alone before a writer repairs a combined failure.
This check collects correctness evidence. Repair prompts include failed values,
error limits, the isolation result and the accepted kernel sources. Repaired
kernels must pass local checks and a fresh complete-model test.
Unused tokens pass to later tasks.

The worker caches successful tuning samples by kernel, launch configuration,
input layout, GPU and compiler identity. It reuses matching samples to choose
settings. Each candidate still receives its own acceptance tests.

The generation config selects `measurement.model_sampling="saved_reference"`.
Each GPU runs the original compiled model for three warmups and three timed
generations. Each new candidate must pass correctness before its three warmups
and three timed generations. A GPU reuses its saved original and accepted
times for matching settings. If a winner came from the other GPU, this GPU
measures it once before reuse. Only decode time counts toward the speed test.
Unstable timed samples cannot pass. The final combined model gets a fresh test.
The PyTorch reference also uses three warmups and three timed generations.
The harness runs the initial profiles one GPU at a time when this reference is
enabled. This prevents a profile on one card from delaying eager CPU dispatch
on the other. Kernel tests and finalist model tests still use both GPUs.
Unstable PyTorch samples remain saved and block preparation.

When a request exceeds its task budget, the controller records the full required
reservation. It uses that amount in later admission checks. This prevents repeated
token counts for a request that cannot fit. The total token limit stays fixed.

During kernel tests, a short screen samples the first, middle and last sites
of each matching operation and layout. Before rejecting an entire design,
the harness checks the other sites for a local gain. Confirmation starts
with five timing pairs. It uses the full trial count when results vary or
lie near a speed limit. Each GPU can reuse the compiled reference graph.
Full raw results remain on the worker. To copy a job's full result and log,
use `python -m sako fetch-evidence --config CONFIG --run RUN --job-id ID`.

The standard forward-pass config uses adaptive paired timing.
Its original model runs five times through the normal compiler path. The
profile saves these times, actual input IDs, cache type, checkpoint revision,
GPU identity, software, math settings and compiler settings.

For paired timing, warmup runs have separate records. The first test finds the first stable window
of three warmup runs, up to twelve runs. Later tests reuse that count and extend
warmup if needed. Compilation and warmup stay outside acceptance samples.

Profiling starts with five traces. The harness compares target order in the
first two and last three trials. It also checks variation in the target costs.
An unstable ranking gets another five traces, up to fifteen. The profile reports
uncertainty if the ranking remains unstable. Profiling checks the output scores,
tokens and cache. It reuses the saved starting times. The terminal's baseline
stage also reuses these times and does not load model weights.

For paired timing, final validation checks accuracy and starts with five timed pairs,
five runs of the original model and five runs of the proposed model. Pair order
alternates. Another five pairs run when either timing range exceeds 2%, the
median ratio lies within 0.5% of the acceptance limit, or the estimated interval
crosses that limit. The interval uses paired log ratios and conservative
[Student-t critical values](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm).
It estimates variation under that model. It does not establish independence of
the timing samples.

All collected samples set the result. The harness reports `uncertain` when
ten pairs still do not give clear evidence. An uncertain result cannot replace
the accepted model. Operation races keep their separate trial count. Configs
without the adaptive policy keep fixed sampling.

## Terminal interface

From the project folder, install the interface and start it in a terminal.

```powershell
.venv\Scripts\python -m pip install -e ".[ui]"
.venv\Scripts\python -m sako ui --config .local/my-config.json
```

Select **Optimize a model → Start a Colab session** to request a GPU.
Select the GPU, the action, and a new run folder. Review the settings and press **Start**.
Colab uses compute units during installation and execution. SAKO checks the assigned GPU
and its memory. It installs the GPU dependencies on the runtime. Agent calls run on the controller.
**Verify the GPU worker** runs the hardware and software checks without a model search.

This workspace can use its signed-in project launcher. For a new setup, install the
Colab CLI with the `colab` extra and sign in with `colab usage` before opening the UI.
Install the `agent` extra and set up the provider credentials for a search.
The controller needs OpenSSH. You can set `SAKO_COLAB_CLI` to a Colab executable.
You can also select the configured SSH worker or a prepared Colab connection file.

The controller saves its log and status under `runs/<name>/ui/attempt-*/`.
For Colab, it downloads GPU evidence, checks its SHA-256 hash, and requests release
of its named session. The dashboard shows any error. Press **F** to retry an unconfirmed release.
A new Colab allocation needs a new run and profile.

Use the arrow keys and Enter in setup. A run opens on **Timeline**. It shows each
task, the strategies the coordinator proposes, and the work of each writer.
Kernel tests show the measured ratio and the reason for a failed trial when available.
The full model line shows the result of the test against `torch.compile`.
Scroll up to read earlier events. Press **End** to follow new events.
Press **T** for the timeline, **O** for the overview, **C** for candidates,
**L** for the log, or **R** for the report. **P** requests a stop at a saved boundary.
**Q** closes the view. A started controller continues in the background.
Open a run again with `python -m sako ui --run runs/<name>`. Press **A** for run actions.
A saved run opens without a GPU connection or agent request.

## Default swarm

`configs/project.json` selects `agent.search_mode="swarm"`. One coordinator supplies
distinct hypotheses to at most five active writers. One hypothesis creates one
assignment. Fewer hypotheses create fewer assignments. Extra hypotheses wait in a
saved queue. A free slot can start another experiment before all writers finish.

| Specialty | Experiments |
|---|---|
| Memory | Coalescing, vector loads, cache reuse, and less data movement |
| Compute | Arithmetic, reductions, and supported Tensor Core use |
| Scheduling | Tiles, block sizes, work distribution, and synchronization |
| Fusion | Fewer launches and temporary tensors, with required rounding preserved |
| Registers | Register use, spills, and resource limits |

Specialties guide each experiment. They do not own private candidate archives.
Each assignment records its parent, hypothesis, evidence, specialty, and lessons.
Duplicate changes from the same parent are rejected before generation, even if the
explanation changes. A writer can use any eligible parent from the shared archive.

The coordinator reads hardware facts and measured results. It replaces the separate
hardware adviser, performance analyst, and integration adviser in the default mode.
There is no additional LLM role for profiling, storage, or acceptance.

The API gate permits at most five simultaneous requests. Quota errors lower this
limit temporarily. Successful calls restore it gradually. When writers finish
together, their kernels enter one GPU queue. Free writers can start the next
approved idea while the GPU tests a kernel. The controller saves
timings and correctness results with the candidate ID and test case. By default
the coordinator waits for the batch before writing a new plan. It can also ask
to plan after the next measured result. New assignments read completed lessons.
Writers already at work finish with their original prompt.

On an SSH host with two matching GPUs, set `worker.gpu` and `worker.archive_gpu`
to their device numbers. Each GPU prepares its own profile and saved inputs.
During decode search, each free GPU tests a different candidate against its own
compiled reference. The four versions for each target run in batches of two. The harness
compares the speed ratios from each device. Both devices share one API budget.
For a forward benchmark, the second GPU measures candidates for archive ranking.

The project enables `agent.revisit_with_budget`. After the first pass, the
controller returns to supported tasks while it can fund a writer and a review.
It ranks tasks by their measured GPU cost, divided by one plus their revisit count.
A large ratio on a small operation cannot displace a much larger runtime target.
Each revisit needs a new measured candidate before the coordinator can move on.
The controller saves the visit with the task queue so a stopped run can resume.
A slow parent does not force another refinement when a fresh algorithm is needed.

The coordinator assigns an objective and cites measured evidence. Each writer
chooses its algorithm, launch settings and verification checks. A writer returns
its design and source in one response. It can replace a coordinator suggestion
and record its reason. The evaluator checks the submitted design against the
actual launches. The original proposal remains in the hypothesis history.

The coordinator receives grouped fixture cases and summaries of active evidence.
Writers retain the complete operation contract. Full measurements and code stay
in the evidence files. The default coordinator target is 24,000 characters.

## Search flow

1. Read the GPU facts and software. Select a supported compiler target.
2. Load the model and pin its exact revision.
3. Capture inputs separately from timed profiling.
4. Rank regions by measured GPU time. Search the smallest verified operations inside each region.
5. Verify explicit operation contracts.
6. Let the coordinator assign objectives, specialist perspectives and parents. Writers choose their algorithms.
7. Generate candidates in parallel. Compile, check, and time each on the GPU.
8. Check each writer's declared launch details outside the timed trials.
9. Keep a kernel only when every required case passes accuracy and gains at least 1% throughput over the current kept kernel. The first baseline uses `torch.compile`.
10. Save results, lessons, parent choices, repairs, and stopping reasons. Repeat the kernel search.
11. Test each promoted kernel in the accepted model while its writer can still correct a failure. Check eager layer outputs and complete model outputs. Compare complete model timing with both `torch.compile` and the current accepted model. `integrate` also remains available after search stops.

The default `kernel_search.enabled=true` separates kernel keeps from model acceptance.
It compares each candidate with the current kept kernel in fresh paired GPU trials.
All required accuracy and input checks must pass. Each retained shape must pass the
1.01 speed ratio, trial count, and p95 limit against its current kept kernel. A writer
can refine another archive parent. A failed trial does not replace the kept kernel.
The archive records fresh paired timings against `torch.compile`. For cached
generation, it ranks parents by estimated saved model time from the measured
prefill and decode costs. Repeated seeds and cache samples count once per phase.
A shape variant can keep the original operation in a phase where it loses speed.
With one GPU, it reuses the acceptance timing pair until a kept kernel exists.
It then runs a separate pair against `torch.compile`. With an archive GPU,
all archive timings come from that GPU, including the first pair.
The score is the geometric mean of the per-case ratios. Missing ranking samples
leave the score unknown. Select `score_reference="fixed_anchor"` to use saved
reference times instead.

The search does not run a full model test after each kernel keep. It saves the kept
combination in `manifest.json`. Its final status is `completed_kernel_search`.
Run the later integration step with the same config, run, and prepared profile:

```powershell
.venv/Scripts/python.exe -m sako integrate --config .local/my-config.json --run runs/main
```

Integration tests the current kept combination against `torch.compile`. It declares
target workloads from the manifest before timing. The default model gate requires
a measured positive gain. In a multi-workload run, non-target workloads need a ratio of
at least 1.0. The default run tests only batch 1, sequence 128. Every declared workload
keeps its p95 gate and declared sampling policy. Every replaced layer must match its original
eager output within the configured limits. Attributed failures can remove paths;
the smaller replacement set must pass a fresh complete model test. Reports keep
all failed trials. Export still requires a measured complete model gain.

With `model_trial_base="accepted"`, each new region is tested with the last accepted
model. Final acceptance still requires the configured gain against `torch.compile`.
An added kernel must also improve the accepted model. With adaptive sampling,
the harness compares both measured gains against the same original model. It
uses the saved accepted result and collects no extra generation runs. Overlapping
error ranges keep the accepted model. Fixed sampling uses a separate paired test
against the current model. The tail limit continues to apply. All samples stay
with the trial.
Failed model trials stay in the kernel archive. They do not become the base of
later model trials. Rejected combinations preserve the last accepted model.

The harness ranks supported regions by measured GPU time. A region can contain
library matrix multiplications, generated Triton kernels, or both. For example,
the swarm can search a gate projection and try a separate replacement for its
complete MLP. A target needs a verified operation contract and complete profiler
coverage. If a fused launch crosses the target boundary, the profiler excludes
that target. If the compiled trace yields no mapped targets, it saves the
profile and stops.

New configs omit `optimization_policy`. To start from an old strict config,
remove that field and use a new run and profile. Use `status` or `report` to
inspect the original run.

Deployment diagnostics reuse saved candidates without agent generation. The
worker can compare direct, opaque, and compiled paths under event timing and graph
replay. A rotating control uses real weights from representative layer paths.
Model traces record warmed baseline and candidate forwards after normal timing.
Trace summaries include kernel counts, GPU busy intervals, launch events, and
opaque CPU calls. These diagnostics do not replace the complete model gate.
An original-math wrapper control can identify integration cost. It cannot grant
candidate acceptance, even if its timing ratio passes the speed threshold.
Configs without `kernel_search.enabled=true` retain immediate integration.

## Lessons and candidate archive

The lesson store keeps conclusions and evidence links. Each record contains its
operation, shapes, strides, precision, hardware fingerprint, tested change,
correctness result, and timings. Agent explanations remain labeled hypotheses.
Measurements set the outcome. Full conversations do not enter each prompt.

Lessons use `supported`, `failed`, or `untested` status. Operation and model lessons
have separate scopes. A candidate can pass saved fixtures and fail at another layer. Infrastructure
errors do not become claims that a CUDA method failed. Corrections retain the old lesson and its raw evidence.

Recall requires matching hardware and operation context. The default supplies up to
eight lessons within 12,000 characters. Complete evidence remains in the run folder
and SQLite database. Prompt limits can reduce the recalled set.

The active archive holds up to four correct designs per operation. It keeps the
fastest design and favors different hypotheses or resource patterns for other slots.
A slower correct design can remain as another starting point. Candidates with model
correctness failures require explicit repair. Archive membership does not establish
a model gain. The best accepted model checkpoint remains separate.

The coordinator now starts from archive parents by default. It sees each parent's
measurements and prior use. When an ordinary archive parent is available and no
ordinary refinement is running, a new plan must include an ordinary refinement.
Repairs do not satisfy this rule. When a refinement is required, the controller
places it before fresh designs. A small budget cannot leave all archive work queued.
The coordinator can skip reuse only with `archive_skip`: a concrete reason and
evidence from an available parent. It must also explain each fresh start when
archive parents exist. Reports count ordinary refinements and repairs separately.

## Profiling and numerical diagnosis

The default `ranking_mode="compiled_profiler"` profiles the complete compiled
model after compilation and warm-up. It uses the same compile mode as the model
speed reference. Diagnostic options save source maps and unique kernel names.
They do not enable extra tuning. Compilation stays outside the timing samples.

The adapter links CPU operator self GPU times to Inductor launch order and source
nodes. Exact event names and counts must match. It never adds device events again.
The rank uses a median for each workload, then an equal mean across workloads.
Generated code, source maps, raw samples, and traces remain in the profile folder.
Agents receive compact compiled evidence and bounded generated kernel examples.

A target must contain each complete fusion group that touches its paths. The
adapter does not divide fused kernel time among source layers. A complete MLP can
be selected when it contains fusion across its child operations. Fusion across an
unsupported boundary stays in the report and cannot become a partial replacement.
Missing or uncertain mappings cannot create a priority. Selection uses a tree of
bottlenecks and supported subsections. It rejects overlapping paths. Eager capture supplies the
accuracy reference and verifies the original functional contracts.

The earlier `profiler_events` and `eager_regions` modes remain available. Configs
without a ranking mode retain the old region method. A mode change needs a fresh
profile and run. Ranking does not predict a model gain or measure cached decoding.

The evaluator can observe launch geometry and selected Triton settings for the fixture named in a verification check. It records these observations outside the timed trials.

A separate, untimed numerical diagnostic can compare selected replacement boundaries
under eager execution. It records the first exact difference, first tolerance
failure, and a local comparison using the same input. This is an **eager boundary
trace**. It does not observe internal boundaries in the compiled model. It cannot
accept a candidate. It remains separate from the complete layer checks used for acceptance.

## Setup

To replace the V100 worker with Colab, follow [the Colab setup](docs/colab.md).
Check GPU availability before model setup. Use a new run and profile.

Clone the project and create an environment:

```bash
git clone <repository-url>
cd swarm-agent-kernel-optimization
python -m venv .venv
```

On the Windows controller:

```powershell
.venv\Scripts\python -m pip install -e '.[agent]'
$env:GOOGLE_APPLICATION_CREDENTIALS = 'C:\path\to\service-account.json'
.venv\Scripts\python -m sako api-check --config .local/my-config.json --run runs/api-check
```

Set `worker.host`, `worker.identity`, and `worker.root` in the config. The SSH host
key must already be trusted. Credentials stay on the controller. Keep credentials
and model weights outside Git and candidate files.

Run `scripts/setup_gpu.sh` from the server checkout to create its Python environment.
The current V100 environment uses Python 3.11, PyTorch 2.5.1 with CUDA 12.1 wheels,
Triton 3.1.0, Transformers 4.48.3, and CUDA 12.3 for compilation. Discovery reports
unsupported targets. It does not replace the server environment automatically.

```powershell
.venv\Scripts\python -m sako sync --config .local/my-config.json
.venv\Scripts\python -m sako prepare --config .local/my-config.json --run runs/swarm-001
.venv\Scripts\python -m sako baseline --config .local/my-config.json --run runs/swarm-001
.venv\Scripts\python -m sako search --config .local/my-config.json --run runs/swarm-001
.venv\Scripts\python -m sako integrate --config .local/my-config.json --run runs/swarm-001
.venv\Scripts\python -m sako report --config .local/my-config.json --run runs/swarm-001
```

These commands use your private configuration. The command-line default is the
`configs/project.json` template. `search` prepares missing profiles. `prepare` runs
that stage explicitly. Swarm preparation reads hardware
facts without a separate hardware LLM call. `--profile` selects another prepared
directory. Source, hardware, workload, or test changes require a new profile and run.

`sync` sends an explicit file list. It excludes credentials, Git data, weights, and
run files. Triton builds candidate kernels on the worker. Profile fixtures stay on
the server; the controller stores task descriptions and evidence metadata.

## Acceptance and reports

The default workload has `[batch, sequence]` size `[1,128]`. All three correctness
seeds and the required layer coverage remain active for that workload. The model
gate does not require results for `[1,512]` or `[2,128]` in this run. Those shapes
need separate declared runs before a gain can be claimed for them.
The complete forward pass computes logits for every token. This is not a serving
benchmark or a text generation loop. Agent response time is separate from model time.

The evaluator warms both paths and alternates their order. It records CUDA event
time and synchronized wall time. Compile time is excluded. The ratio is
`compiled_baseline_ms / candidate_ms`. Every case must pass correctness, its
configured speed gate, and a p95 latency ratio no greater than 1.05.

Default operation tests compare three candidate calls with the saved original eager
output for that exact fixture. The capture binds this output to its module path,
workload, seed, and case index. A compiled-baseline difference is diagnostic only.
The default configuration
uses `correctness_scope="replaced_layers"` and `correctness_reference="eager"`.
The evaluator visits every replacement path for every configured workload and seed.
It gives the candidate the exact input and actual weights of that eager module.
It compares the candidate output with the original module output. Original outputs
continue downstream, so earlier candidate errors cannot change later reference inputs.
Missing paths, invalid outputs, and input mutations fail this check. No diagnostic
path limit can reduce acceptance coverage.

The speed reference remains `model_reference="torch_compile"`. After the layer checks,
the evaluator times both complete compiled models with hooks removed. If a rejected
mutation cannot be restored, it stops later checks and all timing on that model.
Weights, input values, precision, seeds, and numerical limits stay fixed.

This policy has no final-logit numerical gate. A local layer pass does not establish
final-output equivalence. Reports identify the accuracy scope and speed reference.
Older runs without these policy keys retain their original final-output gate.
An empty manifest has no replaced layers to check; it remains a timing control.

`sako report` writes `report.md` and `report.json`. They show candidate lineage,
hypotheses, operation gate reasons, model gate reasons, repair outcomes, lesson use,
API use, observed writer overlap, and task stopping decisions.
Each stop identifies whether the coordinator or controller made the decision.
Raw candidate files, timing samples, compiler output, and failed trials remain.

`export` requires a measured complete model gain. Correct kernels and archive parents
can exist even when no export is accepted.

## Budget, stopping, and recovery

The default cap is **1,000,000 total API tokens**. All roles
share one ledger. Concurrent requests reserve tokens before starting. The ledger
counts reported input, output, and reasoning use. Unknown outcomes retain their
reservations. Auxiliary requests and generation attempts have separate counts.

The controller estimates costs from the median of the latest 12 successful calls
for each role, with a 15 percent margin. Before data exists, it estimates twice the
output limit. It admits the next experiment plus a coordinator reserve. It does not
require a three-round migration cycle.

Each task receives an allowance from the remaining tokens and remaining priorities.
This protects later operations from unrestricted early refinement. It cannot guarantee
that all priorities receive a candidate. Actual API use can differ from estimates.
The hard global cap still applies. `max_task_experiments` is an optional guard and is
`null` by default. Legacy generation and migration fields do not impose swarm cycles.

Each logical hypothesis permits **one initial attempt and two corrections**.
Corrections keep the hypothesis ID, failed source, operation, workload, and evidence.
Reassignment to another writer or resume does not reset this limit. Exhaustion closes
that hypothesis. It does not close the operation; the coordinator can propose another
distinct experiment.

A later integration check can reopen the original hypothesis for a numerical
correction. The writer receives the failed boundary, workload, seed, and exact
actual, expected, and allowed values. The original operation result stays intact.
The controller removes an attributable failed keep from future manifests and
saves it with the repair evidence. Other candidates remain available. A correction
must pass the operation gates and revisit integration before model acceptance.
Combined failures require isolated checks before the controller assigns blame.

In the default kernel search, correct measured code closes its error-repair
attempt and becomes an archive parent. A new performance hypothesis can refine
that parent under the same token cap. Noise-aware progress resets stagnation even
before the candidate passes acceptance. At least one ordinary archive refinement
must be measured before the coordinator can skip an available archive.

API, source, compile, execution, and transfer errors do not count as a performance
plateau. Numerical and timing results can update the progress record. A plateau
counter alone does not close an operation. To move on, the coordinator must cite
measured GPU evidence from that operation and explain why further work has low
expected value. This rule applies to every operation type.

The controller also enforces the token budget, user STOP, and optional experiment
cap. Infrastructure and request failures remain distinct stop reasons. Pending and
unfinished work remains visible. Transfer recovery retains the same attempt identity.

The coordinator has a 150,000 input-token ceiling. This is not a target size. Prompt
construction uses compact evidence and a smaller character limit. Every request
checks the provider context limit. Update documented model limits when changing models.
The soft target is 45,000 characters. The hard limit is 90,000. Required contracts
and active repair evidence can exceed the soft target. One evidence registry supplies
both prompt IDs and validator IDs. Completed writers receive GPU evaluation before
the controller asks for another plan. Active writers can still receive new peers.
The coordinator receives summaries of parents, trials, and hypotheses.
Each request and rejected plan keeps a full evidence file. Retry requests use the
same character limit. Writers retain the full source of their selected parents.

The coordinator can declare observable changes to launch geometry and kernel
source before dispatch. The evaluator records each prediction and its result.
The planner uses compiled time per workload to estimate how much model time a
proposal could save. It queues independent trials and waits locally for results.
Search and integration share completed model measurements. Use `model-check`
with a saved manifest to make an independent measurement.

SSH transfers permit three attempts for temporary failures. Permanent authentication,
host-key, or permission errors stop immediately. Transfers use temporary files and
integrity checks. An uncertain disconnect recovers the existing result; it does not
dispatch the same GPU job again. Transfer errors are not CUDA correctness failures.

An existing run can use `python -m scripts.resume_paced` with its saved `--config`,
`--run`, and `--profile`. The driver spaces SSH/SCP starts by at least three seconds
and permits three controller recovery attempts. It preserves job IDs and the ledger.

The default worker uses `persistent=true`. It keeps one SSH stream, the loaded
reference model, and its compiled baseline between jobs. Small configs, candidate
files, and manifests travel in one checked request. Results and full job logs return
on that stream. Large artifacts can use the existing checked file transfer path.
The worker still measures the baseline beside each candidate. It keeps every seed,
workload, warmup, trial, numerical limit, and raw timing sample.

If any required layer check fails, the worker skips compilation and model timing
for all workloads. It keeps all local checks and records why it skipped timing.
A combination that passes accuracy but fails the speed gate does not trigger an
extra isolated model test. Accuracy failures still trigger an isolated test of
the new entry. The saved repair evidence identifies skipped isolated tests.

The resident worker holds the selected GPU lock until the connection closes.
When it holds model weights, a separate process tests kernels on saved fixtures.
Jobs that use the model, including
fixture capture, check its parameters against exact CPU copies. Unsafe state
retires the worker. The default also retires it after 64 jobs or 600 idle seconds.
Normal run completion closes it. Direct scripts must call `worker.close()` when
they finish. This process boundary is not a security sandbox.

Each request retains its ID, input hash, immutable config, and saved result. A
reconnect first recovers the original result. A durable job claim prevents a second
execution. A fixed job timeout saves a failed result and ends that process.
Older configs without `persistent` retain the original process-per-job path.

Raw results now include `worker_total_s`, `stage_times`, `resident_cache`, and
per-case `setup_times`. The first warmup time includes lazy compilation and its
first forward call; it is not a pure compiler timer. These fields stay outside the
measured trial interval. They show model reuse, reference reuse, checks, and setup.

Triton build keys include the source policy, source, and GPU software identity.
Each task keeps its own correctness checks. Build logs retain complete failures.
Request receipts keep each candidate and task identity.

Create `<run>/STOP` to stop new work and let active work reach a saved boundary.
Remove the file and repeat the same search command to resume. Resume reuses saved
responses and measurements. It does not reset the cap. Keep the run config fixed.

## Run compatibility

New runs use one Triton swarm. The earlier island controller and its reference
index are removed. Saved run folders keep their original config and evidence.
Saved report files remain available. Use a fresh run and profile with new source.

The controller audit checks only stopped swarm runs. It cannot convert an old
island run or reset its token ledger.

## Layout and tests

```text
sako/swarm.py          Asynchronous swarm controller
sako/swarm_policy.py   Hypothesis checks and experiment admission
sako/lessons.py        Scoped lessons and immutable evidence
sako/archive.py        Diverse shared candidate archive
sako/profiling.py      Checked transfer of diagnostic artifacts
sako/diagnostics.py    Untimed numerical boundary diagnosis
sako/layer_checks.py   Same-input eager checks at every replacement path
sako/reporting.py      Detailed measured results and stop reports
sako/loop.py           Shared evaluation and model checks
configs/generation.json Cached decode template
configs/project.json  Forward-pass template
configs/validation.json Regression test settings
docs/validation.md    Current validation summary
tests/                Tests that do not require a GPU
runs/                 Local evidence; excluded from Git
.local/               Private settings and old reports; excluded from Git
```

```bash
python -m unittest discover -s tests -v
python -m compileall -q sako scripts tests
```

Generated Triton source runs under the worker account. A process boundary is not
a security sandbox. The GPU lock coordinates SAKO jobs under that account; it
cannot stop jobs from another user. Static checks do not prove arbitrary code safe.

## Design sources

- [AutoKernel](https://github.com/RightNow-AI/autokernel)
- [KernelArc study](https://arxiv.org/html/2608.17071v2)
- [Triton](https://triton-lang.org/)

Five writer slots and four archive entries are configurable starting choices. They
are not proven optimal. Compare agent counts under equal candidate and token budgets.

## Bottleneck tree

The default `agent.target_selection="bottleneck_tree"` starts with the region
that takes the most GPU time. Within that region, writers get the smallest
verified operations with separate compiled timings, most expensive first.
If no smaller operation qualifies, the region itself is the target.
Set `agent.search_parent_regions=true` in a new run to also try a whole-region
replacement alongside the smaller operations.

The controller completes the active region before it visits another region. It
does not divide the token cap equally among a fixed set of four targets. The
coordinator closes a subsection with measured evidence. At a region boundary,
kept kernels receive required eager checks at each replaced layer. Containing MLP
checks are diagnostic unless the candidate replaces the whole MLP.
Only a model that passes the full speed gates can become the accepted model state.
A change to that state triggers a fresh compiled profile when another region
remains. The controller reorders future regions from that profile. It reuses the profile when the model is unchanged.
Saved decisions and the prepared catalog remain immutable during this process.

The budget stop drains already-generated outputs without more API calls. It does
not start new writers or relax any gate. Reports keep region decisions, profile
changes, the token ledger, and full errors. The earlier flat selection remains
available. Changing selection mode requires a new profile and run.

## Tuning, shape selection, and timing controls

The coordinator assigns hypotheses from measured hardware and profile evidence.
Specialist writers generate the first optimization candidates. The search does
not evaluate built-in optimization seeds or supply kernel templates to writers.
The standard configs enable harness tuning. The writer chooses the algorithm.
The harness adds at most six launch configurations to eligible Triton kernels.
Writers can also expose legal block sizes through standard Triton configurations.
The harness preserves the grid and kernel computation. It records a reason when
runtime launch settings cannot be changed safely.

The worker first checks the initial configuration on all required inputs. If it
passes, the worker tunes settings with small CUDA Graph tests. It saves every
trial, including failed configurations and raw timing samples. It then checks
the selected configurations and applies the original independent speed tests.
Tuning samples cannot establish acceptance. Complete model tests remain required.
The tuning policy is part of the saved profile contract.

An untimed call records each compiled kernel's register use, shared memory use,
and reported spills. Writers receive these values with the operation timings.
These compiler and driver values do not measure executed spill traffic or prove
the cause of a slowdown. This collection does not use Nsight Compute.

Kernel acceptance uses CUDA Graph replay to reduce Python launch gaps. Full model
timing still measures the compiled deployment path. A timing audit compares the
same kernel with itself, both directly and through the custom-operation wrapper.
It warms captured graphs before timing and saves event and graph samples.
The timing audit is an explicit diagnostic command. It does not run before the
coordinator or block writer requests. It cannot accept a kernel or relax a speed
limit. The worker doctor uses a trusted copy to test the Triton backend.
That diagnostic never enters the optimization archive. All calibration controls use
at least 100 repeats. This diagnostic count does not change the acceptance trials
or their limits.

Shape selection keeps a candidate only for the exact measured tensor shapes,
strides, dtypes, and scalar values that pass the 1% speed and p95 gates. All supplied
accuracy cases must pass before any shape is retained. Other shapes use an earlier
kept kernel or the original layer. Each shape compares with its current kept
baseline. The default model gate requires a measured positive gain. When a
config declares multiple workloads, its non-target workloads must avoid median
regression. All declared workloads keep the p95 gate. The default config declares
only `[1,128]`.

Before timing a candidate, the worker checks early, middle, and late paths against
saved eager outputs. It includes every configured workload and correctness seed.
The first capture runs eager forwards. Later candidates reuse the saved cases.
These extra cases do not replace the original timing fixtures. The worker also
saves up to three exact failure cases per task for later repairs. It shares saved
weights across cases. All numerical failure records remain available even after
the tensor capture limit is reached. Tensor archives stay on the worker. Writers
receive compact error summaries and an explicit numerical repair objective.

A required layer failure can restrict a kept candidate to paths that passed every
required check. Each restriction changes the manifest identity. The controller
then tests the smaller set against the full model before acceptance. Missing
coverage, input mutation, or state damage prevents this fallback. Operation repairs
still need to pass all supplied correctness cases; a restricted model scope does
not relax that rule.

Google credentials refresh under one shared lock before request dispatch. Temporary
refresh failures receive bounded retries. A proven failure before dispatch uses no
model tokens and does not spend a writer correction. Unknown generation outcomes
retain their token reservation. The controller pauses the same task after the
authentication retry limit.

Use a fresh run and profile after these changes. Old raw evidence remains valid
for its recorded source and software. It cannot certify the new timing path.
Run `python -m unittest discover -s tests -v` for CPU checks. On an authorized GPU,
run `python tests/gpu_search_improvement_checks.py` for the new ABI and timing
controls before a full model search. Synthetic checks do not establish a Qwen gain.


## Combine measured candidates

The default API model is `gemini-4-hydrogen`. The controller uses Google Gen AI
SDK 2.34 or later. The default search token limit is one million.

With `agent.candidate_merging=true`, the controller finds pairs in the active
archive that improve a shared shape. Shape identity includes all tensor shapes,
strides, data types, and scalar values. The task and hardware must match.
Candidates with known model correctness failures cannot serve as merge parents.

The coordinator receives the eligible pairs and their measurements. It can select
a pair through `hypothesis.merge.parent_ids` and `hypothesis.merge.shape_ids`.
It must cite both parents. It must propose an untried merge or explain a measured
conflict through `merge_skip`. Work in flight can finish before another merge.

The writer receives both complete sources and their measured results. It owns
the algorithm and settings. It returns one new implementation and explains each
parent's contribution. The controller rejects unchanged parent code. It records
both parent IDs, saves the exact request for resume, and tests the new source.
Parent speed ratios do not predict the merged kernel's speed. Only the new
measurements can establish a gain. All existing acceptance rules remain active.

## Read speed results in the terminal

Candidates shows one row per kernel. Each row gives the target, writer, GPU,
test result, local speed ratio, and shortlist rank for the target. `USED` marks
a candidate in the selected model. Those rows stay at the top. A ratio above
one means the kernel ran faster. A partial pass gives the number of passing
routes. The run files contain each input and its measurements.

The model table has three columns: PyTorch, torch.compile, and our optimization.
It shows milliseconds per token, tokens per second, and speedup versus PyTorch.
Prompt processing does not enter these decode ratios. A missing PyTorch timing
shows a blank value. Local decode ratios still compare the compiled operation.
