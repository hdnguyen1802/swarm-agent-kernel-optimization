# Colab GPU worker

Colab can replace the V100 worker. Its GPU and software need new measurements.
Use a new run and profile. Do not reuse V100 timings, fixtures, or accepted results.
Keep the model revision, weights, precision, inputs, and test limits fixed.

## Optional direct SSH worker

The existing automated controller uses SSH.
If a reachable SSH worker becomes available, its connection description can use
the separate configuration helper. It must supply `host`, `port`, `user`, `python`,
`cuda_home`, `host_key`, and `host_key_fingerprint`.
Keep the private key on the controller and save the description under `.local/`.
Generate a separate config and host key file:

```powershell
.venv\Scripts\python -m scripts.configure_colab --connection .local/colab_ssh/connection.json --output-dir .local/colab-connections/session-001
```

The helper checks the host key fingerprint supplied through Colab.
It does not accept a key obtained from an unverified network scan.
It changes only worker settings. It preserves the model and evaluator settings.
SSH and SCP use the assigned port and the separate host key file.
Each new runtime needs a new connection directory and host key verification.
The old worker config remains available for its old evidence.

## Check the GPU

Sync the project source:

```powershell
.venv\Scripts\python -m sako sync --config .local/colab-connections/session-001/config.json
```

Before package or model setup, run this command in the remote project folder.
Use the Python and CUDA paths reported by the notebook.
Set `CUDA_HOME` and add its `bin` directory to `PATH`.

```bash
python -m scripts.gpu_preflight --output runs/colab-001/preflight
```

The check does not load a model or call an agent API.
It reads the GPU target, compiles a small Triton kernel, and verifies its output.
Each attempt saves the Triton probe source, commands, output, and status in a new directory.

## Prepare model tests

Check the actual GPU, available memory, driver, compiler, and PyTorch build.
Select a compatible environment before installing the project GPU dependencies.
Do not assume that the V100 environment matches the new GPU.
Keep weights outside the source checkout. Keep agent credentials on the controller.

Run the worker doctor, collect a new profile, and measure a fresh complete baseline.
Use the separate Colab config and `runs/colab-001` for those commands.
The GPU must have enough memory for the original FP16 model and workloads.
Do not use quantization or smaller inputs to make a model gain claim.
Confirm a candidate against the complete model under `torch.compile` before reporting a gain.

A Colab reset can remove remote files. Fetch the evidence before the runtime ends.

[Colab rules](https://research.google.com/colaboratory/faq.html)
