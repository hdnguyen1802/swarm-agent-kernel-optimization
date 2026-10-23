# Project rules

- Use short, clear English sentences. Write in the active voice.
- Keep credentials and model weights outside Git.
- Keep the evaluator separate from agent output.
- Use CUDA kernels. CUTLASS use is allowed and must be recorded.
- Test the complete model against `torch.compile` before you report a model gain.
- Do not change weights, precision, inputs, or test limits to obtain a gain.
- Keep failed trials and raw timing samples.
- Do not claim that a process boundary is a security sandbox.
- Run `python -m unittest discover -s tests -v` after core changes.
