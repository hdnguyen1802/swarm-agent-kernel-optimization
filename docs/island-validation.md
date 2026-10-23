# Island policy validation

The validation uses controlled agents and simulated evaluator results.
It does not establish GPU performance or distinct behavior from the live LLM.

The three-round test runs the actual search controller. It produces 24 proposals.
The policy schedules four transfers after round two. The four designated writers
use these transfers in round three. No writer uses a migration in rounds one or two.

The regression tests also check:

- An empty island starts without a parent, even when foreign candidates exist.
- A normal round selects an eligible local parent.
- Failed local candidates do not enable a foreign fallback.
- Migration rejects failed, blocked, or unknown tasks.
- Repairs keep their source, task, and writer.
- A foreign parent without a matching event fails before the model request.
- Altered tasks, writers, or migration rounds fail validation.
- Resume retains the saved event and parent decision.
- Completed assignments remain available for audit after a later model rejection.
- Independent duplicate proposals retain their origin and island memberships.
- Each island receives different experiment directions.
- The report includes selections even when no proposal returned from the API.

The live Gemini and V100 search was not rerun for this policy change.
The numerical evaluator and its acceptance limits are unchanged.
