# MIRA-Math

**MIRA-Math** is a benchmark for **minimal information requesting and mathematical reasoning**. It evaluates a narrow but important capability: when a mathematical problem is underdetermined from the solver's view, can a model identify the exact missing atomic fact, ask for it precisely, and then use it to compute the correct final answer?

Each instance is generated from a complete latent mathematical state with a unique answer. The solver, called **Agent A** in the code, receives a private view with exactly one necessary atomic fact removed. The information holder, called **Agent B**, receives only that withheld fact. At evaluation time, Agent A may issue natural-language requests under a strict request budget. Agent B is a fixed constrained responder: it must either return an `offer` containing the quoted private constraint when the request semantically matches the fact it holds, or return a `decline` otherwise.

MIRA-Math is therefore **not** a general multi-agent collaboration benchmark. The responder does not solve, explain, negotiate, or volunteer extra hints. The benchmark is designed to isolate three separable skills:

1. recognizing that information is missing,
2. formulating a precise request for the missing fact, and
3. integrating the returned fact into an exact mathematical solution.

The generators, typed hint specifications, instance validators, and final-answer verifiers are deterministic. Request-side metrics are measured under the fixed LLM-mediated responder channel used by the reference runner.

---

## What is in this repository?

This repository contains the reference implementation for MIRA-Math:

- `mira_math/`: Python package for instance generation, validation, prompts, schemas, and scoring.
- `mira_math/families/`: 22 typed mathematical problem families.
- `examples/`: reference LangGraph runners for single-instance and dataset-level evaluation.
- `datasets/sample/` and `examples/sample_instance.json`: small sanity-check artifacts.
- `tests/`: smoke tests for generation, validation, and scoring.
- `.env.template`: environment-variable template for OpenAI, Google GenAI, OpenRouter, and local OpenAI-compatible servers.

The paper release uses the **20/50 typed dataset**: 20 instances per difficulty for each Type-A family and 50 instances per difficulty for each Type-B family, across three difficulty levels. This yields 660 Type-A instances, 1,650 Type-B instances, and 2,310 instances total.

---

## Installation

```bash
git clone https://github.com/cedar-lau/mira-math.git
cd mira-math
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the package:

```bash
# Core package: generators, validators, and scoring
pip install -e .

# Full runner + tests
pip install -e .[agents,dev]
```

The `agents` extra installs LangGraph and LangChain backends used by the reference runner. The `dev` extra installs `pytest`.

---

## API keys and model configuration

Copy the template and fill in the providers you want to use:

```bash
cp .env.template .env
```

Supported backends are selected by model name:

| Model name pattern | Backend | Example |
|---|---|---|
| `gpt-*`, `o*` | OpenAI | `gpt-4o-mini`, `gpt-5.1` |
| `gemini-*` | Google GenAI | `gemini-2.5-flash` |
| `openrouter/<model>` | OpenRouter through an OpenAI-compatible endpoint | `openrouter/google/gemma-4-31b-it` |
| `local/<model>` | Local OpenAI-compatible server | `local/llama-3.1-8b-instruct` |

Model precedence in the runners is:

```text
--model-a / --model-b
MIRA_MATH_MODEL_TIER_A / MIRA_MATH_MODEL_TIER_B
MIRA_MATH_MODEL_A / MIRA_MATH_MODEL_B
MIRA_MATH_MODEL
```

In the paper protocol, Agent B is fixed to `gpt-4o-mini`; Agent A is the solver model being evaluated.

---

## Quick start without API calls

Create local output folders:

```bash
mkdir -p datasets/generated logs
```

Generate a small smoke-test dataset:

```bash
python -m mira_math.generate \
  --family mix \
  --n 5 \
  --difficulties 1,2,3 \
  --seed 0 \
  --out datasets/generated/mix_smoke.jsonl
```

Validate the generated instances:

```bash
python -m mira_math.validate \
  --in datasets/generated/mix_smoke.jsonl
```

Run the tests:

```bash
pytest tests/
```

---

## Generate the paper-style datasets

Zero-shot dataset:

```bash
mkdir -p datasets/generated

python -m mira_math.generate \
  --family typed \
  --n-a 20 \
  --n-b 50 \
  --difficulties 1,2,3 \
  --seed 1234 \
  --out datasets/generated/family_types_20_50.jsonl
```

Four-shot dataset:

```bash
python -m mira_math.generate \
  --family typed \
  --n-a 20 \
  --n-b 50 \
  --difficulties 1,2,3 \
  --seed 1234 \
  --few-shot \
  --out datasets/generated/family_types_20_50_few_shot.jsonl
```

Validate either dataset with:

```bash
python -m mira_math.validate \
  --in datasets/generated/family_types_20_50.jsonl
```

The typed generator creates:

```text
11 Type-A families x 3 difficulties x 20 instances =   660 instances
11 Type-B families x 3 difficulties x 50 instances = 1,650 instances
Total                                                   2,310 instances
```

The evaluated datasets ship with the repository under `datasets/`. Verify them
against `datasets/CHECKSUMS.txt` before a run:

```bash
sha256sum -c datasets/CHECKSUMS.txt
```

Generation is deterministic: the zero-shot command above reproduces
`datasets/family_types_20_50.jsonl` byte for byte on the same platform. It is
written to `datasets/generated/` so that a regenerated copy never overwrites
the distributed, checksummed one.

---

## Run the reference protocol

Run one sample instance:

```bash
python examples/run_single_langgraph.py \
  --instance examples/sample_instance.json \
  --method llm \
  --model-a gpt-4o-mini \
  --model-b gpt-4o-mini \
  --seed 1234 \
  --run-id sample_gpt4omini
```

Run a dataset sequentially:

```bash
python examples/run_dataset_langgraph.py \
  --in datasets/family_types_20_50.jsonl \
  --method llm \
  --model-a gpt-4o-mini \
  --model-b gpt-4o-mini \
  --seed 1234 \
  --log-dir logs \
  --run-id gpt4omini_zs
```

Run a dataset in parallel:

```bash
python examples/run_dataset_langgraph_parallel.py \
  --in datasets/family_types_20_50.jsonl \
  --method llm \
  --model-a gpt-4o-mini \
  --model-b gpt-4o-mini \
  --seed 1234 \
  --workers 8 \
  --instance-timeout 3600 \
  --log-dir logs \
  --run-id gpt4omini_zs_parallel
```

Run a resilient multi-wave evaluation that retries unfinished instances:

```bash
python examples/run_dataset_langgraph_resilient.py \
  --in datasets/family_types_20_50.jsonl \
  --method llm \
  --model-a gpt-4o-mini \
  --model-b gpt-4o-mini \
  --seed 1234 \
  --workers 8 \
  --instance-timeout 1200 \
  --max-waves 20 \
  --wave-pause 30 \
  --log-dir logs \
  --run-id gpt4omini_zs_resilient
```

The runners save per-instance metrics, aggregate metrics, transcripts, and logs under `logs/`.

---

## Oracle-hint baseline

`acc_final` confounds three abilities: noticing that a fact is missing, asking
for it in a way the responder accepts, and solving once the fact is in hand. The
oracle-hint condition isolates the third. Agent A is handed the resolving fact up
front and asked only to solve; Agent B is never called.

```bash
python examples/run_dataset_langgraph_parallel.py \
  --in datasets/family_types_20_50.jsonl \
  --oracle \
  --method llm \
  --model-a gpt-4o-mini \
  --seed 1234 \
  --workers 8 \
  --log-dir logs \
  --run-id gpt4omini_zs_oracle
```

The hint is delivered as a synthetic request/offer pair in Agent A's
`shared_history`, so its prompt matches the state of a real run just after the
responder has made an offer. The text is Agent B's own constraint string, which
is not always how `apply_hint` renders the same fact; `mira_math.oracle`
verifies for every instance that the fact flips Agent A's view from locally
ill-posed to uniquely solvable before it is injected.

Accuracy under this condition is an **upper bound** on protocol accuracy, and
`oracle_accuracy - protocol_accuracy` attributes the gap to the request channel.
It is an unconditional counterfactual over the whole dataset, not accuracy
conditioned on the responder having made an offer -- the latter is biased by
selection on Agent A's own request success.

Request-channel metrics are meaningless without a request phase, so they are
dropped rather than reported as zero: `hit_rate`, `first_request_success`,
`request_attempts`, `decline_count`, `hints_used`, `hint_overuse`, and
`requests_before_offer`. Results files from an oracle run carry
`"condition": "oracle_hint"` and `"model_b": null`; the default protocol runs
carry `"condition": "protocol"`.

---

## Agent methods

The runner supports three Agent-A methods:

| Method | Description |
|---|---|
| `llm` | Single structured LLM call for each request/final-answer step. This is the simplest reference setting. |
| `react` | ReAct-style reasoning, optionally with tools such as the deterministic calculator. |
| `reflexion` | Generate, reflect, and refine before producing the structured request or final answer. |

The paper baselines use the constrained offer/decline responder protocol. When reporting results, specify the Agent-A model, Agent-B model, method, seed, dataset file, prompt regime, temperature, and run date.

---

## Repository layout

```text
mira_math/
  __init__.py
  schema.py            # JSON-first instance schema and JSONL I/O
  agent_schema.py      # Pydantic schemas for requests, offers, declines, and final answers
  prompts.py           # Agent A and Agent B prompts, notation guide, few-shot formatting
  generate.py          # Dataset generation CLI
  validate.py          # Deterministic instance validation CLI
  runner.py            # Backend-agnostic runner skeleton
  scoring.py           # Transcript-level and aggregate metrics
  oracle.py            # Oracle-hint condition: hint selection, verification, metric stripping
  families/            # 22 mathematical family generators
  tools/               # Optional tools, currently including a deterministic calculator
  utils/               # Exact arithmetic, CRT, and polynomial helpers

examples/
  sample_instance.json
  sample_transcript.json
  run_single_langgraph.py
  run_dataset_langgraph.py
  run_dataset_langgraph_parallel.py
  run_dataset_langgraph_resilient.py
  run_helpers.py

datasets/
  sample/
    sample_instance.json
    sample_transcript.json

tests/
  test_generators.py
  test_scoring.py
  test_validators.py
```

---

## Instance format

Each generated JSONL row contains one instance with the following top-level fields:

| Field | Meaning |
|---|---|
| `id` | Unique instance identifier. |
| `family` | Problem-family name. |
| `difficulty` | Difficulty level, usually 1, 2, or 3. |
| `n_agents` | Number of agent views, currently 2. |
| `answer_type` | Expected answer format used by the verifier/scorer. |
| `global_solution` | Exact ground-truth answer. |
| `global_metadata` | Family-specific metadata needed for validation or analysis. |
| `agent_views` | Private views for Agent A and Agent B. |
| `minimal_hint_spec` | Machine-readable typed atomic hint specification. |

The key construction is:

- Agent A's private view is locally underdetermined.
- Agent B holds the atomic fact needed by Agent A.
- The combined information determines a unique answer.
- The final answer is checked with exact family-specific normalization rather than loose string matching.

---

## Family taxonomy

MIRA-Math contains 22 families split into fixed-slot and variable-slot regimes.

### Type A: fixed missing-information slot

In Type-A families, the missing slot is structurally fixed by the family. The solver must identify the needed kind of fact and integrate it correctly.

| Family | Target | Canonical missing fact |
|---|---|---|
| `bayes_missing_prior` | Posterior probability | Prior `P(H)` |
| `crt_reconstruction` | Integer satisfying congruences | Third congruence |
| `geometry_coordinates` | Distance from an intersection point | Second line equation |
| `graph_path_sums` | Triangle edge weights | Third path-sum equation |
| `linear_system_separator` | Full variable assignment | Separator variable value, usually `c` |
| `matrix_completion` | Missing matrix entry | Fixed matrix entry `m[1][0]` |
| `moment_problem` | Three-point distribution | Second-moment equation |
| `phase_retrieval` | Length-4 signal | Sign of `x0` |
| `piecewise_missing_threshold` | Piecewise function value | Threshold `t` |
| `rankdef_linear_shared` | Full linear-system solution | Additional equation |
| `recurrence_missing_init` | Recurrence value | Initial condition `a(1)` |

### Type B: variable missing-information slot

In Type-B families, the missing slot varies by instance. The solver must inspect the current problem, localize the absent slot, and request that exact slot.

| Family | Target | Canonical missing fact |
|---|---|---|
| `birth_death_missing_rate` | Stationary probability | One birth/death rate |
| `circuit_missing_resistance` | Equivalent resistance | One resistor value |
| `deconvolution` | Source signal | One convolution output measurement |
| `discrete_tomography` | Target grid-cell value | One resolving grid-cell value |
| `eigenvector_missing_entry` | Matrix-derived target value | One matrix entry |
| `laplace_grid` | Interior grid value | One boundary value |
| `linear_system_missing_coeff` | Solution component | One matrix coefficient |
| `markov_missing_transition` | Stationary probability | One transition probability |
| `poly_interpolation` | Polynomial value at a query point | One interpolation pair |
| `portfolio_variance_missing_corr` | Portfolio variance | One pairwise correlation |
| `steady_state_missing_emission` | HMM observation probability | One emission probability |

---

## Metrics

`mira_math.scoring.score_transcript(instance, transcript)` computes per-instance metrics including:

| Metric | Meaning |
|---|---|
| `acc_final` | Whether Agent A's final answer matches the exact ground truth. |
| `hit_rate` | Fraction of responder messages that are offers rather than declines. |
| `first_request_success` | Whether the first request was accepted. |
| `request_attempts` | Number of requests issued by Agent A. |
| `decline_count` | Number of responder declines. |
| `hints_used` | Number of offers received. |
| `requests_before_offer` | Number of declined requests before the first offer, if any. |
| `rounds_to_final` | Round of the first submitted final answer. |
| `rounds_to_solve` | Round of the first correct final answer, if any. |
| `token_cost` | Approximate token estimate based on transcript word count. |

`mira_math.scoring.aggregate_metrics(list_of_metrics)` aggregates these over a run. The dataset runners also report type-level, difficulty-level, family-level, difficulty-by-type, and difficulty-by-family summaries.

For the paper-level diagnostic decomposition, use the raw transcripts together with each instance's `minimal_hint_spec` to distinguish:

1. no canonical hint acquired,
2. canonical hint acquired but final answer wrong, and
3. final answer correct.

This distinction is important because request acquisition and mathematical integration can fail independently.

---

## Reproducibility checklist

When reporting results, include:

- repository commit hash,
- dataset file and checksum,
- generator seed and difficulty list,
- whether `--few-shot` was used,
- Agent-A model identifier,
- Agent-B model identifier,
- serving backend,
- decoding temperature,
- inference-time seed,
- request budget rule,
- runner script and method,
- run date,
- transcript checksum or released raw logs.

The default request budget is difficulty-scaled through `get_max_requests`: 3, 6, and 9 attempts for difficulties 1, 2, and 3 respectively.

API-backed model behavior may change over time. Treat reported model rankings as a snapshot under the exact model identifiers, dates, prompts, and runner settings used for that run.

---

## Adding a new family

A new family should live in `mira_math/families/<family_name>.py` and expose:

```python
FAMILY_NAME = "my_family"
FAMILY_TYPE = "A"  # fixed slot, or "B" for variable slot

def generate_instance(rng, difficulty, instance_id) -> dict: ...
def is_locally_illposed(instance, agent_view) -> bool: ...
def is_uniquely_solvable(instance, agent_view) -> bool: ...
def apply_hint(agent_view, hint) -> dict: ...
def solve_global(instance): ...
```

Then register the family in:

1. `mira_math/families/__init__.py`,
2. `mira_math/generate.py`,
3. `mira_math/validate.py`, and
4. `mira_math/prompts.py`.

A valid family should enforce the benchmark invariants by construction: local underdetermination from Agent A's view, sufficiency of the canonical atomic hint, and unique global solvability. The validator should be treated as a safety check, not as a substitute for careful generator design.

Notation consistency is important. The natural-language constraints, machine-readable hint specification, and prompt notation guide should all refer to the same mathematical quantity in compatible ways; otherwise the constrained responder may decline requests that were intended to be correct.

---

## Intended uses

MIRA-Math is intended for:

- evaluating whether models ask for missing mathematical information instead of guessing,
- comparing request precision across models and prompts,
- separating information-acquisition failures from downstream computation failures,
- auditing prompt or solver-loop strategies under partial mathematical observability, and
- studying family-specific failure modes in exact mathematical reasoning.

---

## Out-of-scope uses and limitations

MIRA-Math should not be used as evidence that a model can perform open-ended human clarification, general multi-agent collaboration, tool use, web navigation, social reasoning, or real-world decision-making. The task is synthetic, mathematically structured, and intentionally narrow.

The current release uses one required atomic hint per instance. Many real settings require multiple facts, uncertain evidence, human preferences, or open-ended negotiation. The fixed responder channel improves controllability and reproducibility, but request metrics can still reflect occasional LLM-mediated responder matching errors. Final-answer verification and dataset validation are deterministic.

---

## Citation

If you use MIRA-Math, please cite the paper:

```bibtex
@misc{albateh2026miramath,
  title  = {MIRA-Math: A Benchmark for Minimal Information Requesting and Mathematical Reasoning},
  author = {Al Bateh, Charbel and Saab Jr., Samer},
  year   = {2026},
  note   = {Benchmark and reference implementation},
  url    = {https://github.com/cedar-lau/mira-math}
}
```

---

## License

The code is released under the MIT License; see [`LICENSE`](LICENSE).

If generated datasets are distributed separately, release them with an explicit data license. The paper recommends CC BY 4.0 for generated data unless institutional policy requires different terms.
