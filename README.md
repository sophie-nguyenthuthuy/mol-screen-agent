# mol-screen-agent

An agentic molecule-screening pipeline that filters candidate molecules against
medicinal-chemistry property thresholds and **explains its reasoning** — built
on RDKit for the chemistry, LangGraph for the orchestration, and a **self-hosted
open-source LLM** (via any OpenAI-compatible endpoint — Ollama, vLLM, llama.cpp,
TGI) for intent parsing and per-molecule narration. No cloud LLM, no vendor lock-in.

The design principle: **the LLM never decides pass/fail.** RDKit computes the
properties, deterministic rule sets decide the verdict, and the LLM is used only
for the two things language models are actually good at here — turning a fuzzy
brief into a concrete screening plan, and explaining a verdict to a chemist.

## What it does

Give it a list of SMILES and a natural-language brief:

```bash
mol-screen smiles "CC(=O)Oc1ccccc1C(=O)O" "CCCCCCCCCCCCCCCCCC(=O)O" \
    --brief "oral, drug-like, no PAINS"
```

and it will:

1. **Plan** — the LLM reads the brief and emits a structured plan: which rule
   sets to apply (`lipinski_ro5`, `veber`, `cns_mpo`, …) and any threshold
   overrides the brief implies.
2. **Screen** — RDKit computes ~11 descriptors per molecule (MW, cLogP, HBD/HBA,
   TPSA, rotatable bonds, QED, PAINS alerts, …); the deterministic evaluator
   checks them against the planned thresholds.
3. **Explain** — the LLM narrates each verdict, citing the specific properties
   and thresholds that drove it.
4. **Summarize** — pass / fail / invalid counts.

## Architecture

```
              natural-language brief                 SMILES list
                       │                                  │
                       ▼                                  │
   ┌───────────────────────────────────────┐             │
   │ plan   (open-source LLM, structured)   │             │
   │   brief ─▶ ScreeningPlan               │             │
   │   {rule_sets, threshold overrides}     │             │
   └───────────────────┬───────────────────┘             │
                       ▼                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ screen   (RDKit + deterministic evaluator)  ← the hard facts  │
   │   compute_properties() ─▶ evaluate() ─▶ Verdict (pass/fail)   │
   └───────────────────┬─────────────────────────────────────────┘
                       ▼
   ┌───────────────────────────────────────┐
   │ explain  (open-source LLM)             │
   │   Verdict ─▶ 2-3 sentence rationale    │
   └───────────────────┬───────────────────┘
                       ▼
                  summarize ─▶ report
```

These four nodes are wired as a [LangGraph](https://langchain-ai.github.io/langgraph/)
`StateGraph` (`src/mol_screen/graph.py`). A conditional edge skips the explain
step when no molecule parsed.

| Module | Responsibility | Heavy deps |
| --- | --- | --- |
| `rules.py` | Rule sets + deterministic evaluator | none (pure stdlib) |
| `descriptors.py` | RDKit property computation + PAINS | RDKit |
| `llm.py` | OpenAI-compatible intake + explanation (+ offline fallbacks) | langchain-openai |
| `graph.py` | LangGraph state machine | langgraph |
| `agent.py` / `cli.py` | Public API and CLI | — |

`rules.py` deliberately has **no RDKit dependency** — it consumes a plain
property dict, which keeps the decision logic fully unit-testable on stdlib and
makes the screening verdict reproducible.

## Built-in rule sets

Drug-likeness / absorption: `lipinski_ro5`, `veber`, `ghose`, `egan`, `muegge`,
`gsk_4_400`. Stage-specific: `lead_like`, `rule_of_three` (fragments), `cns_mpo`
(BBB heuristics). Structural-alert filters: `pains` (assay interference) and
`brenk` (reactive / toxicophore fragments) — each backed by its own RDKit
FilterCatalog. List them all with thresholds:

```bash
mol-screen rules
```

The agent picks among these from the brief, and can tighten/loosen individual
thresholds (e.g. "MW under 350") via structured overrides — but it can only
adjust properties that are actually computed, never invent new ones.

## Install

```bash
pip install -r requirements.txt        # or: pip install -e .
```

RDKit is required for the chemistry. If you hit wheel issues on pip, use
conda-forge: `conda install -c conda-forge rdkit`.

## LLM setup (self-hosted, open source)

The agent talks to any OpenAI-compatible server hosting an open-weights model.
The simplest local option is [Ollama](https://ollama.com):

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

That's it — the defaults already point at Ollama. To use a different server or
model, copy `.env.example` and set:

```bash
export MOL_SCREEN_LLM_BASE_URL=http://localhost:11434/v1   # or your vLLM/TGI endpoint
export MOL_SCREEN_LLM_MODEL=qwen2.5:7b-instruct            # any open-weights instruct model
```

For production, point `MOL_SCREEN_LLM_BASE_URL` at a [vLLM](https://github.com/vllm-project/vllm)
or TGI server (e.g. `http://host:8000/v1`) serving Qwen2.5 / Llama 3.1 / Mistral /
DeepSeek. Tool-calling support in the model improves the structured intake step.

### Runs without any LLM, too

If no endpoint is reachable (or `langchain-openai` isn't installed), the agent
degrades gracefully: intake falls back to keyword matching, and explanations
fall back to templated summaries. The RDKit screening — the part that matters —
is unchanged. This makes local development and CI possible with no model server.

## Usage

```bash
# Screen from a file, show the full property table
mol-screen file examples/candidates.smi --brief "CNS-penetrant, lead-like" -p

# As a library
python -c "
from mol_screen import screen
r = screen(['CC(=O)Oc1ccccc1C(=O)O'], brief='oral, drug-like')
for v in r.verdicts:
    print(v.smiles, v.passed, r.explanations[v.smiles])
"
```

## Tests

```bash
pytest                 # core evaluator + offline intake, no RDKit/LLM needed
```

The test suite covers the deterministic contract (thresholds, violation
allowances, PAINS/Brenk alerts, overrides, fail-safe on missing properties) and
the offline planning fallback — all without external dependencies.

A separate opt-in suite exercises the real LLM path and is skipped by default:

```bash
# with a server up, e.g. `ollama serve`
MOL_SCREEN_LIVE_LLM=1 \
    MOL_SCREEN_LLM_BASE_URL=http://localhost:11434/v1 \
    MOL_SCREEN_LLM_MODEL=qwen2.5:7b-instruct \
    pytest tests/test_llm_live.py -v
```

It downgrades to a skip (never a failure) if no LLM client is configured.
