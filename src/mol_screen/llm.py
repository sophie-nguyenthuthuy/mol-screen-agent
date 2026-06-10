"""Open-source LLM integration via an OpenAI-compatible endpoint.

Works with any self-hostable server that speaks the OpenAI API — Ollama, vLLM,
llama.cpp's server, Text-Generation-Inference, LM Studio — so the model stays
open-source and on your own infrastructure. Defaults target a local Ollama
instance running an open-weights instruct model (Qwen2.5 / Llama 3.1 / ...).

Two LLM touch-points, both optional:

1. **Intake** — turn a natural-language screening brief ("CNS-penetrant,
   lead-like, no PAINS") into a structured :class:`ScreeningPlan` (which rule
   sets to apply, plus optional threshold overrides).
2. **Explanation** — narrate a single molecule's deterministic verdict in two
   or three sentences a chemist can act on.

If no LLM endpoint is reachable (or langchain-openai isn't installed), both fall
back to deterministic behavior so the agent still runs end-to-end offline. The
numeric pass/fail decision is *never* delegated to the model — see ``rules.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .descriptors import PROPERTY_LABELS

# Open-source model served over an OpenAI-compatible API. Point these at your
# own server: Ollama (default, http://localhost:11434/v1), vLLM, llama.cpp, TGI.
# Pick any open-weights instruct model with tool-calling support (Qwen2.5,
# Llama 3.1, Mistral, DeepSeek, ...).
DEFAULT_MODEL = os.environ.get("MOL_SCREEN_LLM_MODEL", "qwen2.5:7b-instruct")
DEFAULT_BASE_URL = os.environ.get("MOL_SCREEN_LLM_BASE_URL", "http://localhost:11434/v1")
# Most local servers ignore the key, but the OpenAI client requires a non-empty one.
DEFAULT_API_KEY = os.environ.get("MOL_SCREEN_LLM_API_KEY", "not-needed")


@dataclass
class ScreeningPlan:
    """Structured screening configuration the agent acts on."""

    rule_sets: list[str] = field(default_factory=lambda: ["lipinski_ro5", "veber"])
    overrides: dict[str, dict] = field(default_factory=dict)
    require_all: bool = True
    rationale: str = "default drug-likeness screen"


# Per-process cache of endpoint reachability, keyed by base_url, so a down or
# misconfigured server is probed once (seconds) instead of timing out on every
# LLM call (minutes). Set MOL_SCREEN_LLM_PROBE=0 to disable the pre-flight check.
_REACHABLE: dict[str, bool] = {}
_PROBE_ENABLED = os.environ.get("MOL_SCREEN_LLM_PROBE", "1") != "0"
_PROBE_TIMEOUT = float(os.environ.get("MOL_SCREEN_LLM_PROBE_TIMEOUT", "3"))


def _endpoint_reachable(base_url: str) -> bool:
    """Quick, cached pre-flight: is an OpenAI-compatible server answering at
    ``base_url``? Hits GET /models with a short timeout so the deterministic
    fallback kicks in fast when nothing (usable) is listening."""
    if not _PROBE_ENABLED:
        return True
    if base_url in _REACHABLE:
        return _REACHABLE[base_url]
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {DEFAULT_API_KEY}"})
        with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT) as resp:
            ok = getattr(resp, "status", 200) < 500
    except Exception:
        ok = False
    _REACHABLE[base_url] = ok
    return ok


def _build_chat(model: Optional[str] = None, base_url: Optional[str] = None):
    """Construct a chat client for the OpenAI-compatible endpoint, or return None
    if langchain-openai isn't installed or the endpoint isn't reachable.

    ``max_retries=0`` means a bad call (e.g. model not pulled) fails fast to the
    deterministic fallback rather than retrying for minutes."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None
    url = base_url or DEFAULT_BASE_URL
    if not _endpoint_reachable(url):
        return None
    try:
        return ChatOpenAI(
            model=model or DEFAULT_MODEL,
            base_url=url,
            api_key=DEFAULT_API_KEY,
            temperature=0,
            max_tokens=1024,
            timeout=45,
            max_retries=0,
        )
    except Exception:
        return None


# --------------------------------------------------------------------------
# Intake: natural-language brief -> ScreeningPlan
# --------------------------------------------------------------------------

# Pydantic schema for structured output. Defined lazily so pydantic is only
# required when the LLM path is actually used.
def _plan_schema():
    from pydantic import BaseModel, Field

    class ThresholdOverride(BaseModel):
        prop: str = Field(description="Property name, e.g. mol_weight, logp, tpsa.")
        min: Optional[float] = Field(default=None, description="New lower bound, if any.")
        max: Optional[float] = Field(default=None, description="New upper bound, if any.")

    class PlanSchema(BaseModel):
        rule_sets: list[str] = Field(
            description=(
                "Names of rule sets to apply, chosen from the available list. "
                "Pick the smallest set that matches the brief."
            )
        )
        overrides: list[ThresholdOverride] = Field(
            default_factory=list,
            description="Optional per-property threshold adjustments implied by the brief.",
        )
        require_all: bool = Field(
            default=True,
            description="True if a molecule must pass every applied rule set; False if any one suffices.",
        )
        rationale: str = Field(description="One sentence explaining the chosen plan.")

    return PlanSchema


_INTAKE_SYSTEM = """You are a medicinal-chemistry screening planner. Translate a \
natural-language screening brief into a structured plan that selects molecular \
filter rule sets and, when the brief implies different cutoffs, overrides specific \
thresholds.

Available rule sets:
{rule_set_catalog}

Available properties for overrides: {property_list}

Rules:
- Choose only rule sets from the list above. Prefer the minimal set that matches intent.
- Add a threshold override only when the brief clearly implies a non-default cutoff.
- Never invent properties outside the available list.
- If the brief is vague, default to lipinski_ro5 + veber."""


def plan_from_brief(
    brief: str,
    rule_set_catalog: dict,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ScreeningPlan:
    """Produce a ScreeningPlan from a brief, via the LLM if reachable else default."""
    chat = _build_chat(model, base_url)
    if chat is None or not brief.strip():
        return _fallback_plan(brief, rule_set_catalog)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        catalog_lines = "\n".join(
            f"- {name}: {rs.description}" for name, rs in rule_set_catalog.items()
        )
        system = _INTAKE_SYSTEM.format(
            rule_set_catalog=catalog_lines,
            property_list=", ".join(PROPERTY_LABELS.keys()),
        )
        structured = chat.with_structured_output(_plan_schema())
        result = structured.invoke(
            [SystemMessage(content=system), HumanMessage(content=f"Screening brief: {brief}")]
        )
        overrides = {
            o.prop: {k: v for k, v in (("min", o.min), ("max", o.max)) if v is not None}
            for o in result.overrides
        }
        valid_sets = [s for s in result.rule_sets if s in rule_set_catalog]
        if not valid_sets:
            valid_sets = ["lipinski_ro5", "veber"]
        return ScreeningPlan(
            rule_sets=valid_sets,
            overrides=overrides,
            require_all=result.require_all,
            rationale=result.rationale,
        )
    except Exception:
        return _fallback_plan(brief, rule_set_catalog)


def _fallback_plan(brief: str, rule_set_catalog: dict) -> ScreeningPlan:
    """Keyword-matched plan used when the LLM is unavailable."""
    b = (brief or "").lower()
    selected: list[str] = []
    if any(k in b for k in ("cns", "brain", "bbb", "central nervous")):
        selected.append("cns_mpo")
    if any(k in b for k in ("fragment", "rule of three", "rule of 3")):
        selected.append("rule_of_three")
    if "lead" in b:
        selected.append("lead_like")
    if "ghose" in b:
        selected.append("ghose")
    if "egan" in b:
        selected.append("egan")
    if "muegge" in b:
        selected.append("muegge")
    if any(k in b for k in ("gsk", "4/400", "developab")):
        selected.append("gsk_4_400")
    if any(k in b for k in ("oral", "drug-like", "druglike", "lipinski", "ro5")):
        selected.extend(["lipinski_ro5", "veber"])
    if any(k in b for k in ("pains", "interfere", "promiscu", "false positive")):
        selected.append("pains")
    if any(k in b for k in ("brenk", "reactive", "toxicophore", "unstable")):
        selected.append("brenk")
    if not selected:
        selected = ["lipinski_ro5", "veber"]
    # De-duplicate, preserve order, keep only known sets.
    seen, ordered = set(), []
    for s in selected:
        if s in rule_set_catalog and s not in seen:
            seen.add(s)
            ordered.append(s)
    return ScreeningPlan(
        rule_sets=ordered,
        rationale="keyword-matched plan (LLM unavailable; using offline fallback)",
    )


# --------------------------------------------------------------------------
# Explanation: verdict -> chemist-readable rationale
# --------------------------------------------------------------------------

_EXPLAIN_SYSTEM = """You are a medicinal chemist reviewing an automated screen. \
Given a molecule's computed properties and the exact rule-by-rule results, write \
2-3 sentences explaining the verdict to a colleague. Reference the specific \
properties and thresholds that drove the outcome. Do not recompute or second-guess \
the numbers — they are authoritative. Be concise and concrete."""


def explain_verdict(verdict, model: Optional[str] = None, base_url: Optional[str] = None) -> str:
    """Return a short rationale for a Verdict, via the LLM if reachable else template."""
    chat = _build_chat(model, base_url)
    if chat is None:
        return _fallback_explanation(verdict)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        payload = _verdict_to_prompt(verdict)
        msg = chat.invoke([SystemMessage(content=_EXPLAIN_SYSTEM), HumanMessage(content=payload)])
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        return text.strip() or _fallback_explanation(verdict)
    except Exception:
        return _fallback_explanation(verdict)


def _verdict_to_prompt(verdict) -> str:
    lines = [f"Molecule: {verdict.smiles}"]
    if not verdict.valid:
        return "Molecule could not be parsed as a valid structure: " + verdict.smiles
    p = verdict.properties
    lines.append("Properties:")
    for key, label in PROPERTY_LABELS.items():
        if key in p:
            lines.append(f"  - {label}: {p[key]}")
    if p.get("structural_alerts"):
        lines.append(f"  - PAINS alerts: {', '.join(p['structural_alerts'])}")
    if p.get("brenk_alerts"):
        lines.append(f"  - Brenk alerts: {', '.join(p['brenk_alerts'])}")
    lines.append("Rule results:")
    for r in verdict.rule_set_results:
        status = "PASS" if r.passed else "FAIL"
        detail = f"{r.violations}/{r.max_violations} violations allowed"
        lines.append(f"  - {r.rule_set}: {status} ({detail})")
        for tr in r.failing:
            lines.append(f"      · {tr.threshold.describe()} → {tr.reason}")
        if r.alert_failure:
            lines.append(f"      · structural alert: {r.alert_failure}")
    lines.append(f"Overall verdict: {'PASS' if verdict.passed else 'FAIL'}")
    return "\n".join(lines)


def _fallback_explanation(verdict) -> str:
    if not verdict.valid:
        return "Invalid structure — SMILES could not be parsed, so no properties were computed."
    if verdict.passed:
        sets = ", ".join(verdict.passed_sets) or "all applied filters"
        qed = verdict.properties.get("qed")
        qed_txt = f" QED {qed}." if qed is not None else ""
        return f"Passes {sets} with no disqualifying violations.{qed_txt}"
    reasons = []
    for r in verdict.rule_set_results:
        if r.passed:
            continue
        if r.alert_failure:
            reasons.append(f"{r.rule_set}: structural alert ({r.alert_failure})")
        else:
            bits = "; ".join(f"{tr.threshold.describe()} ({tr.reason})" for tr in r.failing)
            reasons.append(f"{r.rule_set}: {bits}")
    return "Fails — " + " | ".join(reasons)
