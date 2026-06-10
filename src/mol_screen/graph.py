"""LangGraph wiring for the screening agent.

The graph encodes the agentic flow as a small state machine:

    plan ──▶ screen ──▶ explain ──▶ summarize ──▶ END
                │
                └─(no valid molecules)─▶ summarize

* **plan** — LLM turns the natural-language brief into a structured
  :class:`ScreeningPlan` (rule sets + threshold overrides).
* **screen** — deterministic RDKit property computation + threshold evaluation.
  This is the load-bearing, non-hallucinating step.
* **explain** — LLM narrates each molecule's verdict.
* **summarize** — aggregate pass/fail counts.

If LangGraph isn't installed, :func:`run_screen` falls back to calling the same
node functions in sequence, so the agent still runs.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from . import llm as llm_mod
from . import rules as rules_mod
from .descriptors import compute_properties


class ScreenState(TypedDict, total=False):
    brief: str
    smiles_list: list[str]
    model_id: Optional[str]
    region: Optional[str]
    plan: Any  # ScreeningPlan
    resolved_rule_sets: list[Any]  # list[RuleSet] after override application
    verdicts: list[Any]  # list[Verdict]
    explanations: dict[str, str]
    summary: dict


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def plan_node(state: ScreenState) -> ScreenState:
    plan = llm_mod.plan_from_brief(
        state.get("brief", ""),
        rules_mod.BUILTIN_RULE_SETS,
        model_id=state.get("model_id"),
        region=state.get("region"),
    )
    resolved = [
        rules_mod.apply_overrides(rs, plan.overrides)
        for rs in rules_mod.resolve_rule_sets(plan.rule_sets)
    ]
    if not resolved:  # safety net — never screen against an empty plan
        resolved = rules_mod.resolve_rule_sets(["lipinski_ro5", "veber"])
    return {"plan": plan, "resolved_rule_sets": resolved}


def screen_node(state: ScreenState) -> ScreenState:
    resolved = state["resolved_rule_sets"]
    verdicts = []
    for smiles in state.get("smiles_list", []):
        verdicts.append(_screen_one(smiles, resolved))
    return {"verdicts": verdicts}


def _screen_one(smiles: str, resolved_rule_sets: list) -> rules_mod.Verdict:
    props = compute_properties(smiles)
    if not props.get("valid"):
        return rules_mod.Verdict(
            smiles=smiles, valid=False, properties=props,
            passed=False, error="invalid SMILES",
        )
    results = rules_mod.evaluate(props, resolved_rule_sets)
    passed = all(r.passed for r in results) if results else False
    return rules_mod.Verdict(
        smiles=smiles, valid=True, properties=props,
        rule_set_results=results, passed=passed,
    )


def explain_node(state: ScreenState) -> ScreenState:
    explanations = {}
    for v in state.get("verdicts", []):
        explanations[v.smiles] = llm_mod.explain_verdict(
            v, model_id=state.get("model_id"), region=state.get("region")
        )
    return {"explanations": explanations}


def summarize_node(state: ScreenState) -> ScreenState:
    verdicts = state.get("verdicts", [])
    passed = [v for v in verdicts if v.valid and v.passed]
    failed = [v for v in verdicts if v.valid and not v.passed]
    invalid = [v for v in verdicts if not v.valid]
    plan = state.get("plan")
    return {
        "summary": {
            "total": len(verdicts),
            "passed": len(passed),
            "failed": len(failed),
            "invalid": len(invalid),
            "rule_sets": [rs.name for rs in state.get("resolved_rule_sets", [])],
            "require_all": getattr(plan, "require_all", True),
            "plan_rationale": getattr(plan, "rationale", ""),
        }
    }


def _has_valid_molecules(state: ScreenState) -> str:
    return "explain" if any(v.valid for v in state.get("verdicts", [])) else "summarize"


# --------------------------------------------------------------------------
# Graph assembly / execution
# --------------------------------------------------------------------------

def build_graph():
    """Compile the LangGraph StateGraph. Raises ImportError if langgraph missing."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(ScreenState)
    g.add_node("plan", plan_node)
    g.add_node("screen", screen_node)
    g.add_node("explain", explain_node)
    g.add_node("summarize", summarize_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "screen")
    g.add_conditional_edges("screen", _has_valid_molecules,
                            {"explain": "explain", "summarize": "summarize"})
    g.add_edge("explain", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


def run_screen(
    smiles_list: list[str],
    brief: str = "",
    model_id: Optional[str] = None,
    region: Optional[str] = None,
) -> ScreenState:
    """Run the full screening flow. Uses LangGraph when available, otherwise a
    direct sequential fallback over the same node functions."""
    initial: ScreenState = {
        "brief": brief,
        "smiles_list": smiles_list,
        "model_id": model_id,
        "region": region,
    }
    try:
        graph = build_graph()
        return graph.invoke(initial)
    except ImportError:
        # Sequential fallback — identical logic, no LangGraph dependency.
        state: ScreenState = dict(initial)
        state.update(plan_node(state))
        state.update(screen_node(state))
        if _has_valid_molecules(state) == "explain":
            state.update(explain_node(state))
        else:
            state["explanations"] = {}
        state.update(summarize_node(state))
        return state
