"""Offline tests for the LLM client gating — no server, no langchain-openai needed.

These lock in the fast-fail contract: an unreachable endpoint must be detected
(and cached) quickly so screening drops to the deterministic fallback in seconds
rather than timing out per LLM call.
"""

from mol_screen import llm as llm_mod

# A port nothing listens on — connection is refused instantly.
CLOSED = "http://localhost:1/v1"


def test_unreachable_endpoint_reported_and_cached():
    llm_mod._REACHABLE.pop(CLOSED, None)
    assert llm_mod._endpoint_reachable(CLOSED) is False
    # Result is cached so we don't re-probe (and re-pay the timeout) per call.
    assert llm_mod._REACHABLE[CLOSED] is False


def test_build_chat_none_when_unreachable():
    llm_mod._REACHABLE.pop(CLOSED, None)
    assert llm_mod._build_chat(base_url=CLOSED) is None


def test_plan_falls_back_when_unreachable():
    from mol_screen.rules import BUILTIN_RULE_SETS

    llm_mod._REACHABLE.pop(CLOSED, None)
    plan = llm_mod.plan_from_brief("oral drug-like, no PAINS", BUILTIN_RULE_SETS, base_url=CLOSED)
    assert "offline fallback" in plan.rationale
    assert "lipinski_ro5" in plan.rule_sets
