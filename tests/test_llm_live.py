"""Live LLM smoke test — opt-in only.

These exercise the real open-source LLM path (intake + explanation) against a
running OpenAI-compatible endpoint (Ollama, vLLM, llama.cpp, TGI). They are
skipped by default so the normal/offline suite and CI stay green without a
model server.

Run them with a server up and the opt-in flag set::

    # e.g. `ollama serve` then `ollama pull qwen2.5:7b-instruct`
    MOL_SCREEN_LIVE_LLM=1 \
        MOL_SCREEN_LLM_BASE_URL=http://localhost:11434/v1 \
        MOL_SCREEN_LLM_MODEL=qwen2.5:7b-instruct \
        pytest tests/test_llm_live.py -v

A missing client (no langchain-openai) downgrades to a skip rather than a
failure, so an accidental run never reports red.
"""

import os

import pytest

from mol_screen import llm as llm_mod
from mol_screen.descriptors import rdkit_available
from mol_screen.rules import BUILTIN_RULE_SETS, resolve_rule_sets

LIVE = os.environ.get("MOL_SCREEN_LIVE_LLM") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="set MOL_SCREEN_LIVE_LLM=1 (with an OpenAI-compatible server up) to run live LLM tests",
)


def _chat_or_skip():
    chat = llm_mod._build_chat()
    if chat is None:
        pytest.skip("LLM client unavailable (langchain-openai missing or endpoint unreachable)")
    return chat


def test_live_plan_from_brief_hits_the_model():
    _chat_or_skip()
    plan = llm_mod.plan_from_brief("CNS-penetrant, lead-like, exclude PAINS", BUILTIN_RULE_SETS)
    assert plan.rule_sets, "model returned an empty plan"
    assert all(s in BUILTIN_RULE_SETS for s in plan.rule_sets), "model invented an unknown rule set"
    # The offline fallback stamps this marker; its absence proves we used the LLM.
    assert "offline fallback" not in plan.rationale


def test_live_explain_verdict_returns_prose():
    _chat_or_skip()
    if not rdkit_available():
        pytest.skip("RDKit not installed")
    from mol_screen.graph import _screen_one

    verdict = _screen_one("CC(=O)Oc1ccccc1C(=O)O", resolve_rule_sets(["lipinski_ro5", "veber"]))
    text = llm_mod.explain_verdict(verdict)
    assert isinstance(text, str) and len(text.strip()) > 20
