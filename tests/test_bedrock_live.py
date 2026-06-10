"""Live Bedrock smoke test — opt-in only.

These exercise the real Claude-on-Bedrock path (intake + explanation). They are
skipped by default so the normal/offline suite and CI stay green without AWS.

Run them with credentials configured and the opt-in flag set::

    MOL_SCREEN_LIVE_BEDROCK=1 AWS_REGION=us-east-1 \
        BEDROCK_MODEL_ID=us.anthropic.claude-3-5-sonnet-20241022-v2:0 \
        pytest tests/test_bedrock_live.py -v

A missing Bedrock client (no langchain-aws, no creds, bad region) downgrades to
a skip rather than a failure, so an accidental run never reports red.
"""

import os

import pytest

from mol_screen import llm as llm_mod
from mol_screen.descriptors import rdkit_available
from mol_screen.rules import BUILTIN_RULE_SETS, resolve_rule_sets

LIVE = os.environ.get("MOL_SCREEN_LIVE_BEDROCK") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE, reason="set MOL_SCREEN_LIVE_BEDROCK=1 (with AWS creds) to run live Bedrock tests"
)


def _chat_or_skip():
    chat = llm_mod._build_chat()
    if chat is None:
        pytest.skip("Bedrock client unavailable (langchain-aws missing or creds/region not configured)")
    return chat


def test_live_plan_from_brief_hits_the_model():
    _chat_or_skip()
    plan = llm_mod.plan_from_brief("CNS-penetrant, lead-like, exclude PAINS", BUILTIN_RULE_SETS)
    assert plan.rule_sets, "model returned an empty plan"
    assert all(s in BUILTIN_RULE_SETS for s in plan.rule_sets), "model invented an unknown rule set"
    # The offline fallback stamps this marker; its absence proves we used Bedrock.
    assert "offline fallback" not in plan.rationale


def test_live_explain_verdict_returns_prose():
    _chat_or_skip()
    if not rdkit_available():
        pytest.skip("RDKit not installed")
    from mol_screen.graph import _screen_one

    verdict = _screen_one("CC(=O)Oc1ccccc1C(=O)O", resolve_rule_sets(["lipinski_ro5", "veber"]))
    text = llm_mod.explain_verdict(verdict)
    assert isinstance(text, str) and len(text.strip()) > 20
