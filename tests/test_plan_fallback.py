"""Tests for the offline intake fallback (keyword -> ScreeningPlan).

No AWS/Bedrock required: these exercise the deterministic fallback used when
the LLM is unavailable, so they verify the agent degrades gracefully offline.
"""

from mol_screen.llm import _fallback_plan
from mol_screen.rules import BUILTIN_RULE_SETS

CAT = BUILTIN_RULE_SETS


def test_vague_brief_defaults_to_ro5_and_veber():
    plan = _fallback_plan("", CAT)
    assert plan.rule_sets == ["lipinski_ro5", "veber"]


def test_cns_brief_selects_cns_mpo():
    plan = _fallback_plan("CNS-penetrant scaffolds for a brain target", CAT)
    assert "cns_mpo" in plan.rule_sets


def test_fragment_brief_selects_rule_of_three():
    plan = _fallback_plan("fragment screening library, rule of three", CAT)
    assert "rule_of_three" in plan.rule_sets


def test_pains_brief_selects_pains():
    plan = _fallback_plan("drug-like, exclude PAINS false positives", CAT)
    assert "pains" in plan.rule_sets
    assert "lipinski_ro5" in plan.rule_sets


def test_only_known_rule_sets_returned():
    plan = _fallback_plan("lead-like and ghose and oral", CAT)
    for name in plan.rule_sets:
        assert name in CAT


def test_no_duplicate_rule_sets():
    plan = _fallback_plan("oral drug-like lipinski ro5", CAT)
    assert len(plan.rule_sets) == len(set(plan.rule_sets))


def test_brenk_brief_selects_brenk():
    plan = _fallback_plan("remove reactive and toxicophore fragments", CAT)
    assert "brenk" in plan.rule_sets


def test_egan_and_muegge_briefs():
    assert "egan" in _fallback_plan("apply the egan egg", CAT).rule_sets
    assert "muegge" in _fallback_plan("muegge drug-likeness", CAT).rule_sets


def test_gsk_brief_selects_gsk_4_400():
    plan = _fallback_plan("favor developability, gsk 4/400", CAT)
    assert "gsk_4_400" in plan.rule_sets
