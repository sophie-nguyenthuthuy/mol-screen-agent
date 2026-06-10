"""Unit tests for the deterministic rule evaluator.

These run on stdlib alone — no RDKit, LangGraph, or AWS required — because the
evaluator consumes a plain property dict. This is the core correctness contract:
the LLM never touches pass/fail.
"""

from mol_screen.rules import (
    BUILTIN_RULE_SETS,
    RuleSet,
    Threshold,
    apply_overrides,
    evaluate,
    evaluate_rule_set,
)


# Aspirin-ish property dict (well within Ro5).
ASPIRIN = {
    "mol_weight": 180.16,
    "logp": 1.31,
    "h_donors": 1,
    "h_acceptors": 4,
    "tpsa": 63.6,
    "rotatable_bonds": 3,
    "heavy_atoms": 13,
    "structural_alerts": [],
}

# A large, greasy molecule that breaks several Ro5 criteria.
BIG_GREASY = {
    "mol_weight": 720.0,
    "logp": 8.4,
    "h_donors": 7,
    "h_acceptors": 12,
    "tpsa": 180.0,
    "rotatable_bonds": 18,
    "heavy_atoms": 55,
    "structural_alerts": [],
}


def test_threshold_within_range():
    t = Threshold("mol_weight", max=500)
    r = t.check(180.16)
    assert r.passed
    assert "within range" in r.reason


def test_threshold_above_max():
    t = Threshold("mol_weight", max=500)
    r = t.check(720.0)
    assert not r.passed
    assert "above maximum" in r.reason


def test_threshold_below_min():
    t = Threshold("mol_weight", min=160)
    r = t.check(120.0)
    assert not r.passed
    assert "below minimum" in r.reason


def test_threshold_missing_property_fails_safe():
    # A missing property must FAIL, never silently pass.
    t = Threshold("mol_weight", max=500)
    r = t.check(None)
    assert not r.passed
    assert "unavailable" in r.reason


def test_lipinski_pass_for_aspirin():
    res = evaluate_rule_set(ASPIRIN, BUILTIN_RULE_SETS["lipinski_ro5"])
    assert res.passed
    assert res.violations == 0


def test_lipinski_allows_one_violation():
    # MW slightly over 500 but everything else fine → 1 violation, still passes.
    props = dict(ASPIRIN, mol_weight=520.0)
    res = evaluate_rule_set(props, BUILTIN_RULE_SETS["lipinski_ro5"])
    assert res.violations == 1
    assert res.passed


def test_lipinski_fails_with_two_violations():
    props = dict(ASPIRIN, mol_weight=520.0, logp=6.0)
    res = evaluate_rule_set(props, BUILTIN_RULE_SETS["lipinski_ro5"])
    assert res.violations == 2
    assert not res.passed


def test_big_greasy_fails_lipinski_and_veber():
    results = evaluate(
        BIG_GREASY,
        [BUILTIN_RULE_SETS["lipinski_ro5"], BUILTIN_RULE_SETS["veber"]],
    )
    assert all(not r.passed for r in results)


def test_pains_alert_fails_structural_set():
    props = dict(ASPIRIN, structural_alerts=["catechol_A(92)"])
    res = evaluate_rule_set(props, BUILTIN_RULE_SETS["pains"])
    assert not res.passed
    assert res.alert_failure == "catechol_A(92)"


def test_pains_clean_passes():
    res = evaluate_rule_set(ASPIRIN, BUILTIN_RULE_SETS["pains"])
    assert res.passed


def test_apply_overrides_tightens_threshold():
    rs = BUILTIN_RULE_SETS["lipinski_ro5"]
    tightened = apply_overrides(rs, {"mol_weight": {"max": 300}})
    # Aspirin (180) still passes; a 400 MW molecule now fails the MW threshold.
    props = dict(ASPIRIN, mol_weight=400.0)
    res = evaluate_rule_set(props, tightened)
    assert res.violations == 1
    mw_result = next(t for t in res.threshold_results if t.threshold.prop == "mol_weight")
    assert not mw_result.passed


def test_apply_overrides_ignores_unknown_property():
    # Overriding a property not in the set should not add a new constraint.
    rs = BUILTIN_RULE_SETS["veber"]
    same = apply_overrides(rs, {"qed": {"min": 0.5}})
    assert len(same.thresholds) == len(rs.thresholds)


def test_apply_overrides_preserves_alert_property():
    # The Brenk set carries a non-default alert_property; overrides must keep it.
    rs = BUILTIN_RULE_SETS["brenk"]
    copy = apply_overrides(rs, {"mol_weight": {"max": 300}})  # no-op, brenk has no thresholds
    assert copy.alert_property == "brenk_alerts"
    assert copy.forbid_structural_alerts


def test_brenk_alert_uses_its_own_property():
    # A Brenk alert fails the brenk set; a PAINS-only hit does not.
    props = dict(ASPIRIN, brenk_alerts=["aldehyde"], structural_alerts=[])
    res = evaluate_rule_set(props, BUILTIN_RULE_SETS["brenk"])
    assert not res.passed
    assert res.alert_failure == "aldehyde"

    props_pains_only = dict(ASPIRIN, brenk_alerts=[], structural_alerts=["catechol_A(92)"])
    assert evaluate_rule_set(props_pains_only, BUILTIN_RULE_SETS["brenk"]).passed


def test_egan_passes_aspirin_fails_greasy():
    assert evaluate_rule_set(ASPIRIN, BUILTIN_RULE_SETS["egan"]).passed
    assert not evaluate_rule_set(BIG_GREASY, BUILTIN_RULE_SETS["egan"]).passed


def test_muegge_rejects_too_small_molecule():
    # Muegge requires MW >= 200; aspirin (180) is just under the floor.
    res = evaluate_rule_set(ASPIRIN, BUILTIN_RULE_SETS["muegge"])
    assert not res.passed
    mw = next(t for t in res.threshold_results if t.threshold.prop == "mol_weight")
    assert not mw.passed and "below minimum" in mw.reason


def test_gsk_4_400_caps_weight_and_logp():
    res = evaluate_rule_set(ASPIRIN, BUILTIN_RULE_SETS["gsk_4_400"])
    assert res.passed
    assert not evaluate_rule_set(BIG_GREASY, BUILTIN_RULE_SETS["gsk_4_400"]).passed
