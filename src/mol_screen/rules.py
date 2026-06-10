"""Property-threshold rule sets and the deterministic evaluator.

This module is intentionally free of any RDKit dependency: it operates on a
plain ``dict`` of pre-computed molecular properties. That keeps the decision
logic fully unit-testable on stdlib alone and makes the "hard facts" path
deterministic — the LLM never decides whether a molecule passes a threshold,
it only chooses which rule sets to apply and narrates the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Threshold:
    """A single numeric constraint on one molecular property.

    ``min`` and ``max`` are both optional; either or both may be set. A
    property satisfies the threshold when it lies within the closed interval
    implied by whichever bounds are present.
    """

    prop: str
    min: Optional[float] = None
    max: Optional[float] = None
    # Human-readable unit/label shown in explanations, e.g. "g/mol".
    unit: str = ""

    def describe(self) -> str:
        if self.min is not None and self.max is not None:
            body = f"{self.min}–{self.max}"
        elif self.max is not None:
            body = f"≤ {self.max}"
        elif self.min is not None:
            body = f"≥ {self.min}"
        else:  # pragma: no cover - a threshold with no bounds is a config error
            body = "(no bound)"
        return f"{self.prop} {body}{(' ' + self.unit) if self.unit else ''}".strip()

    def check(self, value: Optional[float]) -> "ThresholdResult":
        if value is None:
            return ThresholdResult(self, value, passed=False, reason="property unavailable")
        passed = True
        reason = ""
        if self.min is not None and value < self.min:
            passed = False
            reason = f"{value:.2f} below minimum {self.min}"
        if self.max is not None and value > self.max:
            passed = False
            reason = f"{value:.2f} above maximum {self.max}"
        if passed:
            reason = "within range"
        return ThresholdResult(self, value, passed=passed, reason=reason)


@dataclass(frozen=True)
class ThresholdResult:
    threshold: Threshold
    value: Optional[float]
    passed: bool
    reason: str


@dataclass(frozen=True)
class RuleSet:
    """A named collection of thresholds plus a violation allowance.

    ``max_violations`` models filters like Lipinski's Rule of Five, which is
    conventionally treated as satisfied when at most one of its four
    sub-criteria is broken.
    """

    name: str
    description: str
    thresholds: list[Threshold]
    max_violations: int = 0
    # When True, the presence of any structural alert (e.g. PAINS) fails the
    # set regardless of numeric thresholds. Handled by the evaluator using the
    # ``structural_alerts`` property.
    forbid_structural_alerts: bool = False


@dataclass
class RuleSetResult:
    rule_set: str
    passed: bool
    violations: int
    max_violations: int
    threshold_results: list[ThresholdResult] = field(default_factory=list)
    alert_failure: Optional[str] = None

    @property
    def failing(self) -> list[ThresholdResult]:
        return [t for t in self.threshold_results if not t.passed]


@dataclass
class Verdict:
    """Aggregate screening outcome for one molecule across all applied sets."""

    smiles: str
    valid: bool
    properties: dict
    rule_set_results: list[RuleSetResult] = field(default_factory=list)
    passed: bool = False
    error: Optional[str] = None

    @property
    def passed_sets(self) -> list[str]:
        return [r.rule_set for r in self.rule_set_results if r.passed]

    @property
    def failed_sets(self) -> list[str]:
        return [r.rule_set for r in self.rule_set_results if not r.passed]


def evaluate_rule_set(properties: dict, rule_set: RuleSet) -> RuleSetResult:
    """Evaluate one molecule's properties against one rule set."""
    results = [t.check(properties.get(t.prop)) for t in rule_set.thresholds]
    violations = sum(1 for r in results if not r.passed)

    alert_failure = None
    passed = violations <= rule_set.max_violations
    if rule_set.forbid_structural_alerts:
        alerts = properties.get("structural_alerts") or []
        if alerts:
            passed = False
            alert_failure = ", ".join(str(a) for a in alerts)

    return RuleSetResult(
        rule_set=rule_set.name,
        passed=passed,
        violations=violations,
        max_violations=rule_set.max_violations,
        threshold_results=results,
        alert_failure=alert_failure,
    )


def evaluate(properties: dict, rule_sets: list[RuleSet], require_all: bool = True) -> list[RuleSetResult]:
    """Evaluate against several rule sets. ``require_all`` is enforced by the caller
    when computing the final pass/fail; here we just return per-set results."""
    return [evaluate_rule_set(properties, rs) for rs in rule_sets]


# --------------------------------------------------------------------------
# Built-in rule sets. Thresholds reflect widely-cited medicinal-chemistry
# filters. They are starting points an agent can override per screening brief.
# --------------------------------------------------------------------------

BUILTIN_RULE_SETS: dict[str, RuleSet] = {
    "lipinski_ro5": RuleSet(
        name="lipinski_ro5",
        description="Lipinski's Rule of Five — oral drug-likeness. Up to one violation tolerated.",
        thresholds=[
            Threshold("mol_weight", max=500, unit="g/mol"),
            Threshold("logp", max=5),
            Threshold("h_donors", max=5),
            Threshold("h_acceptors", max=10),
        ],
        max_violations=1,
    ),
    "veber": RuleSet(
        name="veber",
        description="Veber rules — oral bioavailability via flexibility and polar surface area.",
        thresholds=[
            Threshold("rotatable_bonds", max=10),
            Threshold("tpsa", max=140, unit="Å²"),
        ],
        max_violations=0,
    ),
    "ghose": RuleSet(
        name="ghose",
        description="Ghose filter — drug-like range on weight, lipophilicity and atom count.",
        thresholds=[
            Threshold("mol_weight", min=160, max=480, unit="g/mol"),
            Threshold("logp", min=-0.4, max=5.6),
            Threshold("heavy_atoms", min=20, max=70),
        ],
        max_violations=0,
    ),
    "lead_like": RuleSet(
        name="lead_like",
        description="Lead-likeness — tighter than Ro5, leaving room to optimize toward a drug.",
        thresholds=[
            Threshold("mol_weight", max=350, unit="g/mol"),
            Threshold("logp", max=3.5),
            Threshold("rotatable_bonds", max=7),
        ],
        max_violations=0,
    ),
    "rule_of_three": RuleSet(
        name="rule_of_three",
        description="Astex Rule of Three — fragment-screening starting points.",
        thresholds=[
            Threshold("mol_weight", max=300, unit="g/mol"),
            Threshold("logp", max=3),
            Threshold("h_donors", max=3),
            Threshold("h_acceptors", max=3),
            Threshold("rotatable_bonds", max=3),
        ],
        max_violations=0,
    ),
    "cns_mpo": RuleSet(
        name="cns_mpo",
        description="Simplified CNS / blood-brain-barrier permeability heuristics.",
        thresholds=[
            Threshold("mol_weight", max=450, unit="g/mol"),
            Threshold("logp", min=1, max=4),
            Threshold("tpsa", max=90, unit="Å²"),
            Threshold("h_donors", max=3),
        ],
        max_violations=1,
    ),
    "pains": RuleSet(
        name="pains",
        description="Pan-Assay Interference compounds — flags promiscuous false-positive scaffolds.",
        thresholds=[],
        forbid_structural_alerts=True,
    ),
}


def resolve_rule_sets(names: list[str]) -> list[RuleSet]:
    """Map rule-set names to RuleSet objects, ignoring unknown names."""
    resolved = []
    for n in names:
        rs = BUILTIN_RULE_SETS.get(n)
        if rs is not None:
            resolved.append(rs)
    return resolved


def apply_overrides(rule_set: RuleSet, overrides: dict[str, dict]) -> RuleSet:
    """Return a copy of ``rule_set`` with per-property threshold overrides applied.

    ``overrides`` maps a property name to ``{"min": ..., "max": ...}``. Only
    properties already present in the rule set are adjusted; this prevents the
    LLM from inventing constraints on properties we never computed.
    """
    if not overrides:
        return rule_set
    new_thresholds = []
    for t in rule_set.thresholds:
        if t.prop in overrides:
            o = overrides[t.prop]
            new_thresholds.append(
                Threshold(
                    prop=t.prop,
                    min=o.get("min", t.min),
                    max=o.get("max", t.max),
                    unit=t.unit,
                )
            )
        else:
            new_thresholds.append(t)
    return RuleSet(
        name=rule_set.name,
        description=rule_set.description,
        thresholds=new_thresholds,
        max_violations=rule_set.max_violations,
        forbid_structural_alerts=rule_set.forbid_structural_alerts,
    )
