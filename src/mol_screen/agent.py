"""Public, friendly API over the LangGraph screening flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .graph import run_screen


@dataclass
class ScreeningReport:
    """Everything a caller needs after a screen."""

    verdicts: list  # list[rules.Verdict]
    explanations: dict  # smiles -> str
    summary: dict
    plan: object  # llm.ScreeningPlan

    def passed(self) -> list:
        return [v for v in self.verdicts if v.valid and v.passed]

    def failed(self) -> list:
        return [v for v in self.verdicts if v.valid and not v.passed]

    def invalid(self) -> list:
        return [v for v in self.verdicts if not v.valid]


def screen(
    smiles_list: list[str],
    brief: str = "",
    model_id: Optional[str] = None,
    region: Optional[str] = None,
) -> ScreeningReport:
    """Screen a list of SMILES against a natural-language brief.

    Example::

        report = screen(
            ["CC(=O)Oc1ccccc1C(=O)O", "CCN(CC)CC"],
            brief="CNS-penetrant, lead-like, no PAINS",
        )
        for v in report.passed():
            print(v.smiles, report.explanations[v.smiles])
    """
    state = run_screen(smiles_list, brief=brief, model_id=model_id, region=region)
    return ScreeningReport(
        verdicts=state.get("verdicts", []),
        explanations=state.get("explanations", {}),
        summary=state.get("summary", {}),
        plan=state.get("plan"),
    )


class ScreeningAgent:
    """Stateful wrapper that pins a model/region across multiple screens."""

    def __init__(self, model_id: Optional[str] = None, region: Optional[str] = None):
        self.model_id = model_id
        self.region = region

    def screen(self, smiles_list: list[str], brief: str = "") -> ScreeningReport:
        return screen(smiles_list, brief=brief, model_id=self.model_id, region=self.region)
