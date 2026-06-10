"""mol_screen — agentic molecule screening.

RDKit computes the properties, deterministic rule sets decide pass/fail, and a
self-hosted open-source LLM (via LangGraph, over any OpenAI-compatible endpoint)
plans the screen from a natural-language brief and explains each verdict.
"""

from .agent import ScreeningAgent, screen
from .rules import BUILTIN_RULE_SETS, RuleSet, Threshold, Verdict

__all__ = [
    "ScreeningAgent",
    "screen",
    "BUILTIN_RULE_SETS",
    "RuleSet",
    "Threshold",
    "Verdict",
]

__version__ = "0.1.0"
