"""Command-line interface for the screening agent.

Examples
--------
    mol-screen smiles "CC(=O)Oc1ccccc1C(=O)O" "CCN(CC)CC" \
        --brief "oral, drug-like, no PAINS"

    mol-screen file examples/candidates.smi --brief "CNS-penetrant, lead-like"

    mol-screen rules            # list built-in rule sets
"""

from __future__ import annotations

from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import screen
from .descriptors import PROPERTY_LABELS, PROPERTY_ORDER, rdkit_available
from .rules import BUILTIN_RULE_SETS

app = typer.Typer(add_completion=False, help="Agentic molecule screening (RDKit + LangGraph + Bedrock).")
console = Console()


def _read_smiles_file(path: str) -> List[str]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Accept "SMILES" or "SMILES name" (.smi format); take first token.
            out.append(line.split()[0])
    return out


def _render(report, show_properties: bool) -> None:
    s = report.summary
    rationale = s.get("plan_rationale", "")
    mode = "ALL rule sets" if s.get("require_all", True) else "ANY rule set"
    console.print(
        Panel(
            f"[bold]Plan:[/bold] {', '.join(s.get('rule_sets', []))}  "
            f"([italic]must pass {mode}[/italic])\n[dim]{rationale}[/dim]",
            title="Screening plan",
            border_style="cyan",
        )
    )

    table = Table(show_lines=False, header_style="bold")
    table.add_column("SMILES", overflow="fold", max_width=34)
    table.add_column("Verdict", justify="center")
    if show_properties:
        for key in PROPERTY_ORDER:
            table.add_column(key, justify="right")
    table.add_column("Rationale", overflow="fold")

    for v in report.verdicts:
        if not v.valid:
            verdict_cell = "[yellow]INVALID[/yellow]"
        elif v.passed:
            verdict_cell = "[green]PASS[/green]"
        else:
            verdict_cell = "[red]FAIL[/red]"
        row = [v.smiles, verdict_cell]
        if show_properties:
            for key in PROPERTY_ORDER:
                row.append(str(v.properties.get(key, "—")))
        row.append(report.explanations.get(v.smiles, ""))
        table.add_row(*row)

    console.print(table)
    console.print(
        f"\n[bold]Summary:[/bold] {s.get('passed', 0)} passed · "
        f"{s.get('failed', 0)} failed · {s.get('invalid', 0)} invalid "
        f"(of {s.get('total', 0)})"
    )


def _check_rdkit() -> None:
    if not rdkit_available():
        console.print(
            "[red]RDKit is not installed.[/red] Install it with "
            "[bold]pip install rdkit[/bold] (or conda-forge) and retry."
        )
        raise typer.Exit(code=1)


@app.command()
def smiles(
    smiles: List[str] = typer.Argument(..., help="One or more SMILES strings."),
    brief: str = typer.Option("", "--brief", "-b", help="Natural-language screening brief."),
    model_id: Optional[str] = typer.Option(None, "--model", help="Bedrock model id override."),
    region: Optional[str] = typer.Option(None, "--region", help="AWS region override."),
    show_properties: bool = typer.Option(False, "--properties", "-p", help="Show the full property table."),
):
    """Screen SMILES passed on the command line."""
    _check_rdkit()
    report = screen(list(smiles), brief=brief, model_id=model_id, region=region)
    _render(report, show_properties)


@app.command()
def file(
    path: str = typer.Argument(..., help="Path to a .smi/.txt file (one SMILES per line)."),
    brief: str = typer.Option("", "--brief", "-b", help="Natural-language screening brief."),
    model_id: Optional[str] = typer.Option(None, "--model", help="Bedrock model id override."),
    region: Optional[str] = typer.Option(None, "--region", help="AWS region override."),
    show_properties: bool = typer.Option(False, "--properties", "-p", help="Show the full property table."),
):
    """Screen SMILES read from a file."""
    _check_rdkit()
    smiles_list = _read_smiles_file(path)
    if not smiles_list:
        console.print(f"[yellow]No SMILES found in {path}.[/yellow]")
        raise typer.Exit(code=1)
    report = screen(smiles_list, brief=brief, model_id=model_id, region=region)
    _render(report, show_properties)


@app.command()
def rules():
    """List the built-in rule sets and their thresholds."""
    for name, rs in BUILTIN_RULE_SETS.items():
        table = Table(title=f"{name}", title_style="bold cyan", show_header=False, title_justify="left")
        table.add_column("k")
        table.add_column("v")
        table.add_row("description", rs.description)
        if rs.thresholds:
            table.add_row("thresholds", "\n".join(t.describe() for t in rs.thresholds))
        if rs.forbid_structural_alerts:
            table.add_row("structural alerts", "any match → fail")
        table.add_row("violations allowed", str(rs.max_violations))
        console.print(table)
        console.print()


if __name__ == "__main__":
    app()
