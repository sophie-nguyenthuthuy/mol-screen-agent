"""RDKit-backed molecular property computation.

The only place RDKit is imported. Everything downstream consumes the plain
``dict`` produced by :func:`compute_properties`, so the decision logic and the
LangGraph wiring stay testable without RDKit installed.
"""

from __future__ import annotations

from typing import Optional

# Properties we compute, in display order. Kept here so the CLI, the LLM
# prompts, and the rule sets all reference one canonical list.
PROPERTY_ORDER = [
    "mol_weight",
    "logp",
    "h_donors",
    "h_acceptors",
    "tpsa",
    "rotatable_bonds",
    "rings",
    "aromatic_rings",
    "heavy_atoms",
    "fraction_csp3",
    "qed",
]

PROPERTY_LABELS = {
    "mol_weight": "Molecular weight (g/mol)",
    "logp": "cLogP",
    "h_donors": "H-bond donors",
    "h_acceptors": "H-bond acceptors",
    "tpsa": "TPSA (Å²)",
    "rotatable_bonds": "Rotatable bonds",
    "rings": "Ring count",
    "aromatic_rings": "Aromatic rings",
    "heavy_atoms": "Heavy atoms",
    "fraction_csp3": "Fraction sp³ C",
    "qed": "QED (drug-likeness)",
}


class RDKitNotAvailable(RuntimeError):
    pass


def _require_rdkit():
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import (  # noqa: F401
            Crippen,
            Descriptors,
            Lipinski,
            QED,
            rdMolDescriptors,
        )
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise RDKitNotAvailable(
            "RDKit is not installed. Install it with `pip install rdkit` "
            "(or `conda install -c conda-forge rdkit`)."
        ) from e


def parse_smiles(smiles: str):
    """Return an RDKit Mol or ``None`` if the SMILES is invalid."""
    _require_rdkit()
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles)


_PAINS_CATALOG = None


def _pains_catalog():
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        from rdkit.Chem import FilterCatalog
        from rdkit.Chem.FilterCatalog import FilterCatalogParams

        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
    return _PAINS_CATALOG


def structural_alerts(mol) -> list[str]:
    """Return the names of any PAINS substructure alerts matched by ``mol``."""
    catalog = _pains_catalog()
    matches = catalog.GetMatches(mol)
    return [m.GetDescription() for m in matches]


def compute_properties(smiles: str) -> dict:
    """Compute the full property dict for one SMILES string.

    Always returns a dict. On an invalid SMILES, ``valid`` is False and the
    numeric properties are absent so thresholds report "property unavailable"
    rather than silently passing.
    """
    _require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles, "valid": False}

    props = {
        "smiles": smiles,
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mol_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Crippen.MolLogP(mol), 2),
        "h_donors": Lipinski.NumHDonors(mol),
        "h_acceptors": Lipinski.NumHAcceptors(mol),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": Lipinski.NumAromaticRings(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "fraction_csp3": round(Descriptors.FractionCSP3(mol), 3),
        "qed": round(QED.qed(mol), 3),
    }
    props["structural_alerts"] = structural_alerts(mol)
    return props


def rdkit_available() -> bool:
    try:
        _require_rdkit()
        return True
    except RDKitNotAvailable:
        return False
