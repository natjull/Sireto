"""Mapping des codes SIRENE vers catégories métier Pipe V6."""

from __future__ import annotations

import logging
from typing import Dict

# Codes categorieJuridiqueUniteLegale (4 chiffres) -> catégorie métier
# Catégories cibles : PUBLIC, PRIVE, INCONNU.
LEGAL_NATURE_MAPPING: Dict[str, str] = {
    # Secteur public - administrations et collectivités
    "7111": "PUBLIC",
    "7112": "PUBLIC",
    "7120": "PUBLIC",
    "7150": "PUBLIC",
    "7160": "PUBLIC",
    # Secteur public - établissements publics
    "7210": "PUBLIC",
    "7220": "PUBLIC",
    "7230": "PUBLIC",
    "7312": "PUBLIC",
    "7313": "PUBLIC",
    "7314": "PUBLIC",
    "7321": "PUBLIC",
    "7331": "PUBLIC",
    # Secteur public - structures spécifiques
    "7344": "PUBLIC",
    "7345": "PUBLIC",
    "7346": "PUBLIC",
    "7347": "PUBLIC",
    "7348": "PUBLIC",
    "7354": "PUBLIC",
    "7389": "PUBLIC",
    # Secteur public - SEM / SPL
    "5532": "PUBLIC",
    "5542": "PUBLIC",
    "5552": "PUBLIC",
    # Secteur privé - sociétés commerciales
    "5410": "PRIVE",
    "5415": "PRIVE",
    "5499": "PRIVE",
    "5505": "PRIVE",
    "5510": "PRIVE",
    "5560": "PRIVE",
    "5599": "PRIVE",
    "5710": "PRIVE",
    "5720": "PRIVE",
    "5770": "PRIVE",
    "5785": "PRIVE",
    # Secteur privé - entrepreneurs et professions libérales
    "1000": "PRIVE",
    "1100": "PRIVE",
    "1200": "PRIVE",
    "1300": "PRIVE",
    "1400": "PRIVE",
    "1500": "PRIVE",
    "1600": "PRIVE",
    # Secteur privé - associations et organismes de droit privé
    "9220": "PRIVE",
    "9230": "PRIVE",
    "9240": "PRIVE",
    "9260": "PRIVE",
    "9300": "PRIVE",
}


def map_legal_to_category(
    legal_nature: str | None, logger: logging.Logger | None = None
) -> str:
    """
    Convertit un code categorieJuridiqueUniteLegale SIRENE en catégorie métier.

    Args:
        legal_nature: Code SIRENE sur 4 chiffres (str) ou None.
        logger: Logger optionnel pour tracer les codes non mappés.

    Returns:
        "PUBLIC", "PRIVE" ou "INCONNU".
    """
    if not legal_nature:
        return "INCONNU"

    category = LEGAL_NATURE_MAPPING.get(legal_nature)
    if category is None:
        if logger:
            logger.warning("Code legal_nature non mappé: '%s' → INCONNU", legal_nature)
        return "INCONNU"

    return category


__all__ = ["LEGAL_NATURE_MAPPING", "map_legal_to_category"]
