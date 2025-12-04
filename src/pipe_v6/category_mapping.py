"""Mapping des codes SIRENE vers catégories métier Pipe V6."""

from __future__ import annotations

import logging
from typing import Dict

# Codes categorieJuridiqueUniteLegale (4 chiffres) -> catégorie métier
# Catégories cibles : PUBLIC, PRIVE, INCONNU.
LEGAL_NATURE_MAPPING: Dict[str, str] = {
    # Ajouts ciblés (logs non mappés)
    # 2110 : Indivision entre personnes physiques -> personne(s) privée(s)
    "2110": "PRIVE",
    # 5660 : Autre SA coopérative à directoire (coopérative ≈ privé/ESS)
    "5660": "PRIVE",
    # 6596 : Coopérative (catégorie juridique listée dans les familles ESS)
    "6596": "PRIVE",
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
    "5558": "PUBLIC",  # Société publique locale (SPL) - variante
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

    # Manquants identifiés dans les logs / cache (SIRENE v3.11)
    # Sociétés / coop / mutuelles / finance
    "5202": "PRIVE",  # Société en commandite par actions
    "5485": "PRIVE",  # Coopérative agricole (code détaillé)
    "5699": "PRIVE",  # Autre société de financement / non codée
    "5699": "PRIVE",
    "6540": "PRIVE",  # Mutuelle / union (code rare)
    "6541": "PRIVE",  # Mutuelle / union (variante)
    "6589": "PRIVE",  # Organisme d’assurance divers
    "6599": "PRIVE",  # Autre organisme financier
    "6220": "PRIVE",  # Société en nom collectif (SNC) activité bancaire
    "7171": "PUBLIC", # Syndicat mixte fermé (assimilé public)
    "7353": "PUBLIC", # Établissement public industriel ou commercial local
    "7361": "PUBLIC", # Régie autonome (eau/énergie/transport)
    "9150": "PRIVE",  # Association intermédiaire / économie sociale (par défaut privé)
    "3120": "PRIVE",  # Exploitant agricole personne morale
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
