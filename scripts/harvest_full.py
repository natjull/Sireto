#!/usr/bin/env python3
"""
HARVEST COMPLET - API Recherche d'entreprises
==============================================
Siphonne TOUTES les données disponibles de l'API sans rien oublier.

Fonctionnalités:
- Récupère TOUT: entreprises, sièges, compléments, dirigeants, finances, établissements
- Drill-down récursif automatique pour contourner la limite de 10k résultats
- Stop/Resume: Ctrl+C propre, reprend exactement où on s'est arrêté
- Compression SQLite native (VACUUM + page_size optimisé)
- Statistiques en temps réel

Usage:
    python scripts/harvest_full.py              # Lancer ou reprendre
    python scripts/harvest_full.py --reset      # Repartir de zéro
    python scripts/harvest_full.py --stats      # Afficher les stats
"""

from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import sys
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import requests
from tqdm import tqdm

# ============================================================================ #
#                                 CONFIGURATION                                #
# ============================================================================ #

BASE_URL = "https://recherche-entreprises.api.gouv.fr/search"
DB_PATH = Path("data/harvest_full.sqlite")

# Rate limiting
REQUESTS_PER_SECOND = 7
REQUEST_INTERVAL = 1.0 / REQUESTS_PER_SECOND  # ~0.143s

# API limits
PER_PAGE = 25
MAX_RESULTS = 10_000
MAX_PAGES = MAX_RESULTS // PER_PAGE  # 400

# Retry config
MAX_RETRIES = 5
BACKOFF_BASE = 1.0

# Départements (tous, y compris DOM-TOM)
DEPARTEMENTS = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "21",
    "22", "23", "24", "25", "26", "27", "28", "29", "2A", "2B",
    "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
    "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
    "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
    "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
    "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
    "80", "81", "82", "83", "84", "85", "86", "87", "88", "89",
    "90", "91", "92", "93", "94", "95", "971", "972", "973", "974",
    "975", "976", "977", "978", "984", "986", "987", "988", "989"
]

NAF_SECTIONS = list("ABCDEFGHIJKLMNOPQRSTU")

TRANCHES_EFFECTIF = [
    "NN", "00", "01", "02", "03", "11", "12", "21", "22", 
    "31", "32", "41", "42", "51", "52", "53"
]

# Natures juridiques pour entreprises étrangères ou spéciales (sans localisation française)
# Source: https://www.insee.fr/fr/information/2028129
NATURES_JURIDIQUES_SPECIALES = [
    # Personnes morales de droit étranger
    "3110", "3120",  # Représentation d'États étrangers
    "3210", "3220",  # Sociétés commerciales étrangères
    "3290",  # Autres personnes morales de droit étranger
    # Organismes internationaux
    "4110", "4120", "4130", "4140", "4150", "4160",
    # Et aussi recherche par catégorie entreprise pour compléter
]

# ============================================================================ #
#                                   LOGGING                                    #
# ============================================================================ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger(__name__)

# ============================================================================ #
#                              GRACEFUL SHUTDOWN                               #
# ============================================================================ #

class GracefulShutdown:
    """Handle Ctrl+C gracefully - finish current operation then exit."""
    
    def __init__(self):
        self.should_stop = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)
    
    def _handler(self, signum, frame):
        if self.should_stop:
            LOG.warning("Force quit!")
            sys.exit(1)
        LOG.warning("\n⚠️  Arrêt demandé - Fin de l'opération en cours...")
        self.should_stop = True

SHUTDOWN = GracefulShutdown()

# ============================================================================ #
#                                RATE LIMITER                                  #
# ============================================================================ #

class RateLimiter:
    def __init__(self, interval: float):
        self.interval = interval
        self.last_request = 0.0
    
    def acquire(self):
        now = time.monotonic()
        elapsed = now - self.last_request
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_request = time.monotonic()

# ============================================================================ #
#                                   SCOPE                                      #
# ============================================================================ #

@dataclass
class Scope:
    """Represents a query scope with filters for drill-down."""
    departement: Optional[str] = None
    code_postal: Optional[str] = None
    section_activite: Optional[str] = None
    activite_principale: Optional[str] = None
    tranche_effectif: Optional[str] = None
    etat_administratif: Optional[str] = None  # A ou C
    nature_juridique: Optional[str] = None  # Pour entreprises étrangères
    
    @property
    def key(self) -> str:
        parts = []
        if self.departement:
            parts.append(f"D:{self.departement}")
        if self.code_postal:
            parts.append(f"CP:{self.code_postal}")
        if self.section_activite:
            parts.append(f"S:{self.section_activite}")
        if self.activite_principale:
            parts.append(f"NAF:{self.activite_principale}")
        if self.tranche_effectif:
            parts.append(f"TE:{self.tranche_effectif}")
        if self.etat_administratif:
            parts.append(f"EA:{self.etat_administratif}")
        if self.nature_juridique:
            parts.append(f"NJ:{self.nature_juridique}")
        return "|".join(parts) if parts else "ALL"
    
    @property
    def depth(self) -> int:
        count = 0
        if self.departement: count += 1
        if self.code_postal: count += 1
        if self.section_activite: count += 1
        if self.activite_principale: count += 1
        if self.tranche_effectif: count += 1
        if self.etat_administratif: count += 1
        if self.nature_juridique: count += 1
        return count
    
    def to_params(self) -> Dict[str, Any]:
        params = {}
        if self.departement:
            params["departement"] = self.departement
        if self.code_postal:
            params["code_postal"] = self.code_postal
        if self.section_activite:
            params["section_activite_principale"] = self.section_activite
        if self.activite_principale:
            params["activite_principale"] = self.activite_principale
        if self.tranche_effectif:
            params["tranche_effectif_salarie"] = self.tranche_effectif
        if self.etat_administratif:
            params["etat_administratif"] = self.etat_administratif
        if self.nature_juridique:
            params["nature_juridique"] = self.nature_juridique
        return params

# ============================================================================ #
#                                  DATABASE                                    #
# ============================================================================ #

def init_db(conn: sqlite3.Connection):
    """Create all tables with optimized settings."""
    
    # Optimizations for performance + compression
    conn.execute("PRAGMA page_size = 4096;")
    conn.execute("PRAGMA journal_mode = WAL;")  # Write-ahead logging
    conn.execute("PRAGMA synchronous = NORMAL;")  # Good balance speed/safety
    conn.execute("PRAGMA cache_size = -64000;")  # 64MB cache
    conn.execute("PRAGMA temp_store = MEMORY;")
    
    # Entreprises (données principales)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entreprises (
            siren TEXT PRIMARY KEY,
            nom_complet TEXT,
            nom_raison_sociale TEXT,
            sigle TEXT,
            nature_juridique TEXT,
            activite_principale TEXT,
            section_activite_principale TEXT,
            categorie_entreprise TEXT,
            annee_categorie_entreprise TEXT,
            tranche_effectif TEXT,
            annee_tranche_effectif TEXT,
            caractere_employeur TEXT,
            etat_administratif TEXT,
            statut_diffusion TEXT,
            date_creation TEXT,
            date_fermeture TEXT,
            date_mise_a_jour TEXT,
            date_mise_a_jour_insee TEXT,
            date_mise_a_jour_rne TEXT,
            nombre_etablissements INTEGER,
            nombre_etablissements_ouverts INTEGER,
            date_capture TEXT DEFAULT (datetime('now'))
        );
    """)
    
    # Sièges (1 par entreprise)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sieges (
            siret TEXT PRIMARY KEY,
            siren TEXT,
            adresse TEXT,
            complement_adresse TEXT,
            numero_voie TEXT,
            indice_repetition TEXT,
            type_voie TEXT,
            libelle_voie TEXT,
            distribution_speciale TEXT,
            code_postal TEXT,
            cedex TEXT,
            libelle_cedex TEXT,
            commune TEXT,
            libelle_commune TEXT,
            libelle_commune_etranger TEXT,
            code_pays_etranger TEXT,
            libelle_pays_etranger TEXT,
            departement TEXT,
            region TEXT,
            epci TEXT,
            latitude TEXT,
            longitude TEXT,
            geo_id TEXT,
            geo_adresse TEXT,
            coordonnees TEXT,
            activite_principale TEXT,
            activite_principale_registre_metier TEXT,
            tranche_effectif TEXT,
            annee_tranche_effectif TEXT,
            caractere_employeur TEXT,
            etat_administratif TEXT,
            est_siege INTEGER,
            date_creation TEXT,
            date_debut_activite TEXT,
            date_fermeture TEXT,
            date_mise_a_jour TEXT,
            date_mise_a_jour_insee TEXT,
            statut_diffusion TEXT,
            nom_commercial TEXT,
            liste_enseignes TEXT,
            liste_finess TEXT,
            liste_idcc TEXT,
            liste_rge TEXT,
            liste_uai TEXT,
            liste_id_bio TEXT,
            liste_id_organisme_formation TEXT
        );
    """)
    
    # Établissements (matching_etablissements - jusqu'à 100 par entreprise)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS etablissements (
            siret TEXT PRIMARY KEY,
            siren TEXT,
            adresse TEXT,
            code_postal TEXT,
            commune TEXT,
            libelle_commune TEXT,
            departement TEXT,
            region TEXT,
            epci TEXT,
            latitude TEXT,
            longitude TEXT,
            geo_id TEXT,
            activite_principale TEXT,
            tranche_effectif TEXT,
            annee_tranche_effectif TEXT,
            caractere_employeur TEXT,
            etat_administratif TEXT,
            est_siege INTEGER,
            ancien_siege INTEGER,
            date_creation TEXT,
            date_debut_activite TEXT,
            date_fermeture TEXT,
            statut_diffusion TEXT,
            nom_commercial TEXT,
            liste_enseignes TEXT,
            liste_finess TEXT,
            liste_idcc TEXT,
            liste_rge TEXT,
            liste_uai TEXT,
            liste_id_bio TEXT,
            liste_id_organisme_formation TEXT
        );
    """)
    
    # Compléments (1 par entreprise)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS complements (
            siren TEXT PRIMARY KEY,
            identifiant_association TEXT,
            est_association INTEGER,
            est_ess INTEGER,
            est_entrepreneur_individuel INTEGER,
            est_societe_mission INTEGER,
            est_service_public INTEGER,
            est_l100_3 INTEGER,
            est_finess INTEGER,
            est_rge INTEGER,
            est_bio INTEGER,
            est_qualiopi INTEGER,
            est_organisme_formation INTEGER,
            est_entrepreneur_spectacle INTEGER,
            est_uai INTEGER,
            est_siae INTEGER,
            est_patrimoine_vivant INTEGER,
            est_achats_responsables INTEGER,
            est_alim_confiance INTEGER,
            convention_collective_renseignee INTEGER,
            egapro_renseignee INTEGER,
            bilan_ges_renseigne INTEGER,
            type_siae TEXT,
            statut_entrepreneur_spectacle TEXT,
            statut_bio TEXT,
            liste_idcc TEXT,
            liste_id_organisme_formation TEXT,
            liste_finess_juridique TEXT,
            collectivite_territoriale TEXT
        );
    """)
    
    # Dirigeants
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dirigeants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            siren TEXT,
            type_dirigeant TEXT,
            nom TEXT,
            prenoms TEXT,
            date_naissance TEXT,
            annee_naissance TEXT,
            nationalite TEXT,
            siren_pm TEXT,
            denomination TEXT,
            qualite TEXT,
            UNIQUE(siren, type_dirigeant, nom, prenoms, qualite, siren_pm, denomination)
        );
    """)
    
    # Finances
    conn.execute("""
        CREATE TABLE IF NOT EXISTS finances (
            siren TEXT,
            annee TEXT,
            chiffre_affaires INTEGER,
            resultat_net INTEGER,
            PRIMARY KEY (siren, annee)
        );
    """)
    
    # Scopes complétés (pour reprise)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scopes_completed (
            scope_key TEXT PRIMARY KEY,
            depth INTEGER,
            total_results INTEGER,
            pages_fetched INTEGER,
            entreprises_saved INTEGER,
            completed_at TEXT DEFAULT (datetime('now'))
        );
    """)
    
    # Scopes en cours (pour reprise au milieu)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scopes_in_progress (
            scope_key TEXT PRIMARY KEY,
            total_results INTEGER,
            last_page INTEGER,
            started_at TEXT DEFAULT (datetime('now'))
        );
    """)
    
    # Index
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sieges_siren ON sieges(siren);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sieges_dept ON sieges(departement);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sieges_cp ON sieges(code_postal);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_etab_siren ON etablissements(siren);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dir_siren ON dirigeants(siren);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dir_siren_pm ON dirigeants(siren_pm);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fin_siren ON finances(siren);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ent_etat ON entreprises(etat_administratif);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ent_dept ON entreprises(section_activite_principale);")
    
    conn.commit()


def is_scope_completed(conn: sqlite3.Connection, scope: Scope) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM scopes_completed WHERE scope_key = ?", 
        (scope.key,)
    )
    return cur.fetchone() is not None


def get_scope_progress(conn: sqlite3.Connection, scope: Scope) -> Optional[tuple]:
    """Return (total_results, last_page) if scope is in progress."""
    cur = conn.execute(
        "SELECT total_results, last_page FROM scopes_in_progress WHERE scope_key = ?",
        (scope.key,)
    )
    return cur.fetchone()


def mark_scope_in_progress(conn: sqlite3.Connection, scope: Scope, 
                            total_results: int, last_page: int):
    conn.execute("""
        INSERT OR REPLACE INTO scopes_in_progress 
        (scope_key, total_results, last_page) VALUES (?, ?, ?)
    """, (scope.key, total_results, last_page))
    conn.commit()


def mark_scope_completed(conn: sqlite3.Connection, scope: Scope, 
                         total_results: int, pages: int, entreprises: int):
    conn.execute("""
        INSERT OR REPLACE INTO scopes_completed 
        (scope_key, depth, total_results, pages_fetched, entreprises_saved)
        VALUES (?, ?, ?, ?, ?)
    """, (scope.key, scope.depth, total_results, pages, entreprises))
    conn.execute("DELETE FROM scopes_in_progress WHERE scope_key = ?", (scope.key,))
    conn.commit()

# ============================================================================ #
#                               DATA EXTRACTION                                #
# ============================================================================ #

def to_json(obj) -> Optional[str]:
    """Convert list/dict to JSON string for storage."""
    if obj is None:
        return None
    if isinstance(obj, (list, dict)):
        return json.dumps(obj, ensure_ascii=False)
    return str(obj)


def save_entreprise(conn: sqlite3.Connection, ent: dict) -> int:
    """Save one entreprise and all its related data. Returns 1 if saved, 0 if skipped."""
    siren = ent.get("siren")
    if not siren:
        return 0
    
    # Entreprise principale
    conn.execute("""
        INSERT OR REPLACE INTO entreprises VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now')
        )
    """, (
        siren,
        ent.get("nom_complet"),
        ent.get("nom_raison_sociale"),
        ent.get("sigle"),
        ent.get("nature_juridique"),
        ent.get("activite_principale"),
        ent.get("section_activite_principale"),
        ent.get("categorie_entreprise"),
        ent.get("annee_categorie_entreprise"),
        ent.get("tranche_effectif_salarie"),
        ent.get("annee_tranche_effectif_salarie"),
        ent.get("caractere_employeur"),
        ent.get("etat_administratif"),
        ent.get("statut_diffusion"),
        ent.get("date_creation"),
        ent.get("date_fermeture"),
        ent.get("date_mise_a_jour"),
        ent.get("date_mise_a_jour_insee"),
        ent.get("date_mise_a_jour_rne"),
        ent.get("nombre_etablissements"),
        ent.get("nombre_etablissements_ouverts"),
    ))
    
    # Siège
    siege = ent.get("siege")
    if siege:
        conn.execute("""
            INSERT OR REPLACE INTO sieges VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            siege.get("siret"),
            siren,
            siege.get("adresse"),
            siege.get("complement_adresse"),
            siege.get("numero_voie"),
            siege.get("indice_repetition"),
            siege.get("type_voie"),
            siege.get("libelle_voie"),
            siege.get("distribution_speciale"),
            siege.get("code_postal"),
            siege.get("cedex"),
            siege.get("libelle_cedex"),
            siege.get("commune"),
            siege.get("libelle_commune"),
            siege.get("libelle_commune_etranger"),
            siege.get("code_pays_etranger"),
            siege.get("libelle_pays_etranger"),
            siege.get("departement"),
            siege.get("region"),
            siege.get("epci"),
            siege.get("latitude"),
            siege.get("longitude"),
            siege.get("geo_id"),
            siege.get("geo_adresse"),
            siege.get("coordonnees"),
            siege.get("activite_principale"),
            siege.get("activite_principale_registre_metier"),
            siege.get("tranche_effectif_salarie"),
            siege.get("annee_tranche_effectif_salarie"),
            siege.get("caractere_employeur"),
            siege.get("etat_administratif"),
            1 if siege.get("est_siege") else 0,
            siege.get("date_creation"),
            siege.get("date_debut_activite"),
            siege.get("date_fermeture"),
            siege.get("date_mise_a_jour"),
            siege.get("date_mise_a_jour_insee"),
            siege.get("statut_diffusion_etablissement"),
            siege.get("nom_commercial"),
            to_json(siege.get("liste_enseignes")),
            to_json(siege.get("liste_finess")),
            to_json(siege.get("liste_idcc")),
            to_json(siege.get("liste_rge")),
            to_json(siege.get("liste_uai")),
            to_json(siege.get("liste_id_bio")),
            to_json(siege.get("liste_id_organisme_formation")),
        ))
    
    # Établissements (matching_etablissements)
    for etab in ent.get("matching_etablissements") or []:
        siret = etab.get("siret")
        if not siret:
            continue
        try:
            conn.execute("""
                INSERT OR REPLACE INTO etablissements VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                siret,
                siren,
                etab.get("adresse"),
                etab.get("code_postal"),
                etab.get("commune"),
                etab.get("libelle_commune"),
                None,  # departement not in matching
                etab.get("region"),
                etab.get("epci"),
                etab.get("latitude"),
                etab.get("longitude"),
                etab.get("geo_id"),
                etab.get("activite_principale"),
                etab.get("tranche_effectif_salarie"),
                etab.get("annee_tranche_effectif_salarie"),
                etab.get("caractere_employeur"),
                etab.get("etat_administratif"),
                1 if etab.get("est_siege") else 0,
                1 if etab.get("ancien_siege") else 0,
                etab.get("date_creation"),
                etab.get("date_debut_activite"),
                etab.get("date_fermeture"),
                etab.get("statut_diffusion_etablissement"),
                etab.get("nom_commercial"),
                to_json(etab.get("liste_enseignes")),
                to_json(etab.get("liste_finess")),
                to_json(etab.get("liste_idcc")),
                to_json(etab.get("liste_rge")),
                to_json(etab.get("liste_uai")),
                to_json(etab.get("liste_id_bio")),
                to_json(etab.get("liste_id_organisme_formation")),
            ))
        except sqlite3.IntegrityError:
            pass  # Duplicate, skip
    
    # Compléments
    comp = ent.get("complements")
    if comp:
        ct = comp.get("collectivite_territoriale")
        conn.execute("""
            INSERT OR REPLACE INTO complements VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            siren,
            comp.get("identifiant_association"),
            1 if comp.get("est_association") else 0,
            1 if comp.get("est_ess") else 0,
            1 if comp.get("est_entrepreneur_individuel") else 0,
            1 if comp.get("est_societe_mission") else 0,
            1 if comp.get("est_service_public") else 0,
            1 if comp.get("est_l100_3") else 0,
            1 if comp.get("est_finess") else 0,
            1 if comp.get("est_rge") else 0,
            1 if comp.get("est_bio") else 0,
            1 if comp.get("est_qualiopi") else 0,
            1 if comp.get("est_organisme_formation") else 0,
            1 if comp.get("est_entrepreneur_spectacle") else 0,
            1 if comp.get("est_uai") else 0,
            1 if comp.get("est_siae") else 0,
            1 if comp.get("est_patrimoine_vivant") else 0,
            1 if comp.get("est_achats_responsables") else 0,
            1 if comp.get("est_alim_confiance") else 0,
            1 if comp.get("convention_collective_renseignee") else 0,
            1 if comp.get("egapro_renseignee") else 0,
            1 if comp.get("bilan_ges_renseigne") else 0,
            comp.get("type_siae"),
            comp.get("statut_entrepreneur_spectacle"),
            comp.get("statut_bio"),
            to_json(comp.get("liste_idcc")),
            to_json(comp.get("liste_id_organisme_formation")),
            to_json(comp.get("liste_finess_juridique")),
            to_json(ct) if ct else None,
        ))
    
    # Dirigeants
    for d in ent.get("dirigeants") or []:
        type_dir = d.get("type_dirigeant", "")
        try:
            if type_dir == "personne morale":
                conn.execute("""
                    INSERT OR IGNORE INTO dirigeants 
                    (siren, type_dirigeant, siren_pm, denomination, qualite)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    siren, type_dir, 
                    d.get("siren"), d.get("denomination"), d.get("qualite")
                ))
            else:
                conn.execute("""
                    INSERT OR IGNORE INTO dirigeants 
                    (siren, type_dirigeant, nom, prenoms, date_naissance, 
                     annee_naissance, nationalite, qualite)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    siren, type_dir,
                    d.get("nom"), d.get("prenoms"), d.get("date_de_naissance"),
                    d.get("annee_de_naissance"), d.get("nationalite"), d.get("qualite")
                ))
        except sqlite3.IntegrityError:
            pass
    
    # Finances
    finances = ent.get("finances")
    if finances:
        for annee, data in finances.items():
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO finances VALUES (?, ?, ?, ?)
                """, (
                    siren, annee,
                    data.get("ca"), data.get("resultat_net")
                ))
            except (sqlite3.IntegrityError, TypeError):
                pass
    
    return 1

# ============================================================================ #
#                                 API CLIENT                                   #
# ============================================================================ #

class APIClient:
    def __init__(self, limiter: RateLimiter):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "Sireto-HarvestFull/1.0",
        })
        self.limiter = limiter
        self.request_count = 0
    
    def fetch(self, scope: Scope, page: int = 1) -> Optional[dict]:
        """Fetch a page. Returns None on unrecoverable error."""
        params = scope.to_params()
        params["page"] = page
        params["per_page"] = PER_PAGE
        params["limite_matching_etablissements"] = 100  # Max établissements
        # NO minimal=True - we want EVERYTHING
        
        for attempt in range(MAX_RETRIES):
            if SHUTDOWN.should_stop:
                return None
            
            self.limiter.acquire()
            self.request_count += 1
            
            try:
                resp = self.session.get(BASE_URL, params=params, timeout=30)
                
                if resp.status_code == 429:
                    delay = float(resp.headers.get("Retry-After", 5.0))
                    LOG.warning("429 Rate Limited, waiting %.1fs...", delay)
                    time.sleep(delay)
                    continue
                
                if resp.status_code == 400:
                    LOG.warning("400 Bad Request: %s - skipping", scope.key)
                    return None
                
                if resp.status_code in (500, 502, 503, 504):
                    delay = BACKOFF_BASE * (2 ** attempt)
                    LOG.warning("%d Server Error, retry in %.1fs", resp.status_code, delay)
                    time.sleep(delay)
                    continue
                
                resp.raise_for_status()
                return resp.json()
                
            except requests.exceptions.Timeout:
                delay = BACKOFF_BASE * (2 ** attempt)
                LOG.warning("Timeout, retry in %.1fs", delay)
                time.sleep(delay)
            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    LOG.error("Request failed: %s", e)
                    return None
                time.sleep(BACKOFF_BASE * (2 ** attempt))
        
        return None
    
    def get_total(self, scope: Scope) -> int:
        """Quick probe to get total_results."""
        # Use minimal for probes to save bandwidth
        params = scope.to_params()
        params["page"] = 1
        params["per_page"] = 1
        params["minimal"] = True
        
        self.limiter.acquire()
        self.request_count += 1
        
        try:
            resp = self.session.get(BASE_URL, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("total_results", 0)
        except Exception:
            pass
        return 0

# ============================================================================ #
#                              DRILL-DOWN LOGIC                                #
# ============================================================================ #

def get_postal_codes(dep: str) -> List[str]:
    """Generate all postal codes for a département."""
    codes = []
    # Corse uses 20xxx for both 2A and 2B
    if dep in ("2A", "2B"):
        for i in range(1000):
            codes.append(f"20{i:03d}")
    elif len(dep) == 2:
        for i in range(1000):
            codes.append(f"{dep}{i:03d}")
    elif len(dep) == 3:  # DOM-TOM
        for i in range(100):
            codes.append(f"{dep}{i:02d}")
    return codes


def discover_naf_codes(client: APIClient, conn: sqlite3.Connection,
                       scope: Scope, stats: dict, pbar: tqdm) -> List[str]:
    """
    Scan pages to discover all NAF codes present in a scope.
    IMPORTANT: Also saves data during the scan to prevent data loss.
    
    Returns list of unique NAF codes found.
    """
    naf_codes = set()
    total = client.get_total(scope)
    
    if total == 0:
        return []
    
    # Scan enough pages to discover most NAF codes (max 400 pages = 10k results)
    pages_to_scan = min((total + PER_PAGE - 1) // PER_PAGE, MAX_PAGES)
    
    LOG.info("🔍 Découverte NAF dans %s (%d pages)...", scope.key, pages_to_scan)
    
    for page in range(1, pages_to_scan + 1):
        if SHUTDOWN.should_stop:
            break
            
        payload = client.fetch(scope, page=page)
        if not payload or not payload.get("results"):
            break
        
        # SAVE DATA during discovery (prevents data loss)
        for ent in payload.get("results", []):
            save_entreprise(conn, ent)
            naf = ent.get("activite_principale")
            if naf:
                naf_codes.add(naf)
        
        conn.commit()
        stats["pages"] += 1
        stats["entreprises"] += len(payload.get("results", []))
        pbar.update(1)
        pbar.set_postfix({"Ent": f"{stats['entreprises']:,}"})
    
    LOG.info("   → %d codes NAF découverts", len(naf_codes))
    return sorted(naf_codes)


def process_scope(client: APIClient, conn: sqlite3.Connection, 
                  scope: Scope, stats: dict, pbar: tqdm) -> None:
    """Process a scope - download all pages or drill down if >10k."""
    
    if SHUTDOWN.should_stop:
        return
    
    # Already done?
    if is_scope_completed(conn, scope):
        return
    
    # Check for in-progress resume
    progress = get_scope_progress(conn, scope)
    if progress:
        total_results, start_page = progress
        start_page += 1  # Resume from next page
    else:
        total_results = client.get_total(scope)
        start_page = 1
    
    if total_results == 0:
        mark_scope_completed(conn, scope, 0, 0, 0)
        return
    
    # Need to drill down?
    if total_results >= MAX_RESULTS:
        drill_down(client, conn, scope, total_results, stats, pbar)
        return
    
    # Download all pages
    total_pages = min((total_results + PER_PAGE - 1) // PER_PAGE, MAX_PAGES)
    ent_saved = 0
    
    for page in range(start_page, total_pages + 1):
        if SHUTDOWN.should_stop:
            mark_scope_in_progress(conn, scope, total_results, page - 1)
            return
        
        # Robust fetch with local retry to avoid silent page loss
        payload = None
        for attempt in range(3):
            payload = client.fetch(scope, page)
            if payload:
                break
            time.sleep(1 + attempt)
        if not payload:
            LOG.error("Page %d failed for scope %s - keeping progress and retry later", page, scope.key)
            mark_scope_in_progress(conn, scope, total_results, page - 1)
            return
        
        for ent in payload.get("results") or []:
            ent_saved += save_entreprise(conn, ent)
        
        conn.commit()
        mark_scope_in_progress(conn, scope, total_results, page)
        
        stats["pages"] += 1
        stats["entreprises"] += len(payload.get("results") or [])
        pbar.update(1)
        pbar.set_postfix({
            "Ent": f"{stats['entreprises']:,}",
            "Req": f"{client.request_count:,}"
        })
        
        if not payload.get("results"):
            break
    
    mark_scope_completed(conn, scope, total_results, total_pages, ent_saved)
    stats["scopes"] += 1


def drill_down(client: APIClient, conn: sqlite3.Connection, 
               scope: Scope, total: int, stats: dict, pbar: tqdm) -> None:
    """Recursively drill down when >10k results."""
    
    LOG.info("🔀 DRILL-DOWN: %s (%d results) → depth %d", scope.key, total, scope.depth + 1)
    
    # For nature_juridique scopes (foreign companies), drill by section NAF first
    if scope.nature_juridique and not scope.section_activite:
        for section in NAF_SECTIONS:
            if SHUTDOWN.should_stop:
                return
            new_scope = Scope(
                nature_juridique=scope.nature_juridique,
                section_activite=section,
            )
            process_scope(client, conn, new_scope, stats, pbar)
        mark_scope_completed(conn, scope, total, 0, 0)
        return
    
    # For nature_juridique + section, drill by tranche effectif
    if scope.nature_juridique and scope.section_activite and not scope.tranche_effectif:
        for te in TRANCHES_EFFECTIF:
            if SHUTDOWN.should_stop:
                return
            new_scope = Scope(
                nature_juridique=scope.nature_juridique,
                section_activite=scope.section_activite,
                tranche_effectif=te,
            )
            process_scope(client, conn, new_scope, stats, pbar)
        mark_scope_completed(conn, scope, total, 0, 0)
        return
    
    # For NJ + Section + TE, drill by etat_administratif
    if scope.nature_juridique and scope.section_activite and scope.tranche_effectif and not scope.etat_administratif:
        for ea in ["A", "C"]:
            if SHUTDOWN.should_stop:
                return
            new_scope = Scope(
                nature_juridique=scope.nature_juridique,
                section_activite=scope.section_activite,
                tranche_effectif=scope.tranche_effectif,
                etat_administratif=ea,
            )
            process_scope(client, conn, new_scope, stats, pbar)
        mark_scope_completed(conn, scope, total, 0, 0)
        return
    
    # Level 1: Département → Code Postal
    if scope.departement and not scope.code_postal:
        for cp in get_postal_codes(scope.departement):
            if SHUTDOWN.should_stop:
                return
            new_scope = Scope(
                departement=scope.departement,
                code_postal=cp,
            )
            process_scope(client, conn, new_scope, stats, pbar)
        mark_scope_completed(conn, scope, total, 0, 0)
        return
    
    # Level 2: → Section NAF
    if not scope.section_activite:
        for section in NAF_SECTIONS:
            if SHUTDOWN.should_stop:
                return
            new_scope = Scope(
                departement=scope.departement,
                code_postal=scope.code_postal,
                section_activite=section,
            )
            process_scope(client, conn, new_scope, stats, pbar)
        mark_scope_completed(conn, scope, total, 0, 0)
        return
    
    # Level 3: → Tranche Effectif
    if not scope.tranche_effectif:
        for te in TRANCHES_EFFECTIF:
            if SHUTDOWN.should_stop:
                return
            new_scope = Scope(
                departement=scope.departement,
                code_postal=scope.code_postal,
                section_activite=scope.section_activite,
                tranche_effectif=te,
            )
            process_scope(client, conn, new_scope, stats, pbar)
        mark_scope_completed(conn, scope, total, 0, 0)
        return
    
    # Level 4: → État Administratif
    if not scope.etat_administratif:
        for ea in ["A", "C"]:
            if SHUTDOWN.should_stop:
                return
            new_scope = Scope(
                departement=scope.departement,
                code_postal=scope.code_postal,
                section_activite=scope.section_activite,
                tranche_effectif=scope.tranche_effectif,
                etat_administratif=ea,
            )
            process_scope(client, conn, new_scope, stats, pbar)
        mark_scope_completed(conn, scope, total, 0, 0)
        return
    
    # Level 5: → NAF codes discovered dynamically
    # (for ultra-dense scopes like Paris 75008 Section G after all drill-downs)
    if not scope.activite_principale:
        # Discover NAF codes present in this scope (also saves data during scan)
        naf_codes = discover_naf_codes(client, conn, scope, stats, pbar)
        
        if naf_codes:
            LOG.info("🔀 DRILL-DOWN NAF: %s → %d codes NAF", scope.key, len(naf_codes))
            for naf in naf_codes:
                if SHUTDOWN.should_stop:
                    return
                new_scope = Scope(
                    departement=scope.departement,
                    code_postal=scope.code_postal,
                    section_activite=scope.section_activite,
                    tranche_effectif=scope.tranche_effectif,
                    etat_administratif=scope.etat_administratif,
                    activite_principale=naf,
                    nature_juridique=scope.nature_juridique,
                )
                process_scope(client, conn, new_scope, stats, pbar)
        
        mark_scope_completed(conn, scope, total, 0, 0)
        return
    
    # Can't drill further - download what we can (max 10k)
    LOG.warning("⚠️ MAX DEPTH: %s has %d results - downloading first 10k only!", scope.key, total)
    total_pages = MAX_PAGES
    ent_saved = 0
    
    for page in range(1, total_pages + 1):
        if SHUTDOWN.should_stop:
            return
        payload = client.fetch(scope, page)
        if not payload or not payload.get("results"):
            break
        for ent in payload.get("results") or []:
            ent_saved += save_entreprise(conn, ent)
        conn.commit()
        stats["pages"] += 1
        stats["entreprises"] += len(payload.get("results") or [])
        pbar.update(1)
    
    mark_scope_completed(conn, scope, total, total_pages, ent_saved)
    stats["scopes"] += 1
    stats["truncated"] += 1

# ============================================================================ #
#                                    MAIN                                      #
# ============================================================================ #

def show_stats(conn: sqlite3.Connection):
    """Display current harvest statistics."""
    print("\n" + "=" * 60)
    print("              STATISTIQUES HARVEST FULL")
    print("=" * 60)
    
    cur = conn.execute("SELECT COUNT(*) FROM entreprises")
    ent = cur.fetchone()[0]
    
    cur = conn.execute("SELECT COUNT(*) FROM sieges")
    sieges = cur.fetchone()[0]
    
    cur = conn.execute("SELECT COUNT(*) FROM etablissements")
    etab = cur.fetchone()[0]
    
    cur = conn.execute("SELECT COUNT(*) FROM dirigeants")
    dir_ = cur.fetchone()[0]
    
    cur = conn.execute("SELECT COUNT(*) FROM finances")
    fin = cur.fetchone()[0]
    
    cur = conn.execute("SELECT COUNT(*) FROM scopes_completed")
    scopes = cur.fetchone()[0]
    
    cur = conn.execute("SELECT COUNT(*) FROM scopes_in_progress")
    in_prog = cur.fetchone()[0]
    
    # DB size
    db_size = DB_PATH.stat().st_size / (1024 * 1024 * 1024)  # GB
    
    print(f"  Entreprises:     {ent:>15,}")
    print(f"  Sièges:          {sieges:>15,}")
    print(f"  Établissements:  {etab:>15,}")
    print(f"  Dirigeants:      {dir_:>15,}")
    print(f"  Finances:        {fin:>15,}")
    print("-" * 60)
    print(f"  Scopes terminés: {scopes:>15,}")
    print(f"  Scopes en cours: {in_prog:>15,}")
    print(f"  Taille DB:       {db_size:>14.2f} GB")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Harvest complet API Recherche entreprises")
    parser.add_argument("--reset", action="store_true", help="Supprimer la DB et repartir de zéro")
    parser.add_argument("--stats", action="store_true", help="Afficher les statistiques seulement")
    parser.add_argument("--compress", action="store_true", help="Compresser la DB (VACUUM)")
    args = parser.parse_args()
    
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if args.reset and DB_PATH.exists():
        LOG.info("🗑️  Suppression de %s...", DB_PATH)
        DB_PATH.unlink()
        wal = DB_PATH.with_suffix(".sqlite-wal")
        shm = DB_PATH.with_suffix(".sqlite-shm")
        if wal.exists():
            wal.unlink()
        if shm.exists():
            shm.unlink()
    
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    init_db(conn)
    
    if args.stats:
        show_stats(conn)
        conn.close()
        return
    
    if args.compress:
        LOG.info("🗜️  Compression en cours (VACUUM)...")
        conn.execute("VACUUM;")
        LOG.info("✅ Compression terminée!")
        show_stats(conn)
        conn.close()
        return
    
    # Main harvest
    LOG.info("🚀 Démarrage du harvest complet...")
    LOG.info("   Départements: %d", len(DEPARTEMENTS))
    LOG.info("   Rate limit: %d req/s", REQUESTS_PER_SECOND)
    LOG.info("   DB: %s", DB_PATH)
    
    limiter = RateLimiter(REQUEST_INTERVAL)
    client = APIClient(limiter)
    
    stats = {"scopes": 0, "pages": 0, "entreprises": 0, "truncated": 0}
    
    # Estimate total pages (rough)
    pbar = tqdm(total=2_000_000, desc="Pages", unit="pg", smoothing=0.01)
    
    try:
        # PHASE 1: Départements français
        LOG.info("📍 PHASE 1: Départements français (%d)", len(DEPARTEMENTS))
        for dep in DEPARTEMENTS:
            if SHUTDOWN.should_stop:
                break
            
            scope = Scope(departement=dep)
            process_scope(client, conn, scope, stats, pbar)
            
            # Periodic checkpoint (every 100k pages)
            if stats["pages"] % 100_000 == 0 and stats["pages"] > 0:
                LOG.info("🔄 Checkpoint...")
                conn.commit()
        
        # PHASE 2: Entreprises étrangères (par nature juridique)
        if not SHUTDOWN.should_stop:
            LOG.info("🌍 PHASE 2: Entreprises étrangères/spéciales")
            for nj in NATURES_JURIDIQUES_SPECIALES:
                if SHUTDOWN.should_stop:
                    break
                scope = Scope(nature_juridique=nj)
                process_scope(client, conn, scope, stats, pbar)
    
    finally:
        pbar.close()
        conn.commit()
        
        LOG.info("")
        LOG.info("=" * 50)
        LOG.info("  RÉSUMÉ")
        LOG.info("=" * 50)
        LOG.info("  Scopes traités:    %d", stats["scopes"])
        LOG.info("  Pages téléchargées: %d", stats["pages"])
        LOG.info("  Entreprises:       %d", stats["entreprises"])
        LOG.info("  Requêtes API:      %d", client.request_count)
        if stats["truncated"]:
            LOG.warning("  ⚠️ Scopes tronqués: %d", stats["truncated"])
        LOG.info("=" * 50)
        
        if SHUTDOWN.should_stop:
            LOG.info("⏸️  Pause demandée. Relancez le script pour reprendre.")
        else:
            LOG.info("✅ Harvest terminé!")
        
        show_stats(conn)
        conn.close()


if __name__ == "__main__":
    main()
