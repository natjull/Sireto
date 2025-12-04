"""LLM #2 candidate filtering, prompting, and arbitration (tasks 7.1-7.4)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from pipe_v6.config import PipelineConfig
from pipe_v6.candidate_store import NormalizedCandidate
from pipe_v6.llm_normalizer import NormalizedCRMEntry
from pipe_v6.llm_utils import create_llm_client

LOGGER = logging.getLogger(__name__)


class MatchDecisionParseError(RuntimeError):
    """Raised when LLM #2 response cannot be parsed or is invalid."""


@dataclass(frozen=True)
class LLMMatchDecision:
    """Structured decision returned by LLM #2."""

    decision: Literal["BEST_MATCH", "NO_MATCH"]
    chosen_siret: str | None
    confidence: float  # 0.0 to 1.0
    reason: str | None


def _valid_sirets(candidates: list[NormalizedCandidate]) -> set[str]:
    return {c.siret for c in candidates if c.siret}


def _val(row, key: str) -> str:
    if hasattr(row, "get"):
        try:
            value = row.get(key, "")
        except Exception:
            value = ""
    elif hasattr(row, "_asdict"):
        try:
            value = row._asdict().get(key, "")  # type: ignore[call-arg]
        except Exception:
            value = ""
    else:
        value = getattr(row, key, "")
    return "" if value is None else str(value)


def build_matcher_prompt(
    row,
    norm_entry: NormalizedCRMEntry,
    candidates: list[NormalizedCandidate],
) -> str:
    """
    Construct the French prompt for LLM #2 arbitration (task 7.3).

    The prompt is intentionally verbose, structured, and ends with strict JSON
    output instructions to maximise parsability.
    """

    crm_name = _val(row, "crm_name")
    street_number = _val(row, "street_number")
    street_name = _val(row, "street_name")
    postcode = _val(row, "postcode")
    city = _val(row, "city")

    lines: list[str] = []

    lines.append(
        "Tu es un expert du matching d’entreprises françaises. Mission : choisir le meilleur SIRET parmi la liste ou conclure NO_MATCH. N’invente jamais de SIRET."
    )
    lines.append("")

    lines.append("# DONNEES CRM ORIGINALES")
    lines.append(f"- Nom brut : {crm_name}")
    lines.append(f"- Adresse : {street_number} {street_name}".strip())
    lines.append(f"- Code postal : {postcode}")
    lines.append(f"- Ville : {city}")
    lines.append("")

    lines.append("# DONNEES CRM NORMALISEES (LLM #1)")
    lines.append(f"- Nom normalisé : {norm_entry.normalized_name}")
    lines.append(f"- Adresse normalisée : {norm_entry.normalized_address}")
    lines.append(f"- Catégorie CRM : {norm_entry.category}")
    lines.append("")

    lines.append("# CANDIDATS SIRENE / SIRET (déjà filtrés)")

    for idx, cand in enumerate(candidates, start=1):
        source_count = len(cand.sources)
        multi_source = "Oui" if source_count >= 2 else "Non"
        lines.append(f"## Candidat {idx}")
        lines.append(f"- SIRET : {cand.siret}")
        lines.append(f"- SIREN : {cand.siren}")
        lines.append(f"- Nom : {cand.name}")
        lines.append(f"- Adresse : {cand.address}")
        lines.append(f"- Code postal : {cand.postcode}")
        lines.append(f"- Ville : {cand.city}")
        lines.append(f"- Code INSEE : {cand.insee_code or 'N/A'}")
        lines.append(f"- Catégorie SIRENE : {cand.category}")
        if getattr(cand, "qwant_ranks", None):
            ranks = ", ".join(
                f"{src.split('_',1)[1].lower()}=rank{rk}"
                for src, rk in sorted(cand.qwant_ranks.items())
            )
            lines.append(f"- Rangs Qwant (0=top) : {ranks}")
        lines.append(
            f"- Sources : {', '.join(cand.sources)} ({source_count} source{'s' if source_count > 1 else ''})"
        )
        lines.append(f"- Multi-source : {multi_source}")
        lines.append("")

    lines.append("# CAS DE CHANGEMENT DE NOM / RACHAT")
    lines.append("- Une entreprise peut changer de nom (rebranding), être rachetée ou fusionner.")
    lines.append("- Si l'adresse (numéro + voie + code postal + ville) est identique et précise, considère qu'il s'agit probablement du même établissement même si le nom a changé.")
    lines.append("- Indices forts de rebranding : même adresse exacte + multi-source (QWANT_*, RNE, DATAGOUV), SIREN/SIRET identique dans plusieurs sources, nom de groupe connu au lieu d'une ancienne enseigne.")
    lines.append("- Si le même SIREN/SIRET ressort dans les premiers résultats des trois requêtes Qwant, c'est un signal très fort, même si le nom diffère.")
    lines.append("")

    lines.append("# REGLES DE MATCHING (applique-les dans cet ordre)")
    lines.append("1) Adresse = critère principal : adresse exacte (numéro+voie+CP+ville) => très forte confiance; adresse partielle => confiance moyenne; adresse différente => très faible.")
    lines.append("2) Multi-source : 2+ sources indépendantes (QWANT_*, RNE, DATAGOUV) augmentent fortement la confiance.")
    lines.append("3) Catégorie : privilégie les candidats cohérents avec la catégorie CRM. Incohérence = pénalité, sauf si l'adresse est parfaite et unique.")
    lines.append("4) Nom : peut être très différent (enseigne vs raison sociale, rachat, sigle). Si l'adresse est parfaite et multi-source, accepte des différences importantes de nom, y compris un nom de groupe.")
    lines.append("5) NO_MATCH seulement si aucun candidat n'a une adresse précise cohérente, ou s'il y a plusieurs candidats avec des adresses fortes mais contradictoires. Ne retourne pas NO_MATCH uniquement parce que le nom a changé si adresse + sources convergent.")
    lines.append("")

    lines.append("# BAREME DE CONFIANCE (indicatif)")
    lines.append("- Adresse exacte (numéro+voie+CP+ville) : +0.50")
    lines.append("- Adresse très proche (même numéro et rue, même ville, CP voisin) : +0.30")
    lines.append("- Nom identique/similaire : +0.30")
    lines.append("- Nom différent mais rebranding/enseigne probable : +0.20")
    lines.append("- Multi-source (≥2 sources) : +0.20")
    lines.append("- Catégorie cohérente : +0.10")
    lines.append("Cas particuliers : un seul candidat avec adresse exacte + multi-source + catégorie cohérente -> confiance typiquement 0.80-0.95, même si le nom a changé.")
    lines.append("Si plusieurs candidats, adresse exacte pour l'un et seulement approximative pour les autres -> donne un score nettement plus élevé à l'adresse exacte.")
    lines.append("Clampe la confiance entre 0 et 1.")
    lines.append("")

    lines.append("# FORMAT DE REPONSE (JSON UNIQUEMENT, aucun texte avant/après, pas de ```)")
    lines.append("Reponds UNIQUEMENT en JSON valide, pas de texte libre.")
    lines.append("{")
    lines.append('  "decision": "BEST_MATCH" ou "NO_MATCH",')
    lines.append('  "chosen_siret": "12345678901234" ou null,')
    lines.append('  "confidence": nombre entre 0 et 1,')
    lines.append('  "reason": "explication courte (adresse/nom/sources)"')
    lines.append("}")

    return "\n".join(lines)


def filter_candidates_by_category(
    crm_category: str,
    candidates: list[NormalizedCandidate],
    config: PipelineConfig,
    logger: logging.Logger | None = None,
) -> list[NormalizedCandidate]:
    """
    Filter candidates according to CRM category with optional fallback.

    Rules (task 7.2):
    - PUBLIC  -> keep candidates.category == PUBLIC
    - PRIVE   -> keep candidates.category == PRIVE
    - EQUIPEMENT_URBAIN -> return [] (LLM will be bypassed later)
    - INCONNU -> no filtering
    - If filtering disabled in config -> no filtering
    - If filtered list is empty but original non-empty and fallback enabled -> use all candidates
    - Limit final list to config.max_candidates_llm_matcher, ordered by source count desc.
    """

    log = logger or LOGGER

    if not config.category_filter_enabled:
        filtered = list(candidates)
    else:
        crm_category = (crm_category or "").upper()

        if crm_category == "PUBLIC":
            filtered = [c for c in candidates if c.category == "PUBLIC"]
        elif crm_category == "PRIVE":
            filtered = [c for c in candidates if c.category == "PRIVE"]
        elif crm_category == "EQUIPEMENT_URBAIN":
            # No legal entity expected; LLM will be short-circuited by caller.
            return []
        else:  # INCONNU or any other -> no filter
            filtered = list(candidates)

        if not filtered and candidates and config.category_filter_fallback:
            log.info(
                "No candidates after category filter (%s). Falling back to all %d candidates.",
                crm_category or "INCONNU",
                len(candidates),
            )
            filtered = list(candidates)

    if len(filtered) > config.max_candidates_llm_matcher:
        filtered = sorted(filtered, key=lambda c: len(c.sources), reverse=True)[
            : config.max_candidates_llm_matcher
        ]
        log.info(
            "Too many candidates (%d), keeping top %d by source count.",
            len(candidates),
            config.max_candidates_llm_matcher,
        )

    return filtered


def parse_match_decision(
    data: dict,
    candidates: list[NormalizedCandidate],
    logger: logging.Logger | None = None,
) -> LLMMatchDecision:
    """
    Parse and validate the JSON payload returned by LLM #2.

    Rules:
    - decision must be "BEST_MATCH" or "NO_MATCH".
    - BEST_MATCH requires a chosen_siret present in the candidate list.
    - confidence is clamped to [0.0, 1.0].
    - Extra keys are ignored.

    On critical validation failure, raises MatchDecisionParseError.
    If chosen_siret is not in candidates, returns a forced NO_MATCH decision.
    """

    log = logger or LOGGER

    if not isinstance(data, dict):
        raise MatchDecisionParseError("LLM response is not a JSON object")

    try:
        decision = data["decision"]
    except KeyError as exc:
        raise MatchDecisionParseError("Missing field 'decision'") from exc

    if decision not in ("BEST_MATCH", "NO_MATCH"):
        raise MatchDecisionParseError(f"Invalid decision: {decision}")

    try:
        confidence_raw = data["confidence"]
    except KeyError as exc:
        raise MatchDecisionParseError("Missing field 'confidence'") from exc

    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError) as exc:
        raise MatchDecisionParseError(f"Invalid confidence: {confidence_raw}") from exc

    confidence = max(0.0, min(1.0, confidence))

    chosen_siret = data.get("chosen_siret")
    reason = data.get("reason")

    if decision == "BEST_MATCH":
        if not chosen_siret:
            raise MatchDecisionParseError("BEST_MATCH requires chosen_siret")

        valid_sirets = _valid_sirets(candidates)
        if chosen_siret not in valid_sirets:
            log.warning("Invalid SIRET from LLM: %s (not in candidates)", chosen_siret)
            return LLMMatchDecision(
                decision="NO_MATCH",
                chosen_siret=None,
                confidence=0.0,
                reason=f"LLM_VALIDATION_ERROR: SIRET {chosen_siret} not in candidates",
            )
    else:
        # Normalize: no SIRET when NO_MATCH
        chosen_siret = None

    return LLMMatchDecision(
        decision=decision,
        chosen_siret=chosen_siret,
        confidence=confidence,
        reason=reason,
    )


def decide_match(
    row,
    norm_entry: NormalizedCRMEntry,
    candidates: list[NormalizedCandidate],
    config: PipelineConfig,
    logger: logging.Logger | None = None,
    client=None,
) -> LLMMatchDecision:
    """
    Call LLM #2 to decide the best candidate SIRET or NO_MATCH (task 7.4).

    Behavior:
    - If no candidates, returns NO_MATCH without calling LLM.
    - Builds the matcher prompt and calls Ollama in JSON mode.
    - Parses and validates the JSON response (parse_match_decision).
    - On any error (LLM call, parsing), returns NO_MATCH with reason.
    """

    log = logger or LOGGER

    if not candidates:
        log.info("No candidates available, returning NO_MATCH without LLM call.")
        return LLMMatchDecision(
            decision="NO_MATCH",
            chosen_siret=None,
            confidence=0.0,
            reason="No candidates found after filtering",
        )

    owns_client = False
    if client is None:
        client = create_llm_client(config, logger=log)
        owns_client = True

    try:
        prompt = build_matcher_prompt(row, norm_entry, candidates)
        prompt_len = len(prompt)
        prompt_tokens_est = int(len(prompt.split()) * 1.3)
        crm_id = getattr(row, "crm_id", None)
        if crm_id is None and hasattr(row, "get"):
            try:
                crm_id = row.get("crm_id")
            except Exception:  # pragma: no cover - safe fallback
                crm_id = None
        log.info(
            "LLM matcher call: crm_id=%s candidates=%s prompt_len=%s chars (~%s tokens)",
            crm_id,
            len(candidates),
            prompt_len,
            prompt_tokens_est,
        )
        model_name = getattr(config, "matcher_model_name", None) or config.model_name
        response = client.call_json(
            prompt,
            num_predict=config.max_tokens_matcher,
            model=model_name,
        )

        if response.parsed_json is None:
            raise MatchDecisionParseError("LLM response missing JSON payload")

        return parse_match_decision(response.parsed_json, candidates, logger=log)

    except Exception as exc:  # catch all to never raise upstream
        log.error("LLM matcher failed: %s", exc, exc_info=True)
        return LLMMatchDecision(
            decision="NO_MATCH",
            chosen_siret=None,
            confidence=0.0,
            reason=f"LLM_ERROR: {type(exc).__name__}",
        )
    finally:
        if owns_client:
            client.close()


def classify_final_status(
    decision: LLMMatchDecision,
    config: PipelineConfig,
) -> Literal["MATCH", "REVIEW", "NO_MATCH"]:
    """
    Convert a raw LLM decision into final status based on confidence thresholds.

    Rules (task 7.5):
    - If decision is NO_MATCH -> return NO_MATCH regardless of confidence.
    - Else (BEST_MATCH):
        - confidence >= confidence_auto_match -> MATCH
        - confidence_review_min <= confidence < confidence_auto_match -> REVIEW
        - otherwise -> NO_MATCH
    """

    if decision.decision == "NO_MATCH":
        return "NO_MATCH"

    conf = decision.confidence

    if conf >= config.confidence_auto_match:
        return "MATCH"
    if conf >= config.confidence_review_min:
        return "REVIEW"
    return "NO_MATCH"


__all__ = [
    "LLMMatchDecision",
    "MatchDecisionParseError",
    "filter_candidates_by_category",
    "build_matcher_prompt",
    "parse_match_decision",
    "decide_match",
    "classify_final_status",
]
