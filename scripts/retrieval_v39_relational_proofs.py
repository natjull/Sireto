"""Typed official paths for initial retrieval, not a candidate reranker.

Identifiers remain joins/provenance, never text or similarity features.
An edge does not establish identity, site binding or name ownership.
"""
from dataclasses import asdict, dataclass
from datetime import date
import re


def identity(kind, value):
    if kind not in ('SIREN', 'SIRET') or not re.fullmatch(r'\d{' + ('9' if kind == 'SIREN' else '14') + '}', str(value or '')):
        raise ValueError('Invalid official identifier')
    return str(value)


def date_value(value):
    if value in (None, ''):
        return None
    return date.fromisoformat(str(value)[:10]).isoformat()


@dataclass(frozen=True)
class ProofEdge:
    source: str
    proof_id: str
    source_record_id: str
    relation_type: str
    from_kind: str
    from_id: str
    to_kind: str
    to_id: str
    reported_date: str | None
    date_semantics: str
    observed_at: str | None
    navigation: str
    scope: str
    role_code: str = ''
    role_label: str = ''

    def __post_init__(self):
        identity(self.from_kind, self.from_id); identity(self.to_kind, self.to_id)
        if not self.proof_id or not self.source_record_id:
            raise ValueError('Missing provenance')
        if self.from_kind == self.to_kind and self.from_id == self.to_id:
            raise ValueError('Self relation')

    def to_dict(self):
        return asdict(self)

    def bindings(self):
        """Logical query-planner bindings, not ranked candidate outputs.

        The required evidence must later be verified, never fabricated by a
        boolean passed from a query's truth. RNE remains context until its
        role dictionary and site grounding have been verified.
        """
        if self.navigation == 'CONTEXT_ONLY':
            return ()
        forward = dict(anchor_kind=self.from_kind, anchor_id=self.from_id,
                       target_kind=self.to_kind, target_id=self.to_id,
                       direction='FORWARD', relation_type=self.relation_type,
                       proof_id=self.proof_id, required_evidence=self.scope)
        if self.navigation == 'FORWARD_STRUCTURAL':
            return (forward,)
        return (forward, dict(forward, anchor_kind=self.to_kind, anchor_id=self.to_id,
                              target_kind=self.from_kind, target_id=self.from_id, direction='REVERSE'))


def membership(row):
    """Require both identifiers from an official row; never derive SIREN."""
    siren = identity('SIREN', row['siren']); siret = identity('SIRET', row['siret'])
    return ProofEdge('SIRENE', row['proof_id'], row['source_record_id'],
        'ESTABLISHMENT_OF', 'SIREN', siren, 'SIRET', siret, None, 'NOT_APPLICABLE',
        date_value(row.get('observed_at')), 'FORWARD_STRUCTURAL', 'OFFICIAL_MEMBERSHIP')


def succession(row):
    # Explicit true only; NULL and truthy strings do not establish continuity.
    if row.get('economic_continuity') is not True:
        return None
    identity('SIREN', row['predecessor_siren']); identity('SIREN', row['successor_siren'])
    return ProofEdge('SIRENE', row['relation_id'], row['relation_id'],
        'ESTABLISHMENT_SUCCESSION', 'SIRET', row['predecessor_siret'],
        'SIRET', row['successor_siret'], date_value(row.get('effective_date')),
        'SOURCE_EFFECTIVE_DATE', date_value(row.get('observed_at')),
        'BIDIRECTIONAL_CONDITIONAL', 'ECONOMIC_CONTINUITY_AND_QUERY_SITE_COMPATIBILITY')


def bodacc_transfer(row):
    if (row.get('source'), row.get('relation_type'), row.get('from_kind'), row.get('to_kind')) != (
            'BODACC', 'ASSET_TRANSFER', 'SIREN', 'SIREN'):
        raise ValueError('Unsupported BODACC relation')
    return ProofEdge('BODACC', row['relation_id'], row['source_record_id'],
        'ASSET_TRANSFER', 'SIREN', row['from_identifier'], 'SIREN', row['to_identifier'],
        date_value(row.get('effective_date')), 'REPORTED_DATE_EFFECTIVITY_UNVERIFIED',
        date_value(row.get('observed_at')), 'BIDIRECTIONAL_CONDITIONAL',
        'EVENT_SITE_BINDING_AND_OFFICIAL_TARGET_MEMBERSHIP')


def rne_role(row):
    if row.get('source') != 'RNE':
        raise ValueError('Unsupported role source')
    return ProofEdge('RNE', row['role_id'], row['source_record_id'],
        'LEGAL_ENTITY_ROLE', 'SIREN', row['subject_siren'], 'SIREN', row['related_siren'],
        date_value(row.get('effective_date')), 'SOURCE_ROLE_DATE_NOT_IDENTITY',
        date_value(row.get('observed_at')), 'CONTEXT_ONLY',
        'UNVERIFIED_ROLE_AND_SITE_SCOPE', str(row.get('role_code') or ''), str(row.get('role_label') or ''))


def site_name_binding_allowed(name, target_siret, official_parent):
    """A parent name is a parent proof, never a fabricated target-site alias."""
    identity('SIRET', target_siret); identity('SIREN', official_parent)
    if name['subject_kind'] == 'SIRET':
        return name['siret'] == target_siret and name['siren'] == official_parent
    if name['subject_kind'] == 'SIREN':
        return name['siren'] == official_parent
    return False


def deduplicate(edges):
    """Repeated official facts are not independent support; conflicts fail."""
    unique = {}
    for edge in edges:
        key = (edge.source, edge.proof_id)
        if key in unique and unique[key] != edge:
            raise ValueError('Conflicting official proof identifier')
        unique[key] = edge
    return tuple(unique[k] for k in sorted(unique))
