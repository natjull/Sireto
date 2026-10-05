"""Complete exact facility-alias source access, not a ranked candidate pool.

Local/source-unknown/nonlocal facts stay distinct. Published place
SIRET is an official join hypothesis, not a certification of physical site.
"""
import json
import re
from scripts import retrieval_v310_culture_source as source
from .retrieval_contained_parent_v300 import fragments
from .retrieval_v20_workbench import record


def code(raw, size):
    value = raw.strip()
    pattern = '[0-9]{5}|2[AB][0-9]{3}' if size == 'insee' else '[0-9]{5}'
    return value if re.fullmatch(pattern, value) else None


class Facilities:
    def __init__(self, files, reference):
        root = files.file(reference).parent
        manifest, payload = source.sealed(root)
        for path in ('launch.json', 'phase/manifest.json', 'phase/supervision.json', 'phase/preflight.json', 'phase/payload.json'):
            files.file(record(root/path))
        for ref in json.loads(files.file(record(root/'launch.json')).read_text())['implementation'].values():
            files.file(ref)
        for entry in payload['source_epochs']:
            files.file(entry['reference'])
        self.c = source.readonly(files.file(manifest['database']))
        self.files, self.stats = files, manifest['stats']

    def facts(self, query):
        spans = fragments(query.name)
        by_endpoint, observations, missing, matched_rows = {}, 0, set(), set()
        for span in spans:
            self.files.guard()
            cursor = self.c.execute('SELECT a.ordinal,a.role,r.explicit_siret,r.code_commune,r.code_postal FROM aliases a '
                'JOIN records r ON r.ordinal=a.ordinal WHERE a.name=? ORDER BY a.ordinal,a.role', (span,))
            for ordinal, role, siret, raw_insee, raw_postcode in cursor:
                observations += 1
                if observations % 1024 == 0:
                    self.files.guard()
                matched_rows.add(ordinal)
                if siret is None:
                    missing.add(ordinal)
                    continue
                if not re.fullmatch('[0-9]{14}', siret) or siret[:9] == '000000000':
                    raise ValueError('Original explicit place SIRET required')
                insee, postal = code(raw_insee, 'insee'), code(raw_postcode, 'postal')
                local = bool((query.insee and insee == query.insee) or (query.postcode and postal == query.postcode))
                comparable = bool((query.insee and insee is not None) or (query.postcode and postal is not None))
                location = 'local' if local else 'nonlocal' if comparable else 'unknown'
                by_endpoint.setdefault(siret, set()).add(location)
        # Preserve every provenance category; a duplicated source endpoint
        # may have several localities. No hard geographic exclusion here.
        groups = {key: {s for s, places in by_endpoint.items() if key in places} for key in ('local', 'unknown', 'nonlocal')}
        self.files.check_epochs()
        return groups, dict(source_records=self.stats['records'], eligible_query_spans=len(spans),
            matched_observations=observations, matched_source_records=len(matched_rows),
            source_records_missing_explicit_siret=len(missing), distinct_explicit_endpoints=len(by_endpoint),
            locality_counts={key: len(values) for key, values in groups.items()},
            source_fact_or_site_cap=None, candidates_emitted=False, candidate_score_computed=False,
            exact_whole_source_alias=True, singleton_alias_expansion=False,
            source_kind='CULTURE_PUBLISHED_PLACE_SIRET',
            official_physical_site_identity_certified=False, owner_inferred_from_municipality=False)

    def close(self):
        self.c.close()
        self.files.check_epochs()
