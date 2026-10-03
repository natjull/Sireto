"""Source-family presence audit, not a recall or temporal-completeness gate.

Old publications remain immutable. In particular a legacy boolean must not
override explicit zero/missing source counts. Presence alone never proves
that all periods, variants, fields or observations were indexed.
"""
from collections.abc import Mapping


REQUIRED = (
    'SIRENE_CURRENT', 'SIRENE_HISTORY', 'SIRENE_SUCCESSION', 'RNE', 'BODACC',
)


def temporal_coverage(manifest: Mapping) -> dict:
    declared = manifest.get('temporal_complete')
    if declared is not None and not isinstance(declared, bool):
        raise ValueError('temporal_complete must be boolean or absent')
    counts = manifest.get('official_source_counts')
    if counts is None:
        return dict(status='UNVERIFIED_SOURCE_FAMILIES', declared=declared,
                    present=None, missing=None, declaration_consistent=None,
                    exhaustive_period_coverage_proven=False)
    if not isinstance(counts, Mapping):
        raise ValueError('official_source_counts must be a mapping')
    for source, count in counts.items():
        if (not isinstance(source, str) or not isinstance(count, int)
                or isinstance(count, bool) or count < 0):
            raise ValueError('source counts must be nonnegative integers')
    present = sorted(source for source, count in counts.items() if count > 0)
    missing = sorted(set(REQUIRED) - set(present))
    contradiction = declared is True and bool(missing)
    return dict(status=('DECLARATION_CONTRADICTED' if contradiction else
                        'SOURCE_FAMILIES_MISSING' if missing else
                        'SOURCE_FAMILIES_PRESENT_NOT_FULL_COVERAGE'),
                declared=declared, present=present, missing=missing,
                declaration_consistent=False if contradiction else None,
                exhaustive_period_coverage_proven=False)


def field_coverage(source_columns, projected_columns) -> dict:
    """Compare explicit fields only; no inference about replacement evidence."""
    available, projected = set(source_columns), set(projected_columns)
    if any(not isinstance(column, str) for column in available | projected):
        raise ValueError('field names must be strings')
    return dict(available=sorted(available), projected=sorted(projected),
                available_not_projected=sorted(available - projected),
                projected_not_available=sorted(projected - available),
                alternative_source_coverage_proven=False)
