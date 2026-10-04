"""V139 exact structured evidence, releasing only this reader's own pages.

Unmeasured national integration; no score, source, field or candidate change.
This removes cumulative mmap residency during a complete V146 site scan.
"""
import json
from pathlib import Path

import numpy as np

from .retrieval_name_pages_v151 import fingerprint
from .retrieval_paged_names_v170 import PagedReboundNames
from .retrieval_site_attributes_v139 import StructuredSites, encode_code, MISSING
from .retrieval_variants_v118 import normalized
from .retrieval_v20_workbench import verify


class PagedStructuredSites(StructuredSites):
    """Same readonly V139 sources/results; all extras remain UNKNOWN."""
    def __init__(self, reference):
        self.valid = False
        self.a = {}
        self.reused = {}
        try:
            super().__init__(reference)
            self.specs = {}
            self.references = []
            for group, arrays, refs in (('a', self.a, self.metadata['artifacts']),
                                        ('reused', self.reused, self.metadata['reused_arrays'])):
                for name, a in arrays.items():
                    if not isinstance(a, np.memmap) or a.flags.writeable:
                        raise ValueError('Only own readonly official structured maps may rotate')
                    ref = refs[name+'.npy'] if group == 'a' else refs[name]
                    self.references.append(ref)
                    self.specs[group, name] = dict(path=Path(ref['path']), dtype=a.dtype,
                                                   shape=a.shape, epoch=fingerprint(ref['path']))
            cat = json.loads(verify(self.metadata['activity_catalog']).read_text())
            self.references.extend([reference, self.metadata['launch'], self.metadata['number_dictionary'],
                                    self.metadata['activity_catalog'], cat['official_labels']])
            self.metadata_epochs = {r['path']: fingerprint(r['path']) for r in self.references}
            self.valid = True
            self.check_live()
        except BaseException:
            self.close()
            raise

    def check_live(self):
        if not self.valid:
            raise ValueError('Closed or failed complete structured reader')
        try:
            for path, epoch in self.metadata_epochs.items():
                if fingerprint(path) != epoch:
                    raise ValueError('Complete original structured source epoch drift')
        except BaseException:
            self.valid = False
            raise

    def rotate(self, names):
        for group, name in names:
            values = self.a if group == 'a' else self.reused
            spec = self.specs[group, name]
            values[name] = PagedReboundNames.remap(values[name])
            a = values[name]
            if (a.dtype != spec['dtype'] or a.shape != spec['shape'] or a.flags.writeable
                    or fingerprint(spec['path']) != spec['epoch']):
                raise ValueError('Every unchanged readonly structured field required')
        self.check_live()

    def flags(self, ids, postcode='', insee='', number=''):
        self.check_live()
        try:
            result = super().flags(ids, postcode, insee, number)
            if any(not a.flags.owndata for a in result):
                raise ValueError('Structured results must own storage before source pages close')
            touched = [('reused', k) for k in ('headquarters_codes', 'nomenclature_codes', 'original_activity_codes')]
            for name, text in (('postcode_codes', postcode), ('insee_codes', insee)):
                if encode_code(text) != MISSING:
                    touched.append(('a', name))
            value = normalized(number)
            if value and value in self.number_lookup:
                touched.append(('a', 'number_codes'))
            self.rotate(touched)
            return result
        except BaseException:
            self.valid = False
            raise

    def coordinates(self, ids):
        self.check_live()
        try:
            result = super().coordinates(ids)
            if not result.flags.owndata:
                raise ValueError('XY results must own storage before source pages close')
            self.rotate([('reused', 'site_xy')])
            return result
        except BaseException:
            self.valid = False
            raise

    def verify_after(self):
        try:
            self.check_live()
            for ref in self.references:
                verify(ref)
            self.check_live()
        except BaseException:
            self.valid = False
            raise

    def close(self):
        self.valid = False
        for arrays in (self.a, self.reused):
            for a in arrays.values():
                if isinstance(a, np.memmap):
                    a._mmap.close()
            arrays.clear()
