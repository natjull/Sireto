import unittest
from scripts import perplexity_late_probe as probe


class ProbeTests(unittest.TestCase):
    def test_query_excludes_identifiers_and_extra_fields(self):
        row = dict(fold=2, crm_name='COMPANY 123456789', crm_number='8', crm_suffix='',
            crm_street='RUE TEST', crm_postcode='75001', truth='98765432100001')
        self.assertNotIn('123456789', probe.query_text(row))
        self.assertNotIn(row['truth'], probe.query_text(row))
        row['fold'] = 1
        with self.assertRaises(ValueError):
            probe.query_text(row)

    def test_only_official_own_and_parent_name_fields(self):
        doc = dict(key='12345678900001', siren='123456789', own_names=['ENSEIGNE'],
            all_addresses=['8 RUE TEST'], postcode='75001',
            extra=dict(legal=['RAISON SOCIALE'], own_site=['ANCIEN NOM'], siblings=['AUTRE SITE']))
        text = probe.official_text(doc)
        for expected in ('ENSEIGNE', 'RAISON SOCIALE', 'ANCIEN NOM', '8 RUE TEST'):
            self.assertIn(expected, text)
        for excluded in ('AUTRE SITE', doc['key'], doc['siren']):
            self.assertNotIn(excluded, text)
