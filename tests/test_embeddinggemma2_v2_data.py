import unittest
from scripts.embeddinggemma2_v2_data import safe,pack_atoms,eligible_other_site


class WordTokenizer:
    def __call__(self,text,add_special_tokens=True):
        return {'input_ids':text.split()+(['END'] if add_special_tokens else [])}
    def decode(self,tokens,skip_special_tokens=True): return ' '.join(tokens)


class DataSafety(unittest.TestCase):
    def test_unicode_identifiers_and_masks_not_features(self):
        self.assertEqual(safe('ＮＤ'),'')
        self.assertEqual(safe('nom １２３４５６７８９'),'nom')
        self.assertEqual(safe('nom 12345678900012 75000'),'nom 75000')

    def test_every_atom_survives_views_not_truncated(self):
        atoms=[f'enseigne numero{i}' for i in range(70)]
        views,over=pack_atoms(atoms,WordTokenizer(),48,'adresse fixe')
        for atom in atoms: self.assertTrue(any(atom in v for v in views),atom)
        self.assertGreater(len(views),1)
        self.assertEqual(over,0)
        self.assertTrue(all(len(WordTokenizer()('title: none | text: '+v)['input_ids'])<=48 for v in views))

    def test_oversized_atom_keeps_all_tokens(self):
        words=[f'word{i}' for i in range(130)]
        views,over=pack_atoms([' '.join(words)],WordTokenizer(),48)
        self.assertEqual(over,1)
        for word in words: self.assertIn(word,' '.join(views).split())

    def test_equivalent_uncertain_or_wrong_label_never_site_negative(self):
        q=dict(crm_number='4',crm_suffix='',crm_street='RUE DE PARIS',crm_insee='75056')
        p=dict(siren='a',key='p',insee='75056',all_addresses=['4 RUE DE PARIS'])
        n=dict(siren='a',key='n',insee='75056',all_addresses=['6 RUE DE PARIS'])
        self.assertTrue(eligible_other_site(q,p,n))
        self.assertFalse(eligible_other_site(q,p,{**n,'all_addresses':['4 RUE DE PARIS']}))
        self.assertFalse(eligible_other_site(q,p,{**n,'all_addresses':['ZONE INDUSTRIELLE']}))
        self.assertFalse(eligible_other_site(q,p,{**n,'all_addresses':['6 RUE DE PARIS','4 RUE DE PARIS']}))
        self.assertFalse(eligible_other_site({**q,'crm_number':'99'},p,n))
