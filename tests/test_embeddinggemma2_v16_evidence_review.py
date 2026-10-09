import unittest
from scripts.embeddinggemma2_v16_evidence_review import distinctive_address_overlap


class EvidenceTests(unittest.TestCase):
    def test_street_type_is_not_distinctive_evidence(self):
        self.assertFalse(distinctive_address_overlap('RUE DE LA MARNE',['8 RUE DE LA CHAPELLE']))

    def test_distinctive_street_word_retained(self):
        self.assertTrue(distinctive_address_overlap('RUE DE GARDES',['ZAC DE GARDES']))


if __name__=='__main__': unittest.main()
