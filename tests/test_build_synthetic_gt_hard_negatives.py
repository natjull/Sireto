import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "hard_negatives", Path(__file__).parents[1] / "scripts" / "build_synthetic_gt_hard_negatives.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(siret, siren, state="A", insee="12345", address="1 RUE TEST", names=None):
    return {
        "siret": siret, "siren": siren, "state": state, "insee": insee,
        "postcode": "75001", "address_signature": address + " 75001 12345",
        "names": names or [], "legal_denomination": "", "denomination_usuelle": "",
    }


def test_selects_families_without_touching_text():
    seed = candidate("12345678900011", "123456789", names=["ALPHA"])
    cards = [{"siret": seed["siret"], "candidates": [
        seed,
        candidate("12345678900022", "123456789", insee="99999", address="2 RUE AUTRE"),
        candidate("22222222200011", "222222222", state="F", address="3 RUE AUTRE"),
        candidate("33333333300011", "333333333", names=["ALPHA"], address="4 RUE AUTRE"),
    ]}]
    accepts = [{"target_siret": seed["siret"], "variant_id": "v1"}]
    pairs = MODULE.build(accepts, cards, per_positive=10)
    assert {pair["family"] for pair in pairs} == {
        "SAME_SIREN_OTHER_SIRET", "ACTIVE_CLOSED", "LOCAL_HOMONYM"
    }
    assert all("crm" not in pair for pair in pairs)
    assert len({pair["negative_siret"] for pair in pairs}) == len(pairs)


def test_cap_and_stable_output():
    seed = candidate("12345678900011", "123456789")
    cards = [{"siret": seed["siret"], "candidates": [seed, candidate("12345678900022", "123456789")]}]
    accepts = [{"target_siret": seed["siret"], "variant_id": "v1"}]
    first = MODULE.build(accepts, cards, per_positive=1)
    second = MODULE.build(accepts, cards, per_positive=1)
    assert first == second
    assert len(first) == 1
