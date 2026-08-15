from scripts.select_synthetic_gt_sirene_seeds import choose_one_per_siren


def row(siret, siren, state="A", name="Entreprise"):
    return {"siret": siret, "siren": siren, "state": state, "name": name, "enseigne": "", "street": "1 RUE TEST", "postcode": "75001", "city": "PARIS"}


def test_excludes_crm_sirens_and_selects_one_siret_per_siren():
    rows = [
        row("12345678900011", "123456789"),
        row("12345678900022", "123456789", state="F"),
        row("22222222200011", "222222222"),
        row("33333333300011", "333333333"),
    ]
    selected = choose_one_per_siren(rows, {"123456789"}, seed=42, limit=10)
    assert {item["siren"] for item in selected} == {"222222222", "333333333"}
    assert len(selected) == 2


def test_selection_is_stable_and_bounded():
    rows = [row(f"{i:09d}00011", f"{i:09d}") for i in range(100000001, 100000006)]
    first = choose_one_per_siren(rows, set(), seed=42, limit=3)
    second = choose_one_per_siren(rows, set(), seed=42, limit=3)
    assert first == second
    assert len(first) == 3
