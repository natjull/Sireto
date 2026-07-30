from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "independent_audit_v412_unit_retrieval",
    ROOT / "scripts/independent_audit_v412_unit_retrieval.py",
)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


@pytest.fixture(scope="module")
def report() -> dict:
    return subject.audit_repository(ROOT)


def test_full_independent_audit_is_synthetic_only_and_go(report: dict) -> None:
    assert report["verdict"] == subject.GO
    assert report["forbidden_runtime_inputs_opened"] is False
    assert set(report["static_checks"]) == subject.STATIC_CHECKS
    assert all(report["static_checks"].values())
    assert set(report["synthetic"]["checks"]) == subject.SYNTHETIC_CHECKS
    assert all(report["synthetic"]["checks"].values())
    assert report["synthetic"]["trace"] == ["worker_final", "parity_started"]


def test_audit_source_hashes_are_closed_to_code_contract_and_profiles(
    report: dict,
) -> None:
    plan = json.loads((ROOT / subject.PLAN).read_text())
    assert set(report["source_hashes"]) == subject._source_closure(plan)
    payload = json.dumps(report, sort_keys=True)
    assert "queries_dev.parquet" not in payload
    assert "/oracles/" not in payload
    assert "/datasets/" not in payload
    assert "/models/" not in payload


def test_audit_never_opens_real_runtime_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = os.open
    forbidden_opened: list[str] = []
    allowed_test_root = str(tmp_path)

    def guarded(path: object, *args: object, **kwargs: object) -> int:
        text = os.fspath(path) if isinstance(path, (str, os.PathLike)) else ""
        if (
            (
                "/Volumes/CATNAT_DATA/" in text
                and not (
                    text == allowed_test_root
                    or text.startswith(allowed_test_root + os.sep)
                )
            )
            or text.startswith(str(ROOT / "models"))
        ):
            forbidden_opened.append(text)
            raise AssertionError(f"forbidden runtime artifact opened: {text}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded)
    monkeypatch.setattr(subject.tempfile, "tempdir", allowed_test_root)
    result = subject.audit_repository(ROOT)
    assert result["verdict"] == subject.GO
    assert forbidden_opened == []


def test_secure_source_reader_rejects_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real.py"
    real.write_text("VALUE = 1\n")
    (tmp_path / "link.py").symlink_to(real)
    with pytest.raises(subject.IndependentAuditStopped, match="cannot open"):
        subject._secure_read(tmp_path, "link.py")


def test_secure_source_reader_rejects_symlink_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "source.py").write_text("VALUE = 1\n")
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    with pytest.raises(subject.IndependentAuditStopped, match="cannot open"):
        subject._secure_read(tmp_path, "linked/source.py")


def test_secure_source_reader_anchors_ancestor_during_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor = tmp_path / "ancestor"
    ancestor.mkdir()
    (ancestor / "source.py").write_bytes(b"ORIGINAL")
    displaced = tmp_path / "displaced"
    original_open = os.open
    substituted = False

    def racing_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal substituted
        if path == "source.py" and dir_fd is not None and not substituted:
            substituted = True
            os.rename(ancestor, displaced)
            ancestor.mkdir()
            (ancestor / "source.py").write_bytes(b"SUBSTITUTED")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_open)
    assert subject._secure_read(tmp_path, "ancestor/source.py") == b"ORIGINAL"
    assert substituted is True
    assert (ancestor / "source.py").read_bytes() == b"SUBSTITUTED"


def test_worker_ast_rejects_historical_or_model_imports() -> None:
    source = b"""
from xgb_matcher import retrieval
_RETRIEVAL_POLICY = {}
_TFIDF_POLICY = {}
class UnitRetrievalResult: pass
def route_query(): pass
def retrieve_unit_query(): pass
def run_child_worker(): pass
def main(): pass
"""
    with pytest.raises(subject.IndependentAuditStopped, match="forbidden"):
        subject._audit_worker_ast(source)


def test_json_and_source_closure_are_fail_closed() -> None:
    with pytest.raises(subject.IndependentAuditStopped, match="duplicate"):
        subject._parse_json(b'{"a":1,"a":2}', "fixture")
    with pytest.raises(subject.IndependentAuditStopped, match="non-finite"):
        subject._parse_json(b'{"a":NaN}', "fixture")
    plan = {
        "parent_sources": ["a"],
        "worker_sources": ["a"],
        "parity_sources": [],
        "test_sources": [],
        "independent_audit_sources": [subject.SELF, subject.SELF_TEST],
    }
    with pytest.raises(subject.IndependentAuditStopped, match="duplicates"):
        subject._source_closure(plan)


def test_worker_before_parity_static_order_is_fail_closed() -> None:
    with pytest.raises(subject.IndependentAuditStopped, match="required order"):
        subject._source_order_check(
            "_validate_outputs(\nresult = subprocess.run(\n"
            "if result.returncode != 0:\n"
            "_promote(output, pending)\n"
            "_promote(pending, final_runtime)\n",
            'WORKER_VERDICT = "SEALED_V412_UNIT_RETRIEVAL"\n'
            "_validate_worker_manifest(\n_validate_worker_integrity(\n"
            "sealed input changed during parity evaluation\n",
        )


def test_synthetic_security_probes_cover_mutation_symlink_and_toctou(
    report: dict,
) -> None:
    checks = report["synthetic"]["checks"]
    assert checks["mutation_rejected"] is True
    assert checks["symlink_rejected"] is True
    assert checks["anchored_fd_survives_path_substitution"] is True
    assert checks["worker_publication"] is True
    assert checks["parity_publication"] is True


def test_cli_returns_a_single_json_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {"verdict": subject.GO}
    monkeypatch.setattr(subject, "audit_repository", lambda _repo: expected)
    assert subject.main(["--repo", str(ROOT)]) == 0
    assert json.loads(capsys.readouterr().out) == expected
