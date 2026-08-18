from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from src.xgb_matcher.official_source_sync import (
    BodaccSyncConfig,
    DEFAULT_BODACC_ODS_V21_URL,
    KeychainLocator,
    OfficialSyncError,
    RneSyncConfig,
    initialize_keychain_secret,
    read_keychain_secret,
    sync_bodacc,
    sync_rne,
)


PLACEHOLDER_SECRET = b"fixture-password"


class FakeFileTransport:
    def __init__(self, payloads=None, *, failure: Exception | None = None):
        self.payloads = dict(payloads or {})
        self.failure = failure
        self.calls = []

    def download(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.failure:
            raise self.failure
        kwargs["destination"].write_bytes(self.payloads[kwargs["remote_path"]])


class FakeHttpTransport:
    def __init__(self, downloads=None, pages=None):
        self.downloads = dict(downloads or {})
        self.pages = list(pages or [])
        self.download_calls = []
        self.json_calls = []

    def download(self, *, url, destination):
        self.download_calls.append(url)
        destination.write_bytes(self.downloads[url])

    def get_json(self, *, url):
        self.json_calls.append(url)
        return self.pages.pop(0)


class FakeRneApiTransport:
    def __init__(self, pages):
        self.pages = list(pages)
        self.login_calls = []
        self.diff_calls = []
        self.token = bytearray()

    def login(self, **kwargs):
        self.login_calls.append(dict(kwargs))
        self.token = bytearray(b"fixture-bearer")
        return self.token

    def fetch_diff(self, **kwargs):
        self.diff_calls.append(dict(kwargs))
        return self.pages.pop(0)


def _rne_config(tmp_path: Path) -> RneSyncConfig:
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("fixture host key\n", encoding="utf-8")
    return RneSyncConfig.from_dict(
        {
            "keychain": {"service": "fixture.service", "account": "fixture-account"},
            "sftp": {"host": "sftp.invalid", "port": 22, "known_hosts": str(known_hosts)},
            "ftps": {"host": "ftps.invalid", "port": 21},
            "files": [
                {"name": "snapshot.zip", "sftp_path": "/secure/snapshot.zip", "ftps_path": "/tls/snapshot.zip"}
            ],
        }
    )


def _rne_api_config() -> RneSyncConfig:
    return RneSyncConfig.from_dict(
        {
            "keychain": {
                "service": "fixture.service",
                "account": "fixture-account",
            },
            "api": {
                "from": "2026-08-16",
                "to": "2026-08-17",
                "page_size": 2,
            },
        }
    )


def test_keychain_read_uses_no_shell_env_or_secret_argument():
    captured = {}

    def runner(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=PLACEHOLDER_SECRET + b"\n", stderr=b"")

    payload = read_keychain_secret(
        KeychainLocator("fixture.service", "fixture-account"), runner=runner
    )
    assert payload == bytearray(PLACEHOLDER_SECRET)
    assert captured["command"] == [
        "/usr/bin/security", "find-generic-password", "-s", "fixture.service",
        "-a", "fixture-account", "-w",
    ]
    assert captured["kwargs"] == {
        "check": False,
        "capture_output": True,
        "text": False,
        "env": {"PATH": "/usr/bin:/bin"},
    }
    assert PLACEHOLDER_SECRET not in " ".join(captured["command"]).encode()


def test_keychain_initialization_uses_secure_prompt_without_secret_argument():
    captured = {}

    def runner(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    initialize_keychain_secret(
        KeychainLocator("fixture.service", "fixture-account"), runner=runner
    )
    assert captured["command"] == [
        "/usr/bin/security", "add-generic-password", "-U", "-s",
        "fixture.service", "-a", "fixture-account", "-w",
    ]
    assert captured["command"][-1] == "-w"
    assert captured["kwargs"] == {
        "check": False,
        "env": {"PATH": "/usr/bin:/bin"},
    }
    assert PLACEHOLDER_SECRET not in " ".join(captured["command"]).encode()


def test_rne_prefers_sftp_seals_atomically_and_zeroizes_secret(tmp_path: Path):
    config = _rne_config(tmp_path)
    secret = bytearray(PLACEHOLDER_SECRET)
    sftp = FakeFileTransport({"/secure/snapshot.zip": b"rne-fixture"})
    ftps = FakeFileTransport({"/tls/snapshot.zip": b"not-used"})
    output = sync_rne(
        config=config,
        output_root=tmp_path / "store",
        keychain_reader=lambda _locator: secret,
        sftp=sftp,
        ftps=ftps,
    )
    assert (output / "snapshot.zip").read_bytes() == b"rne-fixture"
    assert len(sftp.calls) == 1
    assert ftps.calls == []
    assert secret == bytearray(len(PLACEHOLDER_SECRET))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "rne"
    assert manifest["provenance"]["protocol_by_file"] == {"snapshot.zip": "sftp"}
    assert manifest["provenance"]["credential_material_recorded"] is False
    serialized = json.dumps(manifest)
    assert PLACEHOLDER_SECRET.decode() not in serialized
    assert "fixture-account" not in serialized
    assert not list((tmp_path / "store").glob(".rne-stage-*"))


def test_rne_https_api_diff_uses_bearer_cursor_and_seals_jsonl(tmp_path: Path):
    secret = bytearray(PLACEHOLDER_SECRET)
    api = FakeRneApiTransport(
        [
            ([{"siren": "123456789"}, {"siren": "987654321"}], "987654321"),
            ([{"siren": "111222333"}], ""),
        ]
    )
    output = sync_rne(
        config=_rne_api_config(),
        output_root=tmp_path / "store",
        keychain_reader=lambda _locator: secret,
        rne_api=api,
    )
    rows = [
        json.loads(line)
        for line in (output / "rne-formalites-diff.jsonl").read_text().splitlines()
    ]
    assert [row["siren"] for row in rows] == [
        "123456789",
        "987654321",
        "111222333",
    ]
    assert len(api.login_calls) == 1
    assert [call["search_after"] for call in api.diff_calls] == ["", "987654321"]
    assert api.diff_calls[0]["from_date"] == "2026-08-16"
    assert api.diff_calls[0]["to_date"] == "2026-08-17"
    assert secret == bytearray(len(PLACEHOLDER_SECRET))
    assert api.token == bytearray(len(b"fixture-bearer"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    provenance = manifest["provenance"]
    assert provenance["protocol_by_file"] == {
        "rne-formalites-diff.jsonl": "https-rne-formalites-v4-diff"
    }
    assert provenance["records"] == 3
    assert provenance["from_exclusive"] == "2026-08-16"
    assert provenance["to_inclusive"] == "2026-08-17"
    assert "fixture-account" not in json.dumps(manifest)


def test_rne_api_fails_closed_on_unapproved_host_or_missing_cursor(tmp_path: Path):
    with pytest.raises(OfficialSyncError, match="allow-listed"):
        RneSyncConfig.from_dict(
            {
                "keychain": {"service": "fixture", "account": "fixture"},
                "api": {
                    "login_url": "https://example.invalid/api/sso/login",
                    "from": "2026-08-16",
                    "to": "2026-08-17",
                },
            }
        )
    api = FakeRneApiTransport(
        [([{"siren": "123456789"}, {"siren": "987654321"}], "")]
    )
    with pytest.raises(OfficialSyncError, match="without pagination-search-after"):
        sync_rne(
            config=_rne_api_config(),
            output_root=tmp_path / "store",
            keychain_reader=lambda _locator: bytearray(PLACEHOLDER_SECRET),
            rne_api=api,
        )
    assert not (tmp_path / "store" / "rne").exists()


def test_rne_falls_back_to_ftps_and_is_content_addressed(tmp_path: Path):
    config = _rne_config(tmp_path)
    sftp = FakeFileTransport(failure=OfficialSyncError("fixture unavailable"))
    ftps = FakeFileTransport({"/tls/snapshot.zip": b"same-payload"})

    def keychain(_locator):
        return bytearray(PLACEHOLDER_SECRET)

    first = sync_rne(config=config, output_root=tmp_path / "store", keychain_reader=keychain, sftp=sftp, ftps=ftps)
    second = sync_rne(config=config, output_root=tmp_path / "store", keychain_reader=keychain, sftp=sftp, ftps=ftps)
    assert first == second
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["protocol_by_file"] == {"snapshot.zip": "ftps"}


def test_rne_plain_ftp_and_inline_secrets_are_rejected_before_transfer(tmp_path: Path):
    with pytest.raises(OfficialSyncError, match="unencrypted FTP"):
        RneSyncConfig.from_dict(
            {
                "keychain": {"service": "fixture", "account": "fixture"},
                "sftp": {"host": "ftp://plain.invalid", "known_hosts": str(tmp_path / "kh")},
                "ftps": {"host": "ftps.invalid"},
                "files": [{"name": "x", "sftp_path": "/x", "ftps_path": "/x"}],
            }
        )
    with pytest.raises(OfficialSyncError, match="inline credential"):
        RneSyncConfig.from_dict({"password": "prohibited-fixture"})


def test_rne_insecure_ftps_diagnostic_is_preserved_and_no_partial_publish(tmp_path: Path):
    config = _rne_config(tmp_path)
    sftp = FakeFileTransport(failure=OfficialSyncError("fixture unavailable"))
    ftps = FakeFileTransport(
        failure=OfficialSyncError(
            "INSECURE_FTP_UNSUPPORTED: server does not offer AUTH TLS; credentials not sent"
        )
    )
    with pytest.raises(OfficialSyncError, match="INSECURE_FTP_UNSUPPORTED"):
        sync_rne(
            config=config,
            output_root=tmp_path / "store",
            keychain_reader=lambda _locator: bytearray(PLACEHOLDER_SECRET),
            sftp=sftp,
            ftps=ftps,
        )
    assert not (tmp_path / "store" / "rne").exists()
    assert not list((tmp_path / "store").glob(".rne-stage-*"))


def test_bodacc_refuses_http_and_plain_ftp():
    for url in ["http://plain.invalid/archive.zip", "ftp://plain.invalid/archive.zip"]:
        with pytest.raises(OfficialSyncError, match="prohibited"):
            BodaccSyncConfig.from_dict(
                {"backfill": [{"name": "archive.zip", "url": url}]}
            )


def test_bodacc_https_backfill_and_opendatasoft_v21_increment_are_sealed(tmp_path: Path):
    archive_url = "https://official.invalid/archive.zip"
    config = BodaccSyncConfig.from_dict(
        {
            "backfill": [{"name": "archive.zip", "url": archive_url}],
            "incremental": {
                "watermark_field": "dateparution",
                "watermark_type": "date",
                "since": "2026-08-01",
                "page_size": 2,
            },
        }
    )
    assert config.incremental is not None
    assert config.incremental.url == DEFAULT_BODACC_ODS_V21_URL
    http = FakeHttpTransport(
        downloads={archive_url: b"bodacc-backfill"},
        pages=[
            {"results": [{"id": "a", "dateparution": "2026-08-02"}, {"id": "b", "dateparution": "2026-08-03"}]},
            {"results": [{"id": "c", "dateparution": "2026-08-04"}]},
        ],
    )
    output = sync_bodacc(config=config, output_root=tmp_path / "store", https=http)
    assert (output / "archive.zip").read_bytes() == b"bodacc-backfill"
    rows = [json.loads(line) for line in (output / "incremental.jsonl").read_text().splitlines()]
    assert [row["id"] for row in rows] == ["a", "b", "c"]
    assert len(http.json_calls) == 2
    first_query = parse_qs(urlparse(http.json_calls[0]).query)
    second_query = parse_qs(urlparse(http.json_calls[1]).query)
    assert first_query["where"] == ["dateparution > date'2026-08-01'"]
    assert first_query["order_by"] == ["dateparution,id"]
    assert second_query["where"] == [
        "dateparution > date'2026-08-03' OR "
        "(dateparution = date'2026-08-03' AND id > 'b')"
    ]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "bodacc"
    assert manifest["provenance"]["incremental_rows"] == 3
    assert manifest["provenance"]["next_watermark"] == "2026-08-04"
    assert manifest["provenance"]["next_tie_break"] == "c"
    assert manifest["provenance"]["plain_ftp_allowed"] is False
    assert not list((tmp_path / "store").glob(".bodacc-stage-*"))


def test_bodacc_partitioned_integer_cursor_advances_naturally(tmp_path: Path):
    config = BodaccSyncConfig.from_dict(
        {
            "incremental": {
                "watermark_field": "dateparution",
                "watermark_type": "date",
                "since": "2026-08-17",
                "tie_break_field": "numeroannonce",
                "tie_break_type": "integer",
                "partition_where": "publicationavis = 'C' AND parution = '20260156'",
                "page_size": 2,
            }
        }
    )
    http = FakeHttpTransport(
        pages=[
            {
                "results": [
                    {"id": "C9", "dateparution": "2026-08-18", "numeroannonce": 9},
                    {"id": "C10", "dateparution": "2026-08-18", "numeroannonce": 10},
                ]
            },
            {
                "results": [
                    {"id": "C11", "dateparution": "2026-08-18", "numeroannonce": 11}
                ]
            },
        ]
    )
    output = sync_bodacc(config=config, output_root=tmp_path / "store", https=http)
    rows = [json.loads(line) for line in (output / "incremental.jsonl").read_text().splitlines()]
    assert [row["numeroannonce"] for row in rows] == [9, 10, 11]
    second_query = parse_qs(urlparse(http.json_calls[1]).query)
    assert "numeroannonce > 10" in second_query["where"][0]
    assert "dateparution = date'2026-08-18'" in second_query["where"][0]
    assert "publicationavis = 'C'" in second_query["where"][0]


def test_bodacc_failure_leaves_no_published_or_partial_artifact(tmp_path: Path):
    config = BodaccSyncConfig.from_dict(
        {"backfill": [{"name": "archive.zip", "url": "https://official.invalid/missing.zip"}]}
    )
    http = FakeHttpTransport(downloads={})
    with pytest.raises(KeyError):
        sync_bodacc(config=config, output_root=tmp_path / "store", https=http)
    assert not (tmp_path / "store" / "bodacc").exists()
    assert not list((tmp_path / "store").glob(".bodacc-stage-*"))


def test_bodacc_incremental_fails_closed_when_results_are_missing(tmp_path: Path):
    config = BodaccSyncConfig.from_dict(
        {"incremental": {"watermark_field": "dateparution", "since": "2026-08-01"}}
    )
    http = FakeHttpTransport(pages=[{"error": "fixture malformed response"}])
    with pytest.raises(OfficialSyncError, match="no results array"):
        sync_bodacc(config=config, output_root=tmp_path / "store", https=http)
    assert not (tmp_path / "store" / "bodacc").exists()
    assert not list((tmp_path / "store").glob(".bodacc-stage-*"))
