"""Secure, auditable acquisition of official RNE and BODACC sources.

The only built-in endpoint is the public BODACC Opendatasoft v2.1 records API;
transfer hosts, accounts and file locations remain explicit configuration.
Production passwords are read from a macOS Keychain generic-password item.
Tests inject transport fixtures and never access the network or the Keychain.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import ftplib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import ssl
import subprocess
import tempfile
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


class OfficialSyncError(RuntimeError):
    """Safe acquisition error; messages never include credential material."""


DEFAULT_BODACC_ODS_V21_URL = (
    "https://bodacc-datadila.opendatasoft.com/api/explore/v2.1/catalog/"
    "datasets/annonces-commerciales/records"
)
DEFAULT_RNE_LOGIN_URL = "https://registre-national-entreprises.inpi.fr/api/sso/login"
DEFAULT_RNE_DIFF_URL = "https://registre-national-entreprises.inpi.fr/api/companies/diff"
_ALLOWED_RNE_API_HOSTS = {
    "registre-national-entreprises.inpi.fr",
    "registre-national-entreprises-pprod.inpi.fr",
}


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_filename(name: str) -> str:
    if not name or name in {".", ".."} or Path(name).name != name:
        raise OfficialSyncError("output filename must be a single safe path component")
    return name


def _validate_public_url(url: str, *, schemes: set[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() in {"ftp", "http"}:
        raise OfficialSyncError(
            f"unencrypted {parsed.scheme.upper()} is prohibited; use HTTPS, SFTP or FTPS"
        )
    if parsed.scheme.lower() not in schemes or not parsed.hostname:
        raise OfficialSyncError(f"unsupported official-source URL scheme: {parsed.scheme or 'missing'}")
    if parsed.username or parsed.password:
        raise OfficialSyncError("credentials in URLs are prohibited; use macOS Keychain")


def _validate_host(host: str) -> None:
    if not host or "://" in host or "/" in host or "@" in host:
        if str(host).lower().startswith(("ftp:", "http:")):
            raise OfficialSyncError(
                "unencrypted FTP/HTTP is prohibited; configure an SFTP or FTPS host"
            )
        raise OfficialSyncError("transfer host must be a bare hostname")


def _reject_inline_secrets(raw: Mapping[str, Any]) -> None:
    forbidden = {"password", "secret", "token", "private_key", "api_key"}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).lower() in forbidden:
                    raise OfficialSyncError(
                        f"inline credential field '{key}' is prohibited; use macOS Keychain"
                    )
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(raw)


@dataclass(frozen=True)
class KeychainLocator:
    service: str
    account: str

    def __post_init__(self) -> None:
        if not self.service or not self.account:
            raise OfficialSyncError("Keychain service and account are required")


@dataclass(frozen=True)
class Credentials:
    username: str
    password: str

    @classmethod
    def from_keychain_payload(
        cls, payload: bytearray, *, username: str
    ) -> "Credentials":
        try:
            password = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OfficialSyncError(
                "Keychain password must be valid UTF-8"
            ) from exc
        if not username or not password:
            raise OfficialSyncError("Keychain username/password must be non-empty")
        return cls(username=username, password=password)


KeychainRunner = Callable[..., Any]


def read_keychain_secret(
    locator: KeychainLocator,
    *,
    runner: KeychainRunner = subprocess.run,
) -> bytearray:
    """Read a generic-password item without shell, env or command-line secret."""
    command = [
        "/usr/bin/security",
        "find-generic-password",
        "-s",
        locator.service,
        "-a",
        locator.account,
        "-w",
    ]
    completed = runner(
        command,
        check=False,
        capture_output=True,
        text=False,
        env={"PATH": "/usr/bin:/bin"},
    )
    if int(completed.returncode) != 0:
        raise OfficialSyncError("macOS Keychain item unavailable or access denied")
    payload = bytearray(bytes(completed.stdout).rstrip(b"\r\n"))
    if not payload:
        raise OfficialSyncError("macOS Keychain item is empty")
    return payload


def initialize_keychain_secret(
    locator: KeychainLocator,
    *,
    runner: KeychainRunner = subprocess.run,
) -> None:
    """Create/update the password through macOS' secure interactive prompt.

    ``-w`` is deliberately the final argument with no value: no password ever
    appears in argv, an environment variable, stdin managed by this process, or
    captured output.
    """
    command = [
        "/usr/bin/security",
        "add-generic-password",
        "-U",
        "-s",
        locator.service,
        "-a",
        locator.account,
        "-w",
    ]
    completed = runner(command, check=False, env={"PATH": "/usr/bin:/bin"})
    if int(completed.returncode) != 0:
        raise OfficialSyncError("macOS Keychain password initialization failed")


class FileTransport(Protocol):
    def download(
        self,
        *,
        host: str,
        port: int,
        remote_path: str,
        destination: Path,
        credentials: Credentials | None,
        known_hosts: Path | None = None,
    ) -> None: ...


class SftpTransport:
    def download(
        self,
        *,
        host: str,
        port: int,
        remote_path: str,
        destination: Path,
        credentials: Credentials | None,
        known_hosts: Path | None = None,
    ) -> None:
        if credentials is None:
            raise OfficialSyncError("SFTP credentials are required")
        if known_hosts is None or not known_hosts.is_file():
            raise OfficialSyncError("SFTP known_hosts file is required for host verification")
        try:
            import paramiko  # type: ignore
        except ImportError as exc:
            raise OfficialSyncError("SFTP requires the optional 'paramiko' package") from exc
        client = paramiko.SSHClient()
        client.load_host_keys(str(known_hosts))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            client.connect(
                hostname=host,
                port=port,
                username=credentials.username,
                password=credentials.password,
                look_for_keys=False,
                allow_agent=False,
            )
            sftp = client.open_sftp()
            try:
                with destination.open("wb") as output:
                    sftp.getfo(remote_path, output)
            finally:
                sftp.close()
        except OfficialSyncError:
            raise
        except Exception as exc:
            raise OfficialSyncError("SFTP transfer failed") from exc
        finally:
            client.close()


class FtpsTransport:
    def download(
        self,
        *,
        host: str,
        port: int,
        remote_path: str,
        destination: Path,
        credentials: Credentials | None,
        known_hosts: Path | None = None,
    ) -> None:
        del known_hosts
        context = ssl.create_default_context()
        client = ftplib.FTP_TLS(context=context, timeout=120)
        try:
            client.connect(host=host, port=port)
            try:
                client.auth()
            except Exception as exc:
                # Authentication has not happened yet: credentials were never sent.
                raise OfficialSyncError(
                    "INSECURE_FTP_UNSUPPORTED: server does not offer AUTH TLS; credentials not sent"
                ) from exc
            if credentials is None:
                client.login()
            else:
                client.login(user=credentials.username, passwd=credentials.password)
            client.prot_p()
            with destination.open("wb") as output:
                client.retrbinary(f"RETR {remote_path}", output.write, blocksize=1024 * 1024)
        except OfficialSyncError:
            raise
        except Exception as exc:
            raise OfficialSyncError("FTPS transfer failed") from exc
        finally:
            try:
                client.quit()
            except Exception:
                client.close()


class HttpsTransport:
    def download(self, *, url: str, destination: Path) -> None:
        _validate_public_url(url, schemes={"https"})
        request = Request(url, headers={"User-Agent": "SIRETO-official-source-sync/1"})
        try:
            with urlopen(request, timeout=120, context=ssl.create_default_context()) as response:
                with destination.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
        except Exception as exc:
            raise OfficialSyncError("HTTPS transfer failed") from exc

    def get_json(self, *, url: str) -> Mapping[str, Any]:
        _validate_public_url(url, schemes={"https"})
        request = Request(url, headers={"User-Agent": "SIRETO-official-source-sync/1"})
        try:
            with urlopen(request, timeout=120, context=ssl.create_default_context()) as response:
                value = json.load(response)
        except Exception as exc:
            raise OfficialSyncError("HTTPS JSON request failed") from exc
        if not isinstance(value, Mapping):
            raise OfficialSyncError("official HTTPS endpoint returned a non-object JSON value")
        return value


class RneApiTransport(Protocol):
    def login(self, *, url: str, credentials: Credentials) -> bytearray: ...

    def fetch_diff(
        self,
        *,
        url: str,
        token: bytearray,
        from_date: str,
        to_date: str,
        page_size: int,
        search_after: str,
    ) -> tuple[Sequence[Mapping[str, Any]], str]: ...


def _validate_rne_api_url(url: str) -> None:
    _validate_public_url(url, schemes={"https"})
    if (urlparse(url).hostname or "").lower() not in _ALLOWED_RNE_API_HOSTS:
        raise OfficialSyncError("RNE API host is not an allow-listed official INPI host")


class RneHttpsApiTransport:
    """Official INPI Formalites v4 HTTPS API transport."""

    user_agent = "SIRETO-official-source-sync/1"

    def login(self, *, url: str, credentials: Credentials) -> bytearray:
        _validate_rne_api_url(url)
        body = json.dumps(
            {"username": credentials.username, "password": credentials.password},
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urlopen(
                request, timeout=120, context=ssl.create_default_context()
            ) as response:
                payload = json.load(response)
        except Exception as exc:
            raise OfficialSyncError("RNE HTTPS API authentication failed") from exc
        if not isinstance(payload, Mapping):
            raise OfficialSyncError("RNE HTTPS API login returned malformed JSON")
        token = str(payload.get("token") or "")
        if not token:
            raise OfficialSyncError("RNE HTTPS API login returned no bearer token")
        return bytearray(token.encode("utf-8"))

    def fetch_diff(
        self,
        *,
        url: str,
        token: bytearray,
        from_date: str,
        to_date: str,
        page_size: int,
        search_after: str,
    ) -> tuple[Sequence[Mapping[str, Any]], str]:
        _validate_rne_api_url(url)
        parsed = urlparse(url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query.update({"pageSize": str(page_size), "from": from_date, "to": to_date})
        if search_after:
            query["searchAfter"] = search_after
        request_url = urlunparse(parsed._replace(query=urlencode(query)))
        request = Request(
            request_url,
            headers={
                "Authorization": f"Bearer {bytes(token).decode('utf-8')}",
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urlopen(
                request, timeout=120, context=ssl.create_default_context()
            ) as response:
                payload = json.load(response)
                next_search_after = str(
                    response.headers.get("pagination-search-after") or ""
                ).strip()
        except Exception as exc:
            raise OfficialSyncError("RNE HTTPS API differential request failed") from exc
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, Mapping):
            records = next(
                (
                    payload[key]
                    for key in ("results", "companies", "formalities", "items", "data")
                    if isinstance(payload.get(key), list)
                ),
                None,
            )
            if records is None and "content" in payload:
                records = [payload]
        else:
            records = None
        if records is None or any(not isinstance(item, Mapping) for item in records):
            raise OfficialSyncError("RNE HTTPS API differential response is malformed")
        return tuple(records), next_search_after


def _fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_content_addressed(
    *,
    stage: Path,
    output_root: Path,
    source: str,
    payload_metadata: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> Path:
    """Seal and atomically promote a same-filesystem staged payload tree."""
    identity = {
        "schema_version": "sireto-official-source-manifest-v1",
        "source": source,
        "payload": list(payload_metadata),
        "provenance": dict(provenance),
    }
    build_id = hashlib.sha256(canonical_json(identity)).hexdigest()
    manifest = {**identity, "build_id": build_id, "atomic_promotion": True}
    manifest_path = stage / "manifest.json"
    manifest_path.write_bytes(canonical_json(manifest))
    os.chmod(manifest_path, 0o600)
    for path in stage.iterdir():
        if path.is_file():
            _fsync_path(path)
    _fsync_path(stage)

    source_root = output_root / source
    source_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    final = source_root / build_id[:16]
    if final.exists():
        existing = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("build_id") != build_id:
            raise OfficialSyncError("content-address collision in official source store")
        shutil.rmtree(stage)
        return final
    try:
        os.rename(stage, final)
    except FileExistsError:
        shutil.rmtree(stage)
        return final
    _fsync_path(source_root)
    return final


def _payload_metadata(stage: Path, names: Iterable[str]) -> list[dict[str, Any]]:
    values = []
    for name in sorted(names):
        path = stage / name
        values.append(
            {"name": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return values


@dataclass(frozen=True)
class RneFile:
    name: str
    sftp_path: str
    ftps_path: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RneFile":
        return cls(
            name=_safe_filename(str(raw.get("name") or "")),
            sftp_path=str(raw.get("sftp_path") or ""),
            ftps_path=str(raw.get("ftps_path") or ""),
        )


@dataclass(frozen=True)
class RneApiConfig:
    login_url: str
    diff_url: str
    from_date: str
    to_date: str
    output_name: str = "rne-formalites-diff.jsonl"
    page_size: int = 100
    maximum_pages: int = 1_000_000

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RneApiConfig":
        config = cls(
            login_url=str(raw.get("login_url") or DEFAULT_RNE_LOGIN_URL),
            diff_url=str(raw.get("diff_url") or DEFAULT_RNE_DIFF_URL),
            from_date=str(raw.get("from") or ""),
            to_date=str(raw.get("to") or ""),
            output_name=_safe_filename(
                str(raw.get("output_name") or "rne-formalites-diff.jsonl")
            ),
            page_size=int(raw.get("page_size", 100)),
            maximum_pages=int(raw.get("maximum_pages", 1_000_000)),
        )
        _validate_rne_api_url(config.login_url)
        _validate_rne_api_url(config.diff_url)
        if not config.from_date or not config.to_date:
            raise OfficialSyncError("RNE API differential from/to dates are required")
        try:
            from_day = date.fromisoformat(config.from_date)
            to_day = date.fromisoformat(config.to_date)
        except ValueError as exc:
            raise OfficialSyncError("RNE API differential dates must use YYYY-MM-DD") from exc
        if from_day >= to_day:
            raise OfficialSyncError("RNE API differential from must precede to")
        if not 1 <= config.page_size <= 100:
            raise OfficialSyncError("RNE API page_size must be between 1 and 100")
        if config.maximum_pages < 1:
            raise OfficialSyncError("RNE API maximum_pages must be positive")
        return config


@dataclass(frozen=True)
class RneSyncConfig:
    keychain: KeychainLocator
    api: RneApiConfig | None = None
    sftp_host: str = ""
    sftp_port: int = 22
    known_hosts: Path = Path()
    ftps_host: str = ""
    ftps_port: int = 21
    files: tuple[RneFile, ...] = ()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RneSyncConfig":
        _reject_inline_secrets(raw)
        keychain, sftp, ftps = raw.get("keychain", {}), raw.get("sftp", {}), raw.get("ftps", {})
        api_raw = raw.get("api")
        config = cls(
            keychain=KeychainLocator(str(keychain.get("service") or ""), str(keychain.get("account") or "")),
            api=(RneApiConfig.from_dict(api_raw) if isinstance(api_raw, Mapping) else None),
            sftp_host=str(sftp.get("host") or ""),
            sftp_port=int(sftp.get("port", 22)),
            known_hosts=Path(str(sftp.get("known_hosts") or "")),
            ftps_host=str(ftps.get("host") or ""),
            ftps_port=int(ftps.get("port", 21)),
            files=tuple(RneFile.from_dict(item) for item in raw.get("files", [])),
        )
        if config.api is None:
            _validate_host(config.sftp_host)
            _validate_host(config.ftps_host)
        if config.api is None and (
            not config.files
            or any(not item.sftp_path or not item.ftps_path for item in config.files)
        ):
            raise OfficialSyncError(
                "RNE API config or file list with both secure remote paths is required"
            )
        if len({item.name for item in config.files}) != len(config.files):
            raise OfficialSyncError("RNE output filenames must be unique")
        return config


def sync_rne(
    *,
    config: RneSyncConfig,
    output_root: Path,
    keychain_reader: Callable[[KeychainLocator], bytearray] = read_keychain_secret,
    sftp: FileTransport | None = None,
    ftps: FileTransport | None = None,
    rne_api: RneApiTransport | None = None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage = Path(tempfile.mkdtemp(prefix=".rne-stage-", dir=output_root))
    os.chmod(stage, 0o700)
    secret = bytearray()
    bearer = bytearray()
    protocols: dict[str, str] = {}
    try:
        secret = keychain_reader(config.keychain)
        credentials = Credentials.from_keychain_payload(
            secret, username=config.keychain.account
        )
        if config.api is not None:
            api_transport = rne_api or RneHttpsApiTransport()
            bearer = api_transport.login(
                url=config.api.login_url, credentials=credentials
            )
            destination = stage / config.api.output_name
            search_after = ""
            seen_cursors: set[str] = set()
            record_count = 0
            with destination.open("wb") as output:
                for _page in range(config.api.maximum_pages):
                    records, next_search_after = api_transport.fetch_diff(
                        url=config.api.diff_url,
                        token=bearer,
                        from_date=config.api.from_date,
                        to_date=config.api.to_date,
                        page_size=config.api.page_size,
                        search_after=search_after,
                    )
                    for record in records:
                        output.write(canonical_json(record))
                        record_count += 1
                    if not next_search_after:
                        if len(records) >= config.api.page_size:
                            raise OfficialSyncError(
                                "RNE API returned a full page without pagination-search-after"
                            )
                        break
                    if next_search_after in seen_cursors:
                        raise OfficialSyncError("RNE API pagination cursor repeated")
                    seen_cursors.add(next_search_after)
                    search_after = next_search_after
                else:
                    raise OfficialSyncError("RNE API maximum_pages safety limit reached")
            os.chmod(destination, 0o600)
            protocols[config.api.output_name] = "https-rne-formalites-v4-diff"
            payload_names = [config.api.output_name]
            api_provenance: Mapping[str, Any] = {
                "api_login_url": config.api.login_url,
                "api_diff_url": config.api.diff_url,
                "from_exclusive": config.api.from_date,
                "to_inclusive": config.api.to_date,
                "page_size": config.api.page_size,
                "records": record_count,
                "pagination": "pagination-search-after",
            }
        else:
            sftp_transport = sftp or SftpTransport()
            ftps_transport = ftps or FtpsTransport()
            for item in config.files:
                destination = stage / item.name
                try:
                    sftp_transport.download(
                        host=config.sftp_host,
                        port=config.sftp_port,
                        remote_path=item.sftp_path,
                        destination=destination,
                        credentials=credentials,
                        known_hosts=config.known_hosts,
                    )
                    protocols[item.name] = "sftp"
                except Exception:
                    destination.unlink(missing_ok=True)
                    try:
                        ftps_transport.download(
                            host=config.ftps_host,
                            port=config.ftps_port,
                            remote_path=item.ftps_path,
                            destination=destination,
                            credentials=credentials,
                        )
                        protocols[item.name] = "ftps"
                    except Exception as exc:
                        destination.unlink(missing_ok=True)
                        if "INSECURE_FTP_UNSUPPORTED" in str(exc):
                            raise OfficialSyncError(
                                "INSECURE_FTP_UNSUPPORTED: RNE server offers plaintext FTP only; credentials not sent"
                            ) from exc
                        raise OfficialSyncError(
                            f"RNE secure transfer failed for {item.name}; SFTP and FTPS unavailable"
                        ) from exc
                if not destination.is_file():
                    raise OfficialSyncError(f"RNE transport produced no file for {item.name}")
                os.chmod(destination, 0o600)
            payload_names = [item.name for item in config.files]
            api_provenance = {}
        payload = _payload_metadata(stage, payload_names)
        return publish_content_addressed(
            stage=stage,
            output_root=output_root,
            source="rne",
            payload_metadata=payload,
            provenance={
                "protocol_by_file": protocols,
                "sftp_host": config.sftp_host,
                "ftps_host": config.ftps_host,
                **api_provenance,
                "keychain_locator_sha256": hashlib.sha256(
                    canonical_json(
                        {
                            "service": config.keychain.service,
                            "account": config.keychain.account,
                        }
                    )
                ).hexdigest(),
                "credential_material_recorded": False,
                "plain_ftp_allowed": False,
            },
        )
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    finally:
        for index in range(len(secret)):
            secret[index] = 0
        for index in range(len(bearer)):
            bearer[index] = 0


@dataclass(frozen=True)
class BodaccBackfillFile:
    name: str
    url: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BodaccBackfillFile":
        name, url = _safe_filename(str(raw.get("name") or "")), str(raw.get("url") or "")
        _validate_public_url(url, schemes={"https", "ftps"})
        return cls(name=name, url=url)


@dataclass(frozen=True)
class OdsIncrementalConfig:
    url: str
    watermark_field: str
    since: str
    tie_break_field: str = "id"
    since_tie_break: str = ""
    page_size: int = 100
    partition_where: str = ""
    tie_break_type: str = "text"
    watermark_type: str = "text"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OdsIncrementalConfig":
        url = str(raw.get("url") or DEFAULT_BODACC_ODS_V21_URL)
        _validate_public_url(url, schemes={"https"})
        field = str(raw.get("watermark_field") or "")
        if not field.replace("_", "").isalnum():
            raise OfficialSyncError("Opendatasoft watermark field is invalid")
        tie_break_field = str(raw.get("tie_break_field") or "id")
        if not tie_break_field.replace("_", "").isalnum():
            raise OfficialSyncError("Opendatasoft tie-break field is invalid")
        page_size = int(raw.get("page_size", 100))
        if page_size < 1 or page_size > 100:
            raise OfficialSyncError("Opendatasoft page_size must be between 1 and 100")
        partition_where = str(raw.get("partition_where") or "").strip()
        tie_break_type = str(raw.get("tie_break_type") or "text").lower()
        watermark_type = str(raw.get("watermark_type") or "text").lower()
        if tie_break_type not in {"text", "integer"}:
            raise OfficialSyncError("BODACC tie_break_type must be text or integer")
        if watermark_type not in {"text", "date"}:
            raise OfficialSyncError("BODACC watermark_type must be text or date")
        if any(character in partition_where for character in (";", "\n", "\r")):
            raise OfficialSyncError("BODACC partition_where contains forbidden separators")
        return cls(
            url=url,
            watermark_field=field,
            since=str(raw.get("since") or ""),
            tie_break_field=tie_break_field,
            since_tie_break=str(raw.get("since_tie_break") or ""),
            page_size=page_size,
            partition_where=partition_where,
            tie_break_type=tie_break_type,
            watermark_type=watermark_type,
        )


@dataclass(frozen=True)
class BodaccSyncConfig:
    backfill: tuple[BodaccBackfillFile, ...]
    incremental: OdsIncrementalConfig | None
    keychain: KeychainLocator | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BodaccSyncConfig":
        _reject_inline_secrets(raw)
        keychain_raw = raw.get("keychain")
        keychain = (
            KeychainLocator(str(keychain_raw.get("service") or ""), str(keychain_raw.get("account") or ""))
            if isinstance(keychain_raw, Mapping)
            else None
        )
        incremental_raw = raw.get("incremental")
        config = cls(
            backfill=tuple(BodaccBackfillFile.from_dict(item) for item in raw.get("backfill", [])),
            incremental=(
                OdsIncrementalConfig.from_dict(incremental_raw)
                if isinstance(incremental_raw, Mapping)
                else None
            ),
            keychain=keychain,
        )
        if not config.backfill and config.incremental is None:
            raise OfficialSyncError("BODACC config must contain backfill or incremental acquisition")
        if len({item.name for item in config.backfill}) != len(config.backfill):
            raise OfficialSyncError("BODACC backfill filenames must be unique")
        if any(urlparse(item.url).scheme == "ftps" for item in config.backfill) and keychain is None:
            raise OfficialSyncError("BODACC FTPS requires a macOS Keychain locator")
        return config


def _ftps_url_parts(url: str) -> tuple[str, int, str]:
    parsed = urlparse(url)
    _validate_public_url(url, schemes={"ftps"})
    return str(parsed.hostname), int(parsed.port or 21), str(PurePosixPath(parsed.path))


def _ods_literal(value: str) -> str:
    return value.replace("'", "''")


def _ods_url(
    config: OdsIncrementalConfig,
    *,
    after_watermark: str,
    after_tie_break: str,
) -> str:
    parsed = urlparse(config.url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    cursor_where = ""
    if after_watermark:
        escaped_watermark = _ods_literal(after_watermark)
        watermark_literal = (
            f"date'{escaped_watermark}'"
            if config.watermark_type == "date"
            else f"'{escaped_watermark}'"
        )
        if after_tie_break:
            if config.tie_break_type == "integer":
                if not str(after_tie_break).isdigit():
                    raise OfficialSyncError("BODACC integer cursor is not numeric")
                tie_literal = str(int(after_tie_break))
            else:
                tie_literal = f"'{_ods_literal(after_tie_break)}'"
            cursor_where = (
                f"{config.watermark_field} > {watermark_literal} OR "
                f"({config.watermark_field} = {watermark_literal} AND "
                f"{config.tie_break_field} > {tie_literal})"
            )
        else:
            cursor_where = f"{config.watermark_field} > {watermark_literal}"
    where_parts = [value for value in (config.partition_where, cursor_where) if value]
    if len(where_parts) == 1:
        query["where"] = where_parts[0]
    elif where_parts:
        query["where"] = " AND ".join(f"({value})" for value in where_parts)
    query["order_by"] = f"{config.watermark_field},{config.tie_break_field}"
    query["limit"] = str(config.page_size)
    return urlunparse(parsed._replace(query=urlencode(query)))


class JsonHttpTransport(Protocol):
    def download(self, *, url: str, destination: Path) -> None: ...
    def get_json(self, *, url: str) -> Mapping[str, Any]: ...


def _ods_cursor_key(
    config: OdsIncrementalConfig, watermark: str, tie_break: str
) -> tuple[str, str | int]:
    if config.tie_break_type == "integer":
        try:
            return watermark, int(tie_break or 0)
        except ValueError as exc:
            raise OfficialSyncError("BODACC integer cursor is not numeric") from exc
    return watermark, tie_break


def sync_bodacc(
    *,
    config: BodaccSyncConfig,
    output_root: Path,
    keychain_reader: Callable[[KeychainLocator], bytearray] = read_keychain_secret,
    https: JsonHttpTransport | None = None,
    ftps: FileTransport | None = None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage = Path(tempfile.mkdtemp(prefix=".bodacc-stage-", dir=output_root))
    os.chmod(stage, 0o700)
    secret = bytearray()
    protocols: dict[str, str] = {}
    payload_names: list[str] = []
    http_transport = https or HttpsTransport()
    ftps_transport = ftps or FtpsTransport()
    credentials: Credentials | None = None
    try:
        if any(urlparse(item.url).scheme == "ftps" for item in config.backfill):
            assert config.keychain is not None
            secret = keychain_reader(config.keychain)
            credentials = Credentials.from_keychain_payload(
                secret, username=config.keychain.account
            )
        for item in config.backfill:
            destination = stage / item.name
            scheme = urlparse(item.url).scheme.lower()
            if scheme == "https":
                http_transport.download(url=item.url, destination=destination)
            elif scheme == "ftps":
                host, port, remote_path = _ftps_url_parts(item.url)
                ftps_transport.download(
                    host=host,
                    port=port,
                    remote_path=remote_path,
                    destination=destination,
                    credentials=credentials,
                )
            else:
                raise OfficialSyncError("only HTTPS and FTPS BODACC backfill are allowed")
            if not destination.is_file():
                raise OfficialSyncError(f"BODACC transport produced no file for {item.name}")
            os.chmod(destination, 0o600)
            protocols[item.name] = scheme
            payload_names.append(item.name)

        next_watermark = config.incremental.since if config.incremental else None
        next_tie_break = config.incremental.since_tie_break if config.incremental else None
        incremental_rows = 0
        if config.incremental:
            incremental_name = "incremental.jsonl"
            incremental_path = stage / incremental_name
            with incremental_path.open("wb") as output:
                while True:
                    response = http_transport.get_json(
                        url=_ods_url(
                            config.incremental,
                            after_watermark=next_watermark or "",
                            after_tie_break=next_tie_break or "",
                        )
                    )
                    rows = response.get("results")
                    if not isinstance(rows, list):
                        raise OfficialSyncError("Opendatasoft response has no results array")
                    previous_cursor = _ods_cursor_key(
                        config.incremental,
                        next_watermark or "",
                        next_tie_break or "",
                    )
                    for row in rows:
                        if not isinstance(row, Mapping):
                            raise OfficialSyncError("Opendatasoft result row is not an object")
                        output.write(canonical_json(row))
                        value = row.get(config.incremental.watermark_field)
                        tie_break = row.get(config.incremental.tie_break_field)
                        if value is None or tie_break is None:
                            raise OfficialSyncError(
                                "Opendatasoft result lacks incremental cursor fields"
                            )
                        cursor = (str(value), str(tie_break))
                        if _ods_cursor_key(
                            config.incremental, *cursor
                        ) <= _ods_cursor_key(
                            config.incremental,
                            next_watermark or "",
                            next_tie_break or "",
                        ):
                            raise OfficialSyncError(
                                "Opendatasoft incremental cursor did not advance"
                            )
                        next_watermark, next_tie_break = cursor
                    incremental_rows += len(rows)
                    if len(rows) < config.incremental.page_size:
                        break
                    if _ods_cursor_key(
                        config.incremental,
                        next_watermark or "",
                        next_tie_break or "",
                    ) <= previous_cursor:
                        raise OfficialSyncError(
                            "Opendatasoft pagination made no forward progress"
                        )
            os.chmod(incremental_path, 0o600)
            protocols[incremental_name] = "https-opendatasoft-v2"
            payload_names.append(incremental_name)

        payload = _payload_metadata(stage, payload_names)
        return publish_content_addressed(
            stage=stage,
            output_root=output_root,
            source="bodacc",
            payload_metadata=payload,
            provenance={
                "protocol_by_file": protocols,
                "backfill_sources": [
                    {
                        "scheme": urlparse(item.url).scheme,
                        "host": urlparse(item.url).hostname,
                        "path": urlparse(item.url).path,
                    }
                    for item in config.backfill
                ],
                "incremental_endpoint": (
                    {
                        "scheme": urlparse(config.incremental.url).scheme,
                        "host": urlparse(config.incremental.url).hostname,
                        "path": urlparse(config.incremental.url).path,
                    }
                    if config.incremental
                    else None
                ),
                "incremental_rows": incremental_rows,
                "previous_watermark": config.incremental.since if config.incremental else None,
                "previous_tie_break": (
                    config.incremental.since_tie_break if config.incremental else None
                ),
                "next_watermark": next_watermark,
                "next_tie_break": next_tie_break,
                "credential_material_recorded": False,
                "plain_ftp_allowed": False,
            },
        )
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    finally:
        for index in range(len(secret)):
            secret[index] = 0


def load_json_config(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise OfficialSyncError("sync config root must be a JSON object")
    return value


__all__ = [
    "BodaccSyncConfig",
    "Credentials",
    "FtpsTransport",
    "HttpsTransport",
    "KeychainLocator",
    "OfficialSyncError",
    "RneSyncConfig",
    "RneApiConfig",
    "RneHttpsApiTransport",
    "SftpTransport",
    "load_json_config",
    "initialize_keychain_secret",
    "publish_content_addressed",
    "read_keychain_secret",
    "sync_bodacc",
    "sync_rne",
]
