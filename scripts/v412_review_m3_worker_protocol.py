#!/usr/bin/env python3
"""Closed, bounded stdlib protocol for the future V4.12 M3 worker.

The module is transport-only.  It never creates a socket, process, network
connection, or filesystem path.  Callers must provide an already-connected
AF_UNIX socket (normally an inherited descriptor).
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass
import hashlib
import json
import math
import re
import socket
import struct
import time
from typing import Any, Final


MAGIC: Final = b"S4M3"
PROTOCOL_VERSION: Final = 1
PROTOCOL_NAME: Final = "SIRETO_V412_M3_WORKER"
MAX_FRAME_BYTES: Final = 64 * 1024
HEADER: Final = struct.Struct("!4sBBBBII")
MAX_FRAME_PAYLOAD: Final = MAX_FRAME_BYTES - HEADER.size
MAX_OBJECT_BYTES: Final = 32 * 1024 * 1024
MAX_OBJECT_CHUNKS: Final = (MAX_OBJECT_BYTES + MAX_FRAME_PAYLOAD - 1) // MAX_FRAME_PAYLOAD
MAX_CANDIDATES: Final = 3000
MAX_JSON_DEPTH: Final = 32
MAX_JSON_NODES: Final = 250_000
MAX_SAFE_INTEGER: Final = (1 << 53) - 1
MAX_SEQUENCE: Final = (1 << 32) - 1

ROLE_CONTROLLER: Final = "CONTROLLER"
ROLE_WORKER: Final = "WORKER"
ROLES: Final = frozenset({ROLE_CONTROLLER, ROLE_WORKER})
ROLE_TO_WIRE: Final = {ROLE_CONTROLLER: 1, ROLE_WORKER: 2}
WIRE_TO_ROLE: Final = {value: key for key, value in ROLE_TO_WIRE.items()}

TYPE_HELLO: Final = "HELLO"
TYPE_READY: Final = "READY"
TYPE_GATE: Final = "GATE"
TYPE_BEGIN: Final = "BEGIN"
TYPE_CHUNK: Final = "CHUNK"
TYPE_END: Final = "END"
TYPE_STOP: Final = "STOP"
TYPE_COMPLETE: Final = "COMPLETE"
FRAME_TYPES: Final = (
    TYPE_HELLO,
    TYPE_READY,
    TYPE_GATE,
    TYPE_BEGIN,
    TYPE_CHUNK,
    TYPE_END,
    TYPE_STOP,
    TYPE_COMPLETE,
)
TYPE_TO_WIRE: Final = {name: ordinal for ordinal, name in enumerate(FRAME_TYPES, 1)}
WIRE_TO_TYPE: Final = {value: key for key, value in TYPE_TO_WIRE.items()}

KIND_DTO: Final = "DTO"
KIND_CANDIDATE_BATCH: Final = "CANDIDATE_BATCH"
KIND_RESULT: Final = "RESULT"
OBJECT_KINDS: Final = frozenset({KIND_DTO, KIND_CANDIDATE_BATCH, KIND_RESULT})
OBJECT_DIRECTIONS: Final = {
    KIND_DTO: (ROLE_CONTROLLER, ROLE_WORKER),
    KIND_CANDIDATE_BATCH: (ROLE_CONTROLLER, ROLE_WORKER),
    KIND_RESULT: (ROLE_WORKER, ROLE_CONTROLLER),
}

STATE_NEW: Final = "NEW"
STATE_HELLO_SENT: Final = "HELLO_SENT"
STATE_HELLO_RECEIVED: Final = "HELLO_RECEIVED"
STATE_READY_SENT: Final = "READY_SENT"
STATE_READY: Final = "READY"
STATE_ACTIVE: Final = "ACTIVE"
STATE_STOP_SENT: Final = "STOP_SENT"
STATE_STOP_RECEIVED: Final = "STOP_RECEIVED"
STATE_COMPLETE: Final = "COMPLETE"
STATES: Final = frozenset(
    {
        STATE_NEW,
        STATE_HELLO_SENT,
        STATE_HELLO_RECEIVED,
        STATE_READY_SENT,
        STATE_READY,
        STATE_ACTIVE,
        STATE_STOP_SENT,
        STATE_STOP_RECEIVED,
        STATE_COMPLETE,
    }
)

_OBJECT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


class ProtocolError(RuntimeError):
    """The peer or caller violated a closed protocol invariant."""


class ProtocolTimeout(ProtocolError):
    """The absolute monotonic deadline expired."""


class ProtocolEOF(ProtocolError):
    """The peer closed cleanly between frames."""


class ProtocolTruncated(ProtocolError):
    """The peer closed in the middle of a frame or object."""


class CanonicalJSONError(ProtocolError):
    """A value is not canonical, bounded JSON."""


class SequenceError(ProtocolError):
    """A directional frame sequence was skipped or replayed."""


class StateError(ProtocolError):
    """A message is not legal in the current role/state."""


@dataclass(frozen=True)
class Frame:
    frame_type: str
    sender_role: str
    sequence: int
    payload: bytes


@dataclass(frozen=True)
class ReceivedObject:
    kind: str
    object_id: str
    value: Any
    length: int
    sha256: str


def _reject_constant(value: str) -> None:
    raise CanonicalJSONError(f"non-finite JSON number is forbidden: {value}")


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _scan_json_string(raw: bytes, offset: int) -> int:
    """Return the first offset after one syntactically valid JSON string."""
    if offset >= len(raw) or raw[offset] != 0x22:
        raise CanonicalJSONError("JSON string expected")
    offset += 1
    while offset < len(raw):
        character = raw[offset]
        if character == 0x22:
            return offset + 1
        if character < 0x20:
            raise CanonicalJSONError("unescaped JSON control character")
        if character != 0x5C:
            offset += 1
            continue
        offset += 1
        if offset >= len(raw):
            raise CanonicalJSONError("truncated JSON escape")
        escape = raw[offset]
        if escape in b'"\\/bfnrt':
            offset += 1
            continue
        if escape != 0x75 or offset + 4 >= len(raw):
            raise CanonicalJSONError("invalid JSON escape")
        if any(
            character not in b"0123456789abcdefABCDEF"
            for character in raw[offset + 1 : offset + 5]
        ):
            raise CanonicalJSONError("invalid JSON unicode escape")
        offset += 5
    raise CanonicalJSONError("unterminated JSON string")


def _scan_json_number(raw: bytes, offset: int) -> int:
    """Return the first offset after one strict RFC 8259 JSON number."""
    start = offset
    if raw[offset] == 0x2D:
        offset += 1
        if offset >= len(raw):
            raise CanonicalJSONError("truncated JSON number")
    if raw[offset] == 0x30:
        offset += 1
        if offset < len(raw) and 0x30 <= raw[offset] <= 0x39:
            raise CanonicalJSONError("JSON number has a leading zero")
    elif 0x31 <= raw[offset] <= 0x39:
        offset += 1
        while offset < len(raw) and 0x30 <= raw[offset] <= 0x39:
            offset += 1
    else:
        raise CanonicalJSONError("invalid JSON number")
    if offset < len(raw) and raw[offset] == 0x2E:
        offset += 1
        fraction = offset
        while offset < len(raw) and 0x30 <= raw[offset] <= 0x39:
            offset += 1
        if offset == fraction:
            raise CanonicalJSONError("JSON fraction has no digits")
    if offset < len(raw) and raw[offset] in b"eE":
        offset += 1
        if offset < len(raw) and raw[offset] in b"+-":
            offset += 1
        exponent = offset
        while offset < len(raw) and 0x30 <= raw[offset] <= 0x39:
            offset += 1
        if offset == exponent:
            raise CanonicalJSONError("JSON exponent has no digits")
    if offset == start:
        raise CanonicalJSONError("invalid JSON number")
    return offset


def _prescan_json(raw: bytes) -> None:
    """Validate JSON structure/depth/nodes before invoking ``json.loads``.

    The scanner is iterative and keeps at most ``MAX_JSON_DEPTH`` container
    records.  Brackets inside UTF-8 strings and escaped quotes never affect
    structural depth.
    """
    try:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        view = memoryview(raw)
        for chunk_start in range(0, len(raw), MAX_FRAME_PAYLOAD):
            decoder.decode(view[chunk_start : chunk_start + MAX_FRAME_PAYLOAD], final=False)
        decoder.decode(b"", final=True)
        length = len(raw)
        offset = 0
        stack: list[str] = []
        expectation = "VALUE"
        nodes = 0

        def skip_whitespace(position: int) -> int:
            while position < length and raw[position] in b" \t\r\n":
                position += 1
            return position

        def complete_value() -> str:
            return "COMMA_OR_END" if stack else "EOF"

        while True:
            offset = skip_whitespace(offset)
            if expectation == "EOF":
                if offset != length:
                    raise CanonicalJSONError("extra content after JSON value")
                return
            if offset >= length:
                raise CanonicalJSONError("truncated JSON structure")
            character = raw[offset]

            if expectation in {"VALUE", "VALUE_OR_END"}:
                if expectation == "VALUE_OR_END" and character == 0x5D:
                    offset += 1
                    stack.pop()
                    expectation = complete_value()
                    continue
                nodes += 1
                if nodes > MAX_JSON_NODES:
                    raise CanonicalJSONError("JSON node ceiling exceeded")
                if character == 0x7B:
                    if len(stack) >= MAX_JSON_DEPTH:
                        raise CanonicalJSONError("JSON nesting ceiling exceeded")
                    stack.append("OBJECT")
                    offset += 1
                    expectation = "KEY_OR_END"
                    continue
                if character == 0x5B:
                    if len(stack) >= MAX_JSON_DEPTH:
                        raise CanonicalJSONError("JSON nesting ceiling exceeded")
                    stack.append("ARRAY")
                    offset += 1
                    expectation = "VALUE_OR_END"
                    continue
                if character == 0x22:
                    offset = _scan_json_string(raw, offset)
                elif character == 0x2D or 0x30 <= character <= 0x39:
                    offset = _scan_json_number(raw, offset)
                else:
                    literal = next(
                        (
                            item
                            for item in (b"true", b"false", b"null")
                            if raw.startswith(item, offset)
                        ),
                        None,
                    )
                    if literal is None:
                        raise CanonicalJSONError("invalid JSON value token")
                    offset += len(literal)
                expectation = complete_value()
                continue

            if expectation in {"KEY", "KEY_OR_END"}:
                if expectation == "KEY_OR_END" and character == 0x7D:
                    offset += 1
                    stack.pop()
                    expectation = complete_value()
                    continue
                if character != 0x22:
                    raise CanonicalJSONError("JSON object key must be a string")
                offset = _scan_json_string(raw, offset)
                expectation = "COLON"
                continue

            if expectation == "COLON":
                if character != 0x3A:
                    raise CanonicalJSONError("JSON object key lacks a colon")
                offset += 1
                expectation = "VALUE"
                continue

            if expectation == "COMMA_OR_END":
                if not stack:
                    raise CanonicalJSONError("invalid JSON parser state")
                container = stack[-1]
                closing = 0x7D if container == "OBJECT" else 0x5D
                if character == closing:
                    offset += 1
                    stack.pop()
                    expectation = complete_value()
                    continue
                if character != 0x2C:
                    raise CanonicalJSONError("JSON container lacks comma or closing token")
                offset += 1
                expectation = "KEY" if container == "OBJECT" else "VALUE"
                continue

            raise CanonicalJSONError("invalid JSON scanner state")
    except CanonicalJSONError:
        raise
    except UnicodeDecodeError as exc:
        raise CanonicalJSONError("invalid UTF-8 JSON payload") from exc
    except (MemoryError, RecursionError) as exc:
        raise CanonicalJSONError("JSON pre-scan resource ceiling exceeded") from exc


def _validate_json_tree(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise CanonicalJSONError("JSON node ceiling exceeded")
        if item is None or type(item) in (str, bool):
            continue
        if type(item) is int:
            if abs(item) > MAX_SAFE_INTEGER:
                raise CanonicalJSONError("JSON integer exceeds the closed safe range")
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise CanonicalJSONError("non-finite JSON number is forbidden")
            continue
        if type(item) is list:
            child_depth = depth + 1
            if child_depth > MAX_JSON_DEPTH:
                raise CanonicalJSONError("JSON nesting ceiling exceeded")
            stack.extend((child, child_depth) for child in reversed(item))
            continue
        if type(item) is dict:
            child_depth = depth + 1
            if child_depth > MAX_JSON_DEPTH:
                raise CanonicalJSONError("JSON nesting ceiling exceeded")
            for key, child in reversed(tuple(item.items())):
                if type(key) is not str:
                    raise CanonicalJSONError("JSON object keys must be strings")
                stack.append((child, child_depth))
            continue
        raise CanonicalJSONError(f"non-JSON value type is forbidden: {type(item).__name__}")


def _json_string_size(value: str, ceiling: int) -> int:
    size = 2
    if size > ceiling:
        raise CanonicalJSONError("canonical JSON byte ceiling exceeded")
    for character in value:
        codepoint = ord(character)
        if character in {'"', "\\"} or character in "\b\f\n\r\t":
            size += 2
        elif codepoint < 0x20:
            size += 6
        elif codepoint <= 0x7F:
            size += 1
        elif codepoint <= 0x7FF:
            size += 2
        elif 0xD800 <= codepoint <= 0xDFFF:
            raise CanonicalJSONError("surrogate code point is not valid UTF-8 JSON")
        elif codepoint <= 0xFFFF:
            size += 3
        else:
            size += 4
        if size > ceiling:
            raise CanonicalJSONError("canonical JSON byte ceiling exceeded")
    return size


def _canonical_size(value: Any, ceiling: int) -> int:
    """Compute exact encoded size, aborting before any large JSON allocation."""
    def checked(total: int, increment: int) -> int:
        total += increment
        if total > ceiling:
            raise CanonicalJSONError("canonical JSON byte ceiling exceeded")
        return total

    def scalar(size: int) -> int:
        if size > ceiling:
            raise CanonicalJSONError("canonical JSON byte ceiling exceeded")
        return size

    if value is None:
        return scalar(4)
    if type(value) is bool:
        return scalar(4 if value else 5)
    if type(value) is str:
        return _json_string_size(value, ceiling)
    if type(value) is int:
        return scalar(len(str(value)))
    if type(value) is float:
        return scalar(len(json.dumps(value, allow_nan=False)))
    if type(value) is list:
        total = scalar(2)
        for index, child in enumerate(value):
            total = checked(total, 1 if index else 0)
            total = checked(total, _canonical_size(child, ceiling - total))
        return total
    if type(value) is dict:
        total = scalar(2)
        for index, key in enumerate(sorted(value)):
            total = checked(total, 1 if index else 0)
            total = checked(total, _json_string_size(key, ceiling - total))
            total = checked(total, 1)
            total = checked(total, _canonical_size(value[key], ceiling - total))
        return total
    raise CanonicalJSONError(f"non-JSON value type is forbidden: {type(value).__name__}")


def canonical_json(value: Any, *, ceiling: int = MAX_OBJECT_BYTES) -> bytes:
    """Encode one value as compact UTF-8 canonical JSON without a newline."""
    if type(ceiling) is not int or not 0 <= ceiling <= MAX_OBJECT_BYTES:
        raise CanonicalJSONError("invalid JSON byte ceiling")
    try:
        _validate_json_tree(value)
        expected_size = _canonical_size(value, ceiling)
        encoder = json.JSONEncoder(
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        raw = bytearray()
        for fragment in encoder.iterencode(value):
            encoded = fragment.encode("utf-8", errors="strict")
            if len(raw) + len(encoded) > ceiling:
                raise CanonicalJSONError("canonical JSON byte ceiling exceeded")
            raw.extend(encoded)
        if len(raw) != expected_size:
            raise CanonicalJSONError("canonical JSON size preflight mismatch")
    except CanonicalJSONError:
        raise
    except (TypeError, ValueError, UnicodeEncodeError, MemoryError, RecursionError) as exc:
        raise CanonicalJSONError("value cannot be encoded as canonical UTF-8 JSON") from exc
    return bytes(raw)


def decode_canonical_json(raw: bytes, *, ceiling: int = MAX_OBJECT_BYTES) -> Any:
    """Decode only exact canonical JSON, rejecting duplicates and NaN tokens."""
    if type(raw) is not bytes:
        raise CanonicalJSONError("JSON payload must be bytes")
    if len(raw) > ceiling:
        raise CanonicalJSONError("canonical JSON byte ceiling exceeded")
    try:
        _prescan_json(raw)
        value = json.loads(
            raw,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
        )
    except CanonicalJSONError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        MemoryError,
        RecursionError,
    ) as exc:
        raise CanonicalJSONError("invalid UTF-8 JSON payload") from exc
    try:
        _validate_json_tree(value)
        if canonical_json(value, ceiling=ceiling) != raw:
            raise CanonicalJSONError("JSON payload is not canonical")
    except CanonicalJSONError:
        raise
    except (MemoryError, RecursionError) as exc:
        raise CanonicalJSONError("decoded JSON resource ceiling exceeded") from exc
    return value


def _absolute_remaining(deadline: float) -> float:
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ProtocolTimeout("deadline must be a finite absolute monotonic time")
    remaining = float(deadline) - time.monotonic()
    if remaining <= 0:
        raise ProtocolTimeout("absolute monotonic deadline expired")
    return remaining


def _validate_object_id(object_id: str) -> None:
    if type(object_id) is not str or _OBJECT_ID.fullmatch(object_id) is None:
        raise ProtocolError("object_id is outside the closed identifier grammar")


def _validate_object_value(kind: str, value: Any) -> None:
    if type(kind) is not str or kind not in OBJECT_KINDS:
        raise ProtocolError("unknown object kind")
    if kind == KIND_CANDIDATE_BATCH:
        if type(value) is not list:
            raise ProtocolError("CANDIDATE_BATCH must be a JSON list")
        if len(value) > MAX_CANDIDATES:
            raise ProtocolError("candidate batch exceeds the 3000-item ceiling")
        if any(type(candidate) is not dict for candidate in value):
            raise ProtocolError("each candidate must be a JSON object")
    elif type(value) is not dict:
        raise ProtocolError(f"{kind} must be a JSON object")


class FramedTransport:
    """Binary framing over one caller-owned, connected AF_UNIX socket."""

    def __init__(self, inherited_socket: Any, local_role: str) -> None:
        if type(local_role) is not str or local_role not in ROLES:
            raise ProtocolError("unknown local role")
        if getattr(inherited_socket, "family", None) != socket.AF_UNIX:
            raise ProtocolError("transport requires an inherited AF_UNIX socket")
        socket_type = getattr(inherited_socket, "type", None)
        if not isinstance(socket_type, int) or socket_type & socket.SOCK_STREAM == 0:
            raise ProtocolError("transport requires an inherited AF_UNIX stream socket")
        for method in ("recv", "send", "settimeout"):
            if not callable(getattr(inherited_socket, method, None)):
                raise ProtocolError("inherited socket does not implement the stream contract")
        self.socket = inherited_socket
        self.local_role = local_role
        self.peer_role = ROLE_WORKER if local_role == ROLE_CONTROLLER else ROLE_CONTROLLER
        self.tx_sequence = 0
        self.rx_sequence = 0

    def _send_all(self, raw: bytes, deadline: float) -> None:
        offset = 0
        while offset < len(raw):
            try:
                self.socket.settimeout(_absolute_remaining(deadline))
                count = self.socket.send(raw[offset:])
            except (TimeoutError, socket.timeout) as exc:
                raise ProtocolTimeout("send deadline expired") from exc
            if count is None or count <= 0:
                raise ProtocolTruncated("socket stopped accepting a frame")
            offset += count

    def _recv_exact(self, size: int, deadline: float, *, between_frames: bool = False) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            try:
                self.socket.settimeout(_absolute_remaining(deadline))
                chunk = self.socket.recv(remaining)
            except (TimeoutError, socket.timeout) as exc:
                raise ProtocolTimeout("receive deadline expired") from exc
            if not chunk:
                if between_frames and remaining == size:
                    raise ProtocolEOF("peer closed between frames")
                raise ProtocolTruncated("peer closed inside a frame")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def reserve_sequence_range(self, frame_count: int) -> tuple[int, int]:
        """Fail before I/O unless a complete contiguous frame range remains."""
        if type(frame_count) is not int or frame_count <= 0:
            raise SequenceError("frame reservation count must be a positive integer")
        first = self.tx_sequence
        last = first + frame_count - 1
        if last > MAX_SEQUENCE:
            raise SequenceError("outbound sequence space cannot fit the whole message")
        return first, last

    def send_frame(self, frame_type: str, payload: bytes, *, deadline: float) -> int:
        if type(frame_type) is not str or frame_type not in TYPE_TO_WIRE:
            raise ProtocolError("unknown frame type")
        if type(payload) is not bytes:
            raise ProtocolError("frame payload must be bytes")
        if len(payload) > MAX_FRAME_PAYLOAD:
            raise ProtocolError("frame exceeds the 64 KiB ceiling")
        sequence = self.tx_sequence
        if sequence > MAX_SEQUENCE:
            raise SequenceError("outbound sequence space exhausted")
        header = HEADER.pack(
            MAGIC,
            PROTOCOL_VERSION,
            TYPE_TO_WIRE[frame_type],
            ROLE_TO_WIRE[self.local_role],
            0,
            sequence,
            len(payload),
        )
        self._send_all(header + payload, deadline)
        self.tx_sequence += 1
        return sequence

    def receive_frame(self, *, deadline: float) -> Frame:
        header = self._recv_exact(HEADER.size, deadline, between_frames=True)
        magic, version, wire_type, wire_role, flags, sequence, size = HEADER.unpack(header)
        if magic != MAGIC or version != PROTOCOL_VERSION or flags != 0:
            raise ProtocolError("invalid frame header")
        frame_type = WIRE_TO_TYPE.get(wire_type)
        sender_role = WIRE_TO_ROLE.get(wire_role)
        if frame_type is None or sender_role != self.peer_role:
            raise ProtocolError("unknown frame type or invalid sender role")
        if size > MAX_FRAME_PAYLOAD:
            raise ProtocolError("declared frame exceeds the 64 KiB ceiling")
        if sequence != self.rx_sequence:
            raise SequenceError(
                f"expected peer sequence {self.rx_sequence}, received {sequence}"
            )
        payload = self._recv_exact(size, deadline)
        self.rx_sequence += 1
        return Frame(frame_type, sender_role, sequence, payload)


class ProtocolSession:
    """Role-aware handshake and whole-object transfer state machine."""

    def __init__(self, inherited_socket: Any, role: str) -> None:
        self.transport = FramedTransport(inherited_socket, role)
        self.role = role
        self.state = STATE_NEW

    def _require(self, role: str, state: str) -> None:
        if self.role != role or self.state != state:
            raise StateError(
                f"operation requires {role}/{state}, got {self.role}/{self.state}"
            )

    def _send_control(self, frame_type: str, value: dict[str, Any], deadline: float) -> None:
        raw = canonical_json(value, ceiling=MAX_FRAME_PAYLOAD)
        self.transport.send_frame(frame_type, raw, deadline=deadline)

    def _receive_control(
        self, frame_type: str, expected: dict[str, Any], deadline: float
    ) -> None:
        frame = self.transport.receive_frame(deadline=deadline)
        if frame.frame_type != frame_type:
            raise StateError(f"expected {frame_type}, received {frame.frame_type}")
        decode_canonical_json(frame.payload, ceiling=MAX_FRAME_PAYLOAD)
        if frame.payload != canonical_json(expected, ceiling=MAX_FRAME_PAYLOAD):
            raise StateError(f"invalid {frame_type} payload")

    def send_hello(self, *, deadline: float) -> None:
        self._require(ROLE_CONTROLLER, STATE_NEW)
        self._send_control(
            TYPE_HELLO,
            {"protocol": PROTOCOL_NAME, "version": PROTOCOL_VERSION},
            deadline,
        )
        self.state = STATE_HELLO_SENT

    def receive_hello(self, *, deadline: float) -> None:
        self._require(ROLE_WORKER, STATE_NEW)
        self._receive_control(
            TYPE_HELLO,
            {"protocol": PROTOCOL_NAME, "version": PROTOCOL_VERSION},
            deadline,
        )
        self.state = STATE_HELLO_RECEIVED

    def send_ready(self, *, deadline: float) -> None:
        self._require(ROLE_WORKER, STATE_HELLO_RECEIVED)
        self._send_control(TYPE_READY, {"status": "READY"}, deadline)
        self.state = STATE_READY_SENT

    def receive_ready(self, *, deadline: float) -> None:
        self._require(ROLE_CONTROLLER, STATE_HELLO_SENT)
        self._receive_control(TYPE_READY, {"status": "READY"}, deadline)
        self.state = STATE_READY

    def send_gate(self, *, deadline: float) -> None:
        self._require(ROLE_CONTROLLER, STATE_READY)
        self._send_control(TYPE_GATE, {"gate": "OPEN"}, deadline)
        self.state = STATE_ACTIVE

    def receive_gate(self, *, deadline: float) -> None:
        self._require(ROLE_WORKER, STATE_READY_SENT)
        self._receive_control(TYPE_GATE, {"gate": "OPEN"}, deadline)
        self.state = STATE_ACTIVE

    def send_stop(self, *, deadline: float) -> None:
        self._require(ROLE_CONTROLLER, STATE_ACTIVE)
        self._send_control(TYPE_STOP, {"reason": "NORMAL"}, deadline)
        self.state = STATE_STOP_SENT

    def receive_stop(self, *, deadline: float) -> None:
        self._require(ROLE_WORKER, STATE_ACTIVE)
        self._receive_control(TYPE_STOP, {"reason": "NORMAL"}, deadline)
        self.state = STATE_STOP_RECEIVED

    def send_complete(self, *, deadline: float) -> None:
        self._require(ROLE_WORKER, STATE_STOP_RECEIVED)
        self._send_control(TYPE_COMPLETE, {"status": "OK"}, deadline)
        self.state = STATE_COMPLETE

    def receive_complete(self, *, deadline: float) -> None:
        self._require(ROLE_CONTROLLER, STATE_STOP_SENT)
        self._receive_control(TYPE_COMPLETE, {"status": "OK"}, deadline)
        self.state = STATE_COMPLETE

    def send_object(
        self, kind: str, object_id: str, value: Any, *, deadline: float
    ) -> ReceivedObject:
        if self.state != STATE_ACTIVE:
            raise StateError("objects may only be sent in ACTIVE state")
        _validate_object_id(object_id)
        _validate_object_value(kind, value)
        if OBJECT_DIRECTIONS[kind][0] != self.role:
            raise StateError(f"{kind} cannot be sent by {self.role}")
        raw = canonical_json(value)
        digest = hashlib.sha256(raw).hexdigest()
        chunk_count = max(1, (len(raw) + MAX_FRAME_PAYLOAD - 1) // MAX_FRAME_PAYLOAD)
        self.transport.reserve_sequence_range(chunk_count + 2)
        begin = {
            "chunks": chunk_count,
            "kind": kind,
            "length": len(raw),
            "object_id": object_id,
            "sha256": digest,
        }
        begin_raw = canonical_json(begin, ceiling=MAX_FRAME_PAYLOAD)
        end = {"length": len(raw), "object_id": object_id, "sha256": digest}
        end_raw = canonical_json(end, ceiling=MAX_FRAME_PAYLOAD)
        self.transport.send_frame(
            TYPE_BEGIN,
            begin_raw,
            deadline=deadline,
        )
        for offset in range(0, len(raw), MAX_FRAME_PAYLOAD):
            self.transport.send_frame(
                TYPE_CHUNK, raw[offset : offset + MAX_FRAME_PAYLOAD], deadline=deadline
            )
        if not raw:
            self.transport.send_frame(TYPE_CHUNK, b"", deadline=deadline)
        self.transport.send_frame(
            TYPE_END,
            end_raw,
            deadline=deadline,
        )
        return ReceivedObject(kind, object_id, value, len(raw), digest)

    def receive_object(self, *, deadline: float) -> ReceivedObject:
        if self.state != STATE_ACTIVE:
            raise StateError("objects may only be received in ACTIVE state")
        begin_frame = self.transport.receive_frame(deadline=deadline)
        if begin_frame.frame_type != TYPE_BEGIN:
            raise StateError(f"expected BEGIN, received {begin_frame.frame_type}")
        begin = decode_canonical_json(begin_frame.payload, ceiling=MAX_FRAME_PAYLOAD)
        if type(begin) is not dict or set(begin) != {
            "chunks", "kind", "length", "object_id", "sha256"
        }:
            raise ProtocolError("invalid BEGIN schema")
        kind = begin["kind"]
        object_id = begin["object_id"]
        length = begin["length"]
        expected_hash = begin["sha256"]
        chunks = begin["chunks"]
        _validate_object_id(object_id)
        if (
            type(kind) is not str
            or kind not in OBJECT_KINDS
            or OBJECT_DIRECTIONS[kind][1] != self.role
        ):
            raise StateError("object kind has an invalid direction")
        if (
            type(length) is not int
            or not 0 <= length <= MAX_OBJECT_BYTES
            or type(chunks) is not int
            or not 1 <= chunks <= MAX_OBJECT_CHUNKS
            or chunks != max(1, (length + MAX_FRAME_PAYLOAD - 1) // MAX_FRAME_PAYLOAD)
            or type(expected_hash) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        ):
            raise ProtocolError("invalid BEGIN bounds or digest")
        parts: list[bytes] = []
        observed_length = 0
        digest = hashlib.sha256()
        for index in range(chunks):
            frame = self.transport.receive_frame(deadline=deadline)
            if frame.frame_type != TYPE_CHUNK:
                raise StateError(f"expected CHUNK, received {frame.frame_type}")
            expected_size = min(MAX_FRAME_PAYLOAD, length - observed_length)
            if len(frame.payload) != expected_size:
                raise ProtocolError(f"invalid CHUNK length at index {index}")
            parts.append(frame.payload)
            observed_length += len(frame.payload)
            digest.update(frame.payload)
        end_frame = self.transport.receive_frame(deadline=deadline)
        if end_frame.frame_type != TYPE_END:
            raise StateError(f"expected END, received {end_frame.frame_type}")
        decode_canonical_json(end_frame.payload, ceiling=MAX_FRAME_PAYLOAD)
        expected_end = {
            "length": length,
            "object_id": object_id,
            "sha256": expected_hash,
        }
        if end_frame.payload != canonical_json(
            expected_end, ceiling=MAX_FRAME_PAYLOAD
        ):
            raise ProtocolError("END does not match BEGIN")
        if observed_length != length or digest.hexdigest() != expected_hash:
            raise ProtocolError("object length or SHA-256 mismatch")
        raw = b"".join(parts)
        value = decode_canonical_json(raw)
        _validate_object_value(kind, value)
        return ReceivedObject(kind, object_id, value, length, expected_hash)


__all__ = [
    "CanonicalJSONError",
    "Frame",
    "FramedTransport",
    "KIND_CANDIDATE_BATCH",
    "KIND_DTO",
    "KIND_RESULT",
    "MAX_CANDIDATES",
    "MAX_FRAME_BYTES",
    "MAX_FRAME_PAYLOAD",
    "MAX_OBJECT_BYTES",
    "ProtocolEOF",
    "ProtocolError",
    "ProtocolSession",
    "ProtocolTimeout",
    "ProtocolTruncated",
    "ROLE_CONTROLLER",
    "ROLE_WORKER",
    "ReceivedObject",
    "SequenceError",
    "StateError",
    "canonical_json",
    "decode_canonical_json",
]
