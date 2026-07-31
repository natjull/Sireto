from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import socket
import struct
import threading
import time

import pytest

from scripts import v412_review_m3_worker_protocol as protocol


def deadline(seconds: float = 2.0) -> float:
    return time.monotonic() + seconds


def sessions() -> tuple[
    socket.socket, socket.socket, protocol.ProtocolSession, protocol.ProtocolSession
]:
    controller_socket, worker_socket = socket.socketpair()
    return (
        controller_socket,
        worker_socket,
        protocol.ProtocolSession(controller_socket, protocol.ROLE_CONTROLLER),
        protocol.ProtocolSession(worker_socket, protocol.ROLE_WORKER),
    )


def activate(
    controller: protocol.ProtocolSession, worker: protocol.ProtocolSession
) -> None:
    controller.send_hello(deadline=deadline())
    worker.receive_hello(deadline=deadline())
    worker.send_ready(deadline=deadline())
    controller.receive_ready(deadline=deadline())
    controller.send_gate(deadline=deadline())
    worker.receive_gate(deadline=deadline())


def raw_frame(
    frame_type: str,
    role: str,
    sequence: int,
    payload: bytes,
    *,
    magic: bytes = protocol.MAGIC,
    version: int = protocol.PROTOCOL_VERSION,
    flags: int = 0,
    declared_size: int | None = None,
) -> bytes:
    return protocol.HEADER.pack(
        magic,
        version,
        protocol.TYPE_TO_WIRE[frame_type],
        protocol.ROLE_TO_WIRE[role],
        flags,
        sequence,
        len(payload) if declared_size is None else declared_size,
    ) + payload


def send_raw_object(
    sock: socket.socket,
    *,
    kind: str,
    object_id: str,
    raw: bytes,
    digest: str | None = None,
    length: int | None = None,
    end: dict[str, object] | None = None,
) -> None:
    declared_length = len(raw) if length is None else length
    declared_digest = hashlib.sha256(raw).hexdigest() if digest is None else digest
    begin = protocol.canonical_json(
        {
            "chunks": 1,
            "kind": kind,
            "length": declared_length,
            "object_id": object_id,
            "sha256": declared_digest,
        },
        ceiling=protocol.MAX_FRAME_PAYLOAD,
    )
    terminal = protocol.canonical_json(
        end
        or {
            "length": declared_length,
            "object_id": object_id,
            "sha256": declared_digest,
        },
        ceiling=protocol.MAX_FRAME_PAYLOAD,
    )
    sock.sendall(raw_frame(protocol.TYPE_BEGIN, protocol.ROLE_CONTROLLER, 0, begin))
    sock.sendall(raw_frame(protocol.TYPE_CHUNK, protocol.ROLE_CONTROLLER, 1, raw))
    sock.sendall(raw_frame(protocol.TYPE_END, protocol.ROLE_CONTROLLER, 2, terminal))


def test_module_has_no_socket_process_network_or_filesystem_constructor() -> None:
    source = Path(protocol.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "subprocess" not in {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    forbidden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "socket":
            if node.func.attr in {"socket", "socketpair", "create_connection"}:
                forbidden.append(node.func.attr)
    assert forbidden == []
    assert "open(" not in source
    assert "Path(" not in source


def test_canonical_json_roundtrip_and_closed_value_space() -> None:
    value = {"accent": "École", "items": [None, True, 4, 2.5], "nested": {"a": "b"}}
    raw = protocol.canonical_json(value)
    assert raw == b'{"accent":"\xc3\x89cole","items":[null,true,4,2.5],"nested":{"a":"b"}}'
    assert protocol.decode_canonical_json(raw) == value
    with pytest.raises(protocol.CanonicalJSONError, match="non-JSON"):
        protocol.canonical_json(Path("/tmp/free-path"))
    with pytest.raises(protocol.CanonicalJSONError, match="safe range"):
        protocol.canonical_json(1 << 60)


@pytest.mark.parametrize(
    "raw",
    (
        b'{"a":1,"a":2}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1} ',
        b'{"b":1,"a":2}',
        b'{"x":"\xff"}',
        b'\xef\xbb\xbf{}',
        b'{"x":01}',
    ),
)
def test_decode_rejects_duplicate_nan_noncanonical_and_bad_utf8(raw: bytes) -> None:
    with pytest.raises(protocol.CanonicalJSONError):
        protocol.decode_canonical_json(raw)


def test_prescan_accepts_brackets_escaped_quotes_and_utf8_inside_strings() -> None:
    value = {"text": ("[\\\"]}" * 40) + " École 日本"}
    raw = protocol.canonical_json(value)
    assert protocol.decode_canonical_json(raw) == value


def test_streaming_utf8_validation_accepts_codepoint_split_between_chunks() -> None:
    raw = (
        b'"'
        + (b"a" * (protocol.MAX_FRAME_PAYLOAD - 2))
        + "É".encode("utf-8")
        + b'"'
    )
    assert protocol.decode_canonical_json(raw).endswith("É")


def test_encoder_accepts_depth_32_and_rejects_depth_33() -> None:
    value: object = 0
    for _ in range(protocol.MAX_JSON_DEPTH):
        value = [value]
    protocol.canonical_json(value)
    value = [value]
    with pytest.raises(protocol.CanonicalJSONError, match="nesting"):
        protocol.canonical_json(value)


@pytest.mark.parametrize(
    "raw, message",
    (
        (b"[" * 33 + b"0" + b"]" * 33, "nesting"),
        (b'{"x":"\\q"}', "escape"),
        (b"[0,]", "value token"),
    ),
)
def test_prescan_rejects_depth_and_structure_before_json_loads(
    raw: bytes, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_loads(*args, **kwargs):
        raise AssertionError("json.loads must not see structurally unsafe bytes")

    monkeypatch.setattr(protocol.json, "loads", forbidden_loads)
    with pytest.raises(protocol.CanonicalJSONError, match=message):
        protocol.decode_canonical_json(raw)


def test_prescan_rejects_million_depth_bomb_before_json_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"[" * 1_000_000 + b"0" + b"]" * 1_000_000

    def forbidden_loads(*args, **kwargs):
        raise AssertionError("json.loads must not see the depth bomb")

    monkeypatch.setattr(protocol.json, "loads", forbidden_loads)
    with pytest.raises(protocol.CanonicalJSONError, match="nesting"):
        protocol.decode_canonical_json(raw)


def test_prescan_rejects_node_bomb_before_json_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"[" + (b"0," * (protocol.MAX_JSON_NODES - 1)) + b"0]"

    def forbidden_loads(*args, **kwargs):
        raise AssertionError("json.loads must not see the node bomb")

    monkeypatch.setattr(protocol.json, "loads", forbidden_loads)
    with pytest.raises(protocol.CanonicalJSONError, match="node ceiling"):
        protocol.decode_canonical_json(raw)


def test_encoder_preflight_rejects_huge_string_before_iterencode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = "x" * (2 * 1024 * 1024)

    def forbidden_iterencode(*args, **kwargs):
        raise AssertionError("iterencode must not run after size preflight fails")

    monkeypatch.setattr(protocol.json.JSONEncoder, "iterencode", forbidden_iterencode)
    with pytest.raises(protocol.CanonicalJSONError, match="byte ceiling"):
        protocol.canonical_json(value, ceiling=1024)


@pytest.mark.parametrize("failure", (MemoryError, RecursionError))
def test_decoder_wraps_parser_resource_failures(
    failure: type[BaseException], monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_loads(*args, **kwargs):
        raise failure("injected parser resource failure")

    monkeypatch.setattr(protocol.json, "loads", failing_loads)
    with pytest.raises(protocol.CanonicalJSONError, match="invalid UTF-8 JSON"):
        protocol.decode_canonical_json(b"{}")


@pytest.mark.parametrize("failure", (MemoryError, RecursionError))
def test_encoder_wraps_iterencode_resource_failures(
    failure: type[BaseException], monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_iterencode(*args, **kwargs):
        raise failure("injected encoder resource failure")

    monkeypatch.setattr(protocol.json.JSONEncoder, "iterencode", failing_iterencode)
    with pytest.raises(protocol.CanonicalJSONError, match="cannot be encoded"):
        protocol.canonical_json({"safe": True})


def test_binary_frame_is_versioned_bounded_and_accepts_exact_ceiling() -> None:
    left, right = socket.socketpair()
    try:
        sender = protocol.FramedTransport(left, protocol.ROLE_CONTROLLER)
        receiver = protocol.FramedTransport(right, protocol.ROLE_WORKER)
        payload = b"x" * protocol.MAX_FRAME_PAYLOAD
        thread = threading.Thread(
            target=sender.send_frame,
            args=(protocol.TYPE_CHUNK, payload),
            kwargs={"deadline": deadline()},
        )
        thread.start()
        frame = receiver.receive_frame(deadline=deadline())
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert protocol.HEADER.size + len(frame.payload) == protocol.MAX_FRAME_BYTES
        assert frame.payload == payload and frame.sequence == 0
        with pytest.raises(protocol.ProtocolError, match="64 KiB"):
            sender.send_frame(
                protocol.TYPE_CHUNK, payload + b"x", deadline=deadline()
            )
    finally:
        left.close()
        right.close()


def test_fragmented_header_and_payload_are_reassembled() -> None:
    left, right = socket.socketpair()
    payload = protocol.canonical_json({"status": "READY"})
    raw = raw_frame(protocol.TYPE_READY, protocol.ROLE_WORKER, 0, payload)

    def fragment() -> None:
        for octet in raw:
            left.send(bytes((octet,)))

    thread = threading.Thread(target=fragment)
    thread.start()
    try:
        frame = protocol.FramedTransport(
            right, protocol.ROLE_CONTROLLER
        ).receive_frame(deadline=deadline())
        assert frame.frame_type == protocol.TYPE_READY
        assert frame.payload == payload
    finally:
        thread.join(timeout=2)
        left.close()
        right.close()


def test_absolute_deadline_defeats_slow_drip() -> None:
    left, right = socket.socketpair()
    raw = raw_frame(protocol.TYPE_READY, protocol.ROLE_WORKER, 0, b"{}")

    def drip() -> None:
        try:
            for octet in raw:
                left.send(bytes((octet,)))
                time.sleep(0.025)
        except OSError:
            pass

    thread = threading.Thread(target=drip, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        with pytest.raises(protocol.ProtocolTimeout):
            protocol.FramedTransport(
                right, protocol.ROLE_CONTROLLER
            ).receive_frame(deadline=started + 0.10)
        assert time.monotonic() - started < 0.25
    finally:
        left.close()
        right.close()
        thread.join(timeout=1)


@pytest.mark.parametrize("received_sequence", (1, 17, 0xFFFFFFFF))
def test_directional_sequence_rejects_gap_and_replay(received_sequence: int) -> None:
    left, right = socket.socketpair()
    try:
        left.sendall(
            raw_frame(
                protocol.TYPE_READY,
                protocol.ROLE_WORKER,
                received_sequence,
                b"{}",
            )
        )
        with pytest.raises(protocol.SequenceError, match="expected peer sequence 0"):
            protocol.FramedTransport(
                right, protocol.ROLE_CONTROLLER
            ).receive_frame(deadline=deadline())
    finally:
        left.close()
        right.close()


def test_directional_sequence_rejects_replay_after_valid_frame() -> None:
    left, right = socket.socketpair()
    try:
        receiver = protocol.FramedTransport(right, protocol.ROLE_CONTROLLER)
        valid = raw_frame(protocol.TYPE_READY, protocol.ROLE_WORKER, 0, b"{}")
        left.sendall(valid + valid)
        assert receiver.receive_frame(deadline=deadline()).sequence == 0
        with pytest.raises(protocol.SequenceError, match="expected peer sequence 1"):
            receiver.receive_frame(deadline=deadline())
    finally:
        left.close()
        right.close()


def test_two_directions_have_independent_strict_sequences() -> None:
    left, right = socket.socketpair()
    try:
        controller = protocol.FramedTransport(left, protocol.ROLE_CONTROLLER)
        worker = protocol.FramedTransport(right, protocol.ROLE_WORKER)
        controller.send_frame(protocol.TYPE_HELLO, b"{}", deadline=deadline())
        assert worker.receive_frame(deadline=deadline()).sequence == 0
        worker.send_frame(protocol.TYPE_READY, b"{}", deadline=deadline())
        assert controller.receive_frame(deadline=deadline()).sequence == 0
        controller.send_frame(protocol.TYPE_GATE, b"{}", deadline=deadline())
        assert worker.receive_frame(deadline=deadline()).sequence == 1
    finally:
        left.close()
        right.close()


def test_handshake_state_machine_and_closed_roles_states() -> None:
    left, right, controller, worker = sessions()
    try:
        assert protocol.ROLES == {protocol.ROLE_CONTROLLER, protocol.ROLE_WORKER}
        assert controller.state in protocol.STATES
        with pytest.raises(protocol.StateError):
            controller.send_gate(deadline=deadline())
        activate(controller, worker)
        with pytest.raises(protocol.StateError):
            worker.send_stop(deadline=deadline())
        controller.send_stop(deadline=deadline())
        worker.receive_stop(deadline=deadline())
        worker.send_complete(deadline=deadline())
        controller.receive_complete(deadline=deadline())
        assert controller.state == worker.state == protocol.STATE_COMPLETE
        with pytest.raises(protocol.StateError):
            controller.send_stop(deadline=deadline())
    finally:
        left.close()
        right.close()


def test_control_payload_rejects_numeric_equality_coercion() -> None:
    left, right = socket.socketpair()
    try:
        hello = protocol.canonical_json(
            {"protocol": protocol.PROTOCOL_NAME, "version": 1.0},
            ceiling=protocol.MAX_FRAME_PAYLOAD,
        )
        left.sendall(
            raw_frame(protocol.TYPE_HELLO, protocol.ROLE_CONTROLLER, 0, hello)
        )
        worker = protocol.ProtocolSession(right, protocol.ROLE_WORKER)
        with pytest.raises(protocol.StateError, match="invalid HELLO"):
            worker.receive_hello(deadline=deadline())
    finally:
        left.close()
        right.close()


def test_dto_candidate_batch_3000_and_result_roundtrip_without_pickle_or_path() -> None:
    left, right, controller, worker = sessions()
    try:
        activate(controller, worker)
        dto = {"query_id": "q-1", "crm_name": "ÉCOLE A", "postcode": "75001"}
        controller.send_object(protocol.KIND_DTO, "dto-1", dto, deadline=deadline())
        assert worker.receive_object(deadline=deadline()).value == dto

        candidates = [
            {"rank": index + 1, "score": index / 3000, "siret": f"{index:014d}"}
            for index in range(protocol.MAX_CANDIDATES)
        ]
        sent: list[protocol.ReceivedObject] = []

        def send_batch() -> None:
            sent.append(
                controller.send_object(
                    protocol.KIND_CANDIDATE_BATCH,
                    "batch-3000",
                    candidates,
                    deadline=deadline(5),
                )
            )

        thread = threading.Thread(target=send_batch)
        thread.start()
        received = worker.receive_object(deadline=deadline(5))
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert received.value == candidates
        assert received.length > protocol.MAX_FRAME_PAYLOAD
        assert received.sha256 == sent[0].sha256

        result = {"accepted": False, "reason": "REVIEW"}
        worker.send_object(protocol.KIND_RESULT, "result-1", result, deadline=deadline())
        assert controller.receive_object(deadline=deadline()).value == result
    finally:
        left.close()
        right.close()


def test_object_kinds_directions_batch_ceiling_and_identifier_are_closed() -> None:
    left, right, controller, worker = sessions()
    try:
        activate(controller, worker)
        with pytest.raises(protocol.StateError, match="cannot be sent"):
            controller.send_object(
                protocol.KIND_RESULT, "result", {}, deadline=deadline()
            )
        with pytest.raises(protocol.ProtocolError, match="unknown object kind"):
            controller.send_object("PICKLE", "payload", {}, deadline=deadline())
        with pytest.raises(protocol.ProtocolError, match="identifier grammar"):
            controller.send_object(
                protocol.KIND_DTO, "/tmp/free-path", {}, deadline=deadline()
            )
        with pytest.raises(protocol.ProtocolError, match="3000"):
            controller.send_object(
                protocol.KIND_CANDIDATE_BATCH,
                "too-many",
                [{}] * (protocol.MAX_CANDIDATES + 1),
                deadline=deadline(),
            )
    finally:
        left.close()
        right.close()


def test_non_string_object_kind_fails_with_typed_protocol_error() -> None:
    left, right = socket.socketpair()
    try:
        receiver = protocol.ProtocolSession(right, protocol.ROLE_WORKER)
        receiver.state = protocol.STATE_ACTIVE
        raw = b"{}"
        begin = protocol.canonical_json(
            {
                "chunks": 1,
                "kind": [],
                "length": len(raw),
                "object_id": "dto-1",
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            ceiling=protocol.MAX_FRAME_PAYLOAD,
        )
        left.sendall(raw_frame(protocol.TYPE_BEGIN, protocol.ROLE_CONTROLLER, 0, begin))
        with pytest.raises(protocol.StateError, match="invalid direction"):
            receiver.receive_object(deadline=deadline())
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize(
    "mutation, message",
    (
        ("hash", "SHA-256"),
        ("length", "CHUNK length"),
        ("end", "END does not match"),
    ),
)
def test_object_hash_length_and_end_mismatch_fail_closed(
    mutation: str, message: str
) -> None:
    left, right = socket.socketpair()
    try:
        receiver = protocol.ProtocolSession(right, protocol.ROLE_WORKER)
        receiver.state = protocol.STATE_ACTIVE
        raw = b'{"query_id":"q1"}'
        kwargs: dict[str, object] = {}
        if mutation == "hash":
            kwargs["digest"] = "0" * 64
        elif mutation == "length":
            kwargs["length"] = len(raw) + 1
        else:
            kwargs["end"] = {
                "length": len(raw),
                "object_id": "different",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        send_raw_object(
            left,
            kind=protocol.KIND_DTO,
            object_id="dto-1",
            raw=raw,
            **kwargs,
        )
        with pytest.raises(protocol.ProtocolError, match=message):
            receiver.receive_object(deadline=deadline())
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize(
    "raw",
    (b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":"\xff"}', b'{"x":1} '),
)
def test_invalid_json_inside_chunk_is_rejected_after_integrity_check(raw: bytes) -> None:
    left, right = socket.socketpair()
    try:
        receiver = protocol.ProtocolSession(right, protocol.ROLE_WORKER)
        receiver.state = protocol.STATE_ACTIVE
        send_raw_object(
            left, kind=protocol.KIND_DTO, object_id="dto-bad", raw=raw
        )
        with pytest.raises(protocol.CanonicalJSONError):
            receiver.receive_object(deadline=deadline())
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("cut", (0, 3, protocol.HEADER.size + 1))
def test_eof_and_truncated_frame_are_distinct(cut: int) -> None:
    left, right = socket.socketpair()
    transport = protocol.FramedTransport(right, protocol.ROLE_CONTROLLER)
    payload = b"{}"
    raw = raw_frame(protocol.TYPE_READY, protocol.ROLE_WORKER, 0, payload)
    try:
        left.sendall(raw[:cut])
        left.close()
        expected = protocol.ProtocolEOF if cut == 0 else protocol.ProtocolTruncated
        with pytest.raises(expected):
            transport.receive_frame(deadline=deadline())
    finally:
        right.close()


def test_declared_oversize_is_rejected_before_payload_read() -> None:
    left, right = socket.socketpair()
    try:
        left.sendall(
            raw_frame(
                protocol.TYPE_READY,
                protocol.ROLE_WORKER,
                0,
                b"",
                declared_size=protocol.MAX_FRAME_PAYLOAD + 1,
            )
        )
        with pytest.raises(protocol.ProtocolError, match="declared frame"):
            protocol.FramedTransport(
                right, protocol.ROLE_CONTROLLER
            ).receive_frame(deadline=deadline())
    finally:
        left.close()
        right.close()


def test_invalid_magic_version_role_flags_and_type_are_rejected() -> None:
    mutations = (
        {"magic": b"BAD!"},
        {"version": protocol.PROTOCOL_VERSION + 1},
        {"flags": 1},
    )
    for mutation in mutations:
        left, right = socket.socketpair()
        try:
            left.sendall(
                raw_frame(
                    protocol.TYPE_READY,
                    protocol.ROLE_WORKER,
                    0,
                    b"{}",
                    **mutation,
                )
            )
            with pytest.raises(protocol.ProtocolError):
                protocol.FramedTransport(
                    right, protocol.ROLE_CONTROLLER
                ).receive_frame(deadline=deadline())
        finally:
            left.close()
            right.close()

    for offset, bad_value in (
        (5, 255),
        (6, protocol.ROLE_TO_WIRE[protocol.ROLE_CONTROLLER]),
    ):
        left, right = socket.socketpair()
        try:
            header = bytearray(
                raw_frame(protocol.TYPE_READY, protocol.ROLE_WORKER, 0, b"{}")
            )
            header[offset] = bad_value
            left.sendall(header)
            with pytest.raises(protocol.ProtocolError):
                protocol.FramedTransport(
                    right, protocol.ROLE_CONTROLLER
                ).receive_frame(deadline=deadline())
        finally:
            left.close()
            right.close()


def test_receive_object_rejects_mid_object_order_violation() -> None:
    left, right = socket.socketpair()
    try:
        receiver = protocol.ProtocolSession(right, protocol.ROLE_WORKER)
        receiver.state = protocol.STATE_ACTIVE
        raw = b"{}"
        digest = hashlib.sha256(raw).hexdigest()
        begin = protocol.canonical_json(
            {
                "chunks": 1,
                "kind": protocol.KIND_DTO,
                "length": len(raw),
                "object_id": "dto-1",
                "sha256": digest,
            },
            ceiling=protocol.MAX_FRAME_PAYLOAD,
        )
        left.sendall(raw_frame(protocol.TYPE_BEGIN, protocol.ROLE_CONTROLLER, 0, begin))
        left.sendall(raw_frame(protocol.TYPE_END, protocol.ROLE_CONTROLLER, 1, b"{}"))
        with pytest.raises(protocol.StateError, match="expected CHUNK"):
            receiver.receive_object(deadline=deadline())
    finally:
        left.close()
        right.close()


def test_send_backpressure_obeys_absolute_deadline() -> None:
    left, right = socket.socketpair()
    try:
        left.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
        transport = protocol.FramedTransport(left, protocol.ROLE_CONTROLLER)
        started = time.monotonic()
        with pytest.raises(protocol.ProtocolTimeout):
            transport.send_frame(
                protocol.TYPE_CHUNK,
                b"x" * protocol.MAX_FRAME_PAYLOAD,
                deadline=started + 0.08,
            )
        assert time.monotonic() - started < 0.25
        assert transport.tx_sequence == 0
    finally:
        left.close()
        right.close()


def test_outbound_sequence_space_exhaustion_is_typed_and_sends_nothing() -> None:
    left, right = socket.socketpair()
    try:
        transport = protocol.FramedTransport(left, protocol.ROLE_CONTROLLER)
        transport.tx_sequence = protocol.MAX_SEQUENCE + 1
        with pytest.raises(protocol.SequenceError, match="exhausted"):
            transport.send_frame(protocol.TYPE_HELLO, b"{}", deadline=deadline())
        right.setblocking(False)
        with pytest.raises(BlockingIOError):
            right.recv(1)
    finally:
        left.close()
        right.close()


def test_send_object_reserves_whole_sequence_range_before_first_byte() -> None:
    left, right = socket.socketpair()
    try:
        session = protocol.ProtocolSession(left, protocol.ROLE_CONTROLLER)
        session.state = protocol.STATE_ACTIVE
        session.transport.tx_sequence = protocol.MAX_SEQUENCE
        before = session.transport.tx_sequence
        with pytest.raises(protocol.SequenceError, match="whole message"):
            session.send_object(
                protocol.KIND_DTO,
                "dto-at-sequence-edge",
                {"query_id": "q1"},
                deadline=deadline(),
            )
        assert session.transport.tx_sequence == before
        right.setblocking(False)
        with pytest.raises(BlockingIOError):
            right.recv(1)
    finally:
        left.close()
        right.close()


def test_expired_and_relative_looking_deadlines_fail_before_io() -> None:
    left, right = socket.socketpair()
    try:
        transport = protocol.FramedTransport(left, protocol.ROLE_CONTROLLER)
        with pytest.raises(protocol.ProtocolTimeout):
            transport.send_frame(protocol.TYPE_HELLO, b"{}", deadline=0.1)
        with pytest.raises(protocol.ProtocolTimeout):
            protocol.FramedTransport(
                right, protocol.ROLE_WORKER
            ).receive_frame(deadline=float("nan"))
    finally:
        left.close()
        right.close()
