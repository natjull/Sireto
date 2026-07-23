"""Subprocess-isolated semantic encoder for macOS OpenMP compatibility.

The encoder is launched with ``python -m`` instead of ``multiprocessing``.
With the ``spawn`` multiprocessing context, Python re-imports the caller's
main module in the child.  A caller that imports FAISS at module level would
therefore load FAISS before Torch in the supposedly isolated process and
recreate the duplicate-OpenMP crash this class exists to prevent.
"""

from __future__ import annotations

import argparse
import atexit
import os
import socket
import subprocess
import sys
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _encoder_worker(
    connection: Connection,
    model_path: str,
    device: str,
) -> None:
    try:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()

        from .semantic import _repair_exported_tokenizer, assert_tokenizer_healthy

        encoder = SentenceTransformer(model_path, device=device)
        _repair_exported_tokenizer(encoder, model_path)
        assert_tokenizer_healthy(encoder.tokenizer)
        connection.send(
            {
                "status": "ready",
                "torch": torch.__version__,
                "mps_built": bool(torch.backends.mps.is_built()),
                "mps_available": bool(torch.backends.mps.is_available()),
                "device": device,
            }
        )
        while True:
            request = connection.recv()
            command = request.get("command")
            if command == "close":
                break
            if command != "encode":
                raise ValueError(f"Unsupported semantic worker command: {command}")
            texts = list(request["texts"])
            vectors = encoder.encode(
                texts,
                batch_size=int(request["batch_size"]),
                convert_to_numpy=True,
                normalize_embeddings=bool(request["normalize_embeddings"]),
                show_progress_bar=False,
            ).astype(np.float32, copy=False)
            connection.send({"status": "ok", "vectors": vectors})
    except BaseException as error:
        try:
            connection.send(
                {
                    "status": "error",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
        except BaseException:
            pass
    finally:
        connection.close()


class SemanticProcessClient:
    """SentenceTransformer-compatible client backed by a clean subprocess."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        startup_timeout_seconds: float = 120.0,
    ) -> None:
        if os.name != "posix":
            raise RuntimeError(
                "SemanticProcessClient currently requires a POSIX socketpair"
            )
        self._closed = False
        parent_socket, child_socket = socket.socketpair()
        child_fd = child_socket.fileno()
        project_root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{project_root}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(project_root)
        )
        command = [
            sys.executable,
            "-m",
            "src.xgb_matcher.semantic_process",
            "--worker-fd",
            str(child_fd),
            "--model",
            str(model_path),
            "--device",
            device,
        ]
        try:
            self._process = subprocess.Popen(
                command,
                pass_fds=(child_fd,),
                env=environment,
                cwd=str(project_root),
            )
        except BaseException:
            parent_socket.close()
            child_socket.close()
            raise
        child_socket.close()
        self._connection = Connection(parent_socket.detach())
        if not self._connection.poll(startup_timeout_seconds):
            self.close(force=True)
            raise TimeoutError("Semantic encoder worker startup timed out")
        ready = self._connection.recv()
        if ready.get("status") != "ready":
            self.close(force=True)
            raise RuntimeError(
                "Semantic encoder worker failed: "
                f"{ready.get('type')}: {ready.get('message')}"
            )
        self.runtime_info = dict(ready)
        atexit.register(self.close)

    def encode(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 128,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        **_ignored: Any,
    ) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Semantic encoder worker is closed")
        if not convert_to_numpy:
            raise ValueError("SemanticProcessClient only supports NumPy output")
        materialized = list(texts)
        chunks: list[np.ndarray] = []
        for start in range(0, len(materialized), 2048):
            self._connection.send(
                {
                    "command": "encode",
                    "texts": materialized[start : start + 2048],
                    "batch_size": batch_size,
                    "normalize_embeddings": normalize_embeddings,
                }
            )
            try:
                response = self._connection.recv()
            except EOFError as error:
                raise RuntimeError(
                    "Semantic encoder worker exited unexpectedly "
                    f"(exitcode={self._process.poll()})"
                ) from error
            if response.get("status") != "ok":
                raise RuntimeError(
                    "Semantic encoder worker failed: "
                    f"{response.get('type')}: {response.get('message')}"
                )
            chunks.append(np.asarray(response["vectors"], dtype=np.float32))
        if not chunks:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def close(self, force: bool = False) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        process = getattr(self, "_process", None)
        connection = getattr(self, "_connection", None)
        if process is not None and process.poll() is None and not force:
            try:
                connection.send({"command": "close"})
                process.wait(timeout=10)
            except (BrokenPipeError, EOFError, OSError):
                pass
            except subprocess.TimeoutExpired:
                pass
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if connection is not None:
            connection.close()

    def __enter__(self) -> "SemanticProcessClient":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


__all__ = ["SemanticProcessClient"]


def _worker_main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker-fd", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    _encoder_worker(
        Connection(args.worker_fd),
        model_path=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    _worker_main()
