"""Run FAISS index construction in a clean subprocess.

PyArrow and the Homebrew FAISS wheel initialize incompatible copies of
``libomp`` on the target Mac once they execute native kernels.  Candidate
loading therefore stays in the parent process while FAISS reads a temporary
NumPy array in a dedicated interpreter.
"""

from __future__ import annotations

import argparse
import atexit
import os
import socket
import subprocess
import sys
import tempfile
from collections import OrderedDict
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np


def build_faiss_index_isolated(
    embeddings: np.ndarray,
    output_path: str | Path,
) -> None:
    """Build one immutable FAISS index without loading FAISS in the caller."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(project_root)
    )

    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=".npy",
        delete=False,
    ) as vector_handle:
        vector_path = Path(vector_handle.name)
        np.save(vector_handle, np.asarray(embeddings, dtype=np.float32))
    partial_path = destination.with_name(
        f".{destination.name}.partial.{os.getpid()}"
    )
    command = [
        sys.executable,
        "-m",
        "src.xgb_matcher.faiss_process",
        "--embeddings",
        str(vector_path),
        "--output",
        str(partial_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            env=environment,
            cwd=str(project_root),
        )
        os.replace(partial_path, destination)
    finally:
        vector_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)


def build_faiss_index_file_isolated(
    embeddings_path: str | Path,
    output_path: str | Path,
    *,
    dimension: int | None = None,
    nlist: int | None = None,
    pq_subquantizers: int = 48,
    training_rows: int = 100_000,
    add_batch_size: int = 100_000,
) -> None:
    """Build an index from an existing NPY or raw float32 matrix file."""
    source = Path(embeddings_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(project_root)
    )
    partial_path = destination.with_name(
        f".{destination.name}.partial.{os.getpid()}"
    )
    command = [
        sys.executable,
        "-m",
        "src.xgb_matcher.faiss_process",
        "--embeddings",
        str(source),
        "--output",
        str(partial_path),
        "--pq-subquantizers",
        str(pq_subquantizers),
        "--training-rows",
        str(training_rows),
        "--add-batch-size",
        str(add_batch_size),
    ]
    if dimension is not None:
        command.extend(["--raw-dimension", str(dimension)])
    if nlist is not None:
        command.extend(["--nlist", str(nlist)])
    try:
        subprocess.run(
            command,
            check=True,
            env=environment,
            cwd=str(project_root),
        )
        os.replace(partial_path, destination)
    finally:
        partial_path.unlink(missing_ok=True)


def _search_worker(connection: Connection, max_cache: int) -> None:
    try:
        import faiss

        indexes: OrderedDict[str, object] = OrderedDict()
        connection.send(
            {
                "status": "ready",
                "faiss": getattr(faiss, "__version__", "unknown"),
            }
        )
        while True:
            request = connection.recv()
            command = request.get("command")
            if command == "close":
                break
            path = str(Path(request["path"]).resolve())
            index = indexes.get(path)
            if index is None:
                index = faiss.read_index(path)
                indexes[path] = index
                while len(indexes) > max_cache:
                    indexes.popitem(last=False)
            else:
                indexes.move_to_end(path)
            if command == "describe":
                connection.send(
                    {
                        "status": "ok",
                        "ntotal": int(index.ntotal),
                        "dimension": int(index.d),
                    }
                )
                continue
            if command != "search":
                raise ValueError(f"Unsupported FAISS worker command: {command}")
            query = np.asarray(request["query"], dtype=np.float32)
            if query.ndim == 1:
                query = query.reshape(1, -1)
            k = min(int(request["k"]), int(index.ntotal))
            scores, indices = index.search(query, k)
            connection.send(
                {
                    "status": "ok",
                    "scores": scores[0],
                    "indices": indices[0],
                }
            )
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


class FaissSearchProcessClient:
    """Persistent FAISS search service that never imports FAISS in its caller."""

    def __init__(
        self,
        *,
        max_cache: int = 20,
        startup_timeout_seconds: float = 30.0,
    ) -> None:
        if os.name != "posix":
            raise RuntimeError(
                "FaissSearchProcessClient currently requires a POSIX socketpair"
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
            "src.xgb_matcher.faiss_process",
            "--search-worker-fd",
            str(child_fd),
            "--max-cache",
            str(max_cache),
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
            raise TimeoutError("FAISS search worker startup timed out")
        ready = self._connection.recv()
        if ready.get("status") != "ready":
            self.close(force=True)
            raise RuntimeError(
                "FAISS search worker failed: "
                f"{ready.get('type')}: {ready.get('message')}"
            )
        self.runtime_info = dict(ready)
        atexit.register(self.close)

    def _request(self, payload: dict) -> dict:
        if self._closed:
            raise RuntimeError("FAISS search worker is closed")
        self._connection.send(payload)
        try:
            response = self._connection.recv()
        except EOFError as error:
            raise RuntimeError(
                "FAISS search worker exited unexpectedly "
                f"(exitcode={self._process.poll()})"
            ) from error
        if response.get("status") != "ok":
            raise RuntimeError(
                "FAISS search worker failed: "
                f"{response.get('type')}: {response.get('message')}"
            )
        return response

    def describe(self, path: str | Path) -> tuple[int, int]:
        response = self._request(
            {"command": "describe", "path": str(Path(path).resolve())}
        )
        return int(response["ntotal"]), int(response["dimension"])

    def search(
        self,
        path: str | Path,
        query: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        response = self._request(
            {
                "command": "search",
                "path": str(Path(path).resolve()),
                "query": np.asarray(query, dtype=np.float32),
                "k": int(k),
            }
        )
        return (
            np.asarray(response["scores"], dtype=np.float32),
            np.asarray(response["indices"], dtype=np.int64),
        )

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


def _worker_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--search-worker-fd", type=int)
    parser.add_argument("--max-cache", type=int, default=20)
    parser.add_argument("--raw-dimension", type=int)
    parser.add_argument("--nlist", type=int)
    parser.add_argument("--pq-subquantizers", type=int, default=48)
    parser.add_argument("--training-rows", type=int, default=100_000)
    parser.add_argument("--add-batch-size", type=int, default=100_000)
    args = parser.parse_args()

    if args.search_worker_fd is not None:
        _search_worker(Connection(args.search_worker_fd), args.max_cache)
        return
    if args.embeddings is None or args.output is None:
        parser.error("--embeddings and --output are required in build mode")

    if args.raw_dimension is not None:
        if args.raw_dimension <= 0:
            raise ValueError("--raw-dimension must be positive")
        value_count = args.embeddings.stat().st_size // np.dtype(np.float32).itemsize
        if value_count % args.raw_dimension:
            raise ValueError("Raw embedding file size is not divisible by dimension")
        embeddings = np.memmap(
            args.embeddings,
            dtype=np.float32,
            mode="r",
            shape=(value_count // args.raw_dimension, args.raw_dimension),
        )
    else:
        embeddings = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2 or not len(embeddings):
        raise ValueError("Embeddings must be a non-empty 2D array")
    if args.nlist is None:
        # Imported only in this process: the caller is allowed to keep PyArrow.
        from .dense_retrieval import DenseIndex

        index = DenseIndex(embeddings)
        index.save(args.output)
        return

    import faiss

    dimension = int(embeddings.shape[1])
    if dimension % args.pq_subquantizers:
        raise ValueError(
            f"Embedding dimension {dimension} is not divisible by "
            f"pq-subquantizers={args.pq_subquantizers}"
        )
    training_count = min(args.training_rows, len(embeddings))
    if training_count < 256:
        raise ValueError("At least 256 training vectors are required for IVFPQ")
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        args.nlist,
        args.pq_subquantizers,
        8,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.train(
        np.ascontiguousarray(
            embeddings[:training_count],
            dtype=np.float32,
        )
    )
    index.nprobe = min(32, args.nlist)
    for start in range(0, len(embeddings), args.add_batch_size):
        index.add(
            np.ascontiguousarray(
                embeddings[start : start + args.add_batch_size],
                dtype=np.float32,
            )
        )
    faiss.write_index(index, str(args.output))


if __name__ == "__main__":
    _worker_main()
