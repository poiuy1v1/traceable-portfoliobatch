from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Non-finite JSON value: {value!r}")
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json_text(payload: Any) -> str:
    """Stable, human-readable JSON used for committed scientific artifacts."""
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ) + "\n"


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace a generated file through a same-directory temporary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                temporary = None
                return
            except OSError:
                if attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if temporary is not None and temporary.is_file():
            temporary.unlink()


def write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, canonical_json_text(payload).encode("utf-8"))


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic CSV with stable newlines and float rendering."""
    buffer = io.StringIO(newline="")
    frame.to_csv(
        buffer,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
        na_rep="",
    )
    atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Return the Git SHA-1 blob object ID, including the Git object header."""
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()
