import base64
import json
import os
import re
import tempfile
from pathlib import Path

import blake3

CHUNK = 256 * 1024
MAX_BLOB = 64 * 1024**3


def digest(data):
    return blake3.blake3(data).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("invalid content identifier")
    return value


def safe_path(root, name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("invalid relative path")
    p = Path(name)
    if p.is_absolute() or any(part in ("..", ".") for part in p.parts):
        raise ValueError("path traversal refused")
    target = (root / p).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("path escapes workspace")
    return target


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path):
    return json.loads(path.read_text())


def blob_path(root, value):
    return root / "blobs" / identifier(value)


def write_chunk(root, request):
    path = blob_path(root, request["hash"])
    path.parent.mkdir(parents=True, exist_ok=True)
    total, offset = request["size"], request["offset"]
    data = base64.b64decode(request["data"], validate=True)
    if not isinstance(total, int) or not 0 <= total <= MAX_BLOB or not isinstance(offset, int):
        raise ValueError("invalid size or offset")
    if not 0 <= offset <= total or offset + len(data) > total or len(data) > CHUNK:
        raise ValueError("invalid chunk bounds")
    if path.exists():
        return {"offset": total, "complete": True}
    partial = path.with_suffix(".partial")
    actual = partial.stat().st_size if partial.exists() else 0
    if offset != actual:
        return {"offset": actual, "complete": False}
    with partial.open("ab") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    actual += len(data)
    if actual == total:
        with partial.open("rb") as f:
            h = blake3.blake3()
            while chunk := f.read(CHUNK):
                h.update(chunk)
        if h.hexdigest() != request["hash"]:
            partial.unlink()
            raise ValueError("content hash mismatch")
        os.replace(partial, path)
    return {"offset": actual, "complete": path.exists()}


def read_chunk(path, offset):
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("invalid offset")
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read(CHUNK)
    return {"data": base64.b64encode(data).decode(), "size": path.stat().st_size}


class FileLock:
    def __init__(self, path, blocking=True):
        self.path, self.blocking = path, blocking

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        if os.name == "nt":
            import msvcrt
            self.file.seek(0)
            self.file.write(b"0")
            self.file.flush()
            self.file.seek(0)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_LOCK if self.blocking else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file, fcntl.LOCK_EX | (0 if self.blocking else fcntl.LOCK_NB))
        return self

    def __exit__(self, *args):
        if os.name == "nt":
            import msvcrt
            self.file.seek(0)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()
