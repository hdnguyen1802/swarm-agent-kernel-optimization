"""Small checked packets for the persistent SSH stream."""

import base64
import hashlib
import re
import zlib
from pathlib import Path, PurePosixPath

VERSION = 1
MAX_FRAME = 64 * 1024 * 1024
MAX_FILE = 32 * 1024 * 1024


def workspace_file(root, relative):
    path = PurePosixPath(relative)
    if (path.is_absolute() or not path.parts or path.parts[0] != "runs"
            or ".." in path.parts or not re.fullmatch(r"[A-Za-z0-9_./-]+", relative)):
        raise ValueError("Session files must stay inside runs/.")
    target = (Path(root) / relative).resolve()
    target.relative_to(Path(root).resolve())
    return target


def pack(data):
    if len(data) > MAX_FILE:
        raise ValueError("The session file exceeds its byte limit.")
    encoded = zlib.compress(data, 1) if len(data) >= 8192 else data
    compressed = len(encoded) < len(data)
    return {"data": base64.b64encode(encoded if compressed else data).decode("ascii"),
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            **({"encoding":"zlib"} if compressed else {})}


def unpack(blob):
    if not isinstance(blob, dict) or type(blob.get("bytes")) is not int or not 0 <= blob["bytes"] <= MAX_FILE:
        raise ValueError("Invalid session file size.")
    data = base64.b64decode(blob["data"], validate=True)
    if blob.get("encoding") == "zlib":
        decoder = zlib.decompressobj()
        try:
            data = decoder.decompress(data, blob["bytes"] + 1)
        except zlib.error as error:
            raise ValueError("Invalid compressed session file.") from error
        if not decoder.eof or decoder.unused_data:
            raise ValueError("Invalid compressed session file length.")
    elif "encoding" in blob:
        raise ValueError("Unknown session file encoding.")
    if len(data) != blob["bytes"] or hashlib.sha256(data).hexdigest() != blob.get("sha256"):
        raise ValueError("The session file hash differs.")
    return data


def publish(path, data):
    import uuid
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)
