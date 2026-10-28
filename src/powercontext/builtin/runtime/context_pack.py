# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context Pack and delta encoding for incremental context transport.

A Context Pack carries the prepared context body as a deterministic sequence of
byte chunks plus a digest over those chunks. A delta names how much of a known
base pack the receiver may keep and supplies the remaining suffix, so a client
that already holds one pack can receive only the difference for the next one.

The wire form is an interoperability contract: a receiver must be able to
rebuild the exact body from the serialized bytes alone, without sharing any
process state with the sender.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from powercontext.errors import PowerContextError

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512


class ContextPackError(PowerContextError, ValueError):
    """Raised when a Context Pack or delta violates the wire contract."""

    def __init__(self, code: str) -> None:
        self.code = code
        messages = {
            "base-digest-mismatch": "delta does not apply to this base pack",
            "byte-count-mismatch": "pack byte_count does not match its wire size",
            "chunk-too-small": "chunk_bytes is too small to hold one character",
            "delta-prefix-overflow": "delta prefix_items exceeds the base pack",
            "digest-mismatch": "pack digest does not match its items",
            "invalid-base64": "items contains invalid Base64",
            "invalid-byte-count": "byte_count must be a non-negative integer",
            "invalid-chunk-bytes": "chunk_bytes must be a positive integer",
            "invalid-digest": "digest must be a string",
            "invalid-json": "wire payload is not valid UTF-8 JSON",
            "invalid-prefix": "prefix_items must be a non-negative integer",
            "invalid-schema": "unsupported schema",
            "missing-fields": "wire payload is missing fields",
            "not-an-object": "wire payload must be a JSON object",
            "not-bytes": "pack items must be bytes",
            "not-text": "content must be text",
            "target-digest-mismatch": "delta does not rebuild the target pack",
            "unexpected-fields": "wire payload has unexpected fields",
        }
        super().__init__(messages.get(code, f"invalid Context Pack: {code}"))


def _canonical(payload: dict[str, Any]) -> bytes:
    """Serialize one wire payload the way the digest and the wire bytes require."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_items(items: Sequence[bytes]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode_items(values: object) -> tuple[bytes, ...]:
    if not isinstance(values, list):
        raise ContextPackError("not-an-object")
    decoded: list[bytes] = []
    for value in values:
        if not isinstance(value, str):
            raise ContextPackError("not-an-object")
        try:
            decoded.append(base64.b64decode(value.encode("ascii"), validate=True))
        except (binascii.Error, UnicodeEncodeError, ValueError) as error:
            raise ContextPackError("invalid-base64") from error
    return tuple(decoded)


def _require_mapping(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ContextPackError("not-an-object")
    return payload


def _require_exact_fields(payload: dict[str, Any], expected: set[str]) -> None:
    present = set(payload)
    missing = expected - present
    if missing:
        raise ContextPackError("missing-fields")
    extra = present - expected
    if extra:
        raise ContextPackError("unexpected-fields")


def _digest_of(items: Sequence[bytes]) -> str:
    """Digest the unsigned pack body, exactly as the wire contract defines it."""

    canonical = _canonical({"schema": PACK_SCHEMA, "items": _encode_items(items)})
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class ContextPack:
    """One complete Context Pack: the body as chunks, plus its digest."""

    items: tuple[bytes, ...]
    digest: str

    @classmethod
    def build(cls, items: Iterable[bytes]) -> ContextPack:
        """Build a complete pack from the given byte chunks."""

        normalized = tuple(items)
        for item in normalized:
            if not isinstance(item, bytes):
                raise ContextPackError("not-bytes")
        return cls(items=normalized, digest=_digest_of(normalized))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Split content into chunks that each stay within the byte limit.

        Every chunk must remain valid UTF-8 on its own, so a chunk boundary never
        falls inside a multi-byte character. When the limit cannot hold even one
        character of the content, no legal boundary exists and the call fails.
        """

        if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool):
            raise ContextPackError("invalid-chunk-bytes")
        if chunk_bytes <= 0:
            raise ContextPackError("invalid-chunk-bytes")
        if not isinstance(content, str):
            raise ContextPackError("not-text")
        if not content:
            return cls.build(())

        raw = content.encode("utf-8")
        return cls.build(tuple(_split_bytes(raw, chunk_bytes)))

    def to_bytes(self) -> bytes:
        """Serialize the pack into its wire form."""

        payload = {
            "schema": PACK_SCHEMA,
            "items": _encode_items(self.items),
            "digest": self.digest,
            "byte_count": 0,
        }
        # The byte count covers the whole wire form including the field itself,
        # so grow the number until the encoded length stops changing.
        for _ in range(8):
            encoded = _canonical(payload)
            size = len(encoded)
            if payload["byte_count"] == size:
                return encoded
            payload["byte_count"] = size
        raise ContextPackError("byte-count-mismatch")

    @property
    def byte_count(self) -> int:
        """The exact wire size of this pack, including the byte_count field itself."""

        return len(self.to_bytes())

    @classmethod
    def from_bytes(cls, raw: bytes) -> ContextPack:
        """Decode and validate a serialized pack."""

        payload = _require_mapping(_load_json(raw))
        _require_exact_fields(payload, {"schema", "items", "digest", "byte_count"})
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError("invalid-schema")
        items = _decode_items(payload["items"])
        digest = payload["digest"]
        if not isinstance(digest, str):
            raise ContextPackError("invalid-digest")
        if digest != _digest_of(items):
            raise ContextPackError("digest-mismatch")
        byte_count = payload["byte_count"]
        if type(byte_count) is not int or byte_count < 0:
            raise ContextPackError("invalid-byte-count")
        pack = cls(items=items, digest=digest)
        if pack.byte_count != byte_count:
            raise ContextPackError("byte-count-mismatch")
        return pack


@dataclass(frozen=True)
class ContextPackDelta:
    """The difference between a known base pack and the target pack."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Choose a valid common prefix and carry the target suffix."""

        shared = 0
        limit = min(len(base.items), len(target.items))
        while shared < limit and base.items[shared] == target.items[shared]:
            shared += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=shared,
            appended=target.items[shared:],
        )

    def to_bytes(self) -> bytes:
        """Serialize the delta into its wire form."""

        return _canonical({
            "schema": DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encode_items(self.appended),
        })

    @property
    def byte_count(self) -> int:
        """The exact wire size of this delta."""

        return len(self.to_bytes())

    @classmethod
    def from_bytes(cls, raw: bytes) -> ContextPackDelta:
        """Decode and validate a serialized delta."""

        payload = _require_mapping(_load_json(raw))
        _require_exact_fields(payload, {"schema", "base_digest", "target_digest", "prefix_items", "appended"})
        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError("invalid-schema")
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        if not isinstance(base_digest, str) or not isinstance(target_digest, str):
            raise ContextPackError("invalid-digest")
        prefix_items = payload["prefix_items"]
        if type(prefix_items) is not int or prefix_items < 0:
            raise ContextPackError("invalid-prefix")
        appended = _decode_items(payload["appended"])
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix_items,
            appended=appended,
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from a base pack and a delta."""

    if delta.base_digest != base.digest:
        raise ContextPackError("base-digest-mismatch")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta-prefix-overflow")
    items = base.items[: delta.prefix_items] + delta.appended
    target = ContextPack.build(items)
    if target.digest != delta.target_digest:
        raise ContextPackError("target-digest-mismatch")
    return target


def _split_bytes(raw: bytes, chunk_bytes: int) -> list[bytes]:
    """Split raw bytes into chunks that are each valid UTF-8 and within the limit.

    A chunk boundary never falls inside a multi-byte character, so the chunks
    can be decoded on their own and joined back into the original text.
    """

    chunks: list[bytes] = []
    start = 0
    while start < len(raw):
        end = min(start + chunk_bytes, len(raw))
        # Pull the boundary back to the last character that fits entirely.
        while end > start:
            try:
                raw[start:end].decode("utf-8")
            except UnicodeDecodeError:
                end -= 1
                continue
            break
        else:
            raise ContextPackError("chunk-too-small")
        chunks.append(raw[start:end])
        start = end
    return chunks


def _load_json(raw: bytes) -> Any:
    if not isinstance(raw, bytes):
        raise ContextPackError("not-bytes")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError("invalid-json") from error
