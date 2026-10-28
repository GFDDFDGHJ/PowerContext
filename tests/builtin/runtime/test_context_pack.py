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

"""Tests for the Context Pack wire protocol."""

from __future__ import annotations

import base64
import hashlib
import json

import pytest

from powercontext.builtin.runtime.context_pack import (
    ContextPack,
    ContextPackDelta,
    ContextPackError,
    apply_context_pack_delta,
)


def _canonical(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


class TestContextPackBuild:
    def test_digest_covers_the_unsigned_body(self) -> None:
        items = (b"alpha\n", "海洋".encode())
        unsigned = {
            "schema": "powercontext.context-pack.v1",
            "items": [base64.b64encode(item).decode("ascii") for item in items],
        }
        expected = "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()
        assert ContextPack.build(items).digest == expected

    def test_empty_pack_is_valid(self) -> None:
        empty = ContextPack.build(())
        assert empty.items == ()
        assert ContextPack.from_bytes(empty.to_bytes()) == empty

    def test_rejects_non_bytes_items(self) -> None:
        with pytest.raises(ContextPackError):
            ContextPack.build(("text",))


class TestContextPackFromContent:
    def test_reconstructs_the_original_text(self) -> None:
        pack = ContextPack.from_content("alpha 海洋 beta", chunk_bytes=7)
        assert b"".join(pack.items).decode("utf-8") == "alpha 海洋 beta"

    def test_every_chunk_is_valid_utf8(self) -> None:
        pack = ContextPack.from_content("海洋海洋海洋", chunk_bytes=5)
        for item in pack.items:
            item.decode("utf-8")

    def test_no_chunk_exceeds_the_limit(self) -> None:
        pack = ContextPack.from_content("海洋海洋海洋海洋", chunk_bytes=5)
        assert all(len(item) <= 5 for item in pack.items)

    def test_rejects_a_limit_too_small_for_one_character(self) -> None:
        with pytest.raises(ContextPackError):
            ContextPack.from_content("海", chunk_bytes=2)

    def test_rejects_a_non_positive_limit(self) -> None:
        with pytest.raises(ContextPackError):
            ContextPack.from_content("text", chunk_bytes=0)

    def test_rejects_a_boolean_limit(self) -> None:
        with pytest.raises(ContextPackError):
            ContextPack.from_content("text", chunk_bytes=True)

    def test_empty_content_gives_an_empty_pack(self) -> None:
        assert ContextPack.from_content("").items == ()


class TestContextPackWire:
    def test_round_trip_preserves_the_bytes(self) -> None:
        pack = ContextPack.build((b"alpha", b"beta"))
        assert ContextPack.from_bytes(pack.to_bytes()).to_bytes() == pack.to_bytes()

    def test_byte_count_matches_the_wire_size(self) -> None:
        pack = ContextPack.build((b"alpha",))
        assert pack.byte_count == len(pack.to_bytes())

    def test_rejects_a_tampered_digest(self) -> None:
        pack = ContextPack.build((b"alpha",))
        broken = pack.to_bytes().replace(pack.digest.encode(), b"sha256:" + b"0" * 64)
        with pytest.raises(ContextPackError):
            ContextPack.from_bytes(broken)

    @pytest.mark.parametrize("field", ["schema", "items", "digest", "byte_count"])
    def test_rejects_a_missing_field(self, field: str) -> None:
        pack = ContextPack.build((b"alpha",))
        incomplete = {key: value for key, value in json.loads(pack.to_bytes()).items() if key != field}
        with pytest.raises(ContextPackError):
            ContextPack.from_bytes(json.dumps(incomplete).encode())

    def test_rejects_an_extra_field(self) -> None:
        pack = ContextPack.build((b"alpha",))
        payload = {**json.loads(pack.to_bytes()), "extra": 1}
        with pytest.raises(ContextPackError):
            ContextPack.from_bytes(json.dumps(payload).encode())

    def test_rejects_a_wrong_byte_count(self) -> None:
        pack = ContextPack.build((b"alpha",))
        payload = {**json.loads(pack.to_bytes()), "byte_count": pack.byte_count + 1}
        with pytest.raises(ContextPackError):
            ContextPack.from_bytes(json.dumps(payload).encode())

    def test_rejects_invalid_json(self) -> None:
        with pytest.raises(ContextPackError):
            ContextPack.from_bytes(b"not json")


class TestContextPackDelta:
    def test_carries_the_target_suffix(self) -> None:
        base = ContextPack.build((b"alpha", b"beta"))
        target = ContextPack.build((b"alpha", b"beta", b"gamma"))
        delta = ContextPackDelta.between(base, target)
        assert delta.prefix_items == 2
        assert delta.appended == (b"gamma",)

    def test_round_trip_rebuilds_the_target(self) -> None:
        base = ContextPack.build((b"alpha", b"beta"))
        target = ContextPack.build((b"alpha", b"beta", b"gamma"))
        delta = ContextPackDelta.between(base, target)
        assert apply_context_pack_delta(base, delta).to_bytes() == target.to_bytes()

    def test_byte_count_is_not_a_wire_field(self) -> None:
        base = ContextPack.build((b"alpha",))
        delta = ContextPackDelta.between(base, ContextPack.build((b"alpha", b"beta")))
        payload = json.loads(delta.to_bytes())
        assert set(payload) == {"schema", "base_digest", "target_digest", "prefix_items", "appended"}
        assert delta.byte_count == len(delta.to_bytes())

    @pytest.mark.parametrize("prefix", [True, 1.5, -1])
    def test_rejects_a_non_integer_or_negative_prefix(self, prefix: object) -> None:
        base = ContextPack.build((b"alpha",))
        delta = ContextPackDelta.between(base, ContextPack.build((b"alpha", b"beta")))
        payload = {**json.loads(delta.to_bytes()), "prefix_items": prefix}
        with pytest.raises(ContextPackError):
            ContextPackDelta.from_bytes(json.dumps(payload).encode())

    def test_rejects_a_missing_field(self) -> None:
        base = ContextPack.build((b"alpha",))
        delta = ContextPackDelta.between(base, ContextPack.build((b"alpha", b"beta")))
        payload = {key: value for key, value in json.loads(delta.to_bytes()).items() if key != "prefix_items"}
        with pytest.raises(ContextPackError):
            ContextPackDelta.from_bytes(json.dumps(payload).encode())


class TestApplyContextPackDelta:
    def test_rejects_a_wrong_base(self) -> None:
        base = ContextPack.build((b"one",))
        delta = ContextPackDelta.between(base, ContextPack.build((b"one", b"two")))
        with pytest.raises(ContextPackError):
            apply_context_pack_delta(ContextPack.build((b"other",)), delta)

    def test_rejects_a_prefix_beyond_the_base(self) -> None:
        base = ContextPack.build((b"one",))
        invalid = ContextPackDelta(base.digest, "sha256:" + "0" * 64, 1, ())
        with pytest.raises(ContextPackError):
            apply_context_pack_delta(base, invalid)

    def test_rejects_a_wrong_target_digest(self) -> None:
        base = ContextPack.build((b"one",))
        wrong = ContextPackDelta(base.digest, "sha256:" + "0" * 64, 0, (b"x",))
        with pytest.raises(ContextPackError):
            apply_context_pack_delta(base, wrong)

    @pytest.mark.parametrize(
        "target_items",
        [(b"a", b"b", b"new", b"tail"), (b"new", b"b", b"old"), (b"a", b"new", b"old"), ()],
    )
    def test_rebuilds_a_change_at_any_position(self, target_items: tuple[bytes, ...]) -> None:
        base = ContextPack.build((b"a", b"b", b"old"))
        target = ContextPack.build(target_items)
        delta = ContextPackDelta.between(base, target)
        rebuilt = apply_context_pack_delta(
            ContextPack.from_bytes(base.to_bytes()), ContextPackDelta.from_bytes(delta.to_bytes())
        )
        assert rebuilt.to_bytes() == target.to_bytes()

    def test_a_zero_prefix_still_rebuilds_the_target(self) -> None:
        base = ContextPack.build((b"a", b"b", b"old"))
        target = ContextPack.build((b"new",))
        payload = {
            **json.loads(ContextPackDelta.between(base, target).to_bytes()),
            "prefix_items": 0,
            "appended": [base64.b64encode(item).decode("ascii") for item in target.items],
        }
        decoded = ContextPackDelta.from_bytes(json.dumps(payload).encode())
        assert apply_context_pack_delta(base, decoded).to_bytes() == target.to_bytes()


class TestIncrementalSaving:
    def test_an_unchanged_body_costs_only_the_delta_overhead(self) -> None:
        body = "stable body " * 100
        base = ContextPack.from_content(body, chunk_bytes=128)
        target = ContextPack.from_content(body, chunk_bytes=128)
        delta = ContextPackDelta.between(base, target)
        assert delta.prefix_items == len(base.items)
        assert delta.appended == ()
        assert delta.byte_count < len(body.encode("utf-8"))

    def test_a_changed_body_reuses_the_leading_chunks(self) -> None:
        # The shared prefix has to span at least one whole chunk, otherwise the
        # first chunk already contains the changed bytes and nothing is reusable.
        shared = "shared header " * 40
        base = ContextPack.from_content(shared + "a" * 1000, chunk_bytes=128)
        target = ContextPack.from_content(shared + "b" * 1000, chunk_bytes=128)
        delta = ContextPackDelta.between(base, target)
        assert delta.prefix_items > 0
        assert apply_context_pack_delta(base, delta).to_bytes() == target.to_bytes()

    def test_a_change_inside_the_first_chunk_reuses_nothing(self) -> None:
        base = ContextPack.from_content("header a" * 100, chunk_bytes=128)
        target = ContextPack.from_content("header b" * 100, chunk_bytes=128)
        delta = ContextPackDelta.between(base, target)
        assert delta.prefix_items == 0
        assert apply_context_pack_delta(base, delta).to_bytes() == target.to_bytes()
