# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The seekdb hybrid candidate retrieval answers both channels in one statement."""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, event

from powercontext.builtin.artifacts.memory import EmbeddingProfile, MemoryEntryInput
from powercontext.builtin.inference import EmbeddingResult
from powercontext.builtin.persistence.oceanbase.native_hybrid import native_hybrid_channels
from powercontext.builtin.persistence.seekdb import SeekDBConfig
from powercontext.builtin.runtime import BuiltinConfig, open_builtin_contexts
from powercontext.builtin.tags import MemoryEntryTagTarget, TagFilter

_PROFILE = EmbeddingProfile(
    profile_id="pc11-native-hybrid-test-v1",
    model="deterministic",
    dimension=3,
    distance="l2",
    normalization="unit",
)

_Statement = tuple[str, Any]


class _Embedding:
    profile = _PROFILE

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        return EmbeddingResult(vectors=tuple(_vector(text) for text in texts))


def _vector(text: str) -> tuple[float, ...]:
    folded = text.casefold()
    if "alpha" in folded:
        return (1.0, 0.0, 0.0)
    if "gamma" in folded:
        return (0.0, 0.0, 1.0)
    return (0.0, 1.0, 0.0)


@contextmanager
def _statements() -> Iterator[list[_Statement]]:
    statements: list[_Statement] = []

    def record(_connection, _cursor, statement, parameters, _context, _executemany) -> None:
        statements.append((statement, parameters))

    event.listen(Engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(Engine, "before_cursor_execute", record)


def _candidate_statements(statements: list[_Statement]) -> list[_Statement]:
    return [
        (statement, parameters)
        for statement, parameters in statements
        if "pc_memory_entry_heads" in statement
        and "pc_memory_vector_entries" in statement
        and "AGAINST" in statement.upper()
        and "l2_distance" in statement.lower()
    ]


def test_channels_split_and_rank_both_arms_of_one_statement() -> None:
    from powercontext.artifacts import ArtifactRef

    memory_ref = ArtifactRef(family="memory", artifact_id="memory-1", revision=3)
    rows = [
        {
            "channel": "vector",
            "memory_artifact_id": "memory-1",
            "entry_id": "entry-b",
            "entry_version_id": "version-b",
            "text": "second",
            "score": None,
            "distance": 0.5,
        },
        {
            "channel": "fts",
            "memory_artifact_id": "memory-1",
            "entry_id": "entry-a",
            "entry_version_id": "version-a",
            "text": "first",
            "score": 2.0,
            "distance": None,
        },
        {
            "channel": "vector",
            "memory_artifact_id": "memory-1",
            "entry_id": "entry-a",
            "entry_version_id": "version-a",
            "text": "first",
            "score": None,
            "distance": 0.1,
        },
    ]

    channels = native_hybrid_channels(rows, (memory_ref,))

    assert [hit.entry_id for hit in channels.fts] == ["entry-a"]
    assert [hit.entry_id for hit in channels.vector] == ["entry-a", "entry-b"]
    assert channels.fts[0].distance is None
    assert channels.vector[0].distance == pytest.approx(0.1)
    assert channels.vector[1].distance == pytest.approx(0.5)


def test_hybrid_candidate_uses_one_native_statement() -> None:
    pytest.importorskip("pylibseekdb")

    async def scenario() -> None:
        # Unix socket paths must fit sockaddr_un even in deeply nested CI workspaces.
        directory = tempfile.mkdtemp(prefix="pc11-hybrid-", dir="/tmp")
        async with open_builtin_contexts(
            BuiltinConfig(database=SeekDBConfig(path=Path(directory) / "db")),
            embedding_model=_Embedding(),
        ) as contexts:
            memory = (await contexts.get("project")).artifacts.memory
            head = await memory.remember(
                memory=None,
                entries=(
                    MemoryEntryInput(kind="fact", text="Alpha semantic record."),
                    MemoryEntryInput(kind="fact", text="Beta lexical record."),
                ),
                mode="append",
            )
            assert head is not None
            with _statements() as statements:
                result = await memory.search("alpha", memories=(head,), mode="hybrid", limit=1)

            candidates = _candidate_statements(statements)
            assert len(candidates) == 1
            statement, parameters = candidates[0]
            assert len(result.hits) == 1
            assert result.hits[0].text == "Alpha semantic record."
            assert result.hits[0].matched_by == ("fts", "vector")
            assert result.hits[0].memory_ref == head.as_ref()

            async with contexts.database.transaction() as connection:
                plan = "\n".join(
                    str(row[0]) for row in (await connection.exec_driver_sql(f"EXPLAIN {statement}", parameters)).all()
                )
            assert "TEXT RETRIEVAL SCAN" in plan
            assert "VECTOR INDEX" in plan

    asyncio.run(scenario())


def test_hybrid_candidate_keeps_single_channel_behaviour() -> None:
    pytest.importorskip("pylibseekdb")

    async def scenario() -> None:
        directory = tempfile.mkdtemp(prefix="pc11-hybrid-", dir="/tmp")
        async with open_builtin_contexts(
            BuiltinConfig(database=SeekDBConfig(path=Path(directory) / "db")),
            embedding_model=_Embedding(),
        ) as contexts:
            memory = (await contexts.get("project")).artifacts.memory
            head = await memory.remember(
                memory=None,
                entries=(
                    MemoryEntryInput(kind="fact", text="Alpha semantic record."),
                    MemoryEntryInput(kind="fact", text="Beta lexical record."),
                ),
                mode="append",
            )
            assert head is not None

            with _statements() as statements:
                fts = await memory.search("beta", memories=(head,), mode="fts")
            assert fts.hits[0].text == "Beta lexical record."
            assert not _candidate_statements(statements)

            with _statements() as statements:
                vector = await memory.search("beta", memories=(head,), mode="vector")
            assert vector.hits[0].text == "Beta lexical record."
            assert not _candidate_statements(statements)

    asyncio.run(scenario())


def test_hybrid_candidate_filters_by_tag_inside_the_database() -> None:
    pytest.importorskip("pylibseekdb")

    async def scenario() -> None:
        directory = tempfile.mkdtemp(prefix="pc11-hybrid-", dir="/tmp")
        async with open_builtin_contexts(
            BuiltinConfig(database=SeekDBConfig(path=Path(directory) / "db")),
            embedding_model=_Embedding(),
        ) as contexts:
            memory = (await contexts.get("project")).artifacts.memory
            head = await memory.remember(
                memory=None,
                entries=(
                    MemoryEntryInput(kind="fact", text="Alpha semantic record."),
                    MemoryEntryInput(kind="fact", text="Beta lexical record."),
                ),
                mode="append",
            )
            assert head is not None
            beta = next(entry for entry in await memory.entries(head) if entry.text == "Beta lexical record.")
            target = MemoryEntryTagTarget(artifact_id=head.artifact_id, entry_id=beta.entry_id)
            empty = await contexts.records.get_tags("project", target)
            await contexts.records.replace_tags("project", target, ("selected",), expected_etag=empty.etag)

            with _statements() as statements:
                tagged = await memory.search(
                    "beta",
                    memories=(head,),
                    mode="hybrid",
                    limit=1,
                    tag_filter=TagFilter(tags=("selected",)),
                )
            candidates = _candidate_statements(statements)
            assert len(candidates) == 1
            assert "pc_artifact_tags" in candidates[0][0]
            assert [hit.entry_id for hit in tagged.hits] == [beta.entry_id]

    asyncio.run(scenario())


def test_hybrid_candidate_is_stable_and_scope_isolated() -> None:
    pytest.importorskip("pylibseekdb")

    async def scenario() -> None:
        directory = tempfile.mkdtemp(prefix="pc11-hybrid-", dir="/tmp")
        async with open_builtin_contexts(
            BuiltinConfig(database=SeekDBConfig(path=Path(directory) / "db")),
            embedding_model=_Embedding(),
        ) as contexts:
            memory = (await contexts.get("project")).artifacts.memory
            head = await memory.remember(
                memory=None,
                entries=(MemoryEntryInput(kind="fact", text="Alpha semantic record."),),
                mode="append",
            )
            assert head is not None
            other = (await contexts.get("other-scope")).artifacts.memory
            other_head = await other.remember(
                memory=None,
                entries=(MemoryEntryInput(kind="fact", text="Alpha decoy in another scope."),),
                mode="append",
            )
            assert other_head is not None

            first = await memory.search("alpha", memories=(head,), mode="hybrid", limit=5)
            repeated = await memory.search("alpha", memories=(head,), mode="hybrid", limit=5)
            assert [(hit.entry_id, hit.entry_version_id) for hit in first.hits] == [
                (hit.entry_id, hit.entry_version_id) for hit in repeated.hits
            ]
            assert all(hit.memory_ref == head.as_ref() for hit in first.hits)

            isolated = await other.search("alpha", memories=(other_head,), mode="hybrid", limit=5)
            assert [hit.memory_ref for hit in isolated.hits] == [other_head.as_ref()]

    asyncio.run(scenario())
