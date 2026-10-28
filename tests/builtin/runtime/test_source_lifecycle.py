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

"""Contract tests for the Source lifecycle service."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryCitation, MemoryEntryInput
from powercontext.builtin.artifacts.memory.service import MemoryService
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.records import ArtifactWrite
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CaptureSource,
    GetMemoryEntryRequest,
    RememberMemoryRequest,
    SearchMemoryRequest,
    SearchTopicMemoryRequest,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.source_lifecycle import SourceLifecycleError, SourceLifecycleEvent
from powercontext.builtin.scope import ScopeDraft
from powercontext.sources import SourceRef


def _event(node_id: str, sequence: int, available: bool, *, revision: int = 3, reason: str = "contract") -> SourceLifecycleEvent:
    return SourceLifecycleEvent(
        node_id=node_id,
        revision=revision,
        sequence=sequence,
        available=available,
        reason=reason,
    )


async def _scope(runtime, key: str) -> str:
    scope = await runtime.scopes.create(ScopeDraft(title="Lifecycle", summary="Evidence", idempotency_key=key))
    return scope.scope_id


async def _register_source(runtime, scope_id: str, node_id: str):
    receipt = await runtime.sources.for_scope(scope_id).capture(
        CaptureSource(source_id=node_id, content=f"Origin {node_id}", metadata={})
    )
    lifecycle = runtime.lifecycle.for_scope(scope_id)
    await lifecycle.register_source(node_id, receipt.source_ref, revision=3)
    return receipt.source_ref


async def _register_memory(runtime, scope_id: str, node_id: str, parents: tuple[str, ...], policy: str = "all"):
    created = await runtime.memory.for_scope(scope_id).remember(
        RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text=f"{node_id} evidence"),))
    )
    lifecycle = runtime.lifecycle.for_scope(scope_id)
    await lifecycle.register_memory(node_id, created.entry.citation, parents=parents, policy=policy)
    return created.entry.citation


async def _register_topic(runtime, scope_id: str, node_id: str, parents: tuple[str, ...], policy: str = "all"):
    artifact = await runtime.records.for_scope(scope_id).create_artifact(
        "topic-memory",
        ArtifactWrite(content={"title": node_id, "summary": f"{node_id} summary", "detail": f"{node_id} detail"}),
    )
    reference = ArtifactRef(family="topic-memory", artifact_id=artifact.artifact_id, revision=artifact.revision)
    lifecycle = runtime.lifecycle.for_scope(scope_id)
    await lifecycle.register_topic(node_id, reference, parents=parents, policy=policy)
    return reference


def test_event_validation_rejects_extra_fields_and_blank_values() -> None:
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(node_id="x", revision=1, sequence=1, available=False, reason="ok", extra=True)
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(node_id="   ", revision=1, sequence=1, available=False, reason="ok")
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(node_id="x", revision=0, sequence=1, available=False, reason="ok")
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(node_id="x", revision=1, sequence=0, available=False, reason="ok")
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(node_id="x", revision=1, sequence=1, available=False, reason="  ")


def test_registration_identity_and_dependency_validation() -> None:
    async def run() -> None:
        async with open_builtin_runtime(BuiltinConfig(database=SQLiteConfig())) as runtime:
            scope_id = await _scope(runtime, "registration")
            source_ref = await _register_source(runtime, scope_id, "root")
            lifecycle = runtime.lifecycle.for_scope(scope_id)

            # The same registration is idempotent.
            await lifecycle.register_source("root", source_ref, revision=3)
            # A different binding under the same identity is a conflict.
            with pytest.raises(SourceLifecycleError):
                await lifecycle.register_source("root", source_ref, revision=4)

            citation = await _register_memory(runtime, scope_id, "fact", ("root",))
            await lifecycle.register_memory("fact", citation, parents=("root",), policy="all")

            with pytest.raises(SourceLifecycleError):
                await lifecycle.register_memory("missing-parent", citation, parents=("nope",), policy="all")
            with pytest.raises(SourceLifecycleError):
                await lifecycle.register_memory("no-parent", citation, parents=(), policy="any")
            with pytest.raises(SourceLifecycleError):
                await lifecycle.register_memory("self", citation, parents=("self",), policy="all")
            with pytest.raises(SourceLifecycleError):
                await lifecycle.register_memory("duplicate", citation, parents=("root", "root"), policy="all")
            with pytest.raises(SourceLifecycleError):
                await lifecycle.register_memory("fact", citation, parents=("root",), policy="any")
            with pytest.raises(SourceLifecycleError):
                await lifecycle.register_topic("wrong-family", ArtifactRef(family="memory", artifact_id="x", revision=1))
            with pytest.raises(SourceLifecycleError):
                await lifecycle.register_topic(
                    "missing-artifact",
                    ArtifactRef(family="topic-memory", artifact_id="absent", revision=1),
                    parents=("root",),
                    policy="all",
                )
            with pytest.raises(SourceLifecycleError):
                await lifecycle.register_source(
                    "missing-source",
                    SourceRef(source_type="content", source_id="absent"),
                    revision=1,
                )
            with pytest.raises(SourceLifecycleError):
                await lifecycle.register_memory(
                    "missing-entry",
                    MemoryCitation(
                        memory_ref=ArtifactRef(family="memory", artifact_id="memory", revision=1),
                        entry_id="absent",
                        entry_version_id="absent",
                    ),
                    parents=("root",),
                    policy="all",
                )

    asyncio.run(run())


def test_event_version_and_sequence_are_strict() -> None:
    async def run() -> None:
        async with open_builtin_runtime(BuiltinConfig(database=SQLiteConfig())) as runtime:
            scope_id = await _scope(runtime, "events")
            await _register_source(runtime, scope_id, "root")
            lifecycle = runtime.lifecycle.for_scope(scope_id)

            with pytest.raises(SourceLifecycleError):
                await lifecycle.apply(_event("unknown", 1, True))
            with pytest.raises(SourceLifecycleError):
                await lifecycle.apply(_event("root", 1, True, revision=2))

            assert await lifecycle.apply(_event("root", 2, False)) == []
            # An identical replay is idempotent.
            assert await lifecycle.apply(_event("root", 2, False)) == []
            # A reused sequence with different content is rejected.
            with pytest.raises(SourceLifecycleError):
                await lifecycle.apply(_event("root", 2, True))
            # An older sequence is rejected.
            with pytest.raises(SourceLifecycleError):
                await lifecycle.apply(_event("root", 1, True))
            assert await lifecycle.apply(_event("root", 3, True)) == ["root"]

    asyncio.run(run())


def test_registration_computes_state_immediately_and_propagates() -> None:
    async def run() -> None:
        async with open_builtin_runtime(BuiltinConfig(database=SQLiteConfig())) as runtime:
            scope_id = await _scope(runtime, "registration-state")
            await _register_source(runtime, scope_id, "root")
            lifecycle = runtime.lifecycle.for_scope(scope_id)
            await lifecycle.apply(_event("root", 1, False))

            citation = await _register_memory(runtime, scope_id, "fact", ("root",))
            entries = await runtime.memory.for_scope(scope_id).list(include_inactive=True)
            states = {record.entry.text.split()[0]: record.state for record in entries.entries}
            assert states["fact"] == "inactive"
            page = await runtime.memory.for_scope(scope_id).search(
                SearchMemoryRequest(query="fact", mode="fts", limit=10)
            )
            assert not page.hits
            # The historical body stays readable.
            original = await runtime.memory.for_scope(scope_id).get(GetMemoryEntryRequest(citation=citation))
            assert original.entry.text == "fact evidence"

            await lifecycle.apply(_event("root", 2, True))
            entries = await runtime.memory.for_scope(scope_id).list(include_inactive=True)
            states = {record.entry.text.split()[0]: record.state for record in entries.entries}
            assert states["fact"] == "active"
            page = await runtime.memory.for_scope(scope_id).search(
                SearchMemoryRequest(query="fact", mode="fts", limit=10)
            )
            assert any(hit.text == "fact evidence" for hit in page.hits)

    asyncio.run(run())


def test_all_any_policies_propagate_to_every_descendant() -> None:
    async def run() -> None:
        async with open_builtin_runtime(BuiltinConfig(database=SQLiteConfig())) as runtime:
            scope_id = await _scope(runtime, "policies")
            await _register_source(runtime, scope_id, "left")
            await _register_source(runtime, scope_id, "right")
            lifecycle = runtime.lifecycle.for_scope(scope_id)
            await _register_memory(runtime, scope_id, "needs-both", ("left", "right"), "all")
            await _register_memory(runtime, scope_id, "needs-one", ("left", "right"), "any")
            await _register_topic(runtime, scope_id, "topic", ("needs-both",), "all")

            assert sorted(await lifecycle.apply(_event("left", 1, False))) == ["needs-one", "right"]
            assert sorted(await lifecycle.apply(_event("right", 1, False))) == []
            assert sorted(await lifecycle.apply(_event("left", 2, True))) == ["left", "needs-one"]
            assert sorted(await lifecycle.apply(_event("right", 2, True))) == [
                "left",
                "needs-both",
                "needs-one",
                "right",
                "topic",
            ]

    asyncio.run(run())


def test_topic_leaves_native_search_and_returns_after_restore() -> None:
    async def run() -> None:
        async with open_builtin_runtime(BuiltinConfig(database=SQLiteConfig())) as runtime:
            scope_id = await _scope(runtime, "topics")
            await _register_source(runtime, scope_id, "root")
            lifecycle = runtime.lifecycle.for_scope(scope_id)
            reference = await _register_topic(runtime, scope_id, "topic", ("root",))

            page = await runtime.topic_memory.for_scope(scope_id).search(
                SearchTopicMemoryRequest(query="topic", limit=20)
            )
            assert bool(page.hits) is True

            await lifecycle.apply(_event("root", 1, False))
            page = await runtime.topic_memory.for_scope(scope_id).search(
                SearchTopicMemoryRequest(query="topic", limit=20)
            )
            assert bool(page.hits) is False
            # The original artifact revision stays exactly readable.
            record = await runtime.records.for_scope(scope_id).get_artifact_revision(
                "topic-memory", reference.artifact_id, reference.revision
            )
            assert record.content["title"] == "topic"

            await lifecycle.apply(_event("root", 2, True))
            page = await runtime.topic_memory.for_scope(scope_id).search(
                SearchTopicMemoryRequest(query="topic", limit=20)
            )
            assert bool(page.hits) is True

    asyncio.run(run())


def test_generation_and_audit_purposes_differ() -> None:
    async def run() -> None:
        async with open_builtin_runtime(BuiltinConfig(database=SQLiteConfig())) as runtime:
            scope_id = await _scope(runtime, "purposes")
            source_ref = await _register_source(runtime, scope_id, "root")
            lifecycle = runtime.lifecycle.for_scope(scope_id)
            citation = await _register_memory(runtime, scope_id, "fact", ("root",))

            assert await lifecycle.resolve("fact", purpose="generation") == citation
            assert await lifecycle.resolve("fact", purpose="audit") == citation
            assert await lifecycle.resolve("root", purpose="audit") == source_ref

            await lifecycle.apply(_event("root", 1, False))
            with pytest.raises(SourceLifecycleError):
                await lifecycle.resolve("fact", purpose="generation")
            assert await lifecycle.resolve("fact", purpose="audit") == citation
            with pytest.raises(SourceLifecycleError):
                await lifecycle.resolve("missing", purpose="audit")
            with pytest.raises(SourceLifecycleError):
                await lifecycle.resolve("fact", purpose="unknown")

    asyncio.run(run())


def test_state_persists_across_restart_and_scopes_stay_independent(tmp_path) -> None:
    async def run() -> None:
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'lifecycle.db'}"))
        async with open_builtin_runtime(config) as runtime:
            first = await _scope(runtime, "first")
            second = await _scope(runtime, "second")
            await _register_source(runtime, first, "root")
            await _register_source(runtime, second, "root")
            await _register_memory(runtime, first, "fact", ("root",))
            await _register_memory(runtime, second, "fact", ("root",))
            await runtime.lifecycle.for_scope(first).apply(_event("root", 1, False))

        async with open_builtin_runtime(config) as runtime:
            first = await _scope(runtime, "first")
            second = await _scope(runtime, "second")
            # The same node identifier in another scope keeps its own state.
            with pytest.raises(SourceLifecycleError):
                await runtime.lifecycle.for_scope(first).resolve("root", purpose="generation")
            await runtime.lifecycle.for_scope(second).resolve("root", purpose="generation")
            assert await runtime.lifecycle.for_scope(first).apply(_event("root", 1, False)) == []
            with pytest.raises(SourceLifecycleError):
                await runtime.lifecycle.for_scope(first).resolve("root", purpose="generation")
            await runtime.lifecycle.for_scope(second).resolve("root", purpose="generation")

    asyncio.run(run())


def test_interrupted_native_write_is_completed_by_the_retry(monkeypatch) -> None:
    async def run() -> None:
        async with open_builtin_runtime(BuiltinConfig(database=SQLiteConfig())) as runtime:
            scope_id = await _scope(runtime, "retry")
            await _register_source(runtime, scope_id, "root")
            lifecycle = runtime.lifecycle.for_scope(scope_id)
            await _register_memory(runtime, scope_id, "fact", ("root",))

            original = MemoryService.forget
            calls = {"count": 0}

            async def failing_forget(self, memory, *, entries, reason=None):
                calls["count"] += 1
                raise RuntimeError("memory write failed")  # noqa: TRY003

            monkeypatch.setattr(MemoryService, "forget", failing_forget)
            with pytest.raises(RuntimeError):
                await lifecycle.apply(_event("root", 1, False))
            monkeypatch.setattr(MemoryService, "forget", original)

            assert await lifecycle.apply(_event("root", 1, False)) == []
            entries = await runtime.memory.for_scope(scope_id).list(include_inactive=True)
            states = {record.entry.text.split()[0]: record.state for record in entries.entries}
            assert states["fact"] == "inactive"

    asyncio.run(run())
