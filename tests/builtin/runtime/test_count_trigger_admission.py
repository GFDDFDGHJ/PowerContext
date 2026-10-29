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

"""Count admission for automatic Memory and Topic Memory processing."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from pydantic import ValidationError

from powercontext.builtin.artifacts.topic_memory import TOPIC_MEMORY_SOURCE_WINDOW_BINDING
from powercontext.builtin.persistence.cursors import SourceCursor, SourceCursorRepository
from powercontext.builtin.persistence.processing_intents import ArtifactProcessingIntentRepository
from powercontext.builtin.persistence.sqlite import SQLiteConfig, SQLiteProfile
from powercontext.builtin.persistence.supervision import ArtifactProcessingLeaseRepository
from powercontext.builtin.persistence.tables import SHARED_TABLES
from powercontext.builtin.runtime.artifact_processing import ArtifactProcessingBinding, ArtifactProcessingSupervisor
from powercontext.builtin.runtime.composition import _count_threshold
from powercontext.builtin.runtime.config import BuiltinConfig, RuntimeConfig
from powercontext.builtin.runtime.processing_contracts import ArtifactProcessingWorkerCompletion
from powercontext.builtin.runtime.processing_discovery import pending_source_counts
from powercontext.builtin.triggers.source_window import SOURCE_WINDOW_TRIGGER_NAME
from powercontext.sources import SourceMaterialization
from tests.builtin.persistence.contract import SOURCE_ADAPTERS, NoteSource

MEMORY_BINDING = SOURCE_WINDOW_TRIGGER_NAME
TOPIC_BINDING = TOPIC_MEMORY_SOURCE_WINDOW_BINDING


def _profile(tmp_path, name):
    return SQLiteProfile.open(
        SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / (name + '.db')}"), tables=SHARED_TABLES
    )


async def _add_source(database, scope, name):
    from powercontext.builtin.persistence.sources import SourceRepository

    source = NoteSource(name=name, materialization=SourceMaterialization.CAPTURED, body=name)
    async with database.transaction() as connection:
        await SourceRepository(SOURCE_ADAPTERS).add(connection, scope, source)


async def _counts(database, scope, binding):
    async with database.transaction() as connection:
        counts = await pending_source_counts(connection, (scope,), binding)
    return counts.get(scope, 0)


async def _wait(predicate, timeout_seconds=5):
    async with asyncio.timeout(timeout_seconds):
        while not predicate():  # noqa: ASYNC110 - bounded observation of worker events
            await asyncio.sleep(0.005)


class _Launcher:
    """Accept one automatic invocation and acknowledge it durably."""

    def __init__(self, database):
        self.database = database
        self.assignments = []

    async def start(self, assignment):
        self.assignments.append(assignment)
        return _Handle(self, assignment)


class _Handle:
    def __init__(self, launcher, assignment):
        self.launcher = launcher
        self.assignment = assignment

    async def wait(self):
        work = self.assignment
        async with self.launcher.database.transaction() as connection:
            await ArtifactProcessingLeaseRepository().require_fence(connection, work.fence)
            intents = ArtifactProcessingIntentRepository()
            row = await intents.load(connection, work.scope_id, work.binding_name, for_update=True)
            assert row is not None
            await intents.acknowledge(
                connection,
                work.scope_id,
                work.binding_name,
                work.claimed_request_generation,
                clean_generation=row.dirty_generation,
            )
        return ArtifactProcessingWorkerCompletion()

    async def terminate(self):
        return None


def test_count_threshold_is_a_strict_optional_positive_integer() -> None:
    assert RuntimeConfig().memory_source_count_threshold is None
    assert RuntimeConfig().topic_memory_source_count_threshold is None
    for value in (1, 3, 100):
        assert RuntimeConfig(memory_source_count_threshold=value).memory_source_count_threshold == value
        assert RuntimeConfig(topic_memory_source_count_threshold=value).topic_memory_source_count_threshold == value
    for value in (True, False, 0, -1, 1.0, "3"):
        with pytest.raises(ValidationError):
            RuntimeConfig(memory_source_count_threshold=value)
        with pytest.raises(ValidationError):
            RuntimeConfig(topic_memory_source_count_threshold=value)


def test_count_threshold_is_bound_to_its_own_processing_family() -> None:
    config = BuiltinConfig(
        runtime=RuntimeConfig(memory_source_count_threshold=2, topic_memory_source_count_threshold=3)
    )

    assert _count_threshold(config, "memory") == 2
    assert _count_threshold(config, "topic-memory") == 3
    assert _count_threshold(config, "profile") is None
    assert _count_threshold(BuiltinConfig(), "memory") is None


def test_pending_counts_follow_journal_identities_and_the_committed_cursor(tmp_path) -> None:
    async def scenario() -> None:
        async with _profile(tmp_path, "counts") as profile:
            await _add_source(profile.database, "scope", "one")
            await _add_source(profile.database, "scope", "two")
            await _add_source(profile.database, "scope", "three")
            assert await _counts(profile.database, "scope", MEMORY_BINDING) == 3

            # A replayed identity keeps its journal position, so it adds no
            # pending identity.
            await _add_source(profile.database, "scope", "two")
            assert await _counts(profile.database, "scope", MEMORY_BINDING) == 3

            # Another family keeps its own cursor.
            assert await _counts(profile.database, "scope", TOPIC_BINDING) == 3
            async with profile.database.transaction() as connection:
                await SourceCursorRepository().save(
                    connection, "scope", MEMORY_BINDING, SourceCursor(sequence=2), expected_generation=None
                )
            assert await _counts(profile.database, "scope", MEMORY_BINDING) == 1
            assert await _counts(profile.database, "scope", TOPIC_BINDING) == 3

            # A scope that never committed a cursor starts from zero.
            assert await _counts(profile.database, "other", MEMORY_BINDING) == 0

    asyncio.run(scenario())


def test_automatic_admission_waits_for_the_count_threshold(tmp_path) -> None:
    async def scenario() -> None:
        async with _profile(tmp_path, "admission") as profile:
            launcher = _Launcher(profile.database)
            await _add_source(profile.database, "scope", "one")
            async with ArtifactProcessingSupervisor(
                database=profile.database,
                bindings=(
                    ArtifactProcessingBinding(MEMORY_BINDING, "memory", launcher, count_threshold=2, max_workers=1),
                ),
                lease_mode="single-process",
            ) as supervisor:
                await asyncio.sleep(0.3)
                assert launcher.assignments == []
                assert (await _counts(profile.database, "scope", MEMORY_BINDING)) == 1

                await _add_source(profile.database, "scope", "two")
                await _wait(lambda: len(launcher.assignments) == 1)
                assert launcher.assignments[0].scope_id == "scope"
                await _wait(lambda: supervisor.family_status["memory"]["completed"] == 1)

    asyncio.run(scenario())


def test_explicit_request_bypasses_the_count_threshold(tmp_path) -> None:
    async def scenario() -> None:
        async with _profile(tmp_path, "explicit") as profile:
            launcher = _Launcher(profile.database)
            await _add_source(profile.database, "scope", "one")
            async with profile.database.transaction() as connection:
                await ArtifactProcessingIntentRepository().request(connection, "scope", MEMORY_BINDING)
            async with ArtifactProcessingSupervisor(
                database=profile.database,
                bindings=(
                    ArtifactProcessingBinding(MEMORY_BINDING, "memory", launcher, count_threshold=5, max_workers=1),
                ),
                lease_mode="single-process",
            ):
                await _wait(lambda: len(launcher.assignments) == 1)

    asyncio.run(scenario())


def test_count_trigger_never_postpones_the_time_condition(tmp_path) -> None:
    async def scenario() -> None:
        async with _profile(tmp_path, "schedule") as profile:
            launcher = _Launcher(profile.database)
            await _add_source(profile.database, "scope", "one")
            async with ArtifactProcessingSupervisor(
                database=profile.database,
                bindings=(
                    ArtifactProcessingBinding(
                        MEMORY_BINDING,
                        "memory",
                        launcher,
                        count_threshold=5,
                        max_workers=1,
                        automatic_processing_interval=timedelta(seconds=0.05),
                    ),
                ),
                lease_mode="single-process",
            ):
                # The count is below the threshold, so only the time condition
                # may admit this scope.
                await _wait(lambda: len(launcher.assignments) == 1)

    asyncio.run(scenario())


def test_count_trigger_admits_before_a_long_time_condition(tmp_path) -> None:
    async def scenario() -> None:
        async with _profile(tmp_path, "early") as profile:
            launcher = _Launcher(profile.database)
            async with ArtifactProcessingSupervisor(
                database=profile.database,
                bindings=(
                    ArtifactProcessingBinding(
                        MEMORY_BINDING,
                        "memory",
                        launcher,
                        count_threshold=2,
                        max_workers=1,
                        automatic_processing_interval=timedelta(seconds=3600),
                    ),
                ),
                lease_mode="single-process",
            ):
                # The time condition owns its checkpoint after the first scan.
                await asyncio.sleep(0.3)
                await _add_source(profile.database, "scope", "one")
                await asyncio.sleep(0.3)
                assert launcher.assignments == []
                await _add_source(profile.database, "scope", "two")
                # The count condition admits the Scope long before the hour
                # long time condition would fire.
                await _wait(lambda: len(launcher.assignments) == 1)

    asyncio.run(scenario())
