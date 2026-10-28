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

"""Source dependency lifecycle: withdraw or restore evidence and propagate availability.

A Scope owns one dependency DAG over evidence nodes. Sources are the roots; Memory
entries and Topic Memory artifacts are derived nodes whose usability follows the
``all`` / ``any`` support policy declared at registration. Withdrawing a Source
deactivates the entries and topics that lose their support, restoring it brings them
back, and the historical reference of every node stays readable for audit.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryCitation
from powercontext.builtin.artifacts.memory.errors import (
    InvalidMemoryCitationError,
    MemoryEntryNotFoundError,
)
from powercontext.builtin.artifacts.topic_memory import TopicMemory
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.source_lifecycle import (
    LifecycleReference,
    SourceLifecycleEventRecord,
    SourceLifecycleNode,
    SourceLifecycleRepository,
)
from powercontext.builtin.records import BaseAccessError
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.builtin.sources import validate_scope_id
from powercontext.errors import ArtifactNotFoundError
from powercontext.sources import SourceRef

if TYPE_CHECKING:  # pragma: no cover - the runtime import is resolved for typing only
    from powercontext.builtin.runtime.application import BuiltinRuntime

LIFECYCLE_PURPOSES = ("generation", "audit")

_RegistrationReason = "source lifecycle registration"


class SourceLifecycleError(ValueError):
    """Raised when a lifecycle registration or event breaks the evidence contract."""

    def __init__(self, code: str, detail: object = None) -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if detail is None else f"{code}: {detail}")


class SourceLifecycleEvent(BaseModel):
    """One availability transition for one registered Source."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    node_id: str
    revision: StrictInt = Field(ge=1)
    sequence: StrictInt = Field(ge=1)
    available: StrictBool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def require_meaningful_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")  # noqa: TRY003
        return value


class SourceLifecycleApplication:
    """Select the lifecycle service for one Scope."""

    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase | None) -> None:
        self._runtime = runtime
        self._database = database

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleService:
        """Return the lifecycle service bound to one Scope."""

        if self._database is None:
            raise SourceLifecycleError("lifecycle-storage")
        return ScopedSourceLifecycleService(self._runtime, scope_id, self._database)

    async def inactive_topic_ids(self, scope_id: str, /) -> frozenset[str]:
        """Return the Topic artifacts that lost their support in one Scope."""

        if self._database is None:
            return frozenset()
        service = ScopedSourceLifecycleService(self._runtime, scope_id, self._database)
        return await service.inactive_topic_ids()

class ScopedSourceLifecycleService:
    """Register evidence nodes and propagate Source availability inside one Scope."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str, database: AsyncDatabase) -> None:
        self._runtime = runtime
        self.scope_id = _scope_identifier(scope_id)
        self._database = database
        self._repository = SourceLifecycleRepository()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register_source(self, node_id: str, source_ref: SourceRef, revision: int = 1) -> None:
        """Register one Source root with the lifecycle version it answers to."""

        node_id = _require_identifier(node_id, "node_id")
        revision = _require_positive(revision, "revision")
        if not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("source-reference")
        await self._require_readable_source(source_ref)
        await self._register(
            node_id=node_id,
            family="source",
            reference=LifecycleReference.for_source(source_ref),
            revision=revision,
            parents=(),
            policy="all",
        )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        parents: tuple[str, ...] = (),
        policy: Literal["all", "any"] = "all",
    ) -> None:
        """Register one Memory entry and the evidence it depends on."""

        node_id = _require_identifier(node_id, "node_id")
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("memory-reference")
        await self._require_readable_memory(citation)
        await self._register(
            node_id=node_id,
            family="memory",
            reference=LifecycleReference.for_memory(citation),
            revision=0,
            parents=tuple(parents),
            policy=_require_policy(policy),
        )

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        parents: tuple[str, ...] = (),
        policy: Literal["all", "any"] = "all",
    ) -> None:
        """Register one Topic Memory artifact and the evidence it depends on."""

        node_id = _require_identifier(node_id, "node_id")
        if not isinstance(artifact_ref, ArtifactRef):
            raise SourceLifecycleError("topic-reference")
        if artifact_ref.family != TopicMemory.family:
            raise SourceLifecycleError("topic-family", TopicMemory.family)
        await self._require_readable_artifact(artifact_ref)
        await self._register(
            node_id=node_id,
            family="topic",
            reference=LifecycleReference.for_artifact(artifact_ref),
            revision=0,
            parents=tuple(parents),
            policy=_require_policy(policy),
        )

    async def _register(
        self,
        *,
        node_id: str,
        family: Literal["source", "memory", "topic"],
        reference: LifecycleReference,
        revision: int,
        parents: tuple[str, ...],
        policy: Literal["all", "any"],
    ) -> None:
        async with self._scoped_write():
            nodes = await self._load_nodes()
            existing = nodes.get(node_id)
            if existing is not None:
                if _same_binding(
                    existing,
                    family=family,
                    reference=reference,
                    revision=revision,
                    parents=parents,
                    policy=policy,
                ):
                    return
                raise SourceLifecycleError("node-conflict", node_id)
            if family != "source":
                _require_parents(nodes, node_id, parents)
            availability = _availability_map(nodes)
            available = (
                True
                if family == "source"
                else _supported(availability, parents, policy)
            )
            node = SourceLifecycleNode(
                node_id=node_id,
                family=family,
                revision=revision,
                available=available,
                policy=policy,
                parents=parents,
                reference=reference,
                last_sequence=None,
            )
            # The native state follows the registration immediately: a derived node
            # never starts usable and waits for a later event.
            await self._apply_native_state(
                {**nodes, node_id: node},
                {**availability, node_id: available},
                reason=_RegistrationReason,
            )
            async with self._database.transaction() as connection:
                await self._repository.save_node(connection, self.scope_id, node)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        """Apply one Source availability event and return the usable node ids."""

        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event-type")
        async with self._scoped_write():
            nodes = await self._load_nodes()
            node = nodes.get(event.node_id)
            if node is None or node.family != "source":
                raise SourceLifecycleError("unknown-source", event.node_id)
            if node.revision != event.revision:
                raise SourceLifecycleError("revision-mismatch", (event.node_id, event.revision))
            replay = await self._replay_state(node, event)
            availability = _availability_map(nodes, overrides={event.node_id: event.available})
            await self._apply_native_state(nodes, availability, reason=event.reason)
            last_sequence = node.last_sequence if replay else event.sequence
            updated = {
                node_id: item.model_copy(
                    update={
                        "available": availability[node_id],
                        "last_sequence": last_sequence if node_id == event.node_id else item.last_sequence,
                    }
                )
                for node_id, item in nodes.items()
            }
            async with self._database.transaction() as connection:
                for item in updated.values():
                    await self._repository.save_node(connection, self.scope_id, item)
                if not replay:
                    await self._repository.save_event(
                        connection,
                        self.scope_id,
                        SourceLifecycleEventRecord(
                            node_id=event.node_id,
                            sequence=event.sequence,
                            revision=event.revision,
                            available=event.available,
                            reason=event.reason,
                        ),
                    )
            return sorted(node_id for node_id, available in availability.items() if available)

    async def _replay_state(self, node: SourceLifecycleNode, event: SourceLifecycleEvent) -> bool:
        """Report whether the event repeats an already applied transition."""

        if node.last_sequence is None:
            return False
        if event.sequence < node.last_sequence:
            raise SourceLifecycleError("sequence-order", (event.node_id, event.sequence, node.last_sequence))
        if event.sequence > node.last_sequence:
            return False
        async with self._database.transaction() as connection:
            stored = await self._repository.load_event(
                connection,
                self.scope_id,
                event.node_id,
                event.sequence,
            )
        if stored is None:
            raise SourceLifecycleError("missing-event", (event.node_id, event.sequence))
        if (stored.revision, stored.available, stored.reason) != (event.revision, event.available, event.reason):
            raise SourceLifecycleError("sequence-content", (event.node_id, event.sequence))
        return True

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    async def resolve(
        self,
        node_id: str,
        purpose: Literal["generation", "audit"] = "generation",
    ) -> SourceRef | MemoryCitation | ArtifactRef:
        """Return the original reference of one node for generation or for audit."""

        if purpose not in LIFECYCLE_PURPOSES:
            raise SourceLifecycleError("unknown-purpose", purpose)
        nodes = await self._load_nodes()
        node = nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError("unknown-node", node_id)
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError("node-unavailable", node_id)
        return node.reference.value()

    async def inactive_topic_ids(self) -> frozenset[str]:
        """Return the Topic artifacts that lost their support in this Scope."""

        nodes = await self._load_nodes()
        return frozenset(
            node.reference.artifact.artifact_id
            for node in nodes.values()
            if node.family == "topic" and not node.available and node.reference.artifact is not None
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _scoped_write(self) -> AsyncIterator[None]:
        """Serialize one lifecycle write against the other writers of this Scope."""

        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            yield

    async def _load_nodes(self) -> dict[str, SourceLifecycleNode]:
        async with self._database.transaction() as connection:
            nodes = await self._repository.load_nodes(connection, self.scope_id)
        return {node.node_id: node for node in nodes}

    async def _apply_native_state(
        self,
        nodes: dict[str, SourceLifecycleNode],
        availability: dict[str, bool],
        *,
        reason: str,
    ) -> None:
        """Reflect the computed availability in the native Memory entries."""

        entries = await self._runtime.memory.for_scope(self.scope_id).list(include_inactive=True)
        states = {record.entry.entry_id: record.state for record in entries.entries}
        desired: dict[str, bool] = {}
        for node in nodes.values():
            if node.family != "memory" or node.reference.memory is None:
                continue
            entry_id = node.reference.memory.entry_id
            supported = availability.get(node.node_id, node.available)
            # One entry shared by several nodes stays usable only while every
            # node that claims it keeps its support.
            desired[entry_id] = desired.get(entry_id, True) and supported
        deactivate = tuple(
            entry_id for entry_id, supported in desired.items() if not supported and states.get(entry_id) == "active"
        )
        reactivate = tuple(
            entry_id for entry_id, supported in desired.items() if supported and states.get(entry_id) == "inactive"
        )
        if not deactivate and not reactivate:
            return
        await self._write_entry_states(deactivate, reactivate, reason=reason)

    async def _write_entry_states(
        self,
        deactivate: tuple[str, ...],
        reactivate: tuple[str, ...],
        *,
        reason: str,
    ) -> None:
        """Deactivate and reactivate the selected entries in one scoped write."""

        async with self._runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            try:
                current = await service.head(context.artifacts.memory_artifact_id)
            except ArtifactNotFoundError:
                return
            entries = await service.entries(current)
            if deactivate:
                targets = tuple(item for item in entries if item.entry_id in deactivate)
                if targets:
                    current = await service.forget(current, entries=targets, reason=reason)
                    entries = await service.entries(current)
            if reactivate:
                targets = tuple(item for item in entries if item.entry_id in reactivate)
                if targets:
                    await service.reactivate(current, entries=targets, reason=reason)

    async def _require_readable_source(self, source_ref: SourceRef) -> None:
        try:
            await self._runtime.records.for_scope(self.scope_id).get_source(
                source_ref.source_type,
                source_ref.source_id,
            )
        except BaseAccessError as error:
            raise SourceLifecycleError("unreadable-source", source_ref.source_id) from error

    async def _require_readable_memory(self, citation: MemoryCitation) -> None:
        try:
            await self._runtime.memory.for_scope(self.scope_id).get(
                GetMemoryEntryRequest(citation=citation),
            )
        except (
            ArtifactNotFoundError,
            InvalidMemoryCitationError,
            MemoryEntryNotFoundError,
            BaseAccessError,
        ) as error:
            raise SourceLifecycleError("unreadable-memory", citation.entry_id) from error

    async def _require_readable_artifact(self, artifact_ref: ArtifactRef) -> None:
        try:
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                artifact_ref.family,
                artifact_ref.artifact_id,
                artifact_ref.revision,
            )
        except BaseAccessError as error:
            raise SourceLifecycleError("unreadable-artifact", artifact_ref.artifact_id) from error


def _require_identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceLifecycleError("node-id", field)
    return value


def _scope_identifier(value: object) -> str:
    """Accept one Scope identifier or a Scope descriptor that carries one."""

    if isinstance(value, str):
        return validate_scope_id(value)
    scope_id = getattr(value, "scope_id", None)
    if isinstance(scope_id, str):
        return validate_scope_id(scope_id)
    raise SourceLifecycleError("scope-id")


def _require_positive(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SourceLifecycleError("positive-integer", field)
    return value


def _require_policy(policy: str) -> Literal["all", "any"]:
    if policy not in ("all", "any"):
        raise SourceLifecycleError("support-policy", policy)
    return policy  # type: ignore[return-value]


def _require_parents(
    nodes: dict[str, SourceLifecycleNode],
    node_id: str,
    parents: tuple[str, ...],
) -> None:
    if not parents:
        raise SourceLifecycleError("parents-required", node_id)
    if len(set(parents)) != len(parents):
        raise SourceLifecycleError("duplicate-parent", node_id)
    for parent in parents:
        if parent == node_id:
            raise SourceLifecycleError("self-parent", node_id)
        if parent not in nodes:
            raise SourceLifecycleError("unknown-parent", (node_id, parent))
    if _reaches(nodes, parents, node_id):
        raise SourceLifecycleError("dependency-cycle", node_id)


def _reaches(nodes: dict[str, SourceLifecycleNode], parents: tuple[str, ...], target: str) -> bool:
    """Report whether one of the parents already depends on the target."""

    pending = list(parents)
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if current == target:
            return True
        node = nodes.get(current)
        if node is not None:
            pending.extend(node.parents)
    return False


def _same_binding(
    node: SourceLifecycleNode,
    *,
    family: str,
    reference: LifecycleReference,
    revision: int,
    parents: tuple[str, ...],
    policy: str,
) -> bool:
    return (
        node.family == family
        and node.reference == reference
        and node.revision == revision
        and node.parents == parents
        and node.policy == policy
    )


def _availability_map(
    nodes: dict[str, SourceLifecycleNode],
    overrides: dict[str, bool] | None = None,
) -> dict[str, bool]:
    """Resolve the availability of every node from its parents."""

    resolved: dict[str, bool] = {}
    visiting: set[str] = set()

    def resolve(node_id: str) -> bool:
        if node_id in resolved:
            return resolved[node_id]
        node = nodes[node_id]
        if node.family == "source":
            value = node.available
            if overrides is not None and node_id in overrides:
                value = overrides[node_id]
        else:
            if node_id in visiting:
                raise SourceLifecycleError("dependency-cycle", node_id)
            visiting.add(node_id)
            supports = [resolve(parent) for parent in node.parents]
            visiting.discard(node_id)
            value = all(supports) if node.policy == "all" else any(supports)
        resolved[node_id] = value
        return value

    for node_id in nodes:
        resolve(node_id)
    return resolved


def _supported(availability: dict[str, bool], parents: tuple[str, ...], policy: str) -> bool:
    supports = [availability[parent] for parent in parents]
    return all(supports) if policy == "all" else any(supports)
