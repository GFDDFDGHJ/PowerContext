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

"""Durable Source lifecycle registrations and applied availability events."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryCitation
from powercontext.builtin.persistence.codec import dump_model, load_model
from powercontext.builtin.persistence.errors import InvalidRepositoryArgumentError
from powercontext.builtin.persistence.tables import (
    SOURCE_LIFECYCLE_EVENTS_TABLE,
    SOURCE_LIFECYCLE_NODES_TABLE,
)
from powercontext.sources import SourceRef

LifecycleFamily = Literal["source", "memory", "topic"]
LifecyclePolicy = Literal["all", "any"]


class LifecycleReference(BaseModel):
    """The exact reference one lifecycle node keeps for later historical lookup."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: LifecycleFamily
    source: SourceRef | None = None
    memory: MemoryCitation | None = None
    artifact: ArtifactRef | None = None

    @model_validator(mode="after")
    def require_single_matching_reference(self):
        expected = {
            "source": self.source,
            "memory": self.memory,
            "topic": self.artifact,
        }[self.kind]
        present = [value for value in (self.source, self.memory, self.artifact) if value is not None]
        if expected is None or len(present) != 1:
            raise ValueError("lifecycle reference must carry exactly the reference of its own kind")  # noqa: TRY003
        return self

    @classmethod
    def for_source(cls, reference: SourceRef) -> LifecycleReference:
        return cls(kind="source", source=reference)

    @classmethod
    def for_memory(cls, reference: MemoryCitation) -> LifecycleReference:
        return cls(kind="memory", memory=reference)

    @classmethod
    def for_artifact(cls, reference: ArtifactRef) -> LifecycleReference:
        return cls(kind="topic", artifact=reference)

    def value(self) -> SourceRef | MemoryCitation | ArtifactRef:
        """Return the wrapped reference of this node's family."""

        resolved = {
            "source": self.source,
            "memory": self.memory,
            "topic": self.artifact,
        }[self.kind]
        if resolved is None:  # pragma: no cover - the validator rejects this shape
            raise InvalidRepositoryArgumentError("reference", "lifecycle reference is incomplete")
        return resolved


class SourceLifecycleNode(BaseModel):
    """One registered evidence node together with its current availability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    family: LifecycleFamily
    revision: int = 0
    available: bool = True
    policy: LifecyclePolicy = "all"
    parents: tuple[str, ...] = ()
    reference: LifecycleReference
    last_sequence: int | None = None


class SourceLifecycleEventRecord(BaseModel):
    """One applied availability event kept for exact replay detection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    sequence: int
    revision: int
    available: bool
    reason: str


class SourceLifecycleRepository:
    """Store lifecycle registrations and applied events for one Scope."""

    async def load_nodes(
        self,
        connection: AsyncConnection,
        scope_id: str,
        /,
    ) -> tuple[SourceLifecycleNode, ...]:
        statement = select(SOURCE_LIFECYCLE_NODES_TABLE).where(
            SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
        )
        rows = (await connection.execute(statement)).mappings().all()
        return tuple(_decode_node(row) for row in rows)

    async def load_event(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        sequence: int,
        /,
    ) -> SourceLifecycleEventRecord | None:
        statement = select(SOURCE_LIFECYCLE_EVENTS_TABLE).where(
            SOURCE_LIFECYCLE_EVENTS_TABLE.c.scope_id == scope_id,
            SOURCE_LIFECYCLE_EVENTS_TABLE.c.node_id == node_id,
            SOURCE_LIFECYCLE_EVENTS_TABLE.c.sequence == sequence,
        )
        row = (await connection.execute(statement)).mappings().one_or_none()
        if row is None:
            return None
        return SourceLifecycleEventRecord(
            node_id=str(row["node_id"]),
            sequence=int(row["sequence"]),
            revision=int(row["revision"]),
            available=bool(row["available"]),
            reason=str(row["reason"]),
        )

    async def save_node(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node: SourceLifecycleNode,
        /,
    ) -> None:
        values = {
            "scope_id": scope_id,
            "node_id": node.node_id,
            "family": node.family,
            "revision": node.revision,
            "available": node.available,
            "policy": node.policy,
            "parents": json.dumps(list(node.parents)).encode(),
            "reference": dump_model(node.reference, kind="lifecycle", name="reference"),
            "last_sequence": node.last_sequence,
        }
        dialect = connection.dialect.name
        if dialect == "sqlite":
            statement = sqlite_insert(SOURCE_LIFECYCLE_NODES_TABLE).values(**values)
            statement = statement.on_conflict_do_update(
                index_elements=["scope_id", "node_id"],
                set_={
                    "family": values["family"],
                    "revision": values["revision"],
                    "available": values["available"],
                    "policy": values["policy"],
                    "parents": values["parents"],
                    "reference": values["reference"],
                    "last_sequence": values["last_sequence"],
                },
            )
        elif dialect == "mysql":
            statement = mysql_insert(SOURCE_LIFECYCLE_NODES_TABLE).values(**values)
            statement = statement.on_duplicate_key_update(
                family=values["family"],
                revision=values["revision"],
                available=values["available"],
                policy=values["policy"],
                parents=values["parents"],
                reference=values["reference"],
                last_sequence=values["last_sequence"],
            )
        else:
            raise InvalidRepositoryArgumentError("dialect", f"unsupported database dialect: {dialect}")
        await connection.execute(statement)

    async def save_event(
        self,
        connection: AsyncConnection,
        scope_id: str,
        event: SourceLifecycleEventRecord,
        /,
    ) -> None:
        await connection.execute(
            SOURCE_LIFECYCLE_EVENTS_TABLE.insert().values(
                scope_id=scope_id,
                node_id=event.node_id,
                sequence=event.sequence,
                revision=event.revision,
                available=event.available,
                reason=event.reason,
            )
        )


def _decode_node(row: Mapping[Any, Any]) -> SourceLifecycleNode:
    parents = json.loads(bytes(row["parents"]).decode())
    reference = load_model(
        LifecycleReference,
        row["reference"],
        kind="lifecycle",
        name="reference",
    )
    last_sequence = row["last_sequence"]
    return SourceLifecycleNode(
        node_id=str(row["node_id"]),
        family=str(row["family"]),  # type: ignore[arg-type]
        revision=int(row["revision"]),
        available=bool(row["available"]),
        policy=str(row["policy"]),  # type: ignore[arg-type]
        parents=tuple(str(parent) for parent in parents),
        reference=reference,
        last_sequence=None if last_sequence is None else int(last_sequence),
    )
