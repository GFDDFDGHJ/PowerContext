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

"""One native statement that returns the fulltext and the vector candidates together.

The OceanBase-family engines can answer both channels inside a single statement:
the fulltext arm reads the FULLTEXT index of ``pc_memory_entry_heads`` and the
vector arm reads the vector index of ``pc_memory_vector_entries``. The statement
keeps the scope, Memory artifact, tag and candidate-limit filters inside the
database, so the application layer only fuses the two ordered candidate lists it
gets back instead of issuing and stitching two candidate queries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from pyobvector import VECTOR
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.sql.elements import TextClause

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import (
    CapabilityNotSupportedError,
    EmbeddingProfile,
    MemorySearchChannels,
    MemorySearchRequest,
)
from powercontext.builtin.artifacts.memory.canonical import validate_embedding
from powercontext.builtin.persistence.memory_index import memory_channel_hits
from powercontext.builtin.persistence.tags import memory_tag_parameters, memory_tag_sql

_NATIVE_HYBRID_SEARCH_SQL = """
(SELECT 'fts' AS channel,
        f.memory_artifact_id AS memory_artifact_id,
        f.entry_id AS entry_id,
        f.entry_version_id AS entry_version_id,
        v.text AS text,
        MATCH (f.searchable_text) AGAINST (:query) AS score,
        NULL AS distance
 FROM pc_memory_entry_heads AS f
 JOIN pc_memory_entry_versions AS v
   ON v.scope_id = f.scope_id
  AND v.memory_artifact_id = f.memory_artifact_id
  AND v.entry_id = f.entry_id
  AND v.entry_version_id = f.entry_version_id
 WHERE f.scope_id = :scope_id
   AND f.memory_artifact_id IN :memory_artifact_ids
   AND MATCH (f.searchable_text) AGAINST (:query)
{fts_tag}
 ORDER BY MATCH (f.searchable_text) AGAINST (:query) DESC,
          f.memory_artifact_id, f.entry_id, f.entry_version_id
 LIMIT :candidate_limit)
UNION ALL
(SELECT 'vector' AS channel,
        m.memory_artifact_id AS memory_artifact_id,
        m.entry_id AS entry_id,
        m.entry_version_id AS entry_version_id,
        v.text AS text,
        NULL AS score,
        l2_distance(m.embedding, :query_vector) AS distance
 FROM pc_memory_vector_entries AS m
 JOIN pc_memory_entry_versions AS v
   ON v.scope_id = m.scope_id
  AND v.memory_artifact_id = m.memory_artifact_id
  AND v.entry_id = m.entry_id
  AND v.entry_version_id = m.entry_version_id
 WHERE m.scope_id = :scope_id
   AND m.memory_artifact_id IN :memory_artifact_ids
{vector_tag}
 ORDER BY l2_distance(m.embedding, :query_vector){approximate}
 LIMIT :candidate_limit)
"""

_FTS_CHANNEL = "fts"
_VECTOR_CHANNEL = "vector"


def native_hybrid_sql(*, tagged: bool) -> str:
    """Render the one-statement hybrid candidate query.

    A tag filter is applied inside the statement, so the vector arm keeps its
    exact distance ordering: the approximate index scan answers the limit before
    the tag filter runs and would drop the tagged rows it did not rank first.
    Without a tag filter the arm uses the native vector index.
    """

    return _NATIVE_HYBRID_SEARCH_SQL.format(
        fts_tag=memory_tag_sql("f") if tagged else "",
        vector_tag=memory_tag_sql("m") if tagged else "",
        approximate="" if tagged else " APPROXIMATE",
    )


@lru_cache(maxsize=8)
def native_hybrid_statement(dimension: int, *, tagged: bool) -> TextClause:
    """Return the compiled one-statement hybrid candidate query."""

    statement = text(native_hybrid_sql(tagged=tagged)).bindparams(
        bindparam("memory_artifact_ids", expanding=True),
        bindparam("query_vector", type_=VECTOR(dimension)),
    )
    if tagged:
        statement = statement.bindparams(
            bindparam("tag_keys", expanding=True),
            bindparam("tag_hashes", expanding=True),
        )
    return statement


async def native_hybrid_search(
    connection: AsyncConnection,
    scope_id: str,
    request: MemorySearchRequest,
    profile: EmbeddingProfile,
    /,
) -> MemorySearchChannels:
    """Return both candidate channels from one statement."""

    if request.embedding_profile != profile or request.query_vector is None:
        raise CapabilityNotSupportedError("embedding-profile")
    parameters: dict[str, Any] = {
        "scope_id": scope_id,
        "memory_artifact_ids": tuple(memory.artifact_id for memory in request.memories),
        "query": request.analyzed_query,
        "query_vector": validate_embedding(request.query_vector, dimension=profile.dimension),
        "candidate_limit": request.candidate_limit,
    }
    if request.tag_filter is None:
        statement = native_hybrid_statement(profile.dimension, tagged=False)
    else:
        statement = native_hybrid_statement(profile.dimension, tagged=True)
        parameters.update(memory_tag_parameters(request.tag_filter))
    rows = (await connection.execute(statement, parameters)).mappings().all()
    return native_hybrid_channels(rows, request.memories)


def native_hybrid_channels(
    rows: Sequence[Mapping[Any, Any]],
    memories: tuple[ArtifactRef, ...],
) -> MemorySearchChannels:
    """Split the one statement's rows into the two ordered channel rankings."""

    fts_rows = [row for row in rows if row["channel"] == _FTS_CHANNEL]
    vector_rows = [row for row in rows if row["channel"] == _VECTOR_CHANNEL]
    return MemorySearchChannels(
        fts=memory_channel_hits(_ranked(fts_rows, by_score=True), memories),
        vector=memory_channel_hits(_ranked(vector_rows, by_score=False), memories),
    )


def _ranked(
    rows: Sequence[Mapping[Any, Any]],
    *,
    by_score: bool,
) -> list[Mapping[Any, Any]]:
    """Order one channel exactly like the statement ordered it.

    The database already returned both channels in order, but a union does not
    promise that the arms keep their relative order, so the ranking is rebuilt
    from the values the statement selected. The identity tie-break makes a
    repeated query return the same order.
    """

    if by_score:
        return sorted(rows, key=lambda row: (-float(row["score"]), *_identity(row)))
    return sorted(rows, key=lambda row: (float(row["distance"]), *_identity(row)))


def _identity(row: Mapping[Any, Any]) -> tuple[bytes, bytes, bytes]:
    return (
        str(row["memory_artifact_id"]).encode("utf-8"),
        str(row["entry_id"]).encode("utf-8"),
        str(row["entry_version_id"]).encode("utf-8"),
    )
