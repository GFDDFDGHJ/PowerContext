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

"""HTTP scoring with bounded batches and identity-preserving global ranking."""

from __future__ import annotations

import math
from typing import cast

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Select original Memory ranks using a caller-owned HTTP scoring client."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("rerank batch_size must be a positive integer")  # noqa: TRY003
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("rerank endpoint must be a nonempty string")  # noqa: TRY003
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("rerank policy_id must be a nonempty string")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate once and select by score, then original rank."""

        if not candidates:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if type(limit) is not int or not 1 <= limit <= len(candidates):
            raise ValueError("rerank limit must be an integer between one and the candidate count")  # noqa: TRY003
        if not isinstance(query, str) or not query:
            raise ValueError("rerank query must be a nonempty string")  # noqa: TRY003

        scores: dict[int, int | float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            ranks = set(range(start + 1, start + len(batch) + 1))
            response = await self._client.post(
                self._endpoint,
                follow_redirects=False,
                json={
                    "query": query,
                    "candidates": [
                        {"rank": rank, "text": candidate.text} for rank, candidate in enumerate(batch, start=start + 1)
                    ],
                },
            )
            requests += 1
            response.raise_for_status()
            scores.update(_batch_scores(response.json(object_pairs_hook=_unique_object), ranks))

        return MemoryRerankDecision(
            selected_ranks=tuple(sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("rerank response contains a duplicate JSON field")  # noqa: TRY003
        value[key] = item
    return value


def _batch_scores(payload: object, expected_ranks: set[int]) -> dict[int, int | float]:
    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("rerank response must contain exactly scores")  # noqa: TRY003
    rows = cast(dict[str, object], payload)["scores"]
    if not isinstance(rows, list) or len(rows) != len(expected_ranks):
        raise ValueError("rerank scores must contain one record per batch candidate")  # noqa: TRY003

    scores: dict[int, int | float] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"rank", "score"}:
            raise ValueError("rerank score records must contain exactly rank and score")  # noqa: TRY003
        score_row = cast(dict[str, object], row)
        rank, score = score_row["rank"], score_row["score"]
        if type(rank) is not int or rank not in expected_ranks or rank in scores:
            raise ValueError("rerank scores must cover each rank in the current batch exactly once")  # noqa: TRY003
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            # The response schema contract requires ValueError for invalid types.
            raise ValueError("rerank score must be a finite number")  # noqa: TRY003, TRY004
        if isinstance(score, float) and not math.isfinite(score):
            raise ValueError("rerank score must be a finite number")  # noqa: TRY003
        scores[rank] = score
    return scores


__all__ = ["HTTPMemoryReranker"]
