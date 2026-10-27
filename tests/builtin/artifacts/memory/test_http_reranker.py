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

"""Behavior tests for strict, bounded HTTP Memory reranking."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.http_reranker import HTTPMemoryReranker
from powercontext.builtin.artifacts.memory.models import MemoryHit


def _hit(rank: int) -> MemoryHit:
    return MemoryHit(
        memory_ref=ArtifactRef(family="memory", artifact_id="memory", revision=1),
        entry_id=f"entry-{rank}",
        entry_version_id=f"entry-{rank}-v1",
        text=f"Candidate {rank}",
        score=1 / rank,
        matched_by=("fts",),
    )


def test_global_ranking_preserves_integer_scores_and_reuses_caller_client() -> None:
    scores = {1: 2**60, 2: 2**60 + 1, 3: -2, 4: 2**60 + 1, 5: -1}
    candidates = tuple(_hit(rank) for rank in range(1, 6))
    before = tuple(candidate.model_dump() for candidate in candidates)
    sent: list[list[int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["query"] == "Which record?"
        batch = body["candidates"]
        assert 1 <= len(batch) <= 2
        assert all(item["text"] == candidates[item["rank"] - 1].text for item in batch)
        sent.append([item["rank"] for item in batch])
        return httpx.Response(
            200,
            json={"scores": [{"rank": item["rank"], "score": scores[item["rank"]]} for item in reversed(batch)]},
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker("https://reranker.invalid", client=client, batch_size=2, policy_id="custom")
            for _ in range(2):
                decision = await reranker.rerank("Which record?", candidates, 4)
                assert decision.selected_ranks == (2, 4, 1, 5)
                assert decision.usage.requests == 3
                assert decision.discarded_rank_count == 0
                assert decision.used_fallback is False
                assert not client.is_closed
            assert reranker.policy_id == "custom"
        assert sent == [[1, 2], [3, 4], [5], [1, 2], [3, 4], [5]]
        assert tuple(candidate.model_dump() for candidate in candidates) == before

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"scores": None},
        {"scores": {}},
        {"scores": []},
        {"scores": [None]},
        {"scores": [{"rank": 1}]},
        {"scores": [{"score": 1}]},
        {"scores": [{"rank": 1, "score": 1}], "extra": None},
        {"scores": [{"rank": 1, "score": 1, "extra": None}]},
        *({"scores": [{"rank": rank, "score": 1}]} for rank in (True, 1.0, "1", 0, 2, None)),
        *({"scores": [{"rank": 1, "score": score}]} for score in (True, "1", None, [], float("-inf"), float("nan"))),
    ],
)
def test_rejects_invalid_response_types_without_coercion(payload: object) -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=json.dumps(payload).encode()))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("query", (_hit(1),), 1)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body", [b"not JSON", b'{"scores":[],"scores":[]}', b'{"scores":[{"rank":1,"score":1,"score":2}]}']
)
def test_rejects_malformed_json_and_duplicate_object_fields(body: bytes) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=body))
        ) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("query", (_hit(1),), 1)

    asyncio.run(scenario())


def test_rejects_rank_from_another_batch() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [{"rank": 1, "score": 1}]})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://reranker.invalid", client=client, batch_size=1).rerank(
                    "query", (_hit(1), _hit(2)), 1
                )

    asyncio.run(scenario())


@pytest.mark.parametrize("batch_size", [False, True, 0, -1, 1.5, "2", None])
def test_rejects_invalid_batch_size(batch_size: int) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient() as client:
            with pytest.raises(ValueError):
                HTTPMemoryReranker("https://reranker.invalid", client=client, batch_size=batch_size)

    asyncio.run(scenario())


@pytest.mark.parametrize("limit", [False, True, 0, -1, 3, 1.5, "1", None])
def test_rejects_invalid_limit_before_sending_requests(limit: int) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        pytest.fail("invalid limits must fail before sending HTTP requests")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank(
                    "query", (_hit(1), _hit(2)), limit
                )

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [301, 400, 429, 500])
def test_preserves_http_status_errors_before_parsing_json(status: int) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(status, content=b"not JSON"))
        ) as client:
            with pytest.raises(httpx.HTTPStatusError) as error:
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("query", (_hit(1),), 1)
            assert error.value.response.status_code == status

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_rejects_redirects_even_when_client_enables_them(status: int) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/redirected":
            return httpx.Response(200, json={"scores": [{"rank": 1, "score": 1}]})
        return httpx.Response(status, headers={"Location": "https://reranker.invalid/redirected"})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
            with pytest.raises(httpx.HTTPStatusError) as error:
                await HTTPMemoryReranker("https://reranker.invalid", client=client).rerank("query", (_hit(1),), 1)
            assert error.value.response.status_code == status
            assert len(seen) == 1
            assert seen[0].method == "POST"
            assert client.follow_redirects is True

    asyncio.run(scenario())
