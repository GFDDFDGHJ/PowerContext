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

from __future__ import annotations

import asyncio
import gc
from typing import Any

import pytest

from powercontext.builtin.artifacts.memory.models import EmbeddingProfile
from powercontext.builtin.inference.coalescing import CoalescingEmbeddingModel
from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage
from powercontext.builtin.inference.protocols import embed_query


class Embedding:
    profile = EmbeddingProfile(profile_id="coalescing-test", model="offline", dimension=1)

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        self.calls.append(texts)
        return EmbeddingResult(
            vectors=tuple((float(len(text)),) for text in texts),
            usage=InferenceUsage(requests=3, input_tokens=7, output_tokens=0),
        )


def test_late_waiters_share_inflight_work_even_without_a_cache() -> None:
    class Blocked(Embedding):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
            self.started.set()
            await self.release.wait()
            return await super().embed(texts)

    async def scenario() -> None:
        base = Blocked()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=0)
        first = asyncio.create_task(model.embed(("shared", "shared")))
        await base.started.wait()
        late = asyncio.create_task(model.embed(("shared",)))
        await asyncio.sleep(0)
        base.release.set()
        one, two = await asyncio.gather(first, late)
        assert one.vectors == ((6.0,), (6.0,))
        assert two.vectors == ((6.0,),)
        assert one.usage == two.usage == InferenceUsage(requests=3, input_tokens=7, output_tokens=0)
        assert base.calls == [("shared",)]
        await model.embed(("shared",))
        assert base.calls == [("shared",), ("shared",)]

    asyncio.run(scenario())


def test_all_texts_are_registered_before_independent_batches_start() -> None:
    class Independent(Embedding):
        def __init__(self) -> None:
            super().__init__()
            self.started = {key: asyncio.Event() for key in ("slow", "fast")}
            self.release = {key: asyncio.Event() for key in ("slow", "fast")}

        async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
            key = texts[0]
            self.started[key].set()
            await self.release[key].wait()
            return await super().embed(texts)

    async def scenario() -> None:
        base = Independent()
        model = CoalescingEmbeddingModel(base, max_batch_size=1, coalesce_ms=0, cache_size=0)
        first = asyncio.create_task(model.embed(("slow", "fast", "fast")))
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in base.started.values())), timeout=2)
        late = asyncio.create_task(model.embed(("fast",)))
        await asyncio.sleep(0)
        base.release["fast"].set()
        assert (await asyncio.wait_for(late, timeout=2)).vectors == ((4.0,),)
        assert not first.done()
        base.release["slow"].set()
        result = await first
        assert result.vectors == ((4.0,), (4.0,), (4.0,))
        assert result.usage == InferenceUsage(requests=6, input_tokens=14, output_tokens=0)
        assert sorted(base.calls) == [("fast",), ("slow",)]

    asyncio.run(scenario())


def test_failed_batch_does_not_block_successful_work_and_can_be_retried() -> None:
    class Flaky(Embedding):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
            if texts == ("bad",) and self.fail:
                self.fail = False
                self.calls.append(texts)
                raise RuntimeError("temporary")
            return await super().embed(texts)

    async def scenario() -> None:
        base = Flaky()
        model = CoalescingEmbeddingModel(base, max_batch_size=1, coalesce_ms=0)
        mixed, good = await asyncio.gather(model.embed(("bad", "good")), model.embed(("good",)), return_exceptions=True)
        assert isinstance(mixed, RuntimeError)
        assert isinstance(good, EmbeddingResult)
        assert good.vectors == ((4.0,),)
        assert (await model.embed(("good",))).usage.requests == 0
        assert (await model.embed(("bad",))).vectors == ((3.0,),)
        assert base.calls.count(("bad",)) == 2
        assert base.calls.count(("good",)) == 1

    asyncio.run(scenario())


def test_vector_count_mismatch_does_not_cache_any_part_of_the_batch() -> None:
    class InvalidOnce(Embedding):
        async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
            if not self.calls:
                self.calls.append(texts)
                return EmbeddingResult(vectors=((999.0,),))
            return await super().embed(texts)

    async def scenario() -> None:
        base = InvalidOnce()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0)
        with pytest.raises(ValueError):
            await model.embed(("first", "second"))
        assert (await model.embed(("first", "second"))).vectors == ((5.0,), (6.0,))
        assert base.calls == [("first", "second"), ("first", "second")]

    asyncio.run(scenario())


def test_provider_cancellation_cleans_up_shared_work_for_retry() -> None:
    class CancelOnce(Embedding):
        async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
            if texts == ("cancel",) and texts not in self.calls:
                self.calls.append(texts)
                raise asyncio.CancelledError
            return await super().embed(texts)

    async def scenario() -> None:
        base = CancelOnce()
        model = CoalescingEmbeddingModel(base, max_batch_size=1, coalesce_ms=0)
        cancelled, good = await asyncio.gather(model.embed(("cancel",)), model.embed(("good",)), return_exceptions=True)
        assert isinstance(cancelled, asyncio.CancelledError)
        assert isinstance(good, EmbeddingResult)
        assert good.vectors == ((4.0,),)
        assert (await asyncio.wait_for(model.embed(("cancel",)), timeout=2)).vectors == ((6.0,),)
        assert base.calls.count(("cancel",)) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("fail", [False, True])
def test_cancelled_sole_waiter_leaves_shared_work_safe_and_observes_background_errors(fail: bool) -> None:
    class Blocked(Embedding):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = asyncio.Event()

        async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
            self.started.set()
            await self.release.wait()
            try:
                if fail:
                    raise RuntimeError("unobserved")
                return await super().embed(texts)
            finally:
                self.finished.set()

    async def scenario() -> None:
        base = Blocked()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0)
        loop = asyncio.get_running_loop()
        errors: list[dict[str, Any]] = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        try:
            caller = asyncio.create_task(model.embed(("shared",)))
            await base.started.wait()
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            base.release.set()
            await base.finished.wait()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            gc.collect()
            assert errors == []
            if not fail:
                assert (await model.embed(("shared",))).usage.requests == 0
                assert base.calls == [("shared",)]
        finally:
            loop.set_exception_handler(previous_handler)

    asyncio.run(scenario())


def test_usage_counts_distinct_batches_even_when_their_usage_values_are_identical() -> None:
    async def scenario() -> None:
        base = Embedding()
        model = CoalescingEmbeddingModel(base, max_batch_size=1, coalesce_ms=0)
        first, second = await asyncio.gather(model.embed(("alpha", "beta", "alpha")), model.embed(("beta", "gamma")))
        assert first.usage == second.usage == InferenceUsage(requests=6, input_tokens=14, output_tokens=0)
        mixed = await model.embed(("alpha", "delta", "delta"))
        assert mixed.usage == InferenceUsage(requests=3, input_tokens=7, output_tokens=0)
        assert (await model.embed(("alpha", "delta"))).usage.requests == 0
        assert len(base.calls) == 4

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("token_values", "expected"),
    [((None, None), None), ((None, 7), None), ((0, 0), 0), ((0, 7), 7)],
)
def test_usage_preserves_unknown_and_zero_tokens(token_values: tuple[int | None, ...], expected: int | None) -> None:
    class Metered(Embedding):
        async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
            result = await super().embed(texts)
            tokens = token_values[len(self.calls) - 1]
            return EmbeddingResult(
                vectors=result.vectors,
                usage=InferenceUsage(requests=2, input_tokens=tokens, output_tokens=tokens),
            )

    async def scenario() -> None:
        model = CoalescingEmbeddingModel(Metered(), max_batch_size=1, coalesce_ms=0)
        result = await model.embed(("first", "second", "first"))
        assert result.usage == InferenceUsage(requests=4, input_tokens=expected, output_tokens=expected)

    asyncio.run(scenario())


def test_asymmetric_query_embeddings_use_separate_work_and_cache_entries() -> None:
    class Asymmetric(Embedding):
        def __init__(self) -> None:
            super().__init__()
            self.query_calls: list[tuple[str, ...]] = []

        async def embed_query(self, texts: tuple[str, ...], /) -> EmbeddingResult:
            self.query_calls.append(texts)
            return EmbeddingResult(
                vectors=tuple((-float(len(text)),) for text in texts), usage=InferenceUsage(requests=1)
            )

    async def scenario() -> None:
        base = Asymmetric()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0)
        document, query, shared_query = await asyncio.gather(
            model.embed(("same",)), embed_query(model, ("same",)), model.embed_query(("same", "same"))
        )
        assert document.vectors == ((4.0,),)
        assert query.vectors == ((-4.0,),)
        assert shared_query.vectors == ((-4.0,), (-4.0,))
        assert query.usage.requests == shared_query.usage.requests == 1
        assert (await model.embed(("same",))).usage.requests == 0
        assert (await model.embed_query(("same",))).usage.requests == 0
        assert base.calls == base.query_calls == [("same",)]

    asyncio.run(scenario())


def test_symmetric_query_embeddings_share_document_work_and_cache() -> None:
    async def scenario() -> None:
        base = Embedding()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0)
        document, query = await asyncio.gather(model.embed(("same",)), embed_query(model, ("same",)))
        assert document == query
        assert (await model.embed_query(("same",))).usage.requests == 0
        assert base.calls == [("same",)]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "options",
    [
        {"max_batch_size": 0},
        {"max_batch_size": -1},
        {"max_batch_size": 1.5},
        {"max_batch_size": True},
        {"cache_size": -1},
        {"cache_size": 1.5},
        {"cache_size": True},
        {"coalesce_ms": -1},
        {"coalesce_ms": float("inf")},
        {"coalesce_ms": float("nan")},
        {"coalesce_ms": True},
        {"coalesce_ms": "5"},
        {"coalesce_ms": None},
    ],
)
def test_invalid_limits_are_rejected_before_work_can_be_queued(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CoalescingEmbeddingModel(Embedding(), **options)
