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

"""Share concurrent embedding work and retain successful vectors in a bounded LRU."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, TypeAlias

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel, QueryEmbeddingModel, embed_query

_TextKey: TypeAlias = tuple[bool, str]


@dataclass(frozen=True, slots=True, eq=False)
class _BatchUsage:
    # Identity, rather than usage equality, distinguishes separately billed batches.
    usage: InferenceUsage


@dataclass(frozen=True, slots=True)
class _TextResult:
    vector: EmbeddingVector
    batch: _BatchUsage


_Batch: TypeAlias = tuple[tuple[_TextKey, asyncio.Future[_TextResult]], ...]


def _consume_exception(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


def _token_total(values: tuple[int | None, ...]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


class CoalescingEmbeddingModel:
    """Coalesce embedding calls sharing one event loop without sharing cancellation."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        if (
            isinstance(coalesce_ms, bool)
            or not isinstance(coalesce_ms, (int, float))
            or not math.isfinite(coalesce_ms)
            or coalesce_ms < 0
        ):
            raise ValueError("coalesce_ms must be finite and non-negative")  # noqa: TRY003
        self.profile = model.profile
        self._model = model
        self._separate_query = isinstance(model, QueryEmbeddingModel)
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[_TextKey, EmbeddingVector] = OrderedDict()
        self._inflight: dict[_TextKey, asyncio.Future[_TextResult]] = {}
        self._pending: dict[_TextKey, asyncio.Future[_TextResult]] = {}
        self._dispatcher: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return document vectors in the original order, including duplicate positions."""

        return await self._embed(texts, query=False)

    async def embed_query(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Preserve asymmetric query embedding semantics when the delegate supports them."""

        return await self._embed(texts, query=self._separate_query)

    async def _embed(self, texts: tuple[str, ...], *, query: bool) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=())

        loop = asyncio.get_running_loop()
        items: list[EmbeddingVector | asyncio.Future[_TextResult]] = []
        waiting: dict[asyncio.Future[_TextResult], None] = {}
        # Register every text before yielding, including texts beyond the first batch.
        for text in texts:
            key = (query, text)
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                items.append(cached)
                continue
            future = self._inflight.get(key)
            if future is None:
                future = loop.create_future()
                future.add_done_callback(_consume_exception)
                self._inflight[key] = future
                self._pending[key] = future
            items.append(future)
            waiting[future] = None

        if self._pending and self._dispatcher is None:
            self._dispatcher = self._start(self._dispatch())
        if waiting:
            await asyncio.gather(*(asyncio.shield(future) for future in waiting))

        vectors: list[EmbeddingVector] = []
        batches: set[_BatchUsage] = set()
        for item in items:
            if isinstance(item, asyncio.Future):
                result = item.result()
                vectors.append(result.vector)
                batches.add(result.batch)
            else:
                vectors.append(item)
        usages = tuple(batch.usage for batch in batches)
        return EmbeddingResult(
            vectors=tuple(vectors),
            usage=InferenceUsage(
                requests=sum(usage.requests for usage in usages),
                input_tokens=_token_total(tuple(usage.input_tokens for usage in usages)),
                output_tokens=_token_total(tuple(usage.output_tokens for usage in usages)),
            ),
        )

    def _start(self, coroutine: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(_consume_exception)
        return task

    async def _dispatch(self) -> None:
        try:
            # A zero window still yields so concurrently registered calls can merge.
            await asyncio.sleep(self._coalesce_seconds)
        except asyncio.CancelledError as error:
            self._fail_batch(tuple(self._pending.items()), error)
            self._pending = {}
            self._dispatcher = None
            raise
        pending, self._pending = self._pending, {}
        self._dispatcher = None
        for query in (False, True):
            work = tuple((key, future) for key, future in pending.items() if key[0] == query)
            for start in range(0, len(work), self._max_batch_size):
                self._start(self._run_batch(work[start : start + self._max_batch_size], query=query))

    def _fail_batch(self, batch: _Batch, error: Exception | asyncio.CancelledError) -> None:
        for key, future in batch:
            self._inflight.pop(key, None)
            if isinstance(error, asyncio.CancelledError):
                future.cancel()
            else:
                future.set_exception(error)

    async def _run_batch(self, batch: _Batch, *, query: bool) -> None:
        texts = tuple(key[1] for key, _future in batch)
        try:
            result = await embed_query(self._model, texts) if query else await self._model.embed(texts)
        except asyncio.CancelledError as error:
            self._fail_batch(batch, error)
            raise
        except Exception as error:
            self._fail_batch(batch, error)
            return

        if len(result.vectors) != len(batch):
            self._fail_batch(batch, ValueError("Provider returned the wrong embedding vector count"))
            return
        usage = _BatchUsage(result.usage)
        for (key, future), vector in zip(batch, result.vectors, strict=True):
            if self._cache_size:
                self._cache[key] = vector
                self._cache.move_to_end(key)
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            self._inflight.pop(key, None)
            future.set_result(_TextResult(vector, usage))


__all__ = ["CoalescingEmbeddingModel"]
