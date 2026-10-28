#!/usr/bin/env python3
"""Break the incremental cost down per round so the expensive rounds are visible."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from powercontext.builtin.artifacts.memory import MemoryCandidateRequest, MemoryEntryInput  # noqa: E402
from powercontext.builtin.runtime import (  # noqa: E402
    BuiltinConfig,
    CaptureSource,
    PrepareContextRequest,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.context_pack import ContextPack, ContextPackDelta  # noqa: E402
from powercontext.builtin.scope import ScopeDraft  # noqa: E402
from powercontext.builtin.sources import ContentSource  # noqa: E402

WORKLOAD = (
    HERE.parents[1] / "competition" / "challenges" / "context-02-incremental-pack" / "workload" / "public" / "turns.json"
)
CHUNK_BYTES = 128


class _CandidatePipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        return tuple(
            MemoryEntryInput(kind="conversation-turn", text=source.content, sources=(source,))
            for source in request.sources
            if isinstance(source, ContentSource)
        )


async def main() -> int:
    sample = json.loads(WORKLOAD.read_text(encoding="utf-8"))
    pool = [turn["text"] for turn in sample["turns"]] + [
        f"{sample['turns'][0]['text']} [part {index}]" for index in range(64)
    ]
    queries = ["remote", "video", "amp", "controls"]

    base: ContextPack | None = None
    changed_total = 0
    unchanged_total = 0
    raw_total = 0
    changed_rounds = 0

    async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_CandidatePipeline()) as runtime:
        scope = await runtime.scopes.create(
            ScopeDraft(title="Break", summary="break", idempotency_key="break-1")
        )
        sources = runtime.sources.for_scope(scope.scope_id)
        memory = runtime.memory.for_scope(scope.scope_id)
        context = runtime.context.for_scope(scope.scope_id)
        for index in range(8):
            await sources.capture(CaptureSource(source_id=f"seed-{index}", content=pool[index], metadata={}))
        while (await memory.flush()).processed:
            pass

        for round_index in range(40):
            if round_index in (2, 7, 12, 17, 22, 27, 32, 37):
                await sources.capture(
                    CaptureSource(source_id=f"x-{round_index}", content=pool[8 + round_index], metadata={})
                )
                while (await memory.flush()).processed:
                    pass
            query = queries[min(3, (round_index * 4) // 40)]
            prepared = await context.prepare(PrepareContextRequest(query=query, max_bytes=4096))
            body = prepared.content or ""
            raw_total += len(body.encode("utf-8"))
            target = ContextPack.from_content(body, chunk_bytes=CHUNK_BYTES)
            if base is None:
                cost = len(target.to_bytes())
            else:
                delta = ContextPackDelta.between(base, target)
                cost = len(delta.to_bytes())
                changed = delta.prefix_items < len(base.items)
                if changed:
                    changed_rounds += 1
                    changed_total += cost
                else:
                    unchanged_total += cost
            base = target

    print(f"raw_total         = {raw_total}")
    print(f"changed rounds    = {changed_rounds}, bytes = {changed_total} (avg {changed_total // max(changed_rounds,1)})")
    print(f"unchanged rounds  = {40 - changed_rounds - 1}, bytes = {unchanged_total} (avg {unchanged_total // max(40 - changed_rounds - 1, 1)})")
    print(f"total incremental = {changed_total + unchanged_total + len(base.to_bytes())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
