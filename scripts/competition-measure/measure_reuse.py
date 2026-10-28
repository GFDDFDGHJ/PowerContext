#!/usr/bin/env python3
"""Measure how much of a changed round can actually be reused.

For every pair of consecutive prepared bodies this reports the byte-level
common prefix and the number of leading chunks that survive, which is what the
protocol can actually reuse.
"""

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
from powercontext.builtin.runtime.context_pack import (  # noqa: E402
    ContextPack,
    ContextPackDelta,
    chunk_against_base,
)
from powercontext.builtin.scope import ScopeDraft  # noqa: E402
from powercontext.builtin.sources import ContentSource  # noqa: E402

WORKLOAD = (
    HERE.parents[1] / "competition" / "challenges" / "context-02-incremental-pack" / "workload" / "public" / "turns.json"
)


class _CandidatePipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        return tuple(
            MemoryEntryInput(kind="conversation-turn", text=source.content, sources=(source,))
            for source in request.sources
            if isinstance(source, ContentSource)
        )


def _common_prefix(left: bytes, right: bytes) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


async def main() -> int:
    sample = json.loads(WORKLOAD.read_text(encoding="utf-8"))
    pool = [turn["text"] for turn in sample["turns"]] + [
        f"{sample['turns'][0]['text']} [part {index}]" for index in range(64)
    ]
    queries = ["remote", "video", "amp", "controls"]
    bodies: list[str] = []

    async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_CandidatePipeline()) as runtime:
        scope = await runtime.scopes.create(
            ScopeDraft(title="Reuse", summary="reuse", idempotency_key="reuse-1")
        )
        sources = runtime.sources.for_scope(scope.scope_id)
        memory = runtime.memory.for_scope(scope.scope_id)
        context = runtime.context.for_scope(scope.scope_id)
        for index in range(8):
            await sources.capture(CaptureSource(source_id=f"seed-{index}", content=pool[index], metadata={}))
        while (await memory.flush()).processed:
            pass
        for round_index in range(12):
            if round_index in (2, 7):
                await sources.capture(
                    CaptureSource(source_id=f"x-{round_index}", content=pool[8 + round_index], metadata={})
                )
                while (await memory.flush()).processed:
                    pass
            query = queries[min(3, (round_index * 4) // 12)]
            prepared = await context.prepare(PrepareContextRequest(query=query, max_bytes=4096))
            bodies.append(prepared.content or "")

    for chunk_bytes in (128, 512):
        print(f"=== chunk={chunk_bytes} ===")
        base_pack: ContextPack | None = None
        for index in range(len(bodies) - 1):
            base = base_pack if base_pack is not None else chunk_against_base(bodies[index], None, chunk_bytes=chunk_bytes)
            target = chunk_against_base(bodies[index + 1], base, chunk_bytes=chunk_bytes)
            delta = ContextPackDelta.between(base, target)
            byte_prefix = _common_prefix(
                bodies[index].encode("utf-8"), bodies[index + 1].encode("utf-8")
            )
            reused = sum(len(item) for item in base.items[: delta.prefix_items])
            total = len(b"".join(base.items))
            reused_ratio = reused / max(total, 1)
            print(
                f"  round {index:2d}->{index+1:2d}: byte_prefix={byte_prefix:5d} "
                f"reusable_chunks={delta.prefix_items}/{len(base.items)} reused_bytes={reused:5d} "
                f"({reused_ratio:.1%}) appended={len(delta.appended)} delta_bytes={delta.byte_count}"
            )
            base_pack = target
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
