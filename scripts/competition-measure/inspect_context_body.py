#!/usr/bin/env python3
"""Print the prepared context body for a few rounds so its structure is visible."""

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


async def main() -> int:
    sample = json.loads(WORKLOAD.read_text(encoding="utf-8"))
    turns = [turn["text"] for turn in sample["turns"]]
    async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_CandidatePipeline()) as runtime:
        scope = await runtime.scopes.create(
            ScopeDraft(title="Structure", summary="structure", idempotency_key="structure-1")
        )
        sources = runtime.sources.for_scope(scope.scope_id)
        memory = runtime.memory.for_scope(scope.scope_id)
        context = runtime.context.for_scope(scope.scope_id)
        for index in range(3):
            await sources.capture(
                CaptureSource(source_id=f"turn-{index}", content=turns[index % 2], metadata={})
            )
            while (await memory.flush()).processed:
                pass
            prepared = await context.prepare(PrepareContextRequest(query="remote", max_bytes=4096))
            content = prepared.content or ""
            print(f"=== round {index} len={len(content)} ===")
            print(content[:900])
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
