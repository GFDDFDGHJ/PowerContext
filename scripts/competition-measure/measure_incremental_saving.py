#!/usr/bin/env python3
"""Measure the incremental transport saving with the real runtime.

The public acceptance test only checks the protocol. The platform additionally
scores the net saving over a longer meeting sequence, so this script drives the
real Runtime: it captures turns one by one and measures what an incremental
client would actually send.
"""

from __future__ import annotations

import asyncio
import gzip
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
    apply_context_pack_delta,
)
from powercontext.builtin.scope import ScopeDraft  # noqa: E402
from powercontext.builtin.sources import ContentSource  # noqa: E402

WORKLOAD = (
    HERE
    / "challenges"
    / "context-02-incremental-pack"
    / "workload"
    / "public"
    / "turns.json"
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
    query = sample["query"]

    incremental_bytes = 0
    raw_full_bytes = 0
    gzip_bytes = 0
    base: ContextPack | None = None
    distinct: set[str] = set()

    async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_CandidatePipeline()) as runtime:
        scope = await runtime.scopes.create(
            ScopeDraft(
                title="Saving measurement",
                summary="measure",
                idempotency_key="context-02-incremental-pack-saving",
            )
        )
        sources = runtime.sources.for_scope(scope.scope_id)
        memory = runtime.memory.for_scope(scope.scope_id)
        context = runtime.context.for_scope(scope.scope_id)

        for round_index in range(40):
            turn = turns[round_index % len(turns)]
            await sources.capture(
                CaptureSource(source_id=f"turn-{round_index}", content=turn, metadata={"i": round_index})
            )
            while (await memory.flush()).processed:
                pass

            request = PrepareContextRequest(query=query, max_bytes=4096)
            prepared = await context.prepare(request)
            raw = (prepared.content or "").encode("utf-8")
            raw_full_bytes += len(raw)
            gzip_bytes += len(gzip.compress(raw, compresslevel=9, mtime=0))
            distinct.add(prepared.content or "")

            target = ContextPack.from_content(prepared.content or "")
            if base is None:
                incremental_bytes += len(target.to_bytes())
            else:
                delta = ContextPackDelta.between(base, target)
                incremental_bytes += len(delta.to_bytes())
                rebuilt = apply_context_pack_delta(
                    ContextPack.from_bytes(base.to_bytes()), delta
                )
                if b"".join(rebuilt.items).decode("utf-8") != (prepared.content or ""):
                    print(f"round {round_index}: rebuild mismatch", file=sys.stderr)
                    return 1
            base = target

    saving = 1 - incremental_bytes / raw_full_bytes
    gzip_saving = 1 - incremental_bytes / gzip_bytes
    print(f"distinct contexts = {len(distinct)}")
    print(f"raw_full_bytes    = {raw_full_bytes}")
    print(f"incremental_bytes = {incremental_bytes}")
    print(f"gzip_bytes        = {gzip_bytes}")
    print(f"net saving vs raw  = {saving:.4f} (need >= 0.45)")
    print(f"net saving vs gzip = {gzip_saving:.4f}")
    return 0 if saving >= 0.45 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
