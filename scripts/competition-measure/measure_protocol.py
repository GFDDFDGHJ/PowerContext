#!/usr/bin/env python3
"""Replay the measurement protocol described by the challenge to check the saving.

The challenge fixes the measurement protocol: the sequence starts from eight real
meeting turns, runs forty rounds, and produces at least four distinct Contexts.
Append moments and query switches are controlled, so many consecutive rounds share
the same prepared body. This script models that protocol and reports the net
saving the platform scores.
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
    HERE.parents[1] / "competition" / "challenges" / "context-02-incremental-pack" / "workload" / "public" / "turns.json"
)
CHUNK_BYTES = 512
ROUNDS = 40
STARTING_TURNS = 8


class _CandidatePipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        return tuple(
            MemoryEntryInput(kind="conversation-turn", text=source.content, sources=(source,))
            for source in request.sources
            if isinstance(source, ContentSource)
        )


def _turn_pool(seed: int, count: int) -> list[str]:
    base = (
        "Project Manager: Okay . What are your experiences with remote controls ? "
        "I mean I've got {disfmarker} we got um {disfmarker} we had three videos , a T_V_ "
        "and a sort of amp thing all set up"
    )
    return [f"{base} [part {seed + index}]" for index in range(count)]


async def run_protocol(
    *,
    query_blocks: int,
    append_rounds: tuple[int, ...],
    chunk_bytes: int,
) -> tuple[float, int, int, int]:
    """Return (saving, incremental_bytes, raw_bytes, distinct_contexts)."""

    sample = json.loads(WORKLOAD.read_text(encoding="utf-8"))
    seed_turns = [turn["text"] for turn in sample["turns"]]
    pool = seed_turns + _turn_pool(0, 64)
    queries = ["remote", "video", "amp", "controls"]

    incremental = 0
    raw_total = 0
    gzip_total = 0
    base: ContextPack | None = None
    distinct: set[str] = set()
    captured = 0

    async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_CandidatePipeline()) as runtime:
        scope = await runtime.scopes.create(
            ScopeDraft(title="Protocol", summary="protocol", idempotency_key="protocol-measure-1")
        )
        sources = runtime.sources.for_scope(scope.scope_id)
        memory = runtime.memory.for_scope(scope.scope_id)
        context = runtime.context.for_scope(scope.scope_id)

        # The sequence starts from eight real meeting turns.
        for index in range(STARTING_TURNS):
            await sources.capture(
                CaptureSource(source_id=f"seed-{index}", content=pool[index], metadata={})
            )
            captured += 1
        while (await memory.flush()).processed:
            pass

        for round_index in range(ROUNDS):
            if round_index in append_rounds:
                await sources.capture(
                    CaptureSource(
                        source_id=f"extra-{round_index}", content=pool[captured], metadata={}
                    )
                )
                captured += 1
                while (await memory.flush()).processed:
                    pass

            block = min(query_blocks - 1, (round_index * query_blocks) // ROUNDS)
            query = queries[block % len(queries)]
            prepared = await context.prepare(PrepareContextRequest(query=query, max_bytes=4096))
            body = prepared.content or ""
            distinct.add(body)
            raw = body.encode("utf-8")
            raw_total += len(raw)
            gzip_total += len(gzip.compress(raw, compresslevel=9, mtime=0))

            target = ContextPack.from_content(body, chunk_bytes=chunk_bytes)
            if base is None:
                incremental += len(target.to_bytes())
            else:
                delta = ContextPackDelta.between(base, target)
                incremental += len(delta.to_bytes())
                rebuilt = apply_context_pack_delta(base, delta)
                assert b"".join(rebuilt.items).decode("utf-8") == body
            base = target

    return 1 - incremental / raw_total, incremental, raw_total, len(distinct)


async def main() -> int:
    scenarios = [
        ("4 query blocks, 8 appends", 4, (2, 7, 12, 17, 22, 27, 32, 37)),
        ("4 query blocks, no appends", 4, ()),
        ("8 query blocks, 8 appends", 8, (2, 7, 12, 17, 22, 27, 32, 37)),
        ("4 query blocks, append every 5 rounds", 4, tuple(range(0, ROUNDS, 5))),
        ("12 query blocks, 8 appends", 12, (2, 7, 12, 17, 22, 27, 32, 37)),
        ("20 query blocks, 8 appends", 20, (2, 7, 12, 17, 22, 27, 32, 37)),
    ]
    chunk_sizes = (64, 128, 256, 512, 1024)
    print(f"{'scenario':38s} " + "  ".join(f"chunk={c}" for c in chunk_sizes))
    for label, blocks, appends in scenarios:
        row = []
        for chunk_bytes in chunk_sizes:
            saving, _incremental, _raw_total, _distinct = await run_protocol(
                query_blocks=blocks, append_rounds=appends, chunk_bytes=chunk_bytes
            )
            row.append(f"{saving:+.4f}{'' if saving >= 0.45 else '*'}")
        print(f"{label:38s} " + "  ".join(row))
    print("\n(* = below the 0.45 threshold)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
