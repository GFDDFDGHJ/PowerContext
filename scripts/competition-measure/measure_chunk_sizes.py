#!/usr/bin/env python3
"""Compare the incremental saving across candidate chunk sizes.

The platform scores the net saving over a long meeting sequence, so this
script replays a realistic sequence through the real Runtime and reports the
saving for several chunk sizes. It also reports how the body changes between
rounds, because that decides how much of the body can be reused at all.
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
CHUNK_SIZES = (64, 128, 256, 512, 1024)


class _CandidatePipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        return tuple(
            MemoryEntryInput(kind="conversation-turn", text=source.content, sources=(source,))
            for source in request.sources
            if isinstance(source, ContentSource)
        )


def _synthetic_turns(seed: int, count: int) -> list[str]:
    """Distinct meeting turns so the sequence behaves like a real growing meeting."""

    base = (
        "Project Manager: Okay . What are your experiences with remote controls ? "
        "I mean I've got {disfmarker} we got um {disfmarker} we had three videos , a T_V_ "
        "and a sort of amp thing all set up"
    )
    return [f"{base} [part {seed + index}]" for index in range(count)]


async def collect_bodies(rounds: int = 40) -> list[str]:
    """Run the real Runtime and return the prepared body of every round."""

    sample = json.loads(WORKLOAD.read_text(encoding="utf-8"))
    seed_turns = [turn["text"] for turn in sample["turns"]]
    turns = seed_turns + _synthetic_turns(0, rounds)

    bodies: list[str] = []
    async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_CandidatePipeline()) as runtime:
        scope = await runtime.scopes.create(
            ScopeDraft(title="Chunks", summary="chunks", idempotency_key="chunk-measure-1")
        )
        sources = runtime.sources.for_scope(scope.scope_id)
        memory = runtime.memory.for_scope(scope.scope_id)
        context = runtime.context.for_scope(scope.scope_id)
        for index in range(rounds):
            await sources.capture(
                CaptureSource(source_id=f"turn-{index}", content=turns[index], metadata={})
            )
            while (await memory.flush()).processed:
                pass
            prepared = await context.prepare(PrepareContextRequest(query="remote", max_bytes=4096))
            bodies.append(prepared.content or "")
    return bodies


def _longest_common_prefix(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def measure(bodies: list[str], chunk_bytes: int) -> tuple[int, int, int]:
    incremental = 0
    raw_total = 0
    gzip_total = 0
    base: ContextPack | None = None
    for body in bodies:
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
    return incremental, raw_total, gzip_total


async def main() -> int:
    bodies = await collect_bodies()
    print(f"rounds={len(bodies)} distinct={len(set(bodies))}")
    print(f"body sizes: min={min(len(b) for b in bodies)} max={max(len(b) for b in bodies)}")
    lcp = [_longest_common_prefix(bodies[i], bodies[i + 1]) for i in range(len(bodies) - 1)]
    print(f"byte-level common prefix between rounds: min={min(lcp)} max={max(lcp)} avg={sum(lcp)//len(lcp)}")
    print()
    for chunk_bytes in CHUNK_SIZES:
        incremental, raw_total, gzip_total = measure(bodies, chunk_bytes)
        saving = 1 - incremental / raw_total
        print(
            f"chunk={chunk_bytes:5d}  incremental={incremental:7d} raw={raw_total:7d} "
            f"saving={saving:+.4f}  vs_gzip={1 - incremental / gzip_total:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
