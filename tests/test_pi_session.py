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
import json
from copy import deepcopy
from pathlib import Path

import pytest

from powercontext.sources import SourceDefinitionRegistry, SourceMaterialization, SourceRef, TextEvidence
from powercontext.sources.connectors import ConnectorBinding, ConnectorLifecycle, ConnectorRunStatus
from powercontext.sources.pi_session import (
    PI_EVENT_SOURCE_DEFINITION,
    PiEventInput,
    PiSessionConnector,
    PiSessionInput,
    parse_pi_session,
    pi_event_source_id,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY

_HEADER = {"type": "session", "version": 3, "id": "session-under-test"}


def _archive(*records: dict) -> bytes:
    return "\n".join(json.dumps(record) for record in (_HEADER, *records)).encode()


def _parse(records: list[dict], leaf_id: str | None = None) -> tuple[PiEventInput, ...]:
    return parse_pi_session(PiSessionInput(content=_archive(*records), leaf_id=leaf_id))


def test_selected_branch_keeps_visible_events_and_skips_control_nodes() -> None:
    records = [
        {"type": "message", "id": "a", "parentId": None, "message": {"role": "user", "content": "alpha"}},
        {"type": "thinking_level_change", "id": "b", "parentId": "a"},
        {"type": "message", "id": "c", "parentId": "b", "message": {"role": "assistant", "content": "beta"}},
    ]
    events = _parse(records, "c")
    assert [event.event_id for event in events] == ["a", "c"]
    assert [event.parent_id for event in events] == [None, "b"]
    assert [event.content for event in events] == ["user: alpha", "assistant: beta"]
    assert events[0].event_type == "message"
    assert events[0].record == records[0]


def test_tool_calls_and_results_keep_their_complete_arguments_and_output() -> None:
    records = [
        {
            "type": "message",
            "id": "call",
            "parentId": None,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hidden"},
                    {"type": "toolCall", "name": "read", "id": "call-1", "arguments": {"path": "表.txt", "b": 2}},
                ],
            },
        },
        {
            "type": "message",
            "id": "result",
            "parentId": "call",
            "message": {
                "role": "toolResult",
                "toolName": "read",
                "toolCallId": "call-1",
                "content": [{"type": "text", "text": "文件内容"}],
            },
        },
    ]
    events = _parse(records)
    assert [event.content for event in events] == [
        'assistant: [tool_call read call-1] {"b":2,"path":"表.txt"}',
        "toolResult: [tool_result read call-1] 文件内容",
    ]


def test_bash_custom_and_summary_events_use_their_own_roles() -> None:
    records = [
        {
            "type": "message",
            "id": "bash",
            "parentId": None,
            "message": {"role": "bashExecution", "command": "printf result", "output": "result", "content": []},
        },
        {
            "type": "custom_message",
            "id": "custom",
            "parentId": "bash",
            "content": [{"type": "text", "text": "Injected"}],
        },
        {"type": "compaction", "id": "compaction", "parentId": "custom", "summary": "summary"},
        {"type": "branch_summary", "id": "branch", "parentId": "compaction", "summary": "branch summary"},
    ]
    events = _parse(records)
    assert [event.role for event in events] == ["bashExecution", "custom", "compaction", "branch_summary"]
    assert [event.content for event in events] == [
        "bashExecution: $ printf result\nresult",
        "custom: Injected",
        "compaction: summary",
        "branch_summary: branch summary",
    ]


def test_events_without_a_visible_body_are_ignored_but_keep_their_ancestry() -> None:
    records = [
        {"type": "message", "id": "a", "parentId": None, "message": {"role": "assistant", "content": []}},
        {"type": "model_change", "id": "b", "parentId": "a"},
        {"type": "message", "id": "c", "parentId": "b", "message": {"role": "user", "content": "visible"}},
    ]
    events = _parse(records, "c")
    assert [event.event_id for event in events] == ["c"]
    assert events[0].parent_id == "b"


def test_bom_and_crlf_archives_parse_like_plain_utf8() -> None:
    raw = b"\xef\xbb\xbf" + _archive({
        "type": "message",
        "id": "a",
        "parentId": None,
        "message": {"role": "user", "content": "hello"},
    }).replace(b"\n", b"\r\n")
    events = parse_pi_session(PiSessionInput(content=raw))
    assert [event.content for event in events] == ["user: hello"]


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"{}\n",
        b"[]\n",
        b"null\n",
        b"\xff",
        b'{"type":"session","version":2,"id":"x"}\n',
        b'{"type":"session","version":3,"id":""}\n',
        b'{"type":"session","version":3,"id":"x"}\n{"type":"message","id":"a","parentId":"b"}\n',
        b'{"type":"session","version":3,"id":"x"}\n{"type":"message","id":"a","parentId":"a"}\n',
        b'{"type":"session","version":3,"id":"x"}\n{"type":"message","id":"a","parentId":null}\n'
        b'{"type":"message","id":"a","parentId":null}\n',
        b'{"type":"session","version":3,"id":"x"}\n{"type":"message","id":"a","parentId":null}\n{bad}\n',
    ],
)
def test_invalid_archives_fail_without_partial_results(raw: bytes) -> None:
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=raw))


def test_missing_leaf_is_rejected() -> None:
    with pytest.raises(ValueError):
        _parse([{"type": "message", "id": "a", "parentId": None, "message": {"role": "user", "content": "x"}}], "gone")


def test_identity_is_stable_bounded_and_collision_free() -> None:
    first = pi_event_source_id("session", "event")
    assert first == pi_event_source_id("session", "event")
    assert first != pi_event_source_id("session", "other")
    assert first != pi_event_source_id("other", "event")
    assert len(pi_event_source_id("s" * 200, "e" * 200)) <= 256
    assert pi_event_source_id("s" * 200, "e" * 200) != pi_event_source_id("s" * 200, "f" * 200)


def test_projection_exposes_the_complete_event_text_and_identity() -> None:
    events = _parse([{"type": "message", "id": "a", "parentId": None, "message": {"role": "user", "content": "alpha"}}])
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    source = asyncio.run(registry.resolve(events[0]))
    assert source.materialization is SourceMaterialization.CAPTURED
    assert source.name == source.source_id
    assert asyncio.run(registry.read(source)) == events[0]
    evidence = TextEvidence.model_validate(registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY))
    assert evidence.source_type == "pi-event"
    assert evidence.source_id == source.name
    assert evidence.content == "user: alpha"
    assert evidence.metadata == {
        "session_id": "session-under-test",
        "event_id": "a",
        "parent_id": None,
        "event_type": "message",
        "role": "user",
    }


class _RecordingSink:
    def __init__(self) -> None:
        self.submitted: list[tuple[str, str, object]] = []

    async def submit(self, binding, item_id, definition_name, value, /):
        self.submitted.append((item_id, definition_name, value))
        return SourceRef(source_type=definition_name, source_id=item_id)


class _MemoryCheckpoints:
    def __init__(self) -> None:
        self.value = None

    async def load(self, binding):
        return self.value

    async def save(self, binding, checkpoint, *, expected):
        if self.value != expected:
            raise RuntimeError("checkpoint changed concurrently")  # noqa: TRY003
        self.value = checkpoint


def _write_archive(path: Path, records: list[dict], session_id: str = "session-under-test") -> None:
    header = {"type": "session", "version": 3, "id": session_id}
    path.write_text(
        "\n".join(json.dumps(record) for record in (header, *records)) + "\n",
        encoding="utf-8",
    )


def test_connector_imports_new_branch_events_once_and_keeps_shared_ancestry(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _write_archive(
        path,
        [
            {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "shared"}},
            {
                "type": "message",
                "id": "left",
                "parentId": "root",
                "message": {"role": "assistant", "content": "left"},
            },
            {"type": "model_change", "id": "control", "parentId": "root"},
            {
                "type": "message",
                "id": "right",
                "parentId": "control",
                "message": {"role": "assistant", "content": "right"},
            },
        ],
    )
    sink = _RecordingSink()
    checkpoints = _MemoryCheckpoints()
    binding = ConnectorBinding(
        scope_id="scope", binding_id="pi", connector_name="pi-session-history", connector_version="1"
    )

    async def run(leaf_id: str | None):
        return await ConnectorLifecycle(sink=sink, checkpoints=checkpoints).run(
            PiSessionConnector(path, leaf_id), binding
        )

    first = asyncio.run(run("left"))
    assert [value.content for _, _, value in sink.submitted] == ["user: shared", "assistant: left"]
    assert first.status is ConnectorRunStatus.COMPLETE

    asyncio.run(run("right"))
    assert [value.content for _, _, value in sink.submitted] == [
        "user: shared",
        "assistant: left",
        "assistant: right",
    ]

    replay = asyncio.run(run("right"))
    assert len(sink.submitted) == 3
    assert replay.previous_checkpoint == replay.committed_checkpoint


def test_connector_keeps_progress_when_the_archive_is_broken_or_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    records = [
        {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "shared"}},
        {
            "type": "message",
            "id": "tail",
            "parentId": "root",
            "message": {"role": "assistant", "content": "tail"},
        },
    ]
    _write_archive(path, records)
    sink = _RecordingSink()
    checkpoints = _MemoryCheckpoints()
    binding = ConnectorBinding(
        scope_id="scope", binding_id="pi", connector_name="pi-session-history", connector_version="1"
    )

    async def run():
        return await ConnectorLifecycle(sink=sink, checkpoints=checkpoints).run(PiSessionConnector(path), binding)

    committed = asyncio.run(run()).committed_checkpoint
    assert committed is not None

    path.write_text(path.read_text(encoding="utf-8") + "{bad json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        asyncio.run(run())
    assert checkpoints.value == committed

    _write_archive(path, records)
    rewritten = deepcopy(records)
    rewritten[0]["message"]["content"] = "changed"
    _write_archive(path, rewritten)
    with pytest.raises(ValueError):
        asyncio.run(run())
    assert checkpoints.value == committed

    _write_archive(path, records)
    stable = asyncio.run(run())
    assert stable.previous_checkpoint == committed
    assert stable.items == ()
