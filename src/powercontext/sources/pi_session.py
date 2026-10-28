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

"""Capture the visible events of one Pi session branch as durable evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel, JsonValue, field_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorItemOutcome,
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_SOURCE_NAME = "pi-event"
"""The stable Source Definition name for one captured Pi session event."""

PI_CONNECTOR_NAME = "pi-session-history"
"""The stable Connector name for incremental Pi session branch collection."""

PI_CONNECTOR_VERSION = "1"
"""The durable version of the Pi session Connector contract."""

PI_SOURCE_VERSION = "1"
"""The durable version of the Pi event Source Definition."""

_PI_SESSION_VERSION = 3
_PI_CHECKPOINT_VERSION = 1
_PI_IDENTITY_PREFIX = "pi:"


class PiSessionInput(BaseModel):
    """One complete Pi session archive with its optionally selected leaf node."""

    content: bytes
    leaf_id: str | None = None

    @field_validator("leaf_id")
    @classmethod
    def validate_leaf_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("leaf_id must not be blank when it is provided")  # noqa: TRY003
        return value


class PiEventInput(BaseModel):
    """One visible event of a Pi session branch with its complete input record."""

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]

    @field_validator("session_id", "event_id", "event_type", "role")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Pi event identity parts must be non-empty strings")  # noqa: TRY003
        return value


class PiEventSource(Source):
    """Captured original record for one visible event of a Pi session branch."""

    materialization: Literal[SourceMaterialization.CAPTURED] = SourceMaterialization.CAPTURED
    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Encode one session and event identity without delimiter collisions."""

    if not session_id or not event_id:
        raise ValueError("Pi event identity parts must be non-empty strings")  # noqa: TRY003
    identity = f"{_PI_IDENTITY_PREFIX}{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) <= MAX_SOURCE_ID_LENGTH:
        return identity
    digest = hashlib.sha256(f"{session_id}\x00{event_id}".encode()).hexdigest()
    return f"{_PI_IDENTITY_PREFIX}{digest}"


@dataclass(frozen=True, slots=True)
class _SessionNode:
    """One graph node of the session archive in file order."""

    node_id: str
    parent_id: str | None
    record: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class _ParsedSession:
    """One fully parsed session archive with its selected branch."""

    session_id: str
    events: tuple[PiEventInput, ...]
    records: dict[str, dict[str, JsonValue]]


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse one Pi session archive and return the visible events of the selected branch."""

    return _parse_session(value.content, value.leaf_id).events


def _parse_session(content: bytes, leaf_id: str | None) -> _ParsedSession:
    """Validate the whole archive before exposing any part of the selected branch."""

    session_id, nodes = _decode_nodes(content)
    chain = _select_chain(nodes, leaf_id)
    records = {node.node_id: node.record for node in nodes}
    events = tuple(event for event in (_build_event(session_id, node) for node in chain) if event is not None)
    return _ParsedSession(session_id=session_id, events=events, records=records)


def _decode_nodes(content: bytes) -> tuple[str, list[_SessionNode]]:
    """Decode every archive record and reject an incomplete or inconsistent graph."""

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session content must be UTF-8 encoded") from error  # noqa: TRY003

    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError("Pi session content must contain a session header")  # noqa: TRY003

    session_id = _decode_header(_decode_record(lines[0]))
    nodes: list[_SessionNode] = []
    seen: set[str] = set()
    for line in lines[1:]:
        node = _decode_node(_decode_record(line), seen)
        seen.add(node.node_id)
        nodes.append(node)
    return session_id, nodes


def _decode_header(record: dict[str, JsonValue]) -> str:
    """Validate the session header and return the stable session identity."""

    if record.get("type") != "session":
        raise ValueError("Pi session content must start with a session header")  # noqa: TRY003
    version = record.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != _PI_SESSION_VERSION:
        raise ValueError("Pi session header must declare version 3")  # noqa: TRY003
    session_id = record.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Pi session header must declare a non-empty id")  # noqa: TRY003
    return session_id


def _decode_node(record: dict[str, JsonValue], seen: set[str]) -> _SessionNode:
    """Validate one archive record against the nodes that already appeared."""

    node_id = record.get("id")
    if not isinstance(node_id, str) or not node_id.strip():
        raise ValueError("Pi session records must declare a non-empty id")  # noqa: TRY003
    if node_id in seen:
        raise ValueError("Pi session record ids must be unique")  # noqa: TRY003
    parent_id = record.get("parentId")
    if parent_id is not None:
        if not isinstance(parent_id, str) or not parent_id.strip():
            raise ValueError("Pi session records must declare a non-empty parentId")  # noqa: TRY003
        if parent_id not in seen:
            raise ValueError("Pi session records must reference an already seen parent")  # noqa: TRY003
    return _SessionNode(node_id=node_id, parent_id=parent_id, record=record)


def _decode_record(line: str) -> dict[str, JsonValue]:
    """Decode one JSONL record and require a JSON object."""

    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"Pi session record is not valid JSON: {error}") from error  # noqa: TRY003
    if not isinstance(record, dict):
        raise ValueError("Pi session records must be JSON objects")  # noqa: TRY003, TRY004
    return record


def _select_chain(nodes: list[_SessionNode], leaf_id: str | None) -> list[_SessionNode]:
    """Return the ancestor chain of the selected leaf from its root to the leaf."""

    if not nodes:
        raise ValueError("Pi session content must contain at least one record")  # noqa: TRY003
    by_id = {node.node_id: node for node in nodes}
    if leaf_id is None:
        leaf = nodes[-1]
    else:
        try:
            leaf = by_id[leaf_id]
        except KeyError:
            raise ValueError("Pi session leaf is not present in the archive") from None  # noqa: TRY003

    chain: list[_SessionNode] = []
    visited: set[str] = set()
    current: _SessionNode | None = leaf
    while current is not None:
        if current.node_id in visited:
            raise ValueError("Pi session graph must not contain cycles")  # noqa: TRY003
        visited.add(current.node_id)
        chain.append(current)
        current = by_id.get(current.parent_id) if current.parent_id is not None else None
    chain.reverse()
    return chain


def _build_event(session_id: str, node: _SessionNode) -> PiEventInput | None:
    """Project one graph node into a visible event or ignore it without a visible body."""

    record = node.record
    record_type = record.get("type")
    if record_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            return None
        body = _message_body(role, message)
    elif record_type in {"compaction", "branch_summary"}:
        role = record_type
        summary = record.get("summary")
        body = summary if isinstance(summary, str) else ""
    elif record_type == "custom_message":
        role = "custom"
        body = _visible_text(record.get("content"))
    else:
        return None
    if not body.strip():
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=node.node_id,
        parent_id=node.parent_id,
        event_type=str(record_type),
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _message_body(role: str, message: dict[str, JsonValue]) -> str:
    """Render the visible body of one message record."""

    content = message.get("content")
    if role == "toolResult":
        name = _text_field(message.get("toolName"))
        call_id = _text_field(message.get("toolCallId"))
        return f"[tool_result {name} {call_id}] {_visible_text(content)}"
    if role == "bashExecution":
        return f"$ {_text_field(message.get('command'))}\n{_text_field(message.get('output'))}"
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_content_parts(content))
    return ""


def _content_parts(content: list[object]) -> list[str]:
    """Render every visible block of one message content list."""

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        elif block_type == "toolCall":
            parts.append(_tool_call_text(block))
        elif block_type == "toolResult":
            parts.append(_tool_result_text(block))
    return parts


def _visible_text(content: object) -> str:
    """Return the visible text of a string or a list of text blocks."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        return "\n".join(parts)
    return ""


def _tool_call_text(block: dict[str, JsonValue]) -> str:
    """Render one tool call as compact, sorted JSON arguments."""

    arguments = block.get("arguments")
    if arguments is None:
        arguments = {}
    encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"[tool_call {_text_field(block.get('name'))} {_text_field(block.get('id'))}] {encoded}"


def _tool_result_text(block: dict[str, JsonValue]) -> str:
    """Render one tool result block with its tool identity and complete output."""

    name = _text_field(block.get("toolName"))
    call_id = _text_field(block.get("toolCallId"))
    return f"[tool_result {name} {call_id}] {_visible_text(block.get('content'))}"


def _text_field(value: object) -> str:
    """Return one record field as text without inventing content for other shapes."""

    return value if isinstance(value, str) else ""


class PiEventSourceAdapter:
    """Resolve and read Pi event snapshots through the standard Source interface."""

    input_class = PiEventInput
    name = PI_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        identity = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(name=identity, source_id=identity, **value.model_dump())

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose the complete event text as standard textual evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = AdapterSourceDefinition(
    PiEventSourceAdapter(),
    version=PI_SOURCE_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Import the visible events of one Pi session branch without duplicating evidence."""

    name = PI_CONNECTOR_NAME
    version = PI_CONNECTOR_VERSION
    source_definitions = frozenset({PI_SOURCE_NAME})

    def __init__(self, path: Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        """Parse the archive, submit only new events and report the durable progress."""

        parsed = _parse_session(self._path.read_bytes(), self._leaf_id)
        ledger = _imported_ledger(session.checkpoint, parsed.session_id)
        _reject_rewritten_events(ledger, parsed.records)

        pending = [event for event in parsed.events if event.event_id not in ledger]
        outcomes: list[ConnectorItemOutcome] = []
        for event in pending:
            outcome = await session.submit(
                pi_event_source_id(event.session_id, event.event_id),
                PI_SOURCE_NAME,
                event,
            )
            outcomes.append(outcome)
            if outcome.status is ConnectorSubmissionStatus.ACCEPTED:
                ledger[event.event_id] = _record_digest(event.record)

        if any(outcome.status is not ConnectorSubmissionStatus.ACCEPTED for outcome in outcomes):
            return ConnectorRunCompletion(status=ConnectorRunStatus.INCOMPLETE, checkpoint=session.checkpoint)
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint=_build_checkpoint(parsed.session_id, ledger),
        )


def _imported_ledger(checkpoint: JsonValue | None, session_id: str) -> dict[str, str]:
    """Read the durable record digests of the events imported by earlier runs."""

    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict):
        raise ValueError("Pi session checkpoint must be a JSON object")  # noqa: TRY003, TRY004
    if checkpoint.get("version") != _PI_CHECKPOINT_VERSION:
        raise ValueError("Pi session checkpoint uses an unsupported version")  # noqa: TRY003
    events = checkpoint.get("events")
    if not isinstance(events, dict):
        raise ValueError("Pi session checkpoint must carry an events object")  # noqa: TRY003, TRY004
    if checkpoint.get("session_id") != session_id:
        return {}
    ledger: dict[str, str] = {}
    for event_id, digest in events.items():
        if not isinstance(event_id, str) or not isinstance(digest, str):
            raise ValueError("Pi session checkpoint must map event ids to record digests")  # noqa: TRY003, TRY004
        ledger[event_id] = digest
    return ledger


def _reject_rewritten_events(ledger: dict[str, str], records: dict[str, dict[str, JsonValue]]) -> None:
    """Refuse to continue when an already imported event changed its original record."""

    for event_id, digest in ledger.items():
        record = records.get(event_id)
        if record is not None and _record_digest(record) != digest:
            raise ValueError("Pi session event was rewritten after it was imported")  # noqa: TRY003


def _record_digest(record: dict[str, JsonValue]) -> str:
    """Digest one original record so rewrites are detected across restarts."""

    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _build_checkpoint(session_id: str, ledger: dict[str, str]) -> JsonValue:
    """Describe the durable progress of one session archive."""

    return {
        "version": _PI_CHECKPOINT_VERSION,
        "session_id": session_id,
        "events": dict(sorted(ledger.items())),
    }


__all__ = [
    "PI_CONNECTOR_NAME",
    "PI_CONNECTOR_VERSION",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_SOURCE_NAME",
    "PI_SOURCE_VERSION",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
