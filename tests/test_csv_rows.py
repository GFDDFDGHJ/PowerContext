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

import pytest

from powercontext.sources import SourceDefinitionRegistry, SourceMaterialization, TextEvidence
from powercontext.sources.csv_rows import (
    CSV_ROW_SOURCE_DEFINITION,
    CSVDocumentInput,
    CSVRowInput,
    CSVRowSource,
    csv_row_source_id,
    parse_csv_document,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def test_row_projection_preserves_unicode_column_order_and_string_types() -> None:
    rows = parse_csv_document(
        CSVDocumentInput(
            document_id="机场/表",
            primary_key=" id ",
            content=" id ,id,名称,空值,数字,布尔\r\n key ,other,浦东机场,,001,false\r\n".encode(),
        )
    )
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    source = asyncio.run(registry.resolve(rows[0]))
    assert isinstance(source, CSVRowSource)
    assert source.materialization is SourceMaterialization.CAPTURED
    assert source.name == source.source_id == "csv:%E6%9C%BA%E5%9C%BA%2F%E8%A1%A8:%20key%20"
    assert asyncio.run(registry.read(source)) == rows[0]

    restored = CSVRowSource.model_validate_json(source.model_dump_json())
    evidence = TextEvidence.model_validate(registry.project(restored, TEXT_EVIDENCE_PROJECTION_KEY))
    assert evidence.content == '{" id ":" key ","id":"other","名称":"浦东机场","空值":"","数字":"001","布尔":"false"}'
    assert evidence.metadata == {
        "document_id": "机场/表",
        "primary_key": " id ",
        "primary_key_value": " key ",
        "columns": [" id ", "id", "名称", "空值", "数字", "布尔"],
        "row": {" id ": " key ", "id": "other", "名称": "浦东机场", "空值": "", "数字": "001", "布尔": "false"},
    }


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"])
def test_only_empty_physical_records_are_ignored(line_ending: str) -> None:
    content = line_ending.join(["id,note", "", 'first,""', "", 'second,"a', "", 'b"', ""])
    rows = parse_csv_document(CSVDocumentInput(document_id="doc", primary_key="id", content=content.encode()))
    assert tuple(row.values for row in rows) == (("first", ""), ("second", f"a{line_ending}{line_ending}b"))


@pytest.mark.parametrize(
    "content",
    [
        b"\n",
        b"\nid,note\nfirst,valid\n",
        b'id,note\nfirst,valid\n"",empty-key\n',
        b"id,note\nfirst,valid\n  \t,blank-key\n",
        b"id,note\nfirst,valid\n \n",
        b"id,note\nfirst,valid\n,,\n",
        b'id,note\nfirst,valid\nlast,"closed"garbage\n',
        b'id,note\nfirst,valid\nlast,"not closed\n',
        b"id,note\nfirst,valid\nlast,invalid-utf8-\xff\n",
    ],
)
def test_invalid_records_after_valid_rows_fail_atomically(content: bytes) -> None:
    with pytest.raises(ValueError):
        parse_csv_document(CSVDocumentInput(document_id="doc", primary_key="id", content=content))


@pytest.mark.parametrize("primary_key", ["", "id", " id", "id "])
def test_primary_key_requires_exact_header_even_without_data(primary_key: str) -> None:
    with pytest.raises(ValueError):
        parse_csv_document(CSVDocumentInput(document_id="doc", primary_key=primary_key, content=b" id ,note\n"))


def test_identity_encoding_is_bounded_after_percent_encoding() -> None:
    assert len(csv_row_source_id("d", "x" * 250)) == 256
    assert csv_row_source_id("a:b", "c") != csv_row_source_id("a", "b:c")
    assert csv_row_source_id("d", "%2F") != csv_row_source_id("d", "/")
    with pytest.raises(ValueError):
        csv_row_source_id("d", "x" * 251)
    with pytest.raises(ValueError):
        csv_row_source_id("d", "表" * 28)
    with pytest.raises(ValueError):
        parse_csv_document(CSVDocumentInput(document_id="d", primary_key="id", content=b"id\n" + b"x" * 251))


@pytest.mark.parametrize("primary_key_value", ["", " ", "\t\r\n"])
def test_identity_rejects_blank_keys(primary_key_value: str) -> None:
    with pytest.raises(ValueError):
        csv_row_source_id("doc", primary_key_value)


def test_row_input_rejects_a_key_that_disagrees_with_the_captured_fields() -> None:
    with pytest.raises(ValueError):
        CSVRowInput(
            document_id="doc",
            primary_key="id",
            primary_key_value="other",
            columns=("id", "name"),
            values=("first", "Alpha"),
        )
