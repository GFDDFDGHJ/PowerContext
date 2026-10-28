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
import hashlib
import io

import pytest
from pypdf import PdfWriter

from powercontext.sources import SourceDefinitionRegistry, SourceMaterialization, TextEvidence
from powercontext.sources.pdf_pages import (
    PDF_PAGE_SOURCE_DEFINITION,
    PDFDocumentInput,
    PDFPageSource,
    parse_pdf_document,
    pdf_page_source_id,
    search_pdf_pages,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def _pdf_bytes(pages: list[str]) -> bytes:
    """Build a minimal PDF whose pages carry one line of text each."""

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    page_ids = [4 + 2 * index for index in range(len(pages))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode()
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents " + str(len(objects) + 2).encode() + b" 0 R >>"
        )
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    document = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    start = len(document)
    document += f"xref\n0 {len(objects) + 1}\n".encode()
    document += b"0000000000 65535 f \n"
    for offset in offsets:
        document += f"{offset:010d} 00000 n \n".encode()
    document += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    return bytes(document)


def test_every_physical_page_is_captured_in_order() -> None:
    content = _pdf_bytes(["Consolidated Balance Sheets", "Consolidated Statement of Income", "Item 3 Legal"])
    pages = parse_pdf_document(PDFDocumentInput(document_id="report", content=content))
    assert [page.page_number for page in pages] == [1, 2, 3]
    assert all(page.document_id == "report" for page in pages)
    assert all(page.document_sha256 == hashlib.sha256(content).hexdigest() for page in pages)
    assert "Balance Sheets" in pages[0].text
    assert "Statement of Income" in pages[1].text
    assert "Item 3 Legal" in pages[2].text


def test_blank_pages_keep_their_place() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=200)
    writer.add_blank_page(width=100, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    pages = parse_pdf_document(PDFDocumentInput(document_id="control", content=buffer.getvalue()))
    assert [(page.page_number, page.text) for page in pages] == [(1, ""), (2, "")]


def test_empty_password_permission_marker_is_readable() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=200)
    writer.encrypt(user_password="", owner_password="owner")  # noqa: S106
    buffer = io.BytesIO()
    writer.write(buffer)
    pages = parse_pdf_document(PDFDocumentInput(document_id="permissions", content=buffer.getvalue()))
    assert len(pages) == 1
    assert pages[0].page_number == 1


@pytest.mark.parametrize("raw", [b"", b"not a PDF", b"%PDF-1.7\ntruncated"])
def test_corrupt_documents_fail_atomically(raw: bytes) -> None:
    with pytest.raises(ValueError):
        parse_pdf_document(PDFDocumentInput(document_id="invalid", content=raw))


@pytest.mark.parametrize("document_id", ["", "   "])
def test_blank_document_identity_is_rejected(document_id: str) -> None:
    with pytest.raises(ValueError):
        parse_pdf_document(PDFDocumentInput(document_id=document_id, content=_pdf_bytes(["page"])))


def test_page_projection_keeps_the_document_and_the_physical_page() -> None:
    content = _pdf_bytes(["Item 1 Business", "Item 1A Risk Factors"])
    pages = parse_pdf_document(PDFDocumentInput(document_id="report", content=content))
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    source = asyncio.run(registry.resolve(pages[1]))
    assert isinstance(source, PDFPageSource)
    assert source.materialization is SourceMaterialization.CAPTURED
    assert source.name == source.source_id
    assert asyncio.run(registry.read(source)) == pages[1]

    evidence = TextEvidence.model_validate(registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY))
    assert evidence.source_type == "pdf-page"
    assert evidence.source_id == source.name
    assert evidence.content == f"report\n{pages[1].text}"
    assert evidence.metadata == {
        "document_id": "report",
        "document_sha256": pages[1].document_sha256,
        "page_number": 2,
    }


def test_identity_is_stable_and_changes_with_the_revision() -> None:
    first = parse_pdf_document(PDFDocumentInput(document_id="report", content=_pdf_bytes(["Item 1 Business"])))
    again = parse_pdf_document(PDFDocumentInput(document_id="report", content=_pdf_bytes(["Item 1 Business"])))
    revised = parse_pdf_document(PDFDocumentInput(document_id="report", content=_pdf_bytes(["Item 1A Risk Factors"])))
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    assert asyncio.run(registry.resolve(first[0])).name == asyncio.run(registry.resolve(again[0])).name
    assert asyncio.run(registry.resolve(first[0])).name != asyncio.run(registry.resolve(revised[0])).name


def test_identity_stays_bounded_for_a_long_document_id() -> None:
    identity = pdf_page_source_id("报告" * 200, "a" * 64, 12)
    assert len(identity) <= 256
    assert identity.startswith("pdf-page:")
    assert identity.endswith(":12")


def test_search_returns_distinct_exact_page_citations(tmp_path) -> None:
    from powercontext.builtin.artifacts.memory import MemoryEntryInput
    from powercontext.builtin.persistence.sqlite import SQLiteConfig
    from powercontext.builtin.runtime import (
        BuiltinConfig,
        GetMemoryEntryRequest,
        SubmitSourceObservation,
        open_builtin_runtime,
    )
    from powercontext.builtin.scope import ScopeDraft
    from powercontext.sources import manifest_for_definition, project_source_for_transport

    class _Pipeline:
        async def extract(self, request, /):
            return tuple(
                MemoryEntryInput(
                    kind="source-evidence",
                    text=source.projection(TEXT_EVIDENCE_PROJECTION_KEY)["content"],
                    sources=(source,),
                )
                for source in request.sources
            )

    content = _pdf_bytes([
        "PepsiCo operates in the following geographies North America and Europe",
        "Consolidated Balance Sheets Cash and cash equivalents 4835",
        "Item 3 Legal Proceedings materially important ongoing legal battles",
    ])
    pages = parse_pdf_document(PDFDocumentInput(document_id="PEPSICO_2022_10K", content=content))
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    observations = [project_source_for_transport(registry, asyncio.run(registry.resolve(page))) for page in pages]
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'pdf.db'}"))

    async def run() -> None:
        async with open_builtin_runtime(config, candidate_pipeline=_Pipeline()) as runtime:
            scope = await runtime.scopes.create(
                ScopeDraft(title="Reports", summary="Evidence", idempotency_key="pdf-pages")
            )
            await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
            for observation in observations:
                await runtime.ingestion.submit(
                    SubmitSourceObservation(scope_id=scope.scope_id, observation=observation)
                )
            memory = runtime.memory.for_scope(scope.scope_id)
            for _ in range(len(observations) + 1):
                if not (await memory.flush()).processed:
                    break
            citations = await search_pdf_pages(memory, "Which geographies does PepsiCo operate in?", limit=2)
            assert 1 <= len(citations) <= 2
            pages_seen = []
            for citation in citations:
                record = await memory.get(GetMemoryEntryRequest(citation=citation))
                assert record.entry.sources
                for reference in record.entry.sources:
                    assert reference.source_type == "pdf-page"
                    pages_seen.append(reference.source_id)
            assert len(set(pages_seen)) == len(pages_seen)
            assert any(source_id.endswith(":1") for source_id in pages_seen)

    asyncio.run(run())


def test_search_rejects_an_unusable_request() -> None:
    with pytest.raises(ValueError):
        asyncio.run(search_pdf_pages(None, "   ", limit=5))
    with pytest.raises(ValueError):
        asyncio.run(search_pdf_pages(None, "geographies", limit=0))
