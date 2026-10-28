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

"""Capture every physical page of a report PDF as page-precise evidence."""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote, unquote

from pydantic import BaseModel, field_validator

from powercontext.artifacts import MemoryCitation
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PDF_SOURCE_NAME = "pdf-page"
PDF_SOURCE_VERSION = "1"

# One page is one chunk: the physical page number is both the chunk boundary
# and the identity of the retrieved evidence.
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_NUMBER = re.compile(r"^\d[\d,]*\.?\d*$")
_SHA256 = re.compile(r"[0-9a-f]{64}")

# Function words carry no retrieval signal inside a financial report, and the
# answering instructions that surround a question describe the expected answer
# format rather than the evidence page.
_STOP_WORDS = frozenset([
    "a",
    "about",
    "all",
    "also",
    "am",
    "an",
    "and",
    "answer",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "give",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "me",
    "might",
    "must",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "response",
    "relying",
    "shown",
    "should",
    "show",
    "state",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "us",
    "use",
    "used",
    "using",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
])

# The lexical evidence weight, the morphological evidence weight and the table
# weight of one page. A page that explains little of the question wins on rare
# terms; a page that carries the financial tables wins when the question asks
# for a reported figure.
_LEXICAL_WEIGHT = 8.0
_STEM_WEIGHT = 1.0
_TABLE_WEIGHT = 4.0

# How many pages one lexical pass recalls from the Memory index.
_SEARCH_LIMIT = 64


def _validate_document_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
    return value


def _validate_digest(value: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError("document_sha256 must be a lowercase hexadecimal SHA-256 digest")  # noqa: TRY003
    return value


class PDFDocumentInput(BaseModel):
    """One complete original PDF with its caller-stable identity."""

    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _validate_document_id(value)


class PDFPageInput(BaseModel):
    """The extractable text of exactly one physical page."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _validate_document_id(value)

    @field_validator("document_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _validate_digest(value)

    @field_validator("page_number")
    @classmethod
    def validate_page_number(cls, value: int) -> int:
        if value < 1:
            raise ValueError("page_number must count physical pages from one")  # noqa: TRY003
        return value


class PDFPageSource(Source):
    """One captured physical page of one exact PDF revision."""

    materialization: Literal[SourceMaterialization.CAPTURED] = SourceMaterialization.CAPTURED
    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Encode the document, its revision digest and the physical page."""

    _validate_document_id(document_id)
    _validate_digest(document_sha256)
    if page_number < 1:
        raise ValueError("page_number must count physical pages from one")  # noqa: TRY003
    identity = f"{PDF_SOURCE_NAME}:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        # Long caller identifiers keep their uniqueness through a stable digest.
        digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
        identity = f"{PDF_SOURCE_NAME}:{digest}:{document_sha256}:{page_number}"
    return identity


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Extract every physical page of one original PDF.

    The whole document is parsed before any page is returned, so a broken file
    fails atomically instead of yielding a partial page list.
    """

    _validate_document_id(value.document_id)
    content = value.content
    if not content:
        raise ValueError("PDF content must not be empty")  # noqa: TRY003
    digest = hashlib.sha256(content).hexdigest()
    texts = _extract_pages(content)
    return tuple(
        PDFPageInput(
            document_id=value.document_id,
            document_sha256=digest,
            page_number=number,
            text=text,
        )
        for number, text in enumerate(texts, start=1)
    )


def _extract_pages(content: bytes) -> tuple[str, ...]:
    """Return the extractable text of every physical page, in order."""

    try:
        return _read_pages(content)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Unable to parse the complete PDF document: {error}") from error  # noqa: TRY003


def _read_pages(content: bytes) -> tuple[str, ...]:
    """Read every physical page of one original PDF."""

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    # An empty user password is a permission marker, not a secret.
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF content is encrypted with an unknown password")  # noqa: TRY003
    pages = reader.pages
    if len(pages) == 0:
        raise ValueError("PDF content does not contain any page")  # noqa: TRY003
    return tuple((page.extract_text() or "") for page in pages)


class PDFPageSourceAdapter:
    """Resolve and read page snapshots through the standard Source interface."""

    input_class = PDFPageInput
    name = PDF_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        identity = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(name=identity, source_id=identity, **value.model_dump())

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one physical page as the standard textual evidence projection."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = AdapterSourceDefinition(
    PDFPageSourceAdapter(),
    version=PDF_SOURCE_VERSION,
    projections=(PDFPageTextEvidenceProjection(),),
)


@dataclass(frozen=True, slots=True)
class _Page:
    """One stored page candidate together with its exact citation."""

    citation: MemoryCitation
    text: str
    document_id: str
    page_number: int

    @property
    def identity(self) -> tuple[str, str]:
        """Return the page identity that keeps one physical page unique."""

        if self.page_number > 0:
            return ("page", f"{self.document_id}:{self.page_number}")
        return ("entry", self.citation.entry_version_id)


def _tokenize(value: str) -> list[str]:
    return _TOKEN.findall(value.casefold())


def _stem(word: str) -> str:
    """Reduce one token to the stem that its variants share.

    A report states "geographies" where a question asks about "geographic"
    regions, and "liabilities" where a question asks about a "liability". The
    stem keeps such variants comparable without a language model.
    """

    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "i"
    elif len(word) > 4 and word.endswith("ic"):
        word = word[:-2] + "i"
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    elif len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("y") and not word.endswith(("ay", "ey", "iy", "oy", "uy")):
        word = word[:-1] + "i"
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def _question_terms(question: str) -> tuple[str, ...]:
    """Return the distinct lexical terms that can carry retrieval evidence."""

    terms: list[str] = []
    for token in _tokenize(question):
        if token in _STOP_WORDS or (len(token) == 1 and not token.isdigit()):
            continue
        if token not in terms:
            terms.append(token)
    return tuple(terms)


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Return at most ``limit`` distinct page citations for one question.

    Candidate pages are recalled through the public Memory API and then ranked
    offline: the lexical evidence of one page is weighed against how much of a
    reported table it carries. Only pages that are really stored in the Memory
    are returned, and every returned citation points at one exact page.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")  # noqa: TRY003
    if limit < 1:
        raise ValueError("limit must be a positive number of pages")  # noqa: TRY003
    terms = _question_terms(question)
    if not terms:
        return ()

    recalled = await _recall_pages(memory, terms)
    pages = await _stored_pages(memory, recalled)
    if not pages:
        pages = tuple(recalled.values())
    if not pages:
        return ()

    corpus = _corpus_statistics(pages, terms)
    selected = _select_documents(pages, corpus)
    ranked = _rank_pages(selected, corpus)
    citations: list[MemoryCitation] = []
    seen: set[tuple[str, str]] = set()
    for page in ranked:
        identity = page.identity
        if identity in seen:
            continue
        seen.add(identity)
        citations.append(page.citation)
        if len(citations) >= limit:
            break
    return tuple(citations)


async def _recall_pages(memory, terms: tuple[str, ...]) -> dict[tuple[str, str], _Page]:
    """Recall candidate pages with the public Memory lexical search."""

    from powercontext.builtin.runtime.models import SearchMemoryRequest

    recalled: dict[tuple[str, str], _Page] = {}
    request = SearchMemoryRequest(query=" ".join(terms), mode="fts", limit=_SEARCH_LIMIT)
    try:
        page = await memory.search(request)
    except Exception:
        return recalled
    for hit in page.hits:
        key = (hit.entry_id, hit.entry_version_id)
        if key in recalled:
            continue
        document_id, page_number = _document_and_page(hit.text, None)
        recalled[key] = _Page(
            citation=MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            ),
            text=hit.text,
            document_id=document_id,
            page_number=page_number,
        )
    return recalled


async def _stored_pages(memory, recalled: dict[tuple[str, str], _Page]) -> tuple[_Page, ...]:
    """Read the stored page entries of the scope to complete the candidate pool.

    The Memory index only recalls pages that share query terms; a reported
    figure is often stated in the vocabulary of the statement instead of the
    vocabulary of the question. Reading the scope's stored entries through the
    public Memory API lets the offline rerank see those pages as well.
    """

    try:
        listing = await memory.list()
    except Exception:
        return ()
    pages: list[_Page] = []
    for record in listing.entries:
        entry = record.entry
        source_id = next(
            (
                reference.source_id
                for reference in entry.sources
                if reference.source_type == PDF_SOURCE_NAME or _page_identity(reference.source_id) is not None
            ),
            None,
        )
        if source_id is None:
            continue
        document_id, page_number = _document_and_page(entry.text, source_id)
        pages.append(
            _Page(
                citation=MemoryCitation(
                    memory_ref=record.memory_ref,
                    entry_id=entry.entry_id,
                    entry_version_id=entry.entry_version_id,
                ),
                text=entry.text,
                document_id=document_id,
                page_number=page_number,
            )
        )
    if not pages:
        return ()
    return tuple(pages)


def _page_identity(source_id: str) -> tuple[str, int] | None:
    """Recover the document and the physical page from one captured page identity."""

    parts = source_id.rsplit(":", 2)
    if len(parts) != 3 or not parts[2].isdigit() or not _SHA256.fullmatch(parts[1]):
        return None
    return (unquote(parts[0]), int(parts[2]))


def _document_and_page(text: str, source_id: str | None) -> tuple[str, int]:
    """Recover the document identity and the physical page of one entry."""

    if source_id is not None:
        identity = _page_identity(source_id)
        if identity is not None:
            return identity
    # The projected text starts with the document identity of the page.
    document_id, _, _remainder = text.partition("\n")
    return (document_id, 0)


@dataclass(frozen=True, slots=True)
class _Corpus:
    """The lexical statistics of the pages that the scope stores."""

    weights: dict[str, float]
    evidence_total: float
    stem_weights: dict[str, float]
    stem_evidence_total: float
    table_ratio: dict[tuple[str, str], float]
    highest_ratio: float


def _corpus_statistics(pages: tuple[_Page, ...], terms: tuple[str, ...]) -> _Corpus:
    """Measure how rare the question terms are among the stored pages."""

    frequencies: Counter[str] = Counter()
    stem_frequencies: Counter[str] = Counter()
    ratios: dict[tuple[str, str], float] = {}
    for page in pages:
        tokens = _tokenize(page.text)
        frequencies.update(set(tokens))
        stem_frequencies.update({_stem(token) for token in tokens})
        ratios[page.identity] = _table_ratio(tokens)
    total = len(pages)
    weights = {
        term: (
            math.log(1.0 + (total - frequencies[term] + 0.5) / (frequencies[term] + 0.5)) if frequencies[term] else 0.0
        )
        for term in terms
    }
    stem_weights = {
        _stem(term): (
            math.log(1.0 + (total - stem_frequencies[_stem(term)] + 0.5) / (stem_frequencies[_stem(term)] + 0.5))
            if stem_frequencies[_stem(term)]
            else 0.0
        )
        for term in terms
    }
    highest = max(ratios.values()) if ratios else 0.0
    return _Corpus(
        weights=weights,
        evidence_total=sum(weights.values()),
        stem_weights=stem_weights,
        stem_evidence_total=sum(stem_weights.values()),
        table_ratio=ratios,
        highest_ratio=highest or 1.0,
    )


def _select_documents(pages: tuple[_Page, ...], corpus: _Corpus) -> tuple[_Page, ...]:
    """Keep the pages of the document that the question refers to.

    A report collection holds several documents; the question names the one it
    asks about. The document identity is matched against the question terms, so
    the ranking never spends its page budget on a neighbouring report.
    """

    documents = {page.document_id for page in pages}
    if len(documents) <= 1 or not corpus.evidence_total:
        return pages
    affinity: dict[str, float] = {}
    for document_id in documents:
        identity = document_id.casefold()
        matched = sum(weight for term, weight in corpus.weights.items() if term.casefold() in identity)
        affinity[document_id] = matched / corpus.evidence_total
    best = max(affinity.values())
    if best <= 0:
        return pages
    chosen = {document_id for document_id, score in affinity.items() if score >= best - 1e-9}
    return tuple(page for page in pages if page.document_id in chosen)


def _rank_pages(pages: tuple[_Page, ...], corpus: _Corpus) -> list[_Page]:
    """Order candidate pages by the lexical evidence they carry.

    One page is scored by the share of the question's rare terms that it
    explains and by the share of the page that reports figures. A question
    about a reported figure therefore reaches the statement page, while a
    question about a narrative fact reaches the page that tells the story.
    """

    ranked: list[tuple[float, _Page]] = []
    for page in pages:
        tokens = _tokenize(page.text)
        present = set(tokens)
        stemmed = {_stem(token) for token in tokens}
        evidence = sum(weight for term, weight in corpus.weights.items() if term in present)
        coverage = evidence / corpus.evidence_total if corpus.evidence_total else 0.0
        stem_evidence = sum(weight for term, weight in corpus.stem_weights.items() if term in stemmed)
        stem_coverage = stem_evidence / corpus.stem_evidence_total if corpus.stem_evidence_total else 0.0
        ratio = corpus.table_ratio.get(page.identity, 0.0) / corpus.highest_ratio
        score = _LEXICAL_WEIGHT * coverage + _STEM_WEIGHT * stem_coverage + _TABLE_WEIGHT * ratio
        ranked.append((score, page))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [page for _, page in ranked]


def _table_ratio(tokens: list[str]) -> float:
    """Return the share of one page that is reported figures."""

    if not tokens:
        return 0.0
    figures = sum(1 for token in tokens if _NUMBER.match(token))
    return figures / len(tokens)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
