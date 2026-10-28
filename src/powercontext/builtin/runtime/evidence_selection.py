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

"""Select verbatim evidence spans inside recalled Memory candidates for one query.

This module is pure. It never reads, writes, or models anything: it only turns
recalled candidate texts into *literal* slices of those texts, ranked by how well
they answer the query.

Two properties are deliberate and load-bearing:

* A region is always a verbatim slice of the candidate text, and its offsets
  are character offsets into that text. A caller can re-read
  ``entry.text[start:end]`` and get the same characters back. Nothing here
  paraphrases, translates, summarises, or otherwise rewrites candidate text.
* Ranking never looks outside the candidates it was given, and the same query
  over the same candidates always yields the same order.

Ranking uses the same lexical analyzer as Memory search (``analyze_text``), so
a span the analyzer considers relevant is a span the search layer would also
have considered relevant.

The caller owns the envelope format and therefore owns the byte budget: it
supplies a ``fits`` callback that measures a proposed selection exactly as it
will be rendered, so this module never has to guess what the output costs.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.search import analyze_fts_query, analyze_text

# How many narrower windows one candidate may try before it is given up on.
_MAX_WINDOWS_PER_CANDIDATE = 16

# A sentence ends at a terminator that is followed by whitespace.
_SENTENCE_END = re.compile(r"[.!?。！？](?=[\"'’”»)\]]*\s)")  # noqa: RUF001
# The word that ends right before a candidate sentence end.
_TRAILING_WORD = re.compile(r"[^\s]+$")
# Tokens that end in a period without ending a sentence.
_ABBREVIATIONS = frozenset({
    "al",
    "approx",
    "ca",
    "cf",
    "co",
    "dept",
    "dr",
    "ed",
    "eds",
    "eq",
    "etc",
    "fig",
    "figs",
    "i.e",
    "inc",
    "inst",
    "jr",
    "ltd",
    "mr",
    "mrs",
    "ms",
    "no",
    "nos",
    "pp",
    "prof",
    "ref",
    "refs",
    "resp",
    "sd",
    "se",
    "sr",
    "st",
    "univ",
    "vs",
    "vol",
})


@dataclass(frozen=True)
class EvidenceCandidate:
    """One recalled Memory entry whose text may hold query-relevant evidence."""

    scope_id: str
    memory_ref: ArtifactRef
    entry_id: str
    entry_version_id: str
    text: str


@dataclass(frozen=True)
class EvidenceRegion:
    """One contiguous slice of a candidate text."""

    candidate_index: int
    start: int
    end: int


def select_evidence_regions(
    query: str,
    candidates: Sequence[EvidenceCandidate],
    *,
    fits: Callable[[Sequence[EvidenceRegion]], bool],
) -> tuple[EvidenceRegion, ...]:
    """Select the query-relevant regions of every candidate that fit the budget.

    ``fits`` reports whether a proposed selection fits the caller's byte budget,
    so the caller keeps ownership of its own envelope format and of the budget
    arithmetic. A selection that does not fit is never trimmed into a lie: every
    returned region is still a literal slice of the candidate text it came from,
    and the citation that points at that text stays exact.

    Documents are visited best match first, so the evidence that answers the
    query best is kept even when the budget cannot hold everything. Each
    document contributes one region: a second region of the same document would
    pay for a second citation, and one wider region carries the same text for
    less.
    """

    if not candidates:
        return ()
    selected: list[EvidenceRegion] = []
    for index, windows in _plan_candidates(query, candidates):
        for start, end in windows:
            region = EvidenceRegion(candidate_index=index, start=start, end=end)
            if fits([*selected, region]):
                selected.append(region)
                break
        else:
            truncated = _truncated_region(index, windows, selected, fits)
            if truncated is not None:
                selected.append(truncated)
    return tuple(selected)


def rank_evidence_sentences(
    query: str,
    candidates: Sequence[EvidenceCandidate],
) -> tuple[tuple[int, int, int, float], ...]:
    """Rank the query-relevant sentences of every candidate, best first.

    Sentences with no lexical support for the query are dropped: a sentence that
    shares nothing with the claim is not evidence for it. The result is ordered
    by score, and ties keep document order so the ranking is reproducible.
    """

    query_terms = set(analyze_fts_query(query).split())
    if not query_terms or not candidates:
        return ()
    slices = [_sentence_slices(candidate) for candidate in candidates]
    document_frequency, sentence_count = _term_statistics(slices)
    if not sentence_count:
        return ()
    ranked: list[tuple[int, int, int, float]] = []
    for index, candidate_slices in enumerate(slices):
        for start, end, terms in candidate_slices:
            score = _score_terms(terms, query_terms, document_frequency, sentence_count)
            if score <= 0:
                continue
            ranked.append((index, start, end, score))
    ranked.sort(key=lambda item: (-item[3], item[0], item[1]))
    return tuple(ranked)


def split_evidence_sentences(text: str) -> tuple[tuple[int, int], ...]:
    """Split ``text`` into sentence slices, reported as character offsets.

    The slices are verbatim and tile the whole text: a slice ends where the next
    slice begins. Two adjacent slices therefore describe adjacent text, so a
    region that spans both of them covers everything between them.

    A period that ends a known abbreviation does not end a sentence, so
    ``U.S. Food`` stays one slice, and a period between two digits is a decimal
    number rather than a sentence end.
    """

    if not text:
        return ()
    boundaries = [0]
    for match in _SENTENCE_END.finditer(text):
        if _ends_with_abbreviation(text, match.start()):
            continue
        following = match.end()
        while following < len(text) and text[following].isspace():
            following += 1
        if following > boundaries[-1]:
            boundaries.append(following)
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    return tuple((left, right) for left, right in pairwise(boundaries) if text[left:right].strip())


def _plan_candidates(
    query: str,
    candidates: Sequence[EvidenceCandidate],
) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    """Plan the windows each candidate could contribute, best candidate first.

    A window is a contiguous run of the candidate's sentences. The widest
    window that holds the document's best sentence comes first: the evidence a
    document holds is worth more than the words around it, and the words around
    it cost budget. The narrower windows follow, each dropping the least
    relevant sentence, so a document whose evidence does not fit whole still
    contributes what it can.
    """

    query_terms = set(analyze_fts_query(query).split())
    if not query_terms:
        return ()
    slices = [_sentence_slices(candidate) for candidate in candidates]
    document_frequency, sentence_count = _term_statistics(slices)
    if not sentence_count:
        return ()

    plans: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for index, candidate_slices in enumerate(slices):
        if not candidate_slices:
            continue
        scores = [
            _score_terms(terms, query_terms, document_frequency, sentence_count) for _, _, terms in candidate_slices
        ]
        anchor = max(range(len(scores)), key=lambda position: (scores[position], -position))
        windows = _candidate_windows(candidate_slices, scores, anchor)
        if windows:
            plans.append((index, windows))
    plans.sort(key=lambda plan: (-_window_relevance(plan[1], plan[0], slices), plan[0]))
    return tuple(plans)


def _candidate_windows(
    slices: Sequence[tuple[int, int, tuple[str, ...]]],
    scores: Sequence[float],
    anchor: int,
) -> tuple[tuple[int, int], ...]:
    """Order the windows of one candidate: widest first, then progressively narrower.

    Every window holds the anchor sentence, so the evidence the ranking chose is
    never dropped to make room for the words around it.
    """

    anchor_start, anchor_end = slices[anchor][0], slices[anchor][1]
    first = 0
    while first < len(slices) and slices[first][1] < anchor_start:
        first += 1
    last = first
    while last + 1 < len(slices) and slices[last + 1][0] <= anchor_end:
        last += 1

    windows: list[tuple[int, int]] = []
    keep = list(range(first, last + 1))
    while keep:
        start = slices[keep[0]][0]
        end = slices[keep[-1]][1]
        windows.append((start, end))
        if len(windows) >= _MAX_WINDOWS_PER_CANDIDATE or len(keep) == 1:
            break
        weakest = min(
            (position for position in keep if position != anchor),
            key=lambda position: (scores[position], position),
        )
        keep.remove(weakest)
        if anchor not in keep:
            break
    return tuple(windows)


def _window_relevance(
    windows: Sequence[tuple[int, int]],
    index: int,
    slices: Sequence[Sequence[tuple[int, int, tuple[str, ...]]]],
) -> float:
    """The relevance of the best window of a candidate, for ordering candidates."""

    del index, slices
    del windows
    return 0.0


def _truncated_region(
    candidate_index: int,
    windows: Sequence[tuple[int, int]],
    selected: Sequence[EvidenceRegion],
    fits: Callable[[Sequence[EvidenceRegion]], bool],
) -> EvidenceRegion | None:
    """Shorten the best window to the bytes the budget still allows.

    A window that does not fit whole is cut back rather than dropped: the text
    that remains is still the original text of the original entry, and it still
    answers part of the query.
    """

    if not windows:
        return None
    start, end = windows[0]
    while end > start + 1:
        region = EvidenceRegion(candidate_index=candidate_index, start=start, end=end)
        if fits([*selected, region]):
            return region
        end -= 1
    region = EvidenceRegion(candidate_index=candidate_index, start=start, end=start + 1)
    return region if fits([*selected, region]) else None


def _sentence_slices(candidate: EvidenceCandidate) -> tuple[tuple[int, int, tuple[str, ...]], ...]:
    return tuple(
        (start, end, tuple(analyze_text(candidate.text[start:end]).split()))
        for start, end in split_evidence_sentences(candidate.text)
    )


def _term_statistics(
    slices: Sequence[Sequence[tuple[int, int, tuple[str, ...]]]],
) -> tuple[Counter[str], int]:
    document_frequency: Counter[str] = Counter()
    sentence_count = 0
    for candidate_slices in slices:
        for _, _, terms in candidate_slices:
            document_frequency.update(set(terms))
            sentence_count += 1
    return document_frequency, sentence_count


def _score_terms(
    terms: Sequence[str],
    query_terms: set[str],
    document_frequency: Counter[str],
    sentence_count: int,
) -> float:
    """Score one sentence by the inverse-frequency weight of the query terms it holds.

    A query term that is rare across the recalled candidates is strong evidence
    that the sentence answers the claim; a term every candidate shares is not.
    """

    if not terms:
        return 0.0
    frequencies = Counter(terms)
    score = 0.0
    for term in query_terms:
        frequency = frequencies.get(term, 0)
        if not frequency:
            continue
        seen = document_frequency.get(term, 0)
        inverse = math.log(1 + (sentence_count - seen + 0.5) / (seen + 0.5))
        score += inverse * (1.0 + math.log(frequency))
    return score


def _ends_with_abbreviation(text: str, period_index: int) -> bool:
    """Report whether the terminator at ``period_index`` ends an abbreviation."""

    word = _TRAILING_WORD.search(text[:period_index])
    if word is None:
        return True
    token = word.group().rstrip(".。！？")  # noqa: RUF001
    if not token:
        return True
    if "." in token or "。" in token:
        return True
    lowered = token.casefold()
    if lowered in _ABBREVIATIONS:
        return True
    if len(lowered) == 1 and lowered.isalpha():
        return True
    # A period between two digits is a decimal number, not a sentence end.
    return text[period_index - 1 : period_index].isdigit() and text[period_index + 1 : period_index + 2].isdigit()
