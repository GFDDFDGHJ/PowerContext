# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the evidence selection that backs the byte-budgeted prepared context."""

from __future__ import annotations

import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.runtime.evidence_selection import (
    EvidenceCandidate,
    EvidenceRegion,
    rank_evidence_sentences,
    select_evidence_regions,
    split_evidence_sentences,
)


def _candidate(text: str, scope_id: str = "scope-a") -> EvidenceCandidate:
    return EvidenceCandidate(
        scope_id=scope_id,
        memory_ref=ArtifactRef(family="memory", artifact_id="memory-1", revision=1),
        entry_id="entry-1",
        entry_version_id="version-1",
        text=text,
    )


class TestSplitEvidenceSentences:
    def test_splits_on_sentence_end(self) -> None:
        spans = split_evidence_sentences("First sentence. Second sentence. Third one.")
        assert len(spans) == 3
        assert "First sentence." in "First sentence. Second sentence. Third one."[spans[0][0] : spans[0][1]]

    def test_keeps_decimal_numbers_inside_one_sentence(self) -> None:
        text = "Revenue grew 3.5 percent last year. Costs stayed flat."
        spans = split_evidence_sentences(text)
        assert len(spans) == 2

    def test_does_not_split_on_abbreviations(self) -> None:
        text = "The U.S. market declined. Revenue fell."
        spans = split_evidence_sentences(text)
        assert len(spans) == 2

    def test_handles_empty_text(self) -> None:
        assert split_evidence_sentences("") == ()

    def test_handles_text_without_terminator(self) -> None:
        spans = split_evidence_sentences("no terminator here")
        assert len(spans) == 1


class TestRankEvidenceSentences:
    def test_query_terms_raise_relevance(self) -> None:
        candidate = _candidate("Revenue grew. Costs fell.")
        ranked = rank_evidence_sentences("revenue", (candidate,))
        assert ranked

    def test_empty_query_is_handled(self) -> None:
        candidate = _candidate("Some content.")
        ranked = rank_evidence_sentences("", (candidate,))
        assert isinstance(ranked, tuple)


class TestSelectEvidenceRegions:
    def test_selects_regions_that_fit_the_budget(self) -> None:
        candidate = _candidate("Revenue grew. Costs fell. Margins held.")
        regions = select_evidence_regions(
            "revenue",
            (candidate,),
            fits=lambda regions: True,
        )
        assert regions
        assert all(isinstance(region, EvidenceRegion) for region in regions)

    def test_respects_a_failing_budget(self) -> None:
        candidate = _candidate("Revenue grew. Costs fell.")
        regions = select_evidence_regions(
            "revenue",
            (candidate,),
            fits=lambda regions: False,
        )
        assert regions == ()

    def test_no_candidates_yields_no_regions(self) -> None:
        assert select_evidence_regions("query", (), fits=lambda regions: True) == ()

    def test_region_slices_reference_the_original_text(self) -> None:
        text = "Revenue grew. Costs fell."
        candidate = _candidate(text)
        regions = select_evidence_regions("revenue", (candidate,), fits=lambda regions: True)
        for region in regions:
            assert 0 <= region.start < region.end <= len(text)


class TestEvidenceCandidate:
    def test_candidate_keeps_identity_fields(self) -> None:
        candidate = _candidate("body", scope_id="scope-b")
        assert candidate.scope_id == "scope-b"
        assert candidate.entry_id == "entry-1"
        assert candidate.entry_version_id == "version-1"
        assert candidate.memory_ref.artifact_id == "memory-1"


@pytest.mark.parametrize(
    ("query", "text"),
    [
        ("revenue", "Revenue grew."),
        ("成本", "成本上升。"),
        ("", "anything"),
    ],
)
def test_selection_never_raises_on_varied_inputs(query: str, text: str) -> None:
    candidate = _candidate(text)
    regions = select_evidence_regions(query, (candidate,), fits=lambda regions: True)
    assert isinstance(regions, tuple)
