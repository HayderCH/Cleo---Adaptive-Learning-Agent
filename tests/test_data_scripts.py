"""Unit tests for data preparation scripts."""

from __future__ import annotations

import tempfile
from pathlib import Path

from scripts.chunk_docs import chunk_text
from scripts.make_synthetic_qa import (
    iter_jsonl,
    make_items_from_chunk,
    simple_distractors,
)


class TestChunkDocs:
    def test_chunk_text_basic(self):
        text = "This is a test. It has multiple sentences."
        chunks = list(chunk_text(text, 10))
        assert len(chunks) == 1
        assert "This is a test." in chunks[0]

    def test_chunk_text_split(self):
        text = "Word " * 50  # 50 words
        chunks = list(chunk_text(text, 20))
        assert len(chunks) == 3
        assert len(chunks[0].split()) == 20

    def test_iter_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"test": "data"}\n')
            f.write('{"test2": "data2"}\n')
            path = Path(f.name)

        items = list(iter_jsonl(path))
        assert len(items) == 2
        assert items[0]["test"] == "data"
        path.unlink()


class TestMakeSyntheticQA:
    def test_simple_distractors(self):
        answer = "test answer"
        dist = simple_distractors(answer)
        assert len(dist) == 3
        assert answer not in dist  # Should not include original
        assert all(isinstance(d, str) for d in dist)

    def test_make_items_from_chunk(self):
        chunk = {
            "id": "test-chunk",
            "text": """This is a comprehensive test document that explains programming concepts.
List comprehensions create new lists from iterables. Functions perform tasks.
Classes define object blueprints. Variables store data values.""",
            "concepts": ["test_concept"],
        }
        items = make_items_from_chunk(
            chunk,
            per_chunk=2,
            subject="math",
            bloom="understand",
            difficulty=0.5,
        )
        assert len(items) == 2
        mcq = items[0]
        assert "choices" in mcq
        assert len(mcq["choices"]) == 4  # 1 correct + 3 distractors
        assert mcq["correct_index"] is not None
        assert mcq["meta"]["subject"] == "math"
        assert mcq["meta"]["bloom"] == "understand"
        assert mcq["meta"]["difficulty_est"] == 0.5

    def test_make_items_skip_short_sentences(self):
        chunk = {"text": "Hi. Test."}  # Short sentences
        items = make_items_from_chunk(chunk)
        assert len(items) == 0  # Should skip short fragments

    # Integration test: end-to-end pipeline
    def test_end_to_end_pipeline(self):
        # This would test running chunk_docs then make_synthetic_qa
        # For now, verify the scripts can be imported
        import scripts.chunk_docs
        import scripts.make_synthetic_qa

        assert hasattr(scripts.chunk_docs, "chunk_text")
        assert hasattr(scripts.make_synthetic_qa, "make_items_from_chunk")
