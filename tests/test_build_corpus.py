import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_corpus import build_corpus, clean_text  # noqa: E402


def test_clean_text_removes_gutenberg_boilerplate():
    raw = """
Header
*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***

Chapter 1
Hello    world.



*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***
Footer
"""
    assert clean_text(raw) == "Chapter 1\nHello world."


def test_build_corpus_merges_and_deduplicates(tmp_path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("alpha beta gamma " * 20, encoding="utf-8")
    (input_dir / "b.txt").write_text("alpha beta gamma " * 20, encoding="utf-8")
    (input_dir / "c.txt").write_text("delta epsilon zeta " * 20, encoding="utf-8")

    output_path = tmp_path / "input.txt"
    stats = build_corpus([input_dir], output_path, min_chars=20)
    text = output_path.read_text(encoding="utf-8")

    assert stats["documents"] == 2
    assert "alpha beta gamma" in text
    assert "delta epsilon zeta" in text
