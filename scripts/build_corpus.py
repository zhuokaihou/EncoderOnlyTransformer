import argparse
import hashlib
import re
from pathlib import Path


START_MARKERS = (
    "*** START OF THE PROJECT GUTENBERG EBOOK",
    "*** START OF THIS PROJECT GUTENBERG EBOOK",
    "***START OF THE PROJECT GUTENBERG EBOOK",
)
END_MARKERS = (
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF THIS PROJECT GUTENBERG EBOOK",
    "***END OF THE PROJECT GUTENBERG EBOOK",
)


def iter_text_files(inputs):
    for item in inputs:
        path = Path(item)
        if path.is_file() and path.suffix.lower() == ".txt":
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("*.txt"))
        else:
            raise FileNotFoundError(f"No text file or directory found: {path}")


def strip_gutenberg_boilerplate(text):
    upper = text.upper()
    start_positions = [upper.find(marker) for marker in START_MARKERS]
    start_positions = [pos for pos in start_positions if pos != -1]
    if start_positions:
        start = min(start_positions)
        line_end = text.find("\n", start)
        text = text[line_end + 1 :] if line_end != -1 else text[start:]
        upper = text.upper()

    end_positions = [upper.find(marker) for marker in END_MARKERS]
    end_positions = [pos for pos in end_positions if pos != -1]
    if end_positions:
        text = text[: min(end_positions)]
    return text


def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = strip_gutenberg_boilerplate(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_corpus(input_paths, output_path, min_chars=200, max_chars=None):
    output_path = Path(output_path)
    seen = set()
    documents = []

    for path in iter_text_files(input_paths):
        text = clean_text(path.read_text(encoding="utf-8"))
        if len(text) < min_chars:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        documents.append((path, text))

    if not documents:
        raise ValueError("No usable text documents found after cleaning.")

    corpus = "\n\n".join(text for _, text in documents)
    if max_chars is not None:
        corpus = corpus[:max_chars].rstrip()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(corpus + "\n", encoding="utf-8")
    return {
        "documents": len(documents),
        "characters": len(corpus),
        "output_path": output_path,
        "source_paths": [path for path, _ in documents],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean and merge .txt files into one character-model corpus."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["data/corpus"],
        help="Input .txt files or directories. Defaults to data/corpus.",
    )
    parser.add_argument("--output", default="data/input.txt")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    stats = build_corpus(
        input_paths=args.inputs,
        output_path=args.output,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    print(f"Wrote {stats['characters']} characters from {stats['documents']} documents")
    print(f"Output: {stats['output_path']}")


if __name__ == "__main__":
    main()
