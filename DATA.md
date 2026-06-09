# Expanding the Corpus

The project still supports the original single-file workflow:

```bash
python3 train.py --data-path data/input.txt
```

It now also supports a multi-file corpus directory:

```text
data/corpus/
├── alice.txt
├── shakespeare.txt
└── stories.txt
```

Train directly from the directory:

```bash
python3 train.py --data-path data/corpus
```

Or clean and merge raw `.txt` files into `data/input.txt` first:

```bash
python3 scripts/build_corpus.py data/corpus --output data/input.txt --min-chars 200
```

Recommended sources:

- Public-domain plain text books from Project Gutenberg.
- Small-story corpora such as TinyStories for small language models.
- Your own classroom text examples, as long as the text is plain UTF-8.

Notes:

- Keep the corpus mostly one language and one style for this small character model.
- Avoid duplicating the same file many times; the build script deduplicates exact matches.
- For large corpora, increase training iterations and consider a larger model.
