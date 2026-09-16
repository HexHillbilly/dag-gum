# Corpora

Public-domain style corpora used by the benchmark and the detector. All are
derived from Project Gutenberg plain-text ebooks, public domain in the United
States. Reuse them under the Project Gutenberg License
(https://www.gutenberg.org/policy/license.html) or the underlying public-domain
status of the text, as applicable.

| file | source | sentences | role |
|------|--------|-----------|------|
| `hemingway.txt` | Ernest Hemingway, *In Our Time* (1925) — Gutenberg #61085 | 278 | the style-transfer target |
| `austen.txt` | Jane Austen, *Pride and Prejudice* (1813) — Gutenberg #1342 | 6,488 | non-target human reference for the detector |

Only the author's prose is kept — the Gutenberg header/footer boilerplate is
stripped by `scripts/build_book_corpus.py`. To regenerate:

```bash
python scripts/build_book_corpus.py <raw_gutenberg.txt> docs/corpora/hemingway.txt --start-after "chapter 1"
python scripts/build_book_corpus.py <raw_gutenberg.txt> docs/corpora/austen.txt --start-after "Chapter I.]"
```
