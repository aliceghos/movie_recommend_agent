# What the local knowledge base actually contains

This is an L3 reference — loaded only when you need to know whether a query is worth
retrieving at all.

## Corpus inventory

| Category label | Files | Language |
|---|---|---|
| 类型知识 (genre knowledge) | `赛博朋克电影.txt`, `黑色电影.txt` | Chinese |
| 影评 (reviews) | `inception_review.txt`, `spirited_away_review.txt`, `harry_potter_review.txt`, `jane_eyre_review.txt`, `wuthering_heights_review.txt` | English |

The `data/books/` directory holds a scanned PDF of 《电影艺术词典》 with **no text
layer** — nothing from it is indexed. Do not promise the user dictionary definitions
from it.

## How retrieval behaves

`search_local_knowledge` runs two recall paths and then reranks:

- **Vector recall** matches meaning. It works well for English queries and for
  paraphrase ("what does the neon aesthetic signify").
- **BM25 recall** matches surface terms. It is what actually finds the Chinese genre
  articles, because the embedding model handles Chinese queries noticeably worse than
  English ones on this corpus.

Practical consequence: **query in the source's own language.** A Chinese query about
cyberpunk lands the cyberpunk article via BM25; the same question in English lands it
via vectors. A Chinese query about *Inception* will find less than an English one.

## When not to bother

Only these seven films and two genres are covered. For anything else, skip retrieval
and say the critical layer is unavailable rather than retrieving loosely related
passages and reasoning from them.
