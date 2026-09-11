---
name: film-analysis
description: Analyse a single film in depth — narrative, visual language, genre
  lineage, authorial signature. Use when the user wants to discuss a film rather than
  find one: "why is X good", "what's going on visually in X", director style
  questions, or genre theory questions.
---

# Film Analysis

Analysis means **claims grounded in evidence**. Every assertion you make must trace
back either to TMDB metadata or to a retrieved passage. Unsupported aesthetic
pronouncements are the failure mode to avoid here.

## Procedure

1. **Fix the facts first.** `search_movies` → `get_movie_details`. You need the year,
   genres, runtime, and tagline before you interpret anything. A wrong year wrecks the
   genre-lineage section.

2. **Retrieve the critical layer.** Call `search_local_knowledge` at least once, and
   query it in the language the source is likely written in:
   - Genre-theory questions (noir, cyberpunk) → query in Chinese; those articles are
     Chinese. See `film-analysis/references/knowledge-base.md` for exactly what the
     corpus holds and how it is indexed.
   - A specific film's critical reception → query in English with the title; the
     reviews are English.

   If retrieval comes back empty, say so in the answer and mark the affected section
   as your own reading rather than sourced.

3. **Write four sections.** Keep each to 2–4 sentences; this is an analysis, not an
   essay:

   - **Narrative** — structure and what the structure is doing. Not a plot recap.
   - **Visual language** — light, colour, lens, cutting rhythm, staging. Concrete
     choices only.
   - **Genre coordinates** — where it sits in its genre's history, and which
     conventions it honours versus breaks.
   - **Authorial signature** — what marks it as this director's, cross-referenced
     against their other work.

4. **Cite as you go.** When a claim comes from retrieval, carry the source label
   inline: 【影评 · inception_review.txt】. This is how the user tells your reading
   apart from the corpus's.

5. **End with one viewing prompt** — a specific thing to watch for on a rewatch.

## Rules

- Never invent shot descriptions, crew credits, or box-office numbers. If you do not
  have it, say you do not have it.
- Do not pad with the film's Wikipedia-grade production trivia unless it bears on
  one of the four sections.
- If the user's film is not in the local corpus, the analysis still stands — just be
  explicit that the critical layer is missing.
