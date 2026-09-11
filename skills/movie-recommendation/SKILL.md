---
name: movie-recommendation
description: Produce a personalised shortlist of movies to watch. Use when the user
  asks for something to watch, wants films like one they enjoyed, describes a mood or
  occasion, or asks what is good in a genre or era.
---

# Movie Recommendation

Your job is a **shortlist with reasons**, not a dump of whatever the API returned.

## Procedure

1. **Read the profile before searching.** The long-term profile is already in your
   system prompt. Treat liked genres/tones as soft priors and disliked ones as hard
   filters — never recommend something matching a disliked genre or tone without
   flagging why you made an exception.

2. **Resolve the constraint set.** You need at least one of: genre, era, mood, or a
   reference film. If the request has none of these and the profile is empty, ask
   exactly one clarifying question. Otherwise infer and state your assumption in one
   short line, then proceed — do not interrogate the user.

3. **Choose the retrieval path deliberately:**
   - Reference film named → `search_movies` to get its ID, then `get_recommendations`.
   - Genre / era / rating constraints → `get_genres` for the IDs, then
     `discover_movies` with `min_rating` set (7.0 or above unless the user asked for
     trash).
   - No usable constraints at all → `get_popular_movies`.
   - Mood or aesthetic wording ("neon", "bleak", "cosy") → also call
     `search_local_knowledge`; the genre articles describe visual and tonal registers
     the TMDB metadata does not capture.

4. **Shortlist 3–5 titles.** Never more. Each entry:
   - `**Title** (Year) ⭐ rating — TMDB link`
   - one sentence on what it is
   - one sentence starting with "For you:" tying it to a *specific* thing the user
     said or a *specific* profile entry. If you cannot write that sentence honestly,
     drop the title.

5. **Close with one pivot.** Offer a single concrete alternative direction
   ("want these but lighter?"), not a menu.

## Rules

- Always include the TMDB URL: `https://www.themoviedb.org/movie/{movie_id}`.
- Diversify: at most two titles from the same director or franchise.
- If a tool fails, say what you could not check and recommend from what you have.
  Do not silently substitute guesses for API data.
- If the user names a film you have no data for, say so rather than inventing a
  plot summary.
