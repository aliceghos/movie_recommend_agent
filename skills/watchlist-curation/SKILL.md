---
name: watchlist-curation
description: Read and maintain the user's watchlist files. Use when the user wants to
  save a film for later, asks what is on their list, wants the list pruned or grouped
  by mood, or says they have watched something.
---

# Watchlist Curation

The watchlist lives as markdown files in a sandboxed directory, reachable through the
filesystem tools (`list_directory`, `read_text_file`, `write_file`, `edit_file`,
`search_files`). Those tools are provided by an external MCP server and can only see
that one directory.

## File layout

```
watchlist.md          the main list
watched.md            films the user has finished
moods/<mood>.md       optional themed sub-lists ("rainy-sunday.md", "need-a-cry.md")
```

Entry format — one line each, keep it exactly this shape so the file stays parseable:

```
- [ ] **Title** (Year) — genre, runtime — added YYYY-MM-DD — why: <one clause>
```

Use `- [x]` for watched entries.

## Procedure

1. **Always read before writing.** `list_directory(".")` then `read_text_file` on the
   file you are about to touch. Writing blind will silently destroy existing entries.

2. **Adding a film:**
   - Look it up first (`search_movies` → `get_movie_details`) so the year, genre, and
     runtime in the entry are real rather than remembered.
   - Check for duplicates by title *and* year before appending.
   - Prefer `edit_file` to append a line. Only use `write_file` when creating a file
     from scratch — it overwrites.
   - Fill `why:` from what the user actually said in this conversation. If they gave
     no reason, write `why: user asked to save it` rather than inventing one.

3. **Showing the list:** read the file and render it as-is, grouped by mood file if
   several exist. Report the count. Do not reorder or rewrite entries just to display
   them.

4. **Marking watched:** flip `- [ ]` to `- [x]` in place with `edit_file`, then append
   the line to `watched.md`. Ask whether they liked it — that answer is what feeds the
   long-term profile.

5. **Pruning:** never delete an entry unilaterally. List the candidates you would drop
   with a reason each, and let the user confirm.

## Rules

- If a filesystem tool reports the path is outside the allowed directory, that is the
  sandbox working correctly. Stay inside the watchlist directory.
- If the directory is empty, say the list is empty and offer to start it — do not
  create empty files pre-emptively.
- Never write the user's raw conversation into these files; entries only.
