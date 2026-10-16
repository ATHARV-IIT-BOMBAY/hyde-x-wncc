# progress.md — Agent Work Log

Team: **Attention is All We Need** · Hyde × WnCC AI Builder Hackathon · 17 Oct 2026 · LA001
Task: CPU-only small transformer bot for 2-player simplified Catan (10 VP, 5 resources, bank/ports, Largest Army at 3 knights, Longest Road at 5 segments).

## Rules (mandatory for every agent)

1. Read this whole file before doing anything else.
2. Agents run **one at a time**. Never assume another agent is working in parallel.
3. Before finishing, append a new entry at the **bottom** of the Log. Never edit or delete earlier entries.
4. Each entry must say **what** you did and **why** you did it. Include files touched, decisions made, and anything left unfinished.
5. Never fabricate results. Only record numbers you actually measured.
6. If you hit a blocker or are unsure about something, write it under "Open issues / handoff" instead of guessing.
7. The official problem statement is uploaded by the user into this repo. Do not invent env APIs, action spaces, or rules. Use the statement as the source of truth.
8. Push your changes (including your progress.md entry) directly to the GitHub repo.

## Entry template

```
### [YYYY-MM-DD HH:MM] Agent: <name/model>
**Task given by user:** ...
**What I did:**
- ...
**Why:**
- ...
**Files changed:** `path/a`, `path/b`
**Results / measurements:** ... (or "none")
**Open issues / handoff to next agent:** ...
```

## Context snapshot (as of 2026-10-16)

- Repo: https://github.com/ATHARV-IIT-BOMBAY/hyde-x-wncc
- `scaffold/` has a TinyTransformer (~102k params, d_model=64, n_layer=2, n_head=2), a BC data pipeline, a CPU training loop, and tictactoe/connect4 envs. Verified at ~8k samp/s and 89% win vs random on tictactoe.
- Planned flow once the problem statement is available: wrap the official Catan env into `scaffold/envs/catan.py` -> write `encode()` -> build a heuristic bot -> generate heuristic self-play data -> train the transformer via BC.
- Known issue (not yet fixed; the user said not to touch it for now): in `scaffold/eval.py`, `w == env.player` is always true for decisive games, so losses are always 0. Fix: record `policy_side` before the game loop and compare the winner to it.
- Problem statement drops at **10:30** on 17 Oct. The user will commit it to this repo.

## Log

### [2026-10-16] Agent: Claude (Figma Make session)
**Task given by user:** Create `progress.md` and a reusable prompt for other AI agents, then push it directly to GitHub.
**What I did:**
- Created this file with the rules, entry template, and context snapshot.
- Wrote the onboarding prompt for other agents (delivered in chat).
- Pushed this file directly to `main`.
**Why:**
- Agents will run sequentially, so each needs a shared record of what has been done and why. This prevents repeated or conflicting work.
**Files changed:** `progress.md`
**Results / measurements:** none
**Open issues / handoff to next agent:** No code changed yet. Wait for the problem statement to be committed, then parse it into: env API, action space, observation, constraints, eval format, rules.
