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

### [2026-10-16 19:23 UTC] Agent: Notion AI
**Task given by user:** Update working context from the GitHub repo.
**What I did:**
- Read all 13 tracked files at snapshot d74de973204a4b08ca842193b8e8a7eea958c091: shared instructions, plans, documentation, ignore rules, and scaffold implementation.
- Adopted sequential-agent coordination and append-only handoffs from this file.
- Confirmed by static inspection the known evaluation winner-attribution bug; left it untouched as requested in the existing handoff.
**Why:**
- Align future assistance with actual code and shared decisions rather than the earlier chat summary alone.
**Files changed:** `progress.md` only (this appended entry).
**Results / measurements:** No training, evaluation, or timing benchmarks run. Existing win-rate claims are not reliable until the evaluation bug is fixed and results rerun.
**Open issues / handoff to next agent:**
- Official statement and Catan implementation are not present in this snapshot. Wait for official API/rules; scaffold eligibility still requires organizer confirmation.
- Current model has a fixed action head, not a variable candidate scorer; adapt only after the official action API is known.
- Training splits overlapping trajectory windows across train/validation; split by game before constructing windows for a cleaner validation estimate.
- Both toy solvers assign a negative score to an immediate winning move inside recursive negamax; review teacher correctness before relying on generated expert labels.
- Mixed data generation includes random actions as BC targets, and inference uses random fallback rather than legal-logit masking.
- Preserve the no-code-change request for this context-refresh task. No bugs fixed or new performance claims made.

### [2026-10-17 10:54 IST] Agent: OmniRush.ai (gpt-6-astra)
**Task given by user:** Rebuild the final hackathon execution plan from the official Catan starter repository and push it to GitHub.
**What I did:**
- Read the official problem repository at `lyric-hyde/catan-hackathon-agent` commit `04fffac`, including the problem statement, observation masking, process harness, clock, evaluator, requirements, and F reference agent.
- Added `HACKATHON_PLAN.md` with the direct-agent API, confirmed constraints, v0–v4 milestone order, CPU-budgeted search plan, experiment gates, upload checklist, report requirements, and Git/progress discipline.
- Chose to keep weight tuning after chance-aware search and end-of-turn search: tuning a one-ply policy first could overfit weights that are invalidated by the final search policy.
**Why:**
- The earlier generic Gym/transformer plan does not match the official dynamic `playable_actions` interface or the cumulative one-core CPU clock. The team needs a fault-free F-style baseline before optional learned work.
**Files changed:** `HACKATHON_PLAN.md`, `progress.md`
**Results / measurements:** No application benchmark run; the pinned self-check and v0 evaluation remain the next actions.
**Open issues / handoff to next agent:** Run the official kit's `python -m catan_kit.selfcheck`, then implement and benchmark v0. Do not claim win rates until the evaluator prints them. Keep `scaffold/` out of the submission zip and append actual measurements here after each milestone.

### [2026-10-17 11:19 IST] Agent: OmniRush.ai (gpt-6-astra)
**Task given by user:** Start executing the final plan with the v0 direct-agent baseline.
**What I did:**
- Added `submission/agent.py` implementing the official `Agent.decide(obs, playable_actions) -> int` interface.
- Implemented fresh F-style one-ply search over a single masked sampled world, successor scoring with catanatron's 13-feature value function, deterministic tie-breaking, cumulative-clock budgeting via `obs.cpu_left`, cheap low-clock fallback, and exception-safe legal-index fallback.
- Added public-observation new-game detection using seat colour, turn decreases, and board building/road counts. The private RNG remains persistent because the harness does not expose a game seed.
- Confirmed the team machine environment from the user's terminal output: Python 3.11.16, catanatron 3.3.0, networkx 3.6.1, determinism patch applied, and all official self-checks passed. The reported F-vs-random smoke run was 1.00 over 20 games with 180 ms mean CPU/game, 0 over-budget games, and 0 errors; repeated-run decisions were 976 and 976.
**Why:**
- v0 must validate the real submission interface and safety path before chance averaging or deeper search. It must return a legal index even when a sampled-world branch or feature evaluation fails.
**Files changed:** `submission/agent.py`, `progress.md`
**Results / measurements:** Local `python3 -m py_compile submission/agent.py` passed. The v0 agent has not yet been run through the official evaluator because this assistant shell cannot authenticate to the password-only SSH session (`Permission denied (password)` in non-interactive mode).
**Open issues / handoff to next agent:** Run the v0 agent from the installed team kit with the official `catan_kit.evaluate` against F, then record the actual v0 metrics before attempting the first upload. Do not send or commit the SSH password; either provide an SSH key accessible to this environment or run the remote commands from the team's terminal.

### [2026-10-17 11:29 IST] Agent: OmniRush.ai (gpt-6-astra)
**Task given by user:** Run the first v0 smoke evaluation on the official team machine.
**What I did:**
- Had the user run the official evaluator against `examples/f_agent.py` using the pushed `submission/agent.py`.
- Reviewed the reported result and kept the v0 implementation unchanged because it completed the full smoke run without faults, crashes, timeouts, or clock overruns.
**Why:**
- The smoke run validates the real direct-agent interface and the safety wrapper before spending time on the 400-game gate or an upload.
**Files changed:** `progress.md`
**Results / measurements:** 20 games against F: **8/20 wins, win rate 0.4000 ± 0.2147; mean CPU 150.5 ms/game; worst CPU 230.0 ms/game; 0 games over budget; 0 random moves after exhaustion; 0 errors across 1,056 decisions; 0 timeouts/crashes; load time 0.00 s; wall clock 8.2 s.**
**Open issues / handoff to next agent:** Run the required 400-game v0 gate against F using the same default seed sequence, then decide whether v0 is ready for upload #1. The smoke interval is wide, so do not interpret 0.4000 as a stable strength estimate yet.

### [2026-10-17 11:36 IST] Agent: OmniRush.ai (gpt-6-astra)
**Task given by user:** Review the v0 400-game gate against F.
**What I did:**
- Reviewed the official evaluator output from the team machine.
- Confirmed that v0 remains unchanged and passes the gate for safety and clock behaviour.
**Why:**
- The run is the required same-seed stability check before spending a full 1,000-game evaluation or modifying the policy.
**Files changed:** `progress.md`
**Results / measurements:** 400 games against F: **193/400 wins, win rate 0.4825 ± 0.0490; mean CPU 165.6 ms/game; worst CPU 370.0 ms/game; 0 games over budget; 0 random moves after exhaustion; 0 errors across 22,018 decisions; 0 timeouts/crashes; load time 0.00 s; wall clock 163.5 s.** The interval includes an even matchup, so this is a clean baseline rather than evidence of a strength edge.
**Open issues / handoff to next agent:** Run the required 1,000-game v0 evaluation against F before upload #1. If it remains fault-free and inside the clock, preserve v0 as the fallback and begin v1 chance-aware evaluation.

### [2026-10-17 11:48 IST] Agent: OmniRush.ai (gpt-6-astra)
**Task given by user:** Review and record the required 1,000-game v0 evaluation against F.
**What I did:**
- Reviewed the official evaluator output from the team machine.
- Confirmed v0 is a clean, reproducible fallback with no clock or interface failures.
**Why:**
- The 1,000-game run is the required upload-quality baseline before changing one algorithmic variable for v1.
**Files changed:** `progress.md`
**Results / measurements:** 1,000 games against F: **492/1,000 wins, win rate 0.4920 ± 0.0310; mean CPU 166.5 ms/game; worst CPU 380.0 ms/game; 0 games over budget; 0 random moves after exhaustion; 0 errors across 55,499 decisions; 0 timeouts/crashes; load time 0.00 s; wall clock 412.5 s.**
**Open issues / handoff to next agent:** v0 is ready as the fallback and for an early baseline upload to measure VS O. Preserve this commit/version unchanged, then implement v1 chance-aware one-ply evaluation and compare it against v0 on the same 400-game seeds.

### [2026-10-17 12:13 IST] Agent: OmniRush.ai (gpt-6-astra)
**Task given by user:** Start v1 chance-aware one-ply evaluation instead of uploading v0.
**What I did:**
- Updated `submission/agent.py` from v0's single successor per action to v1's weighted expected successor value while preserving the v0 scorer as a fallback.
- Forced roll outcomes through `ActionRecord` using the exact 2d6 sum distribution (11 sum branches, total weight 36).
- Forced development-card outcomes using the sampled deck composition and robber-steal outcomes using the victim's visible resource frequencies.
- Added CPU-deadline checks, global-random-state restoration around branches, deterministic branch order, and a no-second-search fallback when v1 consumes its decision budget.
- Kept one sampled hidden-information world per decision so v1 changes only chance handling; additional world averaging remains a later experiment.
**Why:**
- v0 is mathematically valid but trusts one roll/card/steal outcome. v1 reduces that variance without changing the value weights or adding recursive search.
**Files changed:** `submission/agent.py`, `progress.md`
**Results / measurements:** `python3 -m py_compile submission/agent.py` passed. A local dependency-free structural stub test passed: 11 roll branches with total weight 36, development branches weighted by deck counts, robber branches weighted by victim hand counts, and a legal integer decision. The official remote v1 evaluator has not run yet.
**Open issues / handoff to next agent:** Pull this commit on the team machine and run v1 against F on the same 400-game seed sequence used for v0. Compare win rate, CPU, over-budget games, and faults before considering v1 an improvement. Do not upload v0 or v1 until the comparison is reviewed.

### [2026-10-17 12:19 IST] Agent: OmniRush.ai (gpt-6-astra)
**Task given by user:** Review the first official 400-game v1 comparison against F.
**What I did:**
- Compared the v1 output with the v0 400-game gate on the same default evaluator seed sequence.
- Confirmed v1 is fault-free and remains well inside the cumulative CPU clock.
**Why:**
- This is the required one-change gate for deciding whether chance-aware scoring is a promising replacement for the v0 fallback.
**Files changed:** `progress.md`
**Results / measurements:** v1 over 400 games against F: **204/400 wins, win rate 0.5100 ± 0.0490; mean CPU 199.4 ms/game; worst CPU 480.0 ms/game; 0 games over budget; 0 random moves after exhaustion; 0 errors across 22,149 decisions; 0 timeouts/crashes; load time 0.00 s; wall clock 177.1 s.** Compared with v0's 0.4825 ± 0.0490 and 165.6 ms/game, v1 is +2.75 percentage points at +33.8 ms/game mean CPU (+20.4%). The confidence intervals overlap, so this is promising evidence, not a statistically conclusive win yet.
**Open issues / handoff to next agent:** Run the 1,000-game v1 evaluation against F before deciding whether the improvement is robust. Keep v0 at commit `9868b58` as the fallback; do not upload yet.

### [2026-10-17 12:25 IST] Agent: OmniRush.ai (gpt-6-astra)
**Task given by user:** Start v2 immediately instead of running the v1 1,000-game confirmation.
**What I did:**
- Extended `submission/agent.py` with v2 shallow own-turn lookahead.
- Added a CPU-budgeted root pass, a four-action beam, and up to six continuation candidates while the sampled successor remains our turn.
- Kept v1's exact chance branches at root and continuation nodes, including forced `ActionRecord` outcomes for rolls, development draws, and robber steals.
- Made v2 fall back to the best completed v1-style root score or a cheap legal choice when the deadline is reached; it never starts a second full search after consuming the current decision budget.
**Why:**
- v1 showed a positive but not yet conclusive 400-game signal. v2 targets the larger weakness—one-ply myopia—while respecting the one-core cumulative clock and preserving v0/v1 in Git history.
**Files changed:** `submission/agent.py`, `progress.md`
**Results / measurements:** `python3 -m py_compile submission/agent.py` passed. The dependency-free v2 structural stub test passed, including exact chance-branch weights and a legal integer decision. The official remote v2 evaluator has not run yet.
**Open issues / handoff to next agent:** Pull the latest commit on the team machine and run a short v2 smoke evaluation first. Watch CPU, over-budget games, errors, and timeouts closely; if v2 is too slow, reduce the beam/continuation widths before any longer run.

### [2026-10-17 12:30 IST] Agent: OmniRush.ai (gpt-6-astra)
**Task given by user:** Review the first official v2 smoke evaluation against F.
**What I did:**
- Reviewed the remote 20-game v2 output and confirmed the new shallow search remains legal and inside the cumulative clock.
- Kept the current four-root/f six-continuation beam unchanged pending a larger same-seed comparison.
**Why:**
- V2 is more expensive than v1, so the first gate is safety: no over-clock games, exceptions, illegal indices, or process failures before measuring strength.
**Files changed:** `progress.md`
**Results / measurements:** 20 games against F: **11/20 wins, win rate 0.5500 ± 0.2180; mean CPU 349.0 ms/game; worst CPU 760.0 ms/game; 0 games over budget; 0 random moves after exhaustion; 0 errors across 1,029 decisions; 0 timeouts/crashes; load time 0.00 s; wall clock 12.9 s.** The win-rate interval is too wide to compare reliably, while CPU is +149.6 ms/game versus v1's 199.4 ms 400-game mean.
**Open issues / handoff to next agent:** Run the same 400-game seed sequence against F. If v2 remains fault-free and improves the point estimate without pushing games over budget, preserve it; otherwise reduce `V2_BEAM_WIDTH` or `V2_CONTINUATION_WIDTH` before further evaluation.

### [2026-10-17 12:37 IST] Agent: OmniRush.ai (gpt-6-astra)
**Task given by user:** Review the official 400-game v2 result and change direction if it does not improve strength.
**What I did:**
- Compared v2 with the measured v0/v1 baselines on the same default 400-game evaluation.
- Rejected the current v2 own-turn beam as the active policy: it costs substantially more CPU without improving the point estimate over v1.
- Chose the next emergency experiment: retain v1's safer chance-aware search, replace F's default value weights with the pinned engine's public contender weights, and measure that change separately.
**Why:**
- The 400-game result is the first reliable v2 signal: deeper own-turn search alone is not buying strength. The leaderboard gap requires a value-policy change, not simply more successor evaluations.
**Files changed:** `progress.md`
**Results / measurements:** v2 over 400 games against F: **200/400 wins, win rate 0.5000 ± 0.0490; mean CPU 334.6 ms/game; worst CPU 880.0 ms/game; 0 games over budget; 0 random moves after exhaustion; 0 errors across 19,067 decisions; 0 timeouts/crashes; load time 0.00 s; wall clock 234.8 s.** Compared with v1's 0.5100 ± 0.0490 and 199.4 ms/game, v2 is -1.0 percentage point at +135.2 ms/game mean CPU (+67.8%).
**Open issues / handoff to next agent:** Keep v2 in Git history but do not submit it. Test the contender-weight/v1 candidate next; if it does not move the result materially, implement opponent-threat/response scoring rather than increasing search depth again.

### [2026-10-17 12:43 IST] Agent: OmniRush.ai (gpt-6-astra)
**Task given by user:** Continue immediately after v2 failed to improve the 400-game baseline and push the next candidate.
**What I did:**
- Switched the active policy back to the faster v1 chance-aware search.
- Replaced F's `DEFAULT_WEIGHTS` with the pinned catanatron `CONTENDER_WEIGHTS`, keeping the change isolated as the next experiment.
- Left v2's own-turn beam in the file and Git history for reference, but it is no longer active in `decide`.
**Why:**
- v2 increased CPU by 67.8% over v1 without improving VS F. The next high-leverage change is value-policy quality, not more depth. The contender weights are public engine material and emphasize reachability, roads, hand synergy, and army progress.
**Files changed:** `submission/agent.py`, `progress.md`
**Results / measurements:** `python3 -m py_compile submission/agent.py` passed. The dependency-free structural test passed with the contender-weight import, exact chance weights, and a valid branch structure. No official remote run has measured the contender candidate yet.
**Open issues / handoff to next agent:** Pull the latest commit and run a 20-game smoke test, then a 400-game same-seed comparison if it is fault-free. Do not claim that the contender weights improve strength until the evaluator measures them.
