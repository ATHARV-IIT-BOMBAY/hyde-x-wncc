# Hyde × WnCC Catan Hackathon — Final Execution Plan

- **Team:** Attention is All We Need
- **Date:** Saturday 17 October 2026
- **Build window:** 11:00–17:00 IST
- **Repository:** `ATHARV-IIT-BOMBAY/hyde-x-wncc`

> This is the execution plan for the official starter kit, not the earlier
> simplified Catan preview. The official problem repository is the source of
> truth for the engine, action API, rules, clock, and evaluation procedure.

## 1. Source of truth and confirmed facts

Read from `lyric-hyde/catan-hackathon-agent` at commit `04fffac`, including
`PROBLEM_STATEMENT.md`, `catan_kit/obs.py`, `catan_kit/match.py`,
`catan_kit/host.py`, `catan_kit/evaluate.py`, and `examples/f_agent.py`.

### Game and observation

- Standard **two-player Catan**, first to 10 victory points.
- The only rule difference from the base game is that **player-to-player
  trading is disabled**. Bank and port trading are enabled.
- The exact engine is the pinned, patched `catanatron` build in the official
  `requirements.txt`. Never install an unrelated PyPI release.
- The agent sees the full board, own resources and development cards, exact
  opponent resource counts, public victory points, bank counts, played
  development cards, turn number, visible history/dice, and `cpu_left`.
- The only meaningful hidden information is the identity of the opponent's
  unplayed development cards, including hidden victory-point cards.
- `obs.sample_world(rng)` returns a complete world consistent with the visible
  observation. It is the permitted basis for imperfect-information search.

### Agent interface

The final submission is direct, not Gym-style:

```python
class Agent:
    def __init__(self):
        # Load weights/tables and warm up expensive imports here.
        pass

    def decide(self, obs, playable_actions) -> int:
        # Return an index into playable_actions, never an Action object.
        pass
```

The harness automatically plays decisions with exactly one legal action. The
agent is called only when there are at least two candidates.

### Evaluation and clock

- Reference opponents are **F** and **O**; other teams' agents are used later.
- Local scoring uses `python -m catan_kit.evaluate`.
- Boards are played from both seats, with the same board/dice pair for the two
  seats. Final evaluation uses private board seeds and a private dice key.
- Each team has at most **7 scored submissions** and **10 uploads total**.
- The agent gets one CPU core and 2 GB memory. No network, subprocesses,
  compiled extensions, or runtime language-model calls.
- Per-game CPU allowance:

  ```text
  min(300 ms + 12.5 ms × number_of_decisions, 1.5 s)
  ```

- The allowance is cumulative for the game. Once exhausted, the harness plays
  random legal moves for the rest of that game.
- A single decision has a 3-second wall-time limit. More than 2% faults or
  illegal indices can disqualify the submission.

## 2. Strategy decision

The order is deliberately:

```text
v0: safe F-style baseline
  → v1: chance-aware one-ply evaluation
  → v2: budgeted end-of-turn/opening search
  → v3: tune weights for the final search policy
  → v4: optional learned value/action scorer
```

**Do not move weight tuning ahead of search.** Weight tuning is only useful for
the policy that will actually be submitted. Search changes which positions and
actions the weights see; tuning first risks overfitting a one-ply policy and
then invalidating the result when search is added. v1 is a low-risk improvement
to F's main blind spot, so it comes before deeper search. If v1 is not cleanly
working within its time box, skip it and continue with v2.

The primary objective is a robust, fault-free agent. A learned model is a
stretch goal, not a reason to replace a measured search agent that wins.

## 3. Repository and submission layout

Keep development code and the final zip separate:

```text
hyde-x-wncc/
├── HACKATHON_PLAN.md       # this plan
├── progress.md             # append-only agent/team log
├── scaffold/               # generic pre-event scaffold; never submit it
├── submission/             # final agent and only files it needs
│   ├── agent.py            # required entry point
│   ├── value.py            # optional value/search helpers
│   ├── search.py           # optional search helpers
│   ├── weights.json        # optional tuned constants
│   └── model.pt            # optional event-trained weights
└── experiments/            # local run logs; include only if useful for report
```

The zip must contain `agent.py` at its top level. Build it from the contents
of `submission/`, not from the repository root. Exclude `.git`, `scaffold/`,
private credentials, caches, and unrelated files.

The final agent must load all files relative to `__file__`; the evaluator's
agent directory is read-only. Temporary files, if genuinely needed, belong in
`tempfile.gettempdir()`.

## 4. Milestone v0 — safe baseline and first upload

**Target:** approximately 11:00–11:30, upload #1.

### Implementation

Write the F logic freshly against the official interface:

1. Import the official engine/kit in the same order as the starter example so
   the determinism patch is applied.
2. In `__init__`, construct the random generator and value function, and warm
   up imports/caches. Do not spend the first-decision budget importing PyTorch
   or building feature orderings.
3. In `decide`, return an index into the supplied action list. For each action:
   - obtain one `obs.sample_world(self.rng)`;
   - copy the sampled world;
   - execute the candidate;
   - score the successor with the official 13-feature value function;
   - skip candidates that fail to execute;
   - use the original action-list index as the deterministic tie-break.
4. Add a cheap fallback path. If `cpu_left` is low, stop searching and choose a
   known legal candidate by a simple priority: immediate winning/building
   action, then the first legal action. Never return an Action object.
5. Wrap the decision path narrowly enough to guarantee a valid integer index.
   If every evaluation fails, return `0` because the harness only calls the
   agent with a non-empty legal list. Log no output from the submitted agent.

### New-game detection

The harness keeps one `Agent` object across many games and does not call a
`reset` method. Maintain a small per-game signature and reset only transient
caches when one of these occurs:

- `obs.me` changes;
- `obs.turn` drops below the previous turn;
- the count of board buildings or roads drops below the previous observation.

Do not assume that a colour change is the only signal: consecutive games can
give the process the same seat colour. Do not infer or access hidden state from
private harness fields; use only the public observation and sampled worlds.

### v0 gate

Before upload:

- Run `python -m catan_kit.selfcheck` and fix environment failures first.
- Run a short smoke match to verify loading, both seats, and no faults.
- Run the official evaluation against F locally, recording win rate, 95%
  interval, mean/worst CPU, over-budget games, errors, timeouts, crashes, and
  load time.
- Upload v0 early even if it is only approximately even with F. This tests the
  complete submission path and preserves the remaining upload budget.

Do not call v0 a success based on a claimed win rate until the actual run has
printed it. Record every number in `experiments/` and `progress.md`.

## 5. Milestone v1 — chance-aware one-ply evaluation

**Target:** approximately 11:30–12:15. **Skip at the time box if unstable.**

F samples one world and then trusts one random outcome. Improve only this
decision, without changing the value weights or the action policy structure.

Use a small action-specific outcome spectrum for stochastic actions:

- `ROLL`, whose result is a dice outcome;
- `BUY_DEVELOPMENT_CARD`, whose result is a card draw;
- robber moves that can steal a random resource;
- any other action whose engine execution has a random result.

For deterministic builds and maritime trades, one successor is enough. For a
stochastic action, execute the same action on several independent copies with
controlled random states, average the successor values, and reduce the sample
count as `cpu_left` falls. Preserve and restore the search process's random
state so search cannot accidentally affect real-game reproducibility.

Use a fixed deterministic tie-break and retain F's fallback. Do not sample so
many worlds that later turns are forced random by clock exhaustion.

### v1 gate

Compare v1 with v0 on the **same 400-game seed set**, with seats alternated.
Keep v1 only if it has zero faults/illegal indices, stays inside the clock,
and shows a meaningful win-rate or CPU-quality improvement. Then run the full
1,000-game F evaluation before considering upload #2.

## 6. Milestone v2 — budgeted search that fits the clock

**Target:** approximately 12:15–13:30.

This is the main algorithmic improvement. Search only while a measured CPU
budget remains; never spend the whole allowance on the opening and then play
randomly later.

### Budget policy

- Treat `obs.cpu_left` as the hard upper bound for the current game.
- Reserve a safety margin for later decisions and Python/harness overhead.
- Use one thread only. If PyTorch is imported, set `torch.set_num_threads(1)`
  and `torch.set_num_interop_threads(1)` before any model work.
- Use process CPU time for local budgeting, not only wall time.
- When the local deadline is reached, return the best completed action; never
  wait for an unfinished branch.

Suggested adaptive policy:

| Remaining CPU | Work allowed |
|---|---|
| Very low | Cheap legal-action priority and immediate-win check |
| Moderate | One sampled world plus chance-aware root scoring |
| High | Multiple worlds, or a shallow beam through our current turn |
| Opening/base allowance | Spend extra time comparing initial settlements/roads |

### Search shape

Extend a root action through the current player's turn when possible instead
of stopping after one build. Stop at the opponent's decision, terminal state,
or the local deadline. Use a narrow beam or top-k continuation list so the
branching factor remains controlled. At chance nodes, use a small outcome
spectrum rather than pretending the sampled roll/card is deterministic.

Opening placement deserves special treatment because the clock starts with the
300 ms base allowance and the action list is broad. Evaluate the initial
settlement and its connected road jointly when the budget allows; fall back to
the one-ply value if not.

### v2 gate

Run the same 400-game comparison against the best of v1/v0. Inspect not only
win rate but also:

- games over budget and random moves after exhaustion;
- worst-game CPU;
- faults, timeouts, and crashes;
- game length and whether the agent collapses late in long games.

Keep v2 only when it improves the complete agent, not merely the first few
decisions. Run 1,000 games against F before upload #3.

## 7. Milestone v3 — tune weights after the policy is stable

**Target:** approximately 13:30–15:00, after the lunch break if necessary.

Tune the 13-feature value weights only for the best final search structure.
Start with a small coordinate/grid search around the default weights. Use
CMA-ES from the already-pinned SciPy only if the simpler search is productive
and there is enough time. Do not launch a large population that consumes the
entire afternoon.

### Tuning discipline

- Keep public-VP/win terms dominant; do not allow a production heuristic to
  override an immediate win.
- Use separate, non-overlapping train/tuning, gate, and held-out evaluation
  seed ranges.
- Compare every candidate against the current best on the same 400-game gate
  seeds. Parallel local games are allowed for experiments, with exactly one
  thread per worker and no shared agent state.
- Accept a candidate only when its confidence interval and CPU/fault profile
  justify the change. A tiny apparent gain inside the interval is not a gain.
- After selecting weights, run a fresh 1,000-game evaluation against F and
  record the exact commit and weight file used.

Weight tuning comes here—not before v2—because the search depth and chance
handling define the distribution of positions being scored.

## 8. Milestone v4 — optional learned scorer

**Target:** only if v3 is stable by approximately 15:00; otherwise stop at v3.

The learned component is a report/demo enhancer, not the safety baseline. Use
the search agent to generate labels from the event's own games, then distil a
small action-conditioned scorer:

- Input: public state features plus an encoding of one candidate action.
- Output: a scalar successor value or action score, not a fixed global action
  index. The legal action list changes every decision.
- Prefer a tiny MLP first. Use the existing causal transformer only if history
  demonstrably helps and the resulting model remains comfortably inside the
  CPU clock.
- Keep the model in the requested small-model range (roughly 80–150k
  parameters) only if that does not hurt latency.
- Batch candidate scoring within one decision, mask nothing by guessing: the
  candidates being scored are already the legal list.
- Load and warm up the model in `__init__`; never train during play.
- If the model raises, is slow, or loses to the tuned search agent on the same
  400-game gate, submit the tuned search agent instead.

No pre-event Catan weights or trajectories may enter the submission. Any
training data, weights, and tuning used in the final agent must be generated
during the event or come from clearly documented public material.

## 9. Experiment protocol and logging

Change exactly one meaningful thing between adjacent versions. Each run gets a
row in `experiments/runs.csv` or `experiments/runs.jsonl`:

```text
run_id, timestamp_ist, commit, variant, opponent, games, seed_start,
wins, win_rate, ci95, mean_cpu_ms, worst_cpu_ms, over_budget,
random_moves, faults, timeouts, crashes, load_s, notes
```

For every comparison, preserve:

- the exact commit and weights/model file;
- engine and Python versions;
- command line and seed range;
- seat alternation setting;
- whether the run was a smoke test, 400-game gate, or 1,000-game upload
  check.

Required gates:

1. **Before keeping a milestone:** 400 games against the previous version on
   the same seeds.
2. **Before every upload:** 1,000 games against F, with no unexplained faults
   or illegal indices.
3. **Before the final upload:** rerun the selected candidate from a clean
   checkout/zip and inspect the zip contents.

The portal's O results and private-board results are authoritative when they
arrive. Do not infer them from local F results.

## 10. Upload and report checklist

### Before each upload

- `python -m catan_kit.selfcheck` passes.
- The candidate has `agent.py` at the zip root and all paths are relative to
  the agent directory.
- The zip contains no `.git`, `scaffold/`, credentials, API keys, caches, or
  unnecessary files and is below 100 MB.
- The official 1,000-game F run has been recorded.
- Faults, illegal indices, timeouts, and over-budget games are reviewed.
- The upload number is recorded; failed uploads count too.

### Report material

Keep the report honest and concise:

1. Interface and constraints: dynamic legal actions, one CPU core, hidden
   development cards, and the cumulative clock.
2. Baseline: fresh F-style one-ply value search.
3. Improvements: chance averaging, budgeted search, and any weight tuning.
4. Experiments: same-seed comparisons, held-out seeds, CPU cost, failures,
   and confidence intervals.
5. Final method: explain why it was selected over any learned alternative.
6. Public material: cite the starter kit, pinned engine, and any literature or
   public code used.

Do not put an unmeasured win rate or a failed experiment in the report as a
success. A robust fallback with clear evidence is better than a fragile model.

## 11. Git and handoff discipline

- Append one entry to `progress.md` at the end of every milestone; never edit
  or delete previous entries.
- Include actual measured results only. If a benchmark was not run, write
  `not run` rather than estimating it.
- Push the plan, progress entry, and milestone code directly to `main` after
  reviewing `git status`, `git diff`, and the recent log.
- Keep each commit focused and use a conventional message, for example
  `docs: add final hackathon execution plan` or `feat: add chance-aware search`.
- Do not commit secrets or local credentials.
- The current generic `scaffold/eval.py` winner-attribution bug is outside this
  Catan submission path. Leave it untouched unless a separate task explicitly
  asks for it.

## 12. Immediate next actions

1. Run the pinned environment self-check from the official starter-kit root.
2. Create the minimal `submission/agent.py` v0 and test it against F.
3. Record the first measured baseline before changing the algorithm.
4. Upload v0 early to validate the complete path.
5. Continue only through the gates above; if a later version is worse or
   unstable, submit the last measured best version.
