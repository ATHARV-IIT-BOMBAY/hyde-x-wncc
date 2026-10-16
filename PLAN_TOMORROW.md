# PLAN — Hyde × WnCC Hackathon, Sat 17 Oct 2026, LA001

## TL;DR
Game = **2-player Catan** (simplified, 10 VP, deck = `Hyde@IITB.pdf`). 10 hours (11:00-17:00 build window). Judged on **win consistency vs reference bots + other teams** + **explain the strategy**. CPU-only, small transformer. **Win condition tomorrow: have a heuristic bot winning by 13:00, then layer transformer on top.**

---

## 0) Tonight (14 Oct night) — 30 min max, then sleep

| Task | Owner | Done? |
|------|-------|-------|
| All 4 sign into `inference.hyde.tech` with IITB email + setup OpenCode (`inference.hyde.tech/docs/opencode`) | Everyone | ☐ |
| Join WhatsApp updates `chat.whatsapp.com/LKkIipw3okmE46AiXl4BHg` | Atharv | ☐ |
| Skim `Hyde@IITB.pdf` visually — Page 6 (Building Costs image) is the only critical page | Everyone | ☐ |
| Charge laptop + charger + phone + mouse/headphones. No dev work. | Everyone | ☐ |
| Agree in WhatsApp: "Meet at [hostel/gate] 8:45, reach LA001 9:15 together" | Atharv | ☐ |

**Do NOT** build a full Catan env tonight — pre-built solutions are disqualified, and the real API drops at 10:30. Scaffold is frozen (`scaffold/` verified: 102k params, tictactoe 89% vs random in 0.6s).

---

## 1) Timeline — The 10 Hours

| Time | What happens | Your focus |
|------|--------------|------------|
| **09:15-09:50** | Registration (name + team name "Attention is All We Need" + college ID, collect attendance card, find placard) | Arrive **together**, all 4 with ID. Sit near power. |
| **10:00-10:30** | Welcome / What is Hyde / IITB@Hyde | Listen, but have laptops open. Note any hiring signals. |
| **10:30-10:50** | **PROBLEM STATEMENT WALKTHROUGH** — *This is the most important 20 min* | **See Section 2 below. One person types, one person screenshots/notes.** |
| **10:50-11:00** | Q&A | **Ask the 3 questions in Section 2.** |
| **11:00-13:00** | **PHASE A — Heuristic baseline (guarantees you don't go 0-wins)** | Build `catan.py` wrapper + dumb-but-winning bot. |
| **13:00-13:30** | Lunch (on them) | Eat fast, keep one person testing eval in background. |
| **13:30-16:00** | **PHASE B — Transformer ranker + self-play BC** | Train small transformer to beat the heuristic. |
| **16:00-17:00** | **PHASE C — Harden + Explainability** | Eval vs reference bots, longest-road/army tuning, robber, prepare 2-min pitch. |
| **17:00-18:00** | Judging (music/snacks) | Be ready, laptop charged, demo loop running. |
| **18:00-19:45** | Top 6 -> Present -> Winners | If in Top 6: present strategy (Section 6). Else stay till end (required). |

---

## 2) The 10:30 Problem Statement — How to Parse in 20 Min

Assign **1 scribe** (Shrot or Ishaan) to capture verbatim. You (Atharv) listen for architecture decisions.

**Extract these 6 fields into a shared doc immediately:**

1.  **Env API:** What do they give you? Expected: `reset()/step(action)/legal_actions()/observation` or Gym-style. Is there a `HydeEnv` pip package? Is the board a graph object or numpy array? Save the example code snippet they show.
2.  **Action Space:** Is it a flat int (e.g. 0..N) or hierarchical dict (`{type: "build_road", edge: 12}`)? How is trading represented? Is player-to-player trading allowed or only bank/ports (deck shows only bank/ports)?
3.  **Observation Space:** What does `obs` contain? Hidden info: do you see opponent's resource counts? Hand size only? Dev card counts? This determines `encode()`.
4.  **Evaluation:** How many games vs whom? Round-robin vs other teams? Elo? Single-elimination? How is stochasticity handled (multiple seeds)? This determines risk tolerance.
5.  **Constraints:** Max model size? Max inference time per move? Must be CPU = what thread count? Is the transformer mandatory or can you win with heuristics + learned value? (Statement says "small transformer" — check if it's graded or just suggested).
6.  **Starter Code / Reference Bots:** Do they give you a random bot / heuristic bot to test against locally?

### 3 Questions to Ask at 10:50 Q&A (if not answered)

1.  "Is the environment deterministic given a seed, and will evaluation be on fixed seeds or random seeds?"
2.  "Is player-to-player trading allowed or only bank/ports as in the deck? And is `legal_actions` already filtered or do we need to generate candidates?"
3.  "Is there an inference time limit per move, and is the model required to be a transformer or is any learned model that beats reference bots acceptable if we explain it?"

---

## 3) PHASE A (11:00-13:00) — Heuristic Baseline

**Goal:** By 13:00, you have a bot that can play full legal games and beats random 70%+ and is testable vs their reference bot. *Do NOT start training a transformer until this works.* Non-obvious tradeoff: A 2-hour heuristic baseline is not wasted time — it becomes your data generator AND your fallback if the transformer fails.

**Parallel work split (4 people, 2 pairs):**

| Pair | Task | Output |
|------|------|--------|
| **Atharv + Barad** | **Env Wrapping (45 min)**. Create `scaffold/envs/catan.py` that wraps THEIR env to OUR interface. See template below. Make `python -m scaffold.train --env catan` not crash. `torch.set_num_threads(4)` already done. | `scaffold/envs/catan.py` + `REGISTRY["catan"]=CatanEnv` |
| **Shrot + Ishaan** | **Heuristic Bot `bots/heuristic.py` (60 min)**. Write a rule-based bot that uses ONLY `legal_actions()`. No ML. | `bots/heuristic.py: act(obs, legal)` |

**`scaffold/envs/catan.py` template (fill in at 11:00):**
```python
# THEIR_ENV is whatever they give you at 10:30
class CatanEnv:
    n_actions = ???  # or dynamic per state
    obs_dim = ???    # after encode()
    def __init__(self): self.env = THEIR_ENV()
    def reset(self): return self.encode(self.env.reset())
    def step(self, a): obs, rew, done, info = self.env.step(a); return self.encode(obs), rew, done, info
    def legal_actions(self): return self.env.legal_actions()
    def is_done(self): return self.env.is_done()
    def winner(self): return self.env.winner()
    def encode(self, obs): # THE critical function — see Section 5
        # Pack: 5 resources + VPs + roads/settlements/cities graph + dev cards + robber + dice history
        pass
```

**Heuristic Bot logic (priority order, simple wins):**
1.  If can build **City** (2 wheat+3 ore) -> do it (fast VP, doubles production).
2.  If can build **Settlement** (brick+wood+sheep+wheat) -> score each legal settlement by `sum(resource weight of adjacent hexes)` (weight = dice probability: 6,8=5 | 5,9=4 | 4,10=3...), pick best.
3.  If can play **Knight** and robber can steal -> play it (progress to Largest Army, 3 knights = +2 VP).
4.  If can build **Road** that extends toward best settlement spot or is 4th/5th road for Longest Road -> build it.
5.  **Trade:** If you need 1 resource to complete City/Settlement and you have 4 of another -> do 4:1. If you have a 2:1 port for a resource you have excess of -> use it. Else pass. (A dumb trader that trades *only* to complete immediate build beats a complex negotiator on CPU).
6.  Robber: place on hex where opponent has most settlements/cities adjacent to high-prob numbers (6/8).

**Test at 12:30:** `python -m scaffold.eval --env catan --opponent random --n_games 50` AND `heuristic vs heuristic` loop. Must see games finish (win/draw) with no illegal moves.

### Fallback at 13:00 if wrapping is stuck
If their env is weird/buggy, freeze Atharv+Barad on fixing ONE env binding. Shrot+Ishaan keep improving heuristic with manual tuning — a well-tuned heuristic CAN win this hackathon if your transformer is late. Do not block everyone on the wrapper.

---

## 4) PHASE B (13:30-16:00) — Transformer Ranker (CPU-realistic)

**Why BC on self-play, not RL/DT:** On CPU with ~3 hours, online RL (PPO) burns compute on rollouts and is sample-inefficient. DT return-conditioning is noisy with dice. **Behavior Cloning on `heuristic vs heuristic` trajectories = 1. Generate trajectories once (CPU cheap), 2. Supervised forward/backward on batches (cache-friendly).** Heuristic is already decent, so BC inherits its level; transformer then learns to *rank* candidates slightly better via history (hidden hands, dice patterns).

**Steps (Atharv leads, Barad assists):**

1.  **Generate data (20 min):** `generate_trajectories(env_fn=CatanEnv, n_games=1000, opponent="mixed")` — Use `bots/heuristic.py` as the expert instead of solver. Change `scaffold/data.py:generate_trajectories` to call `heuristic.act()` not `env.best_action()`. Target: ~15k steps (2P Catan games are ~40-60 moves). If data gen is slow, drop to 500 games — quality > quantity.

2.  **Model (same as scaffold, just retuned):**
    ```python
    TinyTransformer(obs_dim=~50-100, n_actions=dynamic, d_model=64-128, n_layer=2-3, n_head=2-4, max_len=32-64)
    # Keep params 80k-150k. Do NOT go 256d/6L — CPU inference per move must be <50ms.
    # Non-obvious: Use `max_len=32` not 64 — Catan games are short, shorter context = faster + less overfit on dice noise.
    ```
    Change `scaffold/models/transformer.py` to score *candidates* not flat n_actions: Input = `obs_seq (T, obs_dim)`, output = `logits over legal candidates` at last position. Simplest: Keep `head -> n_actions` but mask illegal logits to `-inf` before argmax.

3.  **Train (30-40 min wall time):**
    ```bash
    torch.set_num_threads(4) # already in train.py
    python -m scaffold.train --env catan --n_games 1000 --epochs 8 --d_model 64 --n_layer 2 --batch_size 32
    # Expect: ~5k samp/s on CPU, 8 epochs ~15-20s train time, data gen dominates.
    ```
    Watch `val acc` — should be 0.45-0.65 for Catan (not 0.89 like tictactoe, it's stochastic). If val loss stalls, early stop and eval.

4.  **Inference in game (candidate-filter pattern):**
    ```python
    candidates = env.legal_actions()  # e.g. 15 legal
    # Heuristic pre-filter to top 8 if >20 candidates (CPU win)
    if len(candidates) > 20: candidates = heuristic_topk(candidates, k=8)
    # Transformer scores candidates
    logits = model(obs_seq, mask)  # (1, T, n_actions)
    masked = logits[0,-1][candidates]
    action = candidates[masked.argmax()]
    if action not in candidates: action = random.choice(candidates) # safety
    ```
    This is the key CPU tradeoff: *enumerate -> filter -> rank*. Not *generate*.

5.  **Optional self-play loop (if time after 15:00):** `BC bot vs Heuristic -> generate new trajectories where BC wins -> retrain`. One iteration only. Don't loop indefinitely.

---

## 5) State Encoding — `encode()` Details

This is the highest-leverage 100 lines. Get it wrong = transformer learns nothing.

**Vector layout (example `obs_dim ~ 60-80`):**
```
[0:5]   Your resources: wood, brick, sheep, wheat, ore (normalized /10)
[5:10]  Opponent resource counts (if visible) or [hand_size /10, 0,0,0,0] if hidden
[10:12] Your VPs, Opp VPs (/10)
[12:14] Largest Army flag (you, opp), Longest Road flag (you, opp) (0/1)
[14:16] Your dev cards: knights played, VP cards hidden (count)
[16:18] Robber position (one-hot or coord), dice last roll (one-hot 2-12)
[18:48] Board graph: for each node/edge, 0/1 if you have settlement/city/road, opp has settlement/city/road (flattened adjacency — their env may give you this as graph; just flatten)
[48:60] Port ownership (which 2:1/3:1 you have), turn number, dice history last 6 rolls
```
**Tradeoff:** Do NOT try to encode opponent's hidden hand exactly — you can't. Encode `hand_size` + `dev_card_count` as proxies. Transformer will learn patterns from history.

---

## 6) PHASE C (16:00-17:00) — Harden + Pitch

**Harden (30 min, Ishaan + Shrot):**
- Run `heuristic vs heuristic`, `transformer vs heuristic`, `transformer vs random` — 50 games each. Log win rates. Fix any `Illegal action` crashes (transformer picked masked logit — see fallback above).
- Tune robber placement + road racing: if Longest Road is at 4/5, prioritize road over settlement. If Largest Army at 2/3, prioritize knights.
- If time: `torch.quantization` or just `model.half()` for inference — only if per-move latency is >100ms. Otherwise skip.

**Pitch prep (30 min, Atharv + one other):**
Judging = win consistency + explain strategy. Prepare 2-min talk:

1.  **Problem framing (15s):** "2P Catan is partially observable + stochastic, so we treated it as history-dependent ranking, not perfect-info search."
2.  **Architecture (30s):** "Heuristic candidate filter (15 -> 8) + tiny causal transformer (64d/2L, ~100k params) as ranker over candidates, trained with behavior cloning on 1000 heuristic self-play trajectories. Entirely CPU, 8k samp/s, <20s training."
3.  **Catan insights (30s):** "We exploited 2P structure: 5 roads = Longest Road is short so we race it, 3 knights = Largest Army is cheap so we time knights, and we use a greedy 4:1/2:1 trader that only trades to complete immediate City/Settlement — beats learned trading on limited compute."
4.  **Results (15s):** "Heuristic beats random 70%+, transformer adds X% vs heuristic, zero illegal moves in 100 games."

**Have ready:** One slide or whiteboard diagram: `legal_actions -> heuristic filter -> transformer ranker -> action` + `encode()` vector layout + win-rate table.

---

## 7) Risks & Fallbacks

| Risk | Signal | Fallback |
|------|--------|----------|
| Their env API is nothing like `reset/step/legal_actions` | 11:15 wrapper won't compile | Use their example bot file as template, copy-paste, ignore scaffold interface for that file only. |
| Transformer won't train (data noisy, val acc 0.3) | 14:30 val acc not moving | **Submit the heuristic bot.** A tuned heuristic that wins 60% vs reference bots and you can explain beats a non-training transformer. |
| Teammate env bug blocks training | 13:30 no trajectories | Split: 2 people fix env, 2 people tune heuristic trading/robber — heuristic path still scores. |
| Inference too slow (CPU) | Per-move >200ms | Drop to `d_model=64, n_layer=2, max_len=16`, quantize or prune candidates to top 5. |
| Pre-built disqualification fear | — | Do NOT bring weights from tonight. Generate trajectories + train AFTER 11:00. Scaffold is generic (tictactoe/connect4) so it's allowed. Keep git commits timestamped after 11:00. |

---

## 8) Checklists

**Tonight:** Charge, Hyde Inference login, WhatsApp group, skim PDF page 6, sleep.

**Tomorrow 09:15 bag:** Laptop+charger, Phone+charger, College ID (all 4), Attendance card (get at desk), Headphones/mouse, Water bottle.

**At 10:30, open on one laptop:** This plan + `scaffold/README.md` + their problem statement PDF/slides.

**Git discipline:** `git init` after 11:00 (sandbox may block — use `gh` or manual backup). Commit with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` if using Claude.

---

*Scaffold verified: `torch 2.14.1 CPU (cuda:False, 4 threads)`, `tictactoe 89% win vs random`, `connect4 53s data (solver-bound)`. Swap game = `scaffold/envs/__init__.py:REGISTRY` one line.*
