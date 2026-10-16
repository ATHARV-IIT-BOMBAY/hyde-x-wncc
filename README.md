# Hyde × WnCC AI Builder Hackathon — Team "Attention is All We Need"

> **Status:** Scaffold ready, awaiting official problem statement (drops Sat 17 Oct 10:30 in LA001). Game revealed: **2-player Catan** (simplified, 10 VP). See [`context.md`](context.md) for full context and [`PLAN_TOMORROW.md`](PLAN_TOMORROW.md) for the game-day plan.

## Team
- **Atharv Pratap Singh Chauhan** — 2nd yr Mech, CP Candidate Master, PyTorch/C++, SIH 2026 Nationals ML
- **Shrot Vijayvergiya** — Econ, CV/ML
- **Barad Dhanush** — Electrical Dual, Systems / AI
- **Ishaan Singla** — Civil, Data Analysis

Registered 13 Oct 2026, approved (64 teams). Venue LA001, 17 Oct 10 AM – 8 PM.

## Constraints
- **CPU-only** (`torch` CPU, `cuda:False`, 4 threads) — no GPUs
- **Small transformer** — `d_model 64-128, n_layer 2-3, n_head 2-4, ~80-150k params` — judged on win consistency + explainable strategy
- **All work built DURING the hackathon** (per T&C — generic scaffold below is allowed, no Catan weights yet)

## Contacts & Links
- Luma: https://luma.com/sd8b1f0r · Updates: https://chat.whatsapp.com/LKkIipw3okmE46AiXl4BHg
- Hyde Inference: https://inference.hyde.tech/?auth=signin · OpenCode: https://inference.hyde.tech/docs/opencode
- Rules deck (pre-announce): 11-page `Hyde@IITB.pdf` — 2P Catan, 5 resources, bank/ports trading, 10 VP via settlement/city/VP card/Largest Army (3 knights)/Longest Road (5 segments)

## Structure
```
scaffold/
  envs/__init__.py       # REGISTRY + make(name) — swapping games = one file
  envs/tictactoe.py      # 3×3 + perfect solver (smoke tests)
  envs/connect4.py       # 6×7 + negamax solver (expert data demo)
  models/transformer.py  # TinyTransformer: causal, ~102k @ 64d/2L/2H
  data.py                # generate_trajectories + BCTrajectoryDataset (padded batches)
  train.py               # CPU loop: AdamW, last-token CE, 4 threads
  eval.py                # vs random / vs solver
  README.md              # usage + benchmarks
PLAN_TOMORROW.md         # full game-day plan (11:00-17:00 build phases)
context.md               # mega prompt — full context for any model
```

## Quick start
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium pettingzoo numpy
python -m scaffold.train --env tictactoe --n_games 300 --epochs 3
python -m scaffold.eval  --env tictactoe --n_games 100 --opponent random
# verified: torch 2.14.1 CPU, gymnasium 1.4.0, tictactoe 89% win vs random in 0.6s data + 0.6s train (~8k samp/s)
```

## Tomorrow (17 Oct)
1. **09:15** — All 4 arrive together at LA001 with college IDs
2. **10:30** — Parse problem statement: env API / action space / observation / eval / constraints / reference bots
3. **11:00-13:00** — Phase A: wrap their Catan env → `scaffold/envs/catan.py` + heuristic bot (city > settlement > knight > road + greedy 4:1 trader)
4. **13:30-16:00** — Phase B: generate heuristic self-play data → train tiny transformer as candidate ranker (BC)
5. **16:00-17:00** — Phase C: harden vs reference bots + prepare 2-min pitch (win rate + strategy explanation)

See [`PLAN_TOMORROW.md`](PLAN_TOMORROW.md) for the full timeline, `encode()` vector layout, and fallbacks.

---
*Teams: 64 · Prize: ₹2.5L + internships · Tokens till end 2026 · Schedule 09:15 registration / 10:00 start / 17:00 judging / 18:00 Top 6*
