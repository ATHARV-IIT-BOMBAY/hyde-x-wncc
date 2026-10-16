# Tiny Transformer Game Scaffold

CPU-only. Swapping games is a one-file change via `scaffold/envs`.

## Structure
```
scaffold/
  envs/__init__.py      # REGISTRY + make(name)
  envs/connect4.py      # 6x7, negamax solver for expert data
  envs/tictactoe.py     # 3x3, smoke tests
  models/transformer.py # TinyTransformer (causal, d_model=64 default)
  data.py               # generate_trajectories + BCTrajectoryDataset
  train.py              # BC loop (CPU threads tuned)
  eval.py               # vs random / vs solver
```

## Quick start
```bash
# smoke (tictactoe: 0.6s data + ~0.6s train on CPU)
python -m scaffold.train --env tictactoe --n_games 300 --epochs 3

# eval
python -m scaffold.eval --env tictactoe --n_games 100 --opponent random

# connect4 (slower data gen due to solver; train itself is 0.7s/epoch)
python -m scaffold.train --env connect4 --n_games 300 --epochs 3

# larger model (still CPU-viable: ~5.5k samp/s at 64d/2L, scales linearly)
python -m scaffold.train --env tictactoe --d_model 128 --n_layer 4 --n_head 4 --n_games 800 --epochs 8
```

## Benchmarks (this machine, CPU torch, 4 threads)
| config | params | samp/s | data (300 games) |
|--------|--------|--------|-------------------|
| tictactoe 64/2/2 | 102k | ~8k | 0.6s |
| connect4 64/2/2 | 104k | ~5.8k | 53s (solver depth 3) |

Connect-4 data is solver-bound. Hackathon fix: depth=2 or random/mixed opponent, or pre-generate once.

## Adding a new game (hackathon day)
1. Create `scaffold/envs/mygame.py` with class `MyGameEnv` implementing `reset/step/legal_actions/is_done/winner/encode` (+ optional `best_action` for expert data).
2. Register in `scaffold/envs/__init__.py:REGISTRY`.
3. `python -m scaffold.train --env mygame ...` just works.

## What's not yet (add on demand)
- Decision-transformer conditioning (return-to-go) — same data pipeline, small model change
- Quantization/int8 — `torch.quantization` or `optimum`, only if time left
- Self-play loop (BC -> play -> relabel) for no-expert games
