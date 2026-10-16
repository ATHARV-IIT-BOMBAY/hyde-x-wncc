"""Game-agnostic env interface. Every env exposes the same contract
so swapping games on hackathon day is a one-file change.

Contract:
    obs = env.reset()                 -> np.ndarray (flat or 2D, your encode)
    obs, reward, done, info = env.step(action)
    legal = env.legal_actions()       -> list[int] subset of [0, n_actions)
    done = env.is_done()
    winner = env.winner()             -> 1/-1/0  (None if not terminal)

Obs encoding is the env's job (Board -> vector). The trainer never
sees raw board geometry.

Add a new game by adding <game>.py with a class <Game>Env that
implements the above, then register it in REGISTRY below.
"""
from .connect4 import Connect4Env
from .tictactoe import TicTacToeEnv

REGISTRY = {
    "connect4": Connect4Env,
    "tictactoe": TicTacToeEnv,
}

def make(env_name: str, **kwargs):
    if env_name not in REGISTRY:
        raise ValueError(f"Unknown env {env_name!r}. Available: {list(REGISTRY)}")
    return REGISTRY[env_name](**kwargs)
