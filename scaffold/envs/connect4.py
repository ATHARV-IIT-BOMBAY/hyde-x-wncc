"""Connect-4 env: 6 rows x 7 cols, win=4. Includes negamax solver for expert data."""
import numpy as np

ROWS, COLS = 6, 7
WIN = 4

class Connect4Env:
    n_actions = COLS
    obs_dim = ROWS * COLS  # flat board, values: 0 empty / 1 player 1 / -1 player 2

    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.player = 1
        self.moves = 0

    # --- core API ---
    def reset(self):
        self.board[:] = 0
        self.player = 1
        self.moves = 0
        return self.encode()

    def encode(self):
        """Board from current player's perspective. Flattened float32."""
        return (self.board * self.player).astype(np.float32).reshape(-1)

    def legal_actions(self):
        return [c for c in range(COLS) if self.board[0, c] == 0]

    def is_done(self):
        return self._winner() is not None or self.moves >= ROWS * COLS

    def winner(self):
        w = self._winner()
        if w is None:
            return 0 if self.moves >= ROWS * COLS else None
        return w

    def step(self, action: int):
        assert action in self.legal_actions(), f"Illegal action {action}"
        # drop
        for r in range(ROWS - 1, -1, -1):
            if self.board[r, action] == 0:
                self.board[r, action] = self.player
                break
        self.moves += 1
        w = self._winner()
        if w is not None:
            reward = 1.0  # mover wins (from mover's perspective)
            done = True
        elif self.moves >= ROWS * COLS:
            reward = 0.0
            done = True
        else:
            reward = 0.0
            done = False
            self.player *= -1
        return self.encode(), reward, done, {"winner": w}

    # --- win detection ---
    def _winner(self):
        b = self.board
        for r in range(ROWS):
            for c in range(COLS):
                if b[r, c] == 0:
                    continue
                v = b[r, c]
                for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
                    cnt = 1
                    for k in range(1, WIN):
                        rr, cc = r+dr*k, c+dc*k
                        if 0 <= rr < ROWS and 0 <= cc < COLS and b[rr,cc]==v:
                            cnt += 1
                        else:
                            break
                    if cnt == WIN:
                        return int(v)
        return None

    # --- solver for expert data (negamax with alpha-beta, depth-limited) ---
    def best_action(self, depth=4):
        """Return best action via negamax. Depth 4 is strong enough for expert data."""
        best, best_a = -1e9, self.legal_actions()[0]
        for a in self.legal_actions():
            child = self._clone()
            child.step(a)
            # step already flipped player if not terminal; if terminal, score is immediate
            if child.is_done():
                score = 1 if child.winner() != 0 else 0
                # from parent perspective: mover won
                # child.winner() is absolute, but reward above is mover
                # simplify: terminal win for mover = +1
                if child._winner() is not None:
                    score = 1  # we just won
                else:
                    score = 0
            else:
                score = -self._negamax(child, depth - 1, -1e9, 1e9)
            if score > best:
                best, best_a = score, a
        return best_a

    def _negamax(self, env, depth, alpha, beta):
        if env.is_done() or depth == 0:
            return self._heuristic(env)
        best = -1e9
        for a in env.legal_actions():
            child = env._clone()
            child.step(a)
            if child.is_done():
                w = child._winner()
                val = 1 if w is not None else 0  # win for mover
                # negate because perspective flips each step - but terminal win is for mover
                # if mover won, that's -1 for us (opponent won)
                score = -val if w is not None else 0
            else:
                score = -self._negamax(child, depth - 1, -beta, -alpha)
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def _heuristic(self, env):
        """Simple pattern count heuristic from current player's view."""
        b = env.board * env.player
        score = 0
        for r in range(ROWS):
            for c in range(COLS):
                for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
                    vals = []
                    for k in range(WIN):
                        rr, cc = r+dr*k, c+dc*k
                        if 0 <= rr < ROWS and 0 <= cc < COLS:
                            vals.append(int(b[rr,cc]))
                        else:
                            vals.append(None)
                    if None in vals:
                        continue
                    if vals.count(1) == 3 and vals.count(0) == 1:
                        score += 10
                    elif vals.count(1) == 2 and vals.count(0) == 2:
                        score += 1
                    elif vals.count(-1) == 3 and vals.count(0) == 1:
                        score -= 8
        return score * 0.01

    def _clone(self):
        c = Connect4Env()
        c.board = self.board.copy()
        c.player = self.player
        c.moves = self.moves
        return c
