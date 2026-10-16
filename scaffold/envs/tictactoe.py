"""Tic-tac-toe: 3x3, fast iteration / smoke tests."""
import numpy as np

class TicTacToeEnv:
    n_actions = 9
    obs_dim = 9

    def __init__(self):
        self.board = np.zeros(9, dtype=np.int8)
        self.player = 1
        self.moves = 0

    def reset(self):
        self.board[:] = 0
        self.player = 1
        self.moves = 0
        return self.encode()

    def encode(self):
        return (self.board * self.player).astype(np.float32)

    def legal_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def is_done(self):
        return self._winner() is not None or self.moves >= 9

    def winner(self):
        w = self._winner()
        if w is not None:
            return w
        return 0 if self.moves >= 9 else None

    def step(self, action):
        assert action in self.legal_actions()
        self.board[action] = self.player
        self.moves += 1
        w = self._winner()
        if w is not None:
            return self.encode(), 1.0, True, {"winner": w}
        if self.moves >= 9:
            return self.encode(), 0.0, True, {"winner": 0}
        self.player *= -1
        return self.encode(), 0.0, False, {"winner": None}

    def _winner(self):
        wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a,b,c in wins:
            if self.board[a] != 0 and self.board[a]==self.board[b]==self.board[c]:
                return int(self.board[a])
        return None

    def best_action(self, depth=9):
        # perfect solver via minimax (tiny)
        best, best_a = -1e9, self.legal_actions()[0]
        for a in self.legal_actions():
            child = self._clone()
            child.step(a)
            if child.is_done():
                w = child._winner()
                score = 1 if w is not None else 0
                # we just won if w==self.player before step
                if w is not None:
                    score = 1
                else:
                    score = 0
            else:
                score = -self._negamax(child, depth-1, -1e9, 1e9)
            if score > best:
                best, best_a = score, a
        return best_a

    def _negamax(self, env, depth, alpha, beta):
        if env.is_done() or depth == 0:
            return 0
        best = -1e9
        for a in env.legal_actions():
            child = env._clone()
            child.step(a)
            if child.is_done():
                w = child._winner()
                score = -1 if w is not None else 0
            else:
                score = -self._negamax(child, depth-1, -beta, -alpha)
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def _clone(self):
        c = TicTacToeEnv()
        c.board = self.board.copy()
        c.player = self.player
        c.moves = self.moves
        return c
