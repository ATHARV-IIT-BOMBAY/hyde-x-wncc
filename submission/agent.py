"""Catan agent with a learned win-probability evaluator and budgeted within-turn search.

  * V(s) = P(win | features(s)), a small MLP fitted on self-play outcomes (model.npz).
  * Every legal action is scored by V of its successor (batched forward pass). Chance actions are
    scored in expectation: dice rolls over the exact 2d6 distribution, dev-card purchases over the
    unseen pile, and robber steals over the victim's hand. Ending the turn is evaluated as a real
    successor with the opponent to move.
  * While CPU budget remains, the best candidates are expanded through our own follow-up
    decisions in the same turn (trade -> build, knight -> robber, road building -> roads).
  * The per-decision budget is a fraction of the CPU left on the clock, so spending stabilises.
"""
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import catanfeat as FT  # noqa: E402

from catanatron.apply_action import apply_action  # noqa: E402
from catanatron.models.actions import generate_playable_actions  # noqa: E402
from catanatron.models.board import Board  # noqa: E402
from catanatron.models.enums import (  # noqa: E402
    Action,
    ActionPrompt,
    ActionRecord,
    ActionType,
    RESOURCES,
)
from catanatron.state import State  # noqa: E402

DEVS = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]
WIN = 10.0
AT = ActionType

# Exact 2d6 sum distribution.  ``apply_roll`` consumes the forced dice from
# ActionRecord.result, so these are real state transitions rather than a
# no-op approximation of rolling.
ROLL_OUTCOMES = (
    (1, (1, 1)),
    (2, (1, 2)),
    (3, (1, 3)),
    (4, (1, 4)),
    (5, (1, 5)),
    (6, (1, 6)),
    (5, (2, 6)),
    (4, (3, 6)),
    (3, (4, 6)),
    (2, (5, 6)),
    (1, (6, 6)),
)


def fast_copy(s):
    c = State.__new__(State)
    c.players = s.players
    c.discard_limit = s.discard_limit
    c.friendly_robber = s.friendly_robber
    b0 = s.board
    b = Board.__new__(Board)
    b.map = b0.map
    b.buildings = b0.buildings.copy()
    b.roads = b0.roads.copy()
    cc = defaultdict(list)
    for k, comps in b0.connected_components.items():
        cc[k] = [set(x) for x in comps]
    b.connected_components = cc
    b.board_buildable_ids = b0.board_buildable_ids.copy()
    b.road_lengths = b0.road_lengths.copy()
    b.road_color = b0.road_color
    b.road_length = b0.road_length
    b.robber_coordinate = b0.robber_coordinate
    b.buildable_subgraph = b0.buildable_subgraph
    b.buildable_edges_cache = {}
    b.player_port_resources_cache = {}
    c.board = b
    c.player_state = s.player_state.copy()
    c.color_to_index = s.color_to_index
    c.colors = s.colors
    c.resource_freqdeck = s.resource_freqdeck.copy()
    c.development_listdeck = s.development_listdeck.copy()
    bbc = {}
    for col, d in s.buildings_by_color.items():
        nd = defaultdict(list)
        for t, v in d.items():
            nd[t] = list(v)
        bbc[col] = nd
    c.buildings_by_color = bbc
    c.action_records = s.action_records.copy()
    c.num_turns = s.num_turns
    c.current_player_index = s.current_player_index
    c.current_turn_index = s.current_turn_index
    c.current_prompt = s.current_prompt
    c.is_initial_build_phase = s.is_initial_build_phase
    c.is_discarding = s.is_discarding
    c.discard_counts = s.discard_counts.copy()
    c.is_moving_knight = s.is_moving_knight
    c.is_road_building = s.is_road_building
    c.free_roads_available = s.free_roads_available
    c.is_resolving_trade = s.is_resolving_trade
    c.current_trade = s.current_trade
    c.acceptees = s.acceptees
    return c


class Model:
    def __init__(self, path):
        d = np.load(path)
        self.mu = d['mu']; self.sd = d['sd']
        n = int(d['n_layers'])
        self.layers = [(d[f'W{k}'], d[f'b{k}']) for k in range(n)]
        # fold normalisation into the first layer
        W0, b0 = self.layers[0]
        W0n = W0 / self.sd[:, None]
        b0n = b0 - (self.mu / self.sd) @ W0
        self.layers[0] = (W0n, b0n)

    def logits(self, X):
        h = np.asarray(X, dtype=np.float64)
        L = len(self.layers)
        for k, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if k < L - 1:
                np.maximum(h, 0, out=h)
        return h[:, 0]


class Agent:
    def __init__(self):
        cfg = {}
        cp = os.path.join(_HERE, "config.json")
        if os.path.exists(cp):
            with open(cp) as f:
                cfg = json.load(f)
        self.cfg = cfg
        self.model = Model(os.path.join(_HERE, cfg.get("model", "model.npz")))
        self.depth = cfg.get("depth", 2)
        self.beam = cfg.get("beam", 3)
        self.budget_frac = cfg.get("budget_frac", 0.3)
        self.reserve = cfg.get("reserve", 0.08)
        self.rng = random.Random(4321)
        self.nfeat = len(self.model.mu)
        # warm up numpy / feature code outside the clock
        self.model.logits(np.zeros((3, self.nfeat)))

    # ---------------------------------------------------------------- evaluation
    def feats(self, s, me):
        # This bit is part of the trained 71-feature contract.  The previous
        # candidate hard-coded zero, making every post-END_TURN and
        # same-turn successor look like an identical phase to the MLP.
        me_to_move = int(s.colors[s.current_player_index] == me)
        return FT.features(s, me, me_to_move)

    def is_win(self, s, me):
        return s.player_state[f"P{s.color_to_index[me]}_ACTUAL_VICTORY_POINTS"] >= 10

    def eval_many(self, states, me):
        """Logit of win probability for each state; wins get WIN."""
        rows, idx, out = [], [], [0.0] * len(states)
        for k, s in enumerate(states):
            if self.is_win(s, me):
                out[k] = WIN + 1
            else:
                rows.append(self.feats(s, me)); idx.append(k)
        if rows:
            v = self.model.logits(rows)
            for k, x in zip(idx, v):
                out[k] = float(x)
        return out

    # ---------------------------------------------------------------- successors
    def expand(self, s, a, me):
        """List of (weight, state) outcomes of action a from state s (chance handled explicitly)."""
        at = a.action_type
        if at == AT.ROLL:
            outs = []
            for weight, dice in ROLL_OUTCOMES:
                o = fast_copy(s)
                apply_action(o, a, ActionRecord(a, dice))
                outs.append((weight / 36.0, o))
            return outs
        if at == AT.MOVE_ROBBER and a.value[1] is not None:
            base = fast_copy(s)
            apply_action(base, Action(a.color, at, (a.value[0], None)))
            victim = a.value[1]
            vi = s.color_to_index[victim]; mi_ = s.color_to_index[me]
            hand = [(r, s.player_state[f"P{vi}_{r}_IN_HAND"]) for r in RESOURCES]
            tot = sum(c for _, c in hand)
            if tot == 0:
                return [(1.0, base)]
            outs = []
            for r, c in hand:
                if c == 0:
                    continue
                o = fast_copy(base) if len(outs) < 4 else fast_copy(base)
                o.player_state[f"P{vi}_{r}_IN_HAND"] -= 1
                o.player_state[f"P{mi_}_{r}_IN_HAND"] += 1
                outs.append((c / tot, o))
            return outs
        if at == AT.BUY_DEVELOPMENT_CARD:
            base = fast_copy(s)
            apply_action(base, a)
            i = s.color_to_index[me]
            ps0 = s.player_state
            # the card drawn in this sampled world
            drawn = None
            for d in DEVS:
                if base.player_state[f"P{i}_{d}_IN_HAND"] > ps0[f"P{i}_{d}_IN_HAND"]:
                    drawn = d
            # unseen pile = remaining deck + opponent's (sampled) hand
            j = 1 - i
            pile = list(s.development_listdeck)
            for d in DEVS:
                pile += [d] * ps0[f"P{j}_{d}_IN_HAND"]
            outs = []
            n = len(pile)
            for d in DEVS:
                c = pile.count(d)
                if c == 0:
                    continue
                o = fast_copy(base) if d != drawn else base
                if d != drawn:
                    ps = o.player_state
                    ps[f"P{i}_{drawn}_IN_HAND"] -= 1
                    ps[f"P{i}_{d}_IN_HAND"] += 1
                    if drawn == "VICTORY_POINT":
                        ps[f"P{i}_ACTUAL_VICTORY_POINTS"] -= 1
                    if d == "VICTORY_POINT":
                        ps[f"P{i}_ACTUAL_VICTORY_POINTS"] += 1
                outs.append((c / n, o))
            return outs
        o = fast_copy(s)
        apply_action(o, a)
        return [(1.0, o)]

    def turn_continues(self, s, me):
        if s.colors[s.current_player_index] != me:
            return False
        p = s.current_prompt
        if p == ActionPrompt.PLAY_TURN:
            return not s.is_initial_build_phase
        return p in (ActionPrompt.MOVE_ROBBER, ActionPrompt.DISCARD)

    def score_actions(self, s, acts, me, cur_val):
        """One-ply value of each action. Returns list of (value, deterministic_child_or_None)."""
        groups, states = [], []
        for a in acts:
            at = a.action_type
            try:
                outs = self.expand(s, a, me)
            except Exception:
                groups.append(False)
                continue
            groups.append((len(states), outs))
            states.extend(o for _, o in outs)
        vals = self.eval_many(states, me) if states else []
        res = []
        for a, g in zip(acts, groups):
            if g is None:
                res.append((cur_val, None))
            elif g is False:
                res.append((-1e9, None))
            else:
                k0, outs = g
                if len(outs) == 1:
                    res.append((vals[k0], outs[0][1]))
                else:
                    # expectation in probability space
                    pv = 0.0
                    for t, (w, _) in enumerate(outs):
                        x = vals[k0 + t]
                        pv += w * (1.0 if x > WIN else 1.0 / (1.0 + np.exp(-x)))
                    pv = min(max(pv, 1e-9), 1 - 1e-9)
                    res.append((float(np.log(pv / (1 - pv))), None))
        return res

    def search(self, s, me, depth, deadline):
        cur = self.eval_many([s], me)[0]
        if cur > WIN or depth <= 0 or not self.turn_continues(s, me):
            return cur
        if time.process_time() > deadline:
            return cur
        acts = generate_playable_actions(s)
        if not acts:
            return cur
        sc = self.score_actions(s, acts, me, cur)
        # Road Building leaves current_prompt at PLAY_TURN while the engine
        # requires another road action.  The search must not treat the current
        # state as an optional stopping point in that forced sub-phase.
        must_choose = (
            s.current_prompt != ActionPrompt.PLAY_TURN or s.is_road_building
        )
        best = max(v for v, _ in sc)
        if not must_choose:
            best = max(best, cur)
        if depth > 1:
            cand = sorted([x for x in sc if x[1] is not None], key=lambda x: -x[0])[:self.beam]
            for v, c in cand:
                if time.process_time() > deadline:
                    break
                vv = self.search(c, me, depth - 1, deadline)
                if vv > best:
                    best = vv
        return best

    # ---------------------------------------------------------------- decide
    def decide(self, obs, playable_actions):
        n = len(playable_actions)
        if n == 1:
            return 0
        t0 = time.process_time()
        left = obs.cpu_left if obs.cpu_left is not None else 1.0
        if left < 0.025:
            return self.cheap(playable_actions)
        budget = max(0.0, (left - self.reserve) * self.budget_frac)
        deadline = t0 + budget
        me = obs.me
        game = obs.sample_world(self.rng)
        s = game.state
        if s.is_initial_build_phase:
            return self.initial(s, playable_actions, me)
        cur = self.eval_many([s], me)[0]
        sc = self.score_actions(s, playable_actions, me, cur)
        vals = [v for v, _ in sc]
        if self.depth > 1:
            order = sorted(range(n), key=lambda k: -vals[k])
            for k in order[:self.beam + 1]:
                if time.process_time() > deadline:
                    break
                v, c = sc[k]
                if c is None or v > WIN:
                    continue
                vv = self.search(c, me, self.depth - 1, deadline)
                if vv > vals[k]:
                    vals[k] = vv
        return max(range(n), key=lambda k: (vals[k], -k))

    def initial(self, s, acts, me):
        """Opening placement: F's hand-tuned value function, one ply (the model never saw
        half-built openings)."""
        from catanatron.players.value import base_fn, DEFAULT_WEIGHTS
        from catanatron.game import Game
        if not hasattr(self, "_fv"):
            self._fv = base_fn(DEFAULT_WEIGHTS)
        best, bv = 0, None
        for k, a in enumerate(acts):
            o = fast_copy(s)
            apply_action(o, a)
            g = Game.__new__(Game)
            g.state = o
            v = self._fv(g, me)
            if bv is None or v > bv:
                best, bv = k, v
        return best

    def cheap(self, actions):
        pri = {AT.BUILD_CITY: 0, AT.BUILD_SETTLEMENT: 1, AT.BUY_DEVELOPMENT_CARD: 3, AT.ROLL: 4,
               AT.END_TURN: 5, AT.BUILD_ROAD: 6}
        return min(range(len(actions)), key=lambda i: (pri.get(actions[i].action_type, 7), i))
