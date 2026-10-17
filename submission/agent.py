"""v0 Catan agent: a defensive, one-ply value-function baseline.

The official harness calls ``Agent.decide(obs, playable_actions)`` and expects
an integer index into the supplied action list.  This file deliberately keeps
the first version small and boring: sample one legal world, execute every
candidate on a copy, and choose the successor with catanatron's supplied
13-feature value function.

The safety wrapper is part of the baseline, not a substitute for measuring the
agent.  It guarantees that an engine/search exception cannot turn into an
illegal response, and it stops spending CPU on a full search when the
per-game allowance is nearly exhausted.
"""

from __future__ import annotations

import math
import random
import time

from catanatron.models.enums import ActionType
from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn


# Build the closure once at import time.  The official F agent does the same;
# constructing it inside decide would charge setup work to every game.
VALUE_FN = get_value_fn("base_fn", DEFAULT_WEIGHTS)


class Agent:
    """Fault-tolerant one-ply value search over the current legal actions."""

    # Leave room for the harness and for later decisions in the same game.
    CPU_RESERVE_S = 0.010
    CHEAP_FALLBACK_S = 0.045

    # This is only used when the clock is nearly empty or a search branch
    # fails.  The action list is already legal; this ordering is a cheap
    # policy, not an additional legality check.
    ACTION_PRIORITY = {
        ActionType.BUILD_CITY: 0,
        ActionType.BUILD_SETTLEMENT: 1,
        ActionType.PLAY_KNIGHT_CARD: 2,
        ActionType.BUILD_ROAD: 3,
        ActionType.BUY_DEVELOPMENT_CARD: 4,
        ActionType.PLAY_ROAD_BUILDING: 5,
        ActionType.PLAY_YEAR_OF_PLENTY: 6,
        ActionType.PLAY_MONOPOLY: 7,
        ActionType.MARITIME_TRADE: 8,
        ActionType.ROLL: 9,
        ActionType.END_TURN: 10,
    }

    def __init__(self):
        # The harness does not expose a game seed or a reset callback.  Keep a
        # private RNG for masked-world sampling and detect game boundaries from
        # public observations in _observe_game_boundary.
        self.rng = random.Random(0)
        self.value_fn = VALUE_FN

        self._last_me = None
        self._last_turn = None
        self._last_buildings = None
        self._last_roads = None
        self.game_number = 0
        self.decisions = 0

    @staticmethod
    def _board_counts(obs):
        """Return public building/road counts without touching hidden state."""
        board = getattr(obs, "board", None)
        buildings = getattr(board, "buildings", None)
        roads = getattr(board, "roads", None)

        building_count = len(buildings) if buildings is not None else None
        if roads is None:
            road_count = None
        else:
            # Board.roads stores both edge orientations for convenience.
            road_count = len(roads) // 2
        return building_count, road_count

    def _observe_game_boundary(self, obs):
        """Reset transient per-game state when a new public game is visible."""
        buildings, roads = self._board_counts(obs)
        turn = getattr(obs, "turn", None)
        me = getattr(obs, "me", None)

        is_new = self._last_me is None
        if self._last_me is not None and me != self._last_me:
            is_new = True
        if (
            self._last_turn is not None
            and turn is not None
            and turn < self._last_turn
        ):
            is_new = True
        if (
            self._last_buildings is not None
            and buildings is not None
            and buildings < self._last_buildings
        ):
            is_new = True
        if (
            self._last_roads is not None
            and roads is not None
            and roads < self._last_roads
        ):
            is_new = True

        if is_new:
            self.game_number += 1

        self._last_me = me
        self._last_turn = turn
        self._last_buildings = buildings
        self._last_roads = roads

    @classmethod
    def _cheap_index(cls, actions):
        """Choose a legal index without copying or executing the game."""
        best_index = 0
        best_key = (float("inf"), 0)
        for index, action in enumerate(actions):
            action_type = getattr(action, "action_type", None)
            key = (cls.ACTION_PRIORITY.get(action_type, 50), index)
            if key < best_key:
                best_key = key
                best_index = index
        return best_index

    @staticmethod
    def _remaining_cpu(obs):
        value = getattr(obs, "cpu_left", None)
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return max(0.0, value)

    def _score_actions(self, obs, actions):
        """Run F-style one-ply search and return a legal action index."""
        world = obs.sample_world(self.rng)
        remaining = self._remaining_cpu(obs)
        deadline = None
        if remaining is not None:
            usable = remaining - self.CPU_RESERVE_S
            if usable <= 0.0:
                return self._cheap_index(actions)
            deadline = time.process_time() + usable

        best_index = None
        best_value = None
        for index, action in enumerate(actions):
            if deadline is not None and time.process_time() >= deadline:
                break

            successor = world.copy()
            try:
                successor.execute(action)
                value = float(self.value_fn(successor, obs.me))
            except Exception:
                # The action list should be legal for the real game.  A
                # sampled world can still reject a branch if engine internals
                # disagree; skip it and keep the fallback path available.
                continue

            if not math.isfinite(value):
                continue
            if best_value is None or value > best_value:
                best_index = index
                best_value = value

        return self._cheap_index(actions) if best_index is None else best_index

    def decide(self, obs, playable_actions) -> int:
        """Return a legal index, even when search or observation handling fails."""
        actions = list(playable_actions)
        if not actions:
            # The official harness never calls decide with an empty list, but
            # returning an integer keeps this method total for local probes.
            return 0

        try:
            self._observe_game_boundary(obs)
            self.decisions += 1

            remaining = self._remaining_cpu(obs)
            if remaining is not None and remaining <= self.CHEAP_FALLBACK_S:
                return self._cheap_index(actions)

            index = self._score_actions(obs, actions)
            if type(index) is int and 0 <= index < len(actions):
                return index
        except Exception:
            # Never let a branch, feature, or masked-world error become an
            # illegal response.  The harness's action list is already legal.
            pass

        return self._cheap_index(actions)
