"""v3 candidate: contender-weight chance-aware one-ply search.

The official harness calls ``Agent.decide(obs, playable_actions)`` and expects
an integer index into the supplied action list.  This candidate keeps the v0
safety wrapper and v1 chance handling, but replaces F's default value weights
with catanatron's pinned public contender weights.

Rolls use the two-dice distribution, development-card buys use the sampled
deck composition, and robber steals use the victim's visible resource
frequencies.  The sampled world is still only one hidden development-card
assignment; more worlds belong to a later milestone.  The v2 own-turn beam
remains in the file for comparison but is not active after its 400-game result
failed to improve v1.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter

from catanatron.models.enums import (
    ActionRecord,
    ActionType,
    DEVELOPMENT_CARDS,
    RESOURCES,
)
from catanatron.players.value import CONTENDER_WEIGHTS, get_value_fn
from catanatron.state_functions import player_num_resource_cards


# Build the closure once at import time.  F uses DEFAULT_WEIGHTS; this v3
# candidate uses the pinned engine's public contender weights as the one
# controlled policy change.  Constructing the closure inside decide would
# charge setup work to every game.
VALUE_FN = get_value_fn("contender_fn", CONTENDER_WEIGHTS)


# One representative ordered dice pair per sum is sufficient for the current
# value function: after a roll, the game state depends on the sum, not on the
# ordering of the two dice.  The weights are the exact 2d6 frequencies.
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


class Agent:
    """Fault-tolerant contender-weight chance-aware search."""

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
        ActionType.MOVE_ROBBER: 11,
        ActionType.DISCARD_RESOURCE: 12,
    }

    # Chance expansion is exact for the state information available to the
    # agent, but it can multiply the number of successor evaluations.  These
    # limits keep v1 conservative on robber/card branches and let the clock
    # deadline stop a larger root search cleanly.
    MAX_DEV_BRANCHES = len(DEVELOPMENT_CARDS)
    MAX_ROBBER_BRANCHES = len(RESOURCES)

    # v2 uses a shallow beam: first score root actions, then look through one
    # best continuation while the same decision's CPU deadline remains.  The
    # beam is intentionally small; a complete tree would spend the allowance
    # on one turn and leave later decisions random.
    V2_ROOT_FRACTION = 0.60
    V2_BEAM_WIDTH = 4
    V2_CONTINUATION_WIDTH = 6

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

    def _score_actions_v0(self, obs, actions):
        """Run the v0 F-style one-ply search and return a legal index."""
        remaining = self._remaining_cpu(obs)
        deadline = None
        start_cpu = time.process_time()
        if remaining is not None:
            usable = remaining - self.CPU_RESERVE_S
            if usable <= 0.0:
                return self._cheap_index(actions)
            deadline = start_cpu + usable

        world = obs.sample_world(self.rng)

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

    @classmethod
    def _outcome_branches(cls, world, action):
        """Return ``(weight, ActionRecord|None)`` chance branches for an action.

        ``Game.execute`` only accepts forced stochastic outcomes through an
        ActionRecord.  Supplying a value in Action.value is not enough for
        rolls, development draws, or robber steals in the pinned engine.
        """
        action_type = getattr(action, "action_type", None)

        if action_type == ActionType.ROLL:
            return [
                (
                    weight,
                    ActionRecord(action=action, result=dices),
                )
                for weight, dices in ROLL_OUTCOMES
            ]

        if action_type == ActionType.BUY_DEVELOPMENT_CARD:
            counts = Counter(world.state.development_listdeck)
            branches = []
            for card in DEVELOPMENT_CARDS[: cls.MAX_DEV_BRANCHES]:
                count = counts.get(card, 0)
                if count:
                    branches.append(
                        (
                            count,
                            ActionRecord(action=action, result=card),
                        )
                    )
            return branches

        if action_type == ActionType.MOVE_ROBBER:
            _, victim = action.value
            if victim is None:
                return [(1, None)]

            branches = []
            for resource in RESOURCES[: cls.MAX_ROBBER_BRANCHES]:
                count = player_num_resource_cards(world.state, victim, resource)
                if count:
                    branches.append(
                        (
                            count,
                            ActionRecord(action=action, result=resource),
                        )
                    )
            return branches

        # Builds, trades, discards, and development-card plays have a
        # deterministic immediate transition.  A later forced prompt (for
        # example the robber location after a knight) belongs to v2 search.
        return [(1, None)]

    def _execute_branch(self, world, action, action_record):
        """Execute one forced outcome on an isolated copy."""
        successor = world.copy()
        random_state = random.getstate()
        try:
            if action_record is None:
                successor.execute(action)
            else:
                successor.execute(action, action_record=action_record)
            return successor
        finally:
            # The branch is fully forced, but restoring the agent process's
            # global PRNG also protects against future engine code that draws
            # randomness while evaluating a successor.
            random.setstate(random_state)

    def _branch_value(self, world, action, action_record, obs):
        """Execute one forced outcome on an isolated copy and score it."""
        successor = self._execute_branch(world, action, action_record)
        return float(self.value_fn(successor, obs.me))

    @classmethod
    def _continuation_candidates(cls, actions):
        """Keep a deterministic, diverse subset of a continuation action list."""
        actions = list(actions)
        if len(actions) <= cls.V2_CONTINUATION_WIDTH:
            return actions

        ranked = sorted(
            enumerate(actions),
            key=lambda item: (
                cls.ACTION_PRIORITY.get(
                    getattr(item[1], "action_type", None), 50
                ),
                item[0],
            ),
        )
        selected_indices = {index for index, _ in ranked[: cls.V2_CONTINUATION_WIDTH]}

        # Always retain END_TURN if it exists.  Otherwise the continuation
        # search can spend all its width on builds and never compare passing.
        end_turn_index = next(
            (
                index
                for index, action in enumerate(actions)
                if getattr(action, "action_type", None) == ActionType.END_TURN
            ),
            None,
        )
        if end_turn_index is not None and end_turn_index not in selected_indices:
            selected_indices.pop()
            selected_indices.add(end_turn_index)

        return [action for index, action in enumerate(actions) if index in selected_indices]

    def _evaluate_action_outcomes(
        self, world, action, obs, deadline, collect_successors=False
    ):
        """Return an expected value and optionally its successful successors."""
        branches = self._outcome_branches(world, action)
        if not branches:
            return None, [], False

        weighted_value = 0.0
        successful_weight = 0
        successors = []
        for weight, action_record in branches:
            if deadline is not None and time.process_time() >= deadline:
                return None, [], False
            try:
                successor = self._execute_branch(world, action, action_record)
                value = float(self.value_fn(successor, obs.me))
            except Exception:
                continue
            if not math.isfinite(value):
                continue
            weighted_value += weight * value
            successful_weight += weight
            if collect_successors:
                successors.append((weight, successor, value))

        if successful_weight == 0:
            return None, [], True
        return weighted_value / successful_weight, successors, True

    def _best_continuation_value(self, successor, obs, deadline):
        """Score the best shallow continuation while it is still our turn."""
        if deadline is not None and time.process_time() >= deadline:
            return None, False

        try:
            current_color = successor.state.current_color()
        except Exception:
            return float(self.value_fn(successor, obs.me)), True

        if current_color != obs.me:
            return float(self.value_fn(successor, obs.me)), True

        actions = self._continuation_candidates(successor.playable_actions)
        if not actions:
            return float(self.value_fn(successor, obs.me)), True

        best_value = None
        for action in actions:
            if deadline is not None and time.process_time() >= deadline:
                return None, False
            value, _, complete = self._evaluate_action_outcomes(
                successor, action, obs, deadline, collect_successors=False
            )
            if not complete:
                return None, False
            if value is not None and (best_value is None or value > best_value):
                best_value = value

        if best_value is None:
            return float(self.value_fn(successor, obs.me)), True
        return best_value, True

    def _score_actions_v2(self, obs, actions):
        """Use a shallow beam to look through one continuation of our turn."""
        remaining = self._remaining_cpu(obs)
        start_cpu = time.process_time()
        deadline = None
        root_deadline = None
        if remaining is not None:
            usable = remaining - self.CPU_RESERVE_S
            if usable <= 0.0:
                return self._cheap_index(actions)
            deadline = start_cpu + usable
            root_deadline = start_cpu + usable * self.V2_ROOT_FRACTION

        world = obs.sample_world(self.rng)
        root_results = []
        root_budget_exhausted = False
        for index, action in enumerate(actions):
            if root_deadline is not None and time.process_time() >= root_deadline:
                root_budget_exhausted = True
                break
            value, successors, complete = self._evaluate_action_outcomes(
                world, action, obs, root_deadline, collect_successors=True
            )
            if not complete:
                root_budget_exhausted = True
                break
            if value is not None:
                root_results.append((value, index, successors))

        if not root_results:
            if root_budget_exhausted:
                # v1/v0 would see the original cpu_left snapshot and could
                # start a second full search after this partial v2 pass.
                return self._cheap_index(actions)
            return self._score_actions_v1(obs, actions)

        # The immediate best is always a valid answer if the continuation
        # budget disappears.  This also gives v2 the v1-quality fallback.
        root_results.sort(key=lambda result: (-result[0], result[1]))
        best_root_value, best_root_index, _ = root_results[0]
        best_final_value = None
        best_final_index = best_root_index

        for _, index, branch_successors in root_results[: self.V2_BEAM_WIDTH]:
            if deadline is not None and time.process_time() >= deadline:
                break
            weighted_value = 0.0
            successful_weight = 0
            complete_candidate = True
            for weight, successor, immediate_value in branch_successors:
                continuation_value, complete = self._best_continuation_value(
                    successor, obs, deadline
                )
                if not complete:
                    complete_candidate = False
                    break
                if continuation_value is None:
                    continuation_value = immediate_value
                weighted_value += weight * continuation_value
                successful_weight += weight

            if not complete_candidate or successful_weight == 0:
                break
            final_value = weighted_value / successful_weight
            if best_final_value is None or final_value > best_final_value:
                best_final_value = final_value
                best_final_index = index

        return best_final_index if best_final_value is not None else best_root_index

    def _score_actions_v1(self, obs, actions):
        """Score legal actions by one-sample-world expected successor value."""
        remaining = self._remaining_cpu(obs)
        start_cpu = time.process_time()
        deadline = None
        if remaining is not None:
            usable = remaining - self.CPU_RESERVE_S
            if usable <= 0.0:
                return self._cheap_index(actions)
            deadline = start_cpu + usable

        # Reuse one hidden-information world for all root actions so the
        # comparison does not mix different opponent-card assignments.
        world = obs.sample_world(self.rng)
        best_index = None
        best_value = None
        budget_exhausted = False

        for index, action in enumerate(actions):
            if deadline is not None and time.process_time() >= deadline:
                budget_exhausted = True
                break

            branches = self._outcome_branches(world, action)
            if not branches:
                continue

            weighted_value = 0.0
            successful_weight = 0
            budget_hit = False
            for weight, action_record in branches:
                if deadline is not None and time.process_time() >= deadline:
                    budget_hit = True
                    break
                try:
                    value = self._branch_value(world, action, action_record, obs)
                except Exception:
                    continue
                if math.isfinite(value):
                    weighted_value += weight * value
                    successful_weight += weight

            if budget_hit:
                budget_exhausted = True
                break
            if successful_weight == 0:
                continue

            expected_value = weighted_value / successful_weight
            if best_value is None or expected_value > best_value:
                best_index = index
                best_value = expected_value

        if best_index is not None:
            return best_index

        if budget_exhausted:
            # Do not call v0 after consuming v1's deadline: obs.cpu_left is a
            # snapshot from the start of this decision, so a second full
            # search could overspend the cumulative game allowance.
            return self._cheap_index(actions)

        # Keep the original baseline available if v1 cannot complete a root
        # branch.  Its own deadline and safety checks prevent a second failure
        # from becoming an illegal response.
        return self._score_actions_v0(obs, actions)

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

            # v2's own-turn beam is retained for comparison, but the active
            # emergency candidate is the faster v1 chance search with the
            # contender weights selected above.
            index = self._score_actions_v1(obs, actions)
            if type(index) is int and 0 <= index < len(actions):
                return index
        except Exception:
            # Never let a branch, feature, or masked-world error become an
            # illegal response.  The harness's action list is already legal.
            pass

        return self._cheap_index(actions)
