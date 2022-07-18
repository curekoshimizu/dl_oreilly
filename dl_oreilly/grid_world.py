from __future__ import annotations

import dataclasses
import enum
import math
from collections import defaultdict
from typing import Any, DefaultDict, Iterable

import numpy as np

from . import NDFloatArray


class Action(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


@dataclasses.dataclass(
    order=True,
    eq=True,
    frozen=True,
)
class State:
    x: int
    y: int

    def move(self, action: Action) -> State:
        match action:
            case Action.UP:
                return State(self.x - 1, self.y)
            case Action.DOWN:
                return State(self.x + 1, self.y)
            case Action.LEFT:
                return State(self.x, self.y - 1)
            case Action.RIGHT:
                return State(self.x, self.y + 1)


class Values:
    def __init__(self) -> None:
        self._values: DefaultDict[State, float] = defaultdict(float)

    def copy(self) -> Values:
        new = Values()
        for state in self.keys():
            new.set(state, self.get(state))
        return new

    def keys(self) -> Iterable[State]:
        yield from self._values.keys()

    def set(self, state: State, value: float) -> None:
        self._values[state] = value

    def get(self, state: State) -> float:
        return self._values[state]

    def dump(self) -> None:
        print("== values =======")
        x = 0
        y = 0
        for key, value in sorted(self._values.items()):
            if x != key.x:
                print("")
                x = key.x
                y = 0
            while y < key.y:
                print("None", end="\t")
                y += 1
            y += 1
            print(f"{value:.2f}", end="\t")
        print("")


class GridWorld:
    def __init__(self) -> None:
        self._action_space = [
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
        ]
        self._reward_map: NDFloatArray = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, -math.inf, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.goal_state = State(0, 3)
        self._wall_state = State(1, 1)
        self._start_state = State(2, 0)
        self._agent_state = self._start_state

    @property
    def height(self) -> int:
        return len(self._reward_map)

    @property
    def width(self) -> int:
        return len(self._reward_map[0])

    @property
    def shape(self) -> tuple[int, int]:
        x, y = self._reward_map.shape
        return x, y

    def actions(self) -> list[Action]:
        return self._action_space

    def states(self) -> Iterable[State]:
        for x in range(self.height):
            for y in range(self.width):
                ret = State(x, y)
                if ret == self._wall_state:
                    continue
                yield ret

    def next_state(self, state: State, action: Action) -> State:
        new_state = state.move(action)
        if not ((0 <= new_state.x < self.height) and (0 <= new_state.y < self.width)):
            return state
        elif new_state == self._wall_state:
            return state
        else:
            return new_state

    def reward(self, state: State, action: Action, next_state: State) -> float:
        ret = self._reward_map[dataclasses.astuple(next_state)]
        assert isinstance(ret, float)
        return ret


class ActionProbs:
    """
    pi(a|s) を返すためのクラス
    """

    def __init__(self) -> None:
        init: list[tuple[Action, float]] = [
            (Action.UP, 0.25),
            (Action.DOWN, 0.25),
            (Action.LEFT, 0.25),
            (Action.RIGHT, 0.25),
        ]
        self._pi: DefaultDict[State, list[tuple[Action, float]]] = defaultdict(lambda: init)

    def set_pi(self, state: State, action_prob: list[tuple[Action, float]]) -> None:
        self._pi[state] = action_prob

    def pi(self, state: State) -> list[tuple[Action, float]]:
        """
        (a: Action, pi(a|s)) を返す
        """
        return self._pi[state]

    def keys(self) -> Iterable[State]:
        return self._pi.keys()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ActionProbs):
            return False

        if list(self.keys()) != list(other.keys()):
            return False
        for key in self.keys():
            if self.pi(key) != other.pi(key):
                return False
        return True

    def dump(self) -> None:
        print("== values =======")
        x = 0
        y = 0
        for key, action_probs in sorted(self._pi.items()):
            if x != key.x:
                print("")
                x = key.x
                y = 0
            while y < key.y:
                print("None", end="\t")
                y += 1
            y += 1
            max_action, _ = sorted(action_probs, key=lambda x: -x[1])[0]
            print(f"{str(max_action)[7:]}", end="\t")
        print("")


class DPMethod:
    def policy_iter(self, env: GridWorld, gamma: float = 0.9) -> ActionProbs:
        action_probs = ActionProbs()
        values = Values()

        while True:
            values = self.policy_eval(action_probs, values, env, gamma)
            new_action_probs = self.greedy_policy(values, env, gamma)
            if new_action_probs == action_probs:
                break
            action_probs = new_action_probs
        return action_probs

    def greedy_policy(self, values: Values, env: GridWorld, gamma: float = 0.9) -> ActionProbs:
        action_probs = ActionProbs()
        for state in env.states():
            action_values = []
            for action in env.actions():
                next_state = env.next_state(state, action)
                r = env.reward(state, action, next_state)
                value = r + gamma * values.get(next_state)
                action_values.append((action, value))
            max_action, _ = sorted(action_values, key=lambda x: x[1])[3]

            action_prob = [
                (Action.UP, 0.0),
                (Action.DOWN, 0.0),
                (Action.LEFT, 0.0),
                (Action.RIGHT, 0.0),
            ]
            action_prob[max_action.value] = (max_action, 1.0)
            action_probs.set_pi(state, action_prob)
        return action_probs

    def policy_eval(
        self,
        action_probs: ActionProbs,
        values: Values,
        env: GridWorld,
        gamma: float = 0.9,
        threshold: float = 0.001,
        n_iter: int = 10000,
    ) -> Values:
        for _ in range(n_iter):
            old_values = values.copy()
            new_value = self.eval_onestep(action_probs, values, env, gamma)

            delta = 0.0
            for state in values.keys():
                t = abs(old_values.get(state) - new_value.get(state))
                if delta < t:
                    delta = t
            if delta < threshold:
                break
        return new_value

    def eval_onestep(
        self,
        action_probs: ActionProbs,
        values: Values,
        env: GridWorld,
        gamma: float = 0.9,
    ) -> Values:
        for state in env.states():
            if state == env.goal_state:
                values.set(state, 0.0)
                continue

            new_value = 0.0
            for (action, action_prob) in action_probs.pi(state):
                next_state = env.next_state(state, action)
                r = env.reward(state, action, next_state)
                new_value += action_prob * (r + gamma * values.get(next_state))
            values.set(state, new_value)
        return values
