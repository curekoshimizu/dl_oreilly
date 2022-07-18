from __future__ import annotations

import dataclasses
import enum
import math
from collections import defaultdict
from typing import DefaultDict, Iterator

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
        return dataclasses.replace(self)

    def keys(self) -> Iterator[State]:
        yield from self._values.keys()

    def update(self, state: State, value: float) -> None:
        self._values[state] += value

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
            print(value, end="\t")


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

    def states(self) -> Iterator[State]:
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
        if state == next_state:
            return 0.0
        ret = self._reward_map[dataclasses.astuple(next_state)]
        assert isinstance(ret, float)
        return ret


class Actions:
    """
    pi(a|s) を返すためのクラス
    """

    def pi(self, state: State) -> list[tuple[Action, float]]:
        """
        (a: Action, pi(a|s)) を返す
        """
        return [(Action.UP, 0.25), (Action.DOWN, 0.25), (Action.LEFT, 0.25), (Action.RIGHT, 0.25)]


class DPMethod:
    def policy_eval(
        self, actions: Actions, values: Values, env: GridWorld, gamma: float = 0.9, threshold: float = 0.001
    ) -> Values:
        while True:
            old_values = values.copy()
            new_value = self.eval_onestep(actions, values, env, gamma)

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
        actions: Actions,
        values: Values,
        env: GridWorld,
        gamma: float = 0.9,
    ) -> Values:
        for state in env.states():
            if state == env.goal_state:
                values.update(state, 0.0)
                continue

            new_value = 0.0
            for (action, action_prob) in actions.pi(state):
                next_state = env.next_state(state, action)
                r = env.reward(state, action, next_state)
                new_value += action_prob * (r + gamma * values.get(next_state))
            values.update(state, new_value)
        return values
