import random

from abc import ABC, abstractmethod
from typing import List

from prop.lang import Expr
from prop.tactics import (
    bottom_up_tactics_single,
    bottom_up_tactics_double,
    SingleTactic,
    DoubleTactic,
)


class Heuristic(ABC):
    @staticmethod
    @abstractmethod
    def pick_tactic_single(hypotheses: List[Expr], goal: Expr) -> SingleTactic:
        pass

    @staticmethod
    @abstractmethod
    def pick_tactic_double(hypotheses: List[Expr], goal: Expr) -> DoubleTactic:
        pass


class Naive(Heuristic):
    @staticmethod
    def pick_tactic_single(hypotheses: List[Expr], goal: Expr) -> SingleTactic:
        return random.choice(bottom_up_tactics_single)

    @staticmethod
    def pick_tactic_double(hypotheses: List[Expr], goal: Expr) -> DoubleTactic:
        return random.choice(bottom_up_tactics_double)
