import random

from typing import List

from heuristic.heuristic import Heuristic
from prop.lang import Expr
from prop.tactics import (
    bottom_up_tactics_single,
    bottom_up_tactics_double,
    SingleTactic,
    DoubleTactic,
)


class Naive(Heuristic):
    @staticmethod
    def pick_tactic_single(hypotheses: List[Expr], goal: Expr) -> SingleTactic:
        return random.choice(bottom_up_tactics_single)

    @staticmethod
    def pick_tactic_double(hypotheses: List[Expr], goal: Expr) -> DoubleTactic:
        return random.choice(bottom_up_tactics_double)
