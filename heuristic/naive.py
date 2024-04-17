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
    def pick_tactic_single(self, hypotheses: List[Expr], goal: Expr) -> SingleTactic:
        return random.choice(bottom_up_tactics_single)

    def pick_tactic_double(self, hypotheses: List[Expr], goal: Expr) -> DoubleTactic:
        return random.choice(bottom_up_tactics_double)
