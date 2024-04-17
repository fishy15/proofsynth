import random

from abc import ABC, abstractmethod
from typing import List

from prop.lang import Expr
from prop.tactics import (
    SingleTactic,
    DoubleTactic,
)


class Heuristic(ABC):
    @abstractmethod
    def pick_tactic_single(self, hypotheses: List[Expr], goal: Expr) -> SingleTactic:
        pass

    @abstractmethod
    def pick_tactic_double(self, hypotheses: List[Expr], goal: Expr) -> DoubleTactic:
        pass
