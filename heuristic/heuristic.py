import random

from abc import ABC, abstractmethod
from typing import List

from prop.lang import Expr
from prop.tactics import (
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
