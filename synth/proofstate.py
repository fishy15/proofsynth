from typing import List, Self

from prop.lang import *
from prop.prop_parser import parse_prop


@dataclass(frozen=True, slots=True, eq=True)
class ProofState:
    hypotheses: List[Expr]
    goal: Expr

    @classmethod
    def new(cls, hypotheses: List[str], goal: str) -> Self:
        hypothesis_exprs = [parse_prop(e) for e in hypotheses]
        return cls(hypothesis_exprs, parse_prop(goal))

    @classmethod
    def new_goal(cls, expr: str) -> Self:
        return cls([], parse_prop(expr))
