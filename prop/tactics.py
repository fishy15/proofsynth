from abc import ABC, abstractmethod
from typing import Self

from prop.lang import *
from prop.prop_parser import parse_prop

class Tactic(ABC):
    @abstractmethod
    def eval(self) -> Expr:
        pass

class EvalException(Exception):
    pass

@dataclass(frozen=True, slots=True, eq=True)
class THypothesis(Tactic):
    term: Expr

    def eval(self) -> Expr:
        return self.term

    @staticmethod
    def new(expr: str) -> Self:
        return THypothesis(parse_prop(expr))

@dataclass(frozen=True, slots=True, eq=True)
class TModusPonens(Tactic):
    imply: Tactic
    result: Tactic

    def eval(self) -> Expr:
        imply_expr = self.imply.eval()
        result_expr = self.result.eval()

        match imply_expr:
            case EImplies(left, right):
                if left == result_expr:
                    return right
                raise EvalException("[TModusPonens] Left side of hypothesis does not match input")
            case _:
                raise EvalException("[TModusPonens] Left side is not an implication")

