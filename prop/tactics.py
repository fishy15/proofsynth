from abc import ABC, abstractmethod
from typing import Self

from prop.lang import *
from prop.prop_parser import parse_prop

"""
Implements tactics from https://en.wikipedia.org/wiki/Propositional_calculus#List_of_classically_valid_argument_forms.
"""


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

    @classmethod
    def new(cls, expr: str) -> Self:
        return cls(parse_prop(expr))


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
                raise EvalException(
                    "[TModusPonens] Left side of hypothesis does not match input"
                )
            case _:
                raise EvalException("[TModusPonens] Left side is not an implication")


@dataclass(frozen=True, slots=True, eq=True)
class TModusTollens(Tactic):
    imply: Tactic
    result: Tactic

    def eval(self) -> Expr:
        imply_expr = self.imply.eval()
        result_expr = self.result.eval()

        match imply_expr:
            case EImplies(left, right):
                if ENeg(right) == result_expr:
                    return ENeg(left)
                raise EvalException(
                    "[TModusTollens] Right side of hypothesis does not match negation of input"
                )
            case _:
                raise EvalException("[TModusPonens] Left side is not an implication")


@dataclass(frozen=True, slots=True, eq=True)
class THypotheticalSyllogism(Tactic):
    imply1: Tactic
    imply2: Tactic

    def eval(self) -> Expr:
        imply1_expr = self.imply1.eval()
        imply2_expr = self.imply2.eval()

        match imply1_expr, imply2_expr:
            case EImplies(p, q1), EImplies(q2, r):
                if q1 == q2:
                    return EImplies(p, r)
                raise EvalException(
                    "[THypotheticalSyllogism] Middle terms of implications are not equal"
                )
            case _:
                raise EvalException(
                    "[THypotheticalSyllogism] One of the terms is not an implication"
                )


@dataclass(frozen=True, slots=True, eq=True)
class TDisjunctiveSyllogism(Tactic):
    disjunc: Tactic
    not_p: Tactic

    def eval(self) -> Expr:
        disjunc_expr = self.disjunc.eval()
        not_p_expr = self.not_p.eval()

        match disjunc_expr:
            case EOr(p, q):
                if not_p_expr == ENeg(p):
                    return q
                raise EvalException(
                    "[TDisjunctiveSyllogism] Input is not the negation of first term is disjunction"
                )
            case _:
                raise EvalException(
                    "[TDisjunctiveSyllogism] First term is not a disjunction"
                )
