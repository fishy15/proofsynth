from abc import ABC, abstractmethod
from typing import Callable, List, Self

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


# skipping Constructive Dilemma
# skipping Destructive Dilemma
# skipping Bidirectional Dilemma


@dataclass(frozen=True, slots=True, eq=True)
class TSimplification(Tactic):
    conj: Tactic

    def eval(self) -> Expr:
        conj_expr = self.conj.eval()

        match conj_expr:
            case EAnd(p, _):
                return p
            case _:
                raise EvalException("[TSimplifcation] Term is not a conjunction")


# skipping Addition
# skipping Composition
# skipping De Morgan


@dataclass(frozen=True, slots=True, eq=True)
class TCommuteOr(Tactic):
    disj: Tactic

    def eval(self) -> Expr:
        disj_expr = self.disj.eval()

        match disj_expr:
            case EOr(p, q):
                return EOr(q, p)
            case _:
                raise EvalException("[TCommuteOr] Term is not a disjunction")


@dataclass(frozen=True, slots=True, eq=True)
class TCommuteAnd(Tactic):
    conj: Tactic

    def eval(self) -> Expr:
        conj_expr = self.conj.eval()

        match conj_expr:
            case EAnd(p, q):
                return EAnd(q, p)
            case _:
                raise EvalException("[TCommuteAnd] Term is not a conjunction")


@dataclass(frozen=True, slots=True, eq=True)
class TAssocOrLeft(Tactic):
    disj: Tactic

    def eval(self) -> Expr:
        disj_expr = self.disj.eval()

        match disj_expr:
            case EOr(EOr(p, q), r):
                return EOr(p, EOr(q, r))
            case _:
                raise EvalException("[TAssocOrLeft] Term is not a disjunction")


@dataclass(frozen=True, slots=True, eq=True)
class TAssocAndLeft(Tactic):
    conj: Tactic

    def eval(self) -> Expr:
        conj_expr = self.conj.eval()

        match conj_expr:
            case EAnd(EAnd(p, q), r):
                return EAnd(p, EAnd(q, r))
            case _:
                raise EvalException("[TAssocAndLeft] Term is not a conjunction")


@dataclass(frozen=True, slots=True, eq=True)
class TAssocOrRight(Tactic):
    disj: Tactic

    def eval(self) -> Expr:
        disj_expr = self.disj.eval()

        match disj_expr:
            case EOr(p, EOr(q, r)):
                return EOr(EOr(p, q), r)
            case _:
                raise EvalException("[TAssocOrRight] Term is not a disjunction")


@dataclass(frozen=True, slots=True, eq=True)
class TAssocAndRight(Tactic):
    conj: Tactic

    def eval(self) -> Expr:
        conj_expr = self.conj.eval()

        match conj_expr:
            case EAnd(p, EAnd(q, r)):
                return EAnd(EAnd(p, q), r)
            case _:
                raise EvalException("[TAssocAndRight] Term is not a conjunction")


@dataclass(frozen=True, slots=True, eq=True)
class TDistribAndSingle(Tactic):
    conj: Tactic

    def eval(self) -> Expr:
        conj_expr = self.conj.eval()

        match conj_expr:
            case EAnd(p, EOr(q, r)):
                return EOr(EAnd(p, q), EAnd(p, r))
            case _:
                raise EvalException("[TDistribAndSingle] Term is not a conjunction")


@dataclass(frozen=True, slots=True, eq=True)
class TDistribAndDouble(Tactic):
    conj: Tactic

    def eval(self) -> Expr:
        conj_expr = self.conj.eval()

        match conj_expr:
            case EAnd(EOr(p1, q), EOr(p2, r)):
                if p1 == p2:
                    return EOr(p1, EAnd(q, r))
                else:
                    raise EvalException(
                        "[TDistribAndDouble] First terms are not the same"
                    )
            case _:
                raise EvalException("[TDistribAndSingle] Term is not a conjunction")


@dataclass(frozen=True, slots=True, eq=True)
class TDistribOrSingle(Tactic):
    disj: Tactic

    def eval(self) -> Expr:
        disj_expr = self.disj.eval()

        match disj_expr:
            case EOr(p, EAnd(q, r)):
                return EAnd(EOr(p, q), EOr(p, r))
            case _:
                raise EvalException("[TDistribOrSingle] Term is not a disjunction")


@dataclass(frozen=True, slots=True, eq=True)
class TDistribOrDouble(Tactic):
    disj: Tactic

    def eval(self) -> Expr:
        disj_expr = self.disj.eval()

        match disj_expr:
            case EOr(EAnd(p1, q), EAnd(p2, r)):
                if p1 == p2:
                    return EAnd(p1, EOr(q, r))
                else:
                    raise EvalException(
                        "[TDistribOrDouble] First terms are not the same"
                    )
            case _:
                raise EvalException("[TDistribOrSingle] Term is not a disjunction")


@dataclass(frozen=True, slots=True, eq=True)
class TDoubleNegationAdd(Tactic):
    term: Tactic

    def eval(self) -> Expr:
        return ENeg(ENeg(self.term.eval()))


@dataclass(frozen=True, slots=True, eq=True)
class TDoubleNegationRemove(Tactic):
    term: Tactic

    def eval(self) -> Expr:
        term_expr = self.term.eval()

        match term_expr:
            case ENeg(ENeg(body)):
                return body
            case _:
                raise EvalException(
                    "[TDoubleNegationRemove] Expression does not have a double negation"
                )


@dataclass(frozen=True, slots=True, eq=True)
class TTransposition(Tactic):
    term: Tactic

    def eval(self) -> Expr:
        term_expr = self.term.eval()

        match term_expr:
            case EImplies(p, q):
                return EImplies(ENeg(q), ENeg(p))
            case _:
                raise EvalException("[TTransposition] Expression is not an implication")


SingleTactic = Callable[[Tactic], Tactic]
DoubleTactic = Callable[[Tactic, Tactic], Tactic]

bottom_up_tactics_single: List[SingleTactic] = [
    TSimplification,
    TCommuteOr,
    TCommuteAnd,
    TAssocOrLeft,
    TAssocAndLeft,
    TAssocOrRight,
    TAssocAndRight,
    TDistribAndSingle,
    TDistribAndDouble,
    TDistribOrSingle,
    TDistribOrDouble,
    TDoubleNegationAdd,
    TDoubleNegationRemove,
    TTransposition,
]

bottom_up_tactics_double: List[DoubleTactic] = [
    TModusPonens,
    TModusTollens,
    THypotheticalSyllogism,
    TDisjunctiveSyllogism,
]
