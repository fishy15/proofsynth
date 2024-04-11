from dataclasses import dataclass
from typing import TypeAlias


class Expr:
    pass


Id: TypeAlias = str


@dataclass(frozen=True, slots=True, eq=True)
class EVar(Expr):
    name: Id


@dataclass(frozen=True, slots=True, eq=True)
class ENeg(Expr):
    body: Expr


@dataclass(frozen=True, slots=True, eq=True)
class EImplies(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True, eq=True)
class EAnd(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True, eq=True)
class EOr(Expr):
    left: Expr
    right: Expr
