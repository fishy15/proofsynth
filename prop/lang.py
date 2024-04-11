from dataclasses import dataclass
from typing import TypeAlias


class Expr:
    pass


Id: TypeAlias = str


@dataclass(frozen=True, slots=True, eq=True)
class EVar(Expr):
    name: Id

    def __str__(self) -> str:
        return f"({self.name})"


@dataclass(frozen=True, slots=True, eq=True)
class ENeg(Expr):
    body: Expr

    def __str__(self) -> str:
        return f"(!{self.body})"


@dataclass(frozen=True, slots=True, eq=True)
class EImplies(Expr):
    left: Expr
    right: Expr

    def __str__(self) -> str:
        return f"({self.left} -> {self.right})"


@dataclass(frozen=True, slots=True, eq=True)
class EAnd(Expr):
    left: Expr
    right: Expr

    def __str__(self) -> str:
        return f"({self.left} && {self.right})"


@dataclass(frozen=True, slots=True, eq=True)
class EOr(Expr):
    left: Expr
    right: Expr

    def __str__(self) -> str:
        return f"({self.left} || {self.right})"
