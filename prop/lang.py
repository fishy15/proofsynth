from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias


class Expr(ABC):
    @abstractmethod
    def depth(self) -> int:
        pass


Id: TypeAlias = str


@dataclass(frozen=True, slots=True, eq=True)
class EVar(Expr):
    name: Id

    def __str__(self) -> str:
        return f"({self.name})"

    def depth(self) -> int:
        return 1


@dataclass(frozen=True, slots=True, eq=True)
class ENeg(Expr):
    body: Expr

    def __str__(self) -> str:
        return f"(!{self.body})"

    def depth(self) -> int:
        return self.body.depth() + 1


@dataclass(frozen=True, slots=True, eq=True)
class EImplies(Expr):
    left: Expr
    right: Expr

    def __str__(self) -> str:
        return f"({self.left} -> {self.right})"

    def depth(self) -> int:
        return max(self.left.depth(), self.right.depth()) + 1


@dataclass(frozen=True, slots=True, eq=True)
class EAnd(Expr):
    left: Expr
    right: Expr

    def __str__(self) -> str:
        return f"({self.left} && {self.right})"

    def depth(self) -> int:
        return max(self.left.depth(), self.right.depth()) + 1


@dataclass(frozen=True, slots=True, eq=True)
class EOr(Expr):
    left: Expr
    right: Expr

    def __str__(self) -> str:
        return f"({self.left} || {self.right})"

    def depth(self) -> int:
        return max(self.left.depth(), self.right.depth()) + 1
