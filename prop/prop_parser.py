from pathlib import Path
from lark import Transformer, Lark, Tree

from prop.lang import *


def construct_parser() -> Lark:
    with open(Path(__file__).parent.absolute() / "grammar.lark", "r") as f:
        return Lark(f, start="expr")


def construct_right_associative(cls, tree) -> Expr:
    right: Expr = tree[-1]
    for left in reversed(tree[:-1]):
        right = cls(left, right)
    return right


class PropTransformer(Transformer):
    def __init__(self):
        super(Transformer).__init__()

    def var(self, tree) -> Expr:
        return EVar(tree[0].value)

    def neg(self, tree) -> Expr:
        return ENeg(tree[0])

    def imply(self, tree) -> Expr:
        return construct_right_associative(EImplies, tree)

    def eand(self, tree) -> Expr:
        return construct_right_associative(EAnd, tree)

    def eor(self, tree) -> Expr:
        return construct_right_associative(EOr, tree)


def parse_prop(text: str) -> Expr:
    with open(Path(__file__).parent.absolute() / "grammar.lark", "r") as f:
        parser = Lark(f, start="expr")

    text = "(" + text + ")"
    tree = parser.parse(text)
    return PropTransformer().transform(tree)
