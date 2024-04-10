from pathlib import Path
from lark import Transformer, Lark

from prop.lang import *


def construct_parser() -> Lark:
    with open(Path(__file__).parent.absolute() / "grammar.lark", "r") as f:
        return Lark(f, start="expr")


class PropTransformer(Transformer):
    def __init__(self):
        super(Transformer).__init__()

    def var(self, tree) -> Expr:
        return EVar(tree[0].value)

    def neg(self, tree) -> Expr:
        return ENeg(tree[0])

    def imply(self, tree) -> Expr:
        left, right = tree
        return EImplies(left, right)


def parse_prop(text: str) -> Expr:
    with open(Path(__file__).parent.absolute() / "grammar.lark", "r") as f:
        parser = Lark(f, start="expr")

    tree = parser.parse(text)
    return PropTransformer().transform(tree)
