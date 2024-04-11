import pytest

from prop.lang import *
from prop.prop_parser import parse_prop
from prop.canonicalize import canonicalize


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("((P) -> (Q))", EImplies(EVar("P"), EVar("Q"))),
        ("((P) && (Q))", EAnd(EVar("P"), EVar("Q"))),
        ("((P) || (Q))", EOr(EVar("P"), EVar("Q"))),
        ("(!(P))", ENeg(EVar("P"))),
        ("(!(!(P)))", ENeg(ENeg(EVar("P")))),
    ],
)
def test_prop_parser(expr: str, expected: Expr) -> None:
    assert parse_prop(expr) == expected


@pytest.mark.parametrize(
    "expr,expected",
    [
        (EImplies(EVar("P"), EVar("Q")), "((P) -> (Q))"),
        (EAnd(EVar("P"), EVar("Q")), "((P) && (Q))"),
        (EOr(EVar("P"), EVar("Q")), "((P) || (Q))"),
        (ENeg(EVar("P")), "(!(P))"),
        (ENeg(ENeg(EVar("P"))), "(!(!(P)))"),
    ],
)
def test_prop_str(expr: Expr, expected: str) -> None:
    assert str(expr) == expected

@pytest.mark.parametrize(
    "expr,expected",
    [
        ("(P)", "(P)"),
        ("(!(!(P)))", "(P)"),
        ("((P) -> (Q))", "((P) -> (Q))"),
        ("((P) || (Q))", "((!(P)) -> (Q))"),
        ("((P) && (Q))", "(!((P) -> (!(Q))))"),
        ("((!(P)) || (Q))", "((P) -> (Q))"),
        ("(!((P) && (!(Q))))", "((P) -> (Q))"),
    ],
)
def test_canonicalize(expr: str, expected: Expr) -> None:
    assert str(canonicalize(parse_prop(expr))) == expected