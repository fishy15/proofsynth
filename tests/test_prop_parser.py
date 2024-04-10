import pytest

from prop.lang import *
from prop.prop_parser import parse_prop


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("((P) -> (Q))", EImplies(EVar("P"), EVar("Q"))),
    ],
)
def test_prop_parser(expr: str, expected: Expr) -> None:
    assert parse_prop(expr) == expected
