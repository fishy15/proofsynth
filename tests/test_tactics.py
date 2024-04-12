import pytest

from prop.tactics import *
from prop.prop_parser import parse_prop


@pytest.mark.parametrize(
    "tactics_tree,expected",
    [
        (THypothesis.new("(P)"), "(P)"),
        (TModusPonens(THypothesis.new("((P) -> (Q))"), THypothesis.new("(P)")), "(Q)"),
    ],
)
def test_prop_parser(tactics_tree: Tactic, expected: str) -> None:
    assert tactics_tree.eval() == parse_prop(expected)
