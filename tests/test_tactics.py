import pytest

from prop.tactics import *
from prop.prop_parser import parse_prop


@pytest.mark.parametrize(
    "tactics_tree,expected",
    [
        (THypothesis.new("P"), "P"),
        (TModusPonens(THypothesis.new("P -> Q"), THypothesis.new("P")), "Q"),
        (TModusTollens(THypothesis.new("P -> Q"), THypothesis.new("!Q")), "!P"),
        (
            THypotheticalSyllogism(
                THypothesis.new("P -> Q"), THypothesis.new("Q -> R")
            ),
            "P -> R",
        ),
        (TDisjunctiveSyllogism(THypothesis.new("P || Q"), THypothesis.new("!P")), "Q"),
        (TSimplification(THypothesis.new("P && Q")), "P"),
        (TCommuteOr(THypothesis.new("P || Q")), "Q || P"),
        (TCommuteAnd(THypothesis.new("P && Q")), "Q && P"),
        (TAssocOrLeft(THypothesis.new("(P || Q) || R")), "P || (Q || R)"),
        (TAssocAndLeft(THypothesis.new("(P && Q) && R")), "P && (Q && R)"),
        (TAssocOrRight(THypothesis.new("P || (Q || R)")), "(P || Q) || R"),
        (TAssocAndRight(THypothesis.new("P && (Q && R)")), "(P && Q) && R"),
        (TDistribOrSingle(THypothesis.new("P || (Q && R)")), "(P || Q) && (P || R)"),
        (TDistribAndSingle(THypothesis.new("P && (Q || R)")), "(P && Q) || (P && R)"),
        (TDistribOrDouble(THypothesis.new("(P && Q) || (P && R)")), "P && (Q || R)"),
        (TDistribAndDouble(THypothesis.new("(P || Q) && (P || R)")), "P || (Q && R)"),
        (TDoubleNegationAdd(THypothesis.new("P")), "!!P"),
        (TDoubleNegationRemove(THypothesis.new("!!P")), "P"),
        (TTransposition(THypothesis.new("P -> Q")), "!Q -> !P"),
    ],
)
def test_prop_parser(tactics_tree: Tactic, expected: str) -> None:
    assert tactics_tree.eval() == parse_prop(expected)
