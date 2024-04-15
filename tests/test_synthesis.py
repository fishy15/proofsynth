import pytest

from prop.prop_parser import parse_prop
from prop.lang import *

from synth.synthesis import construct_proof


@pytest.mark.parametrize(
    "goal",
    [
        "P -> P",
        "P -> (P -> Q) -> Q",
        "P -> (P -> Q) -> (Q || R)",
    ],
)
def test_prop_parser(goal: str) -> None:
    goal_expr = parse_prop(goal)
    construct_proof(goal_expr)
