import pytest

from prop.prop_parser import parse_prop
from prop.lang import *

from synth.synthesis import construct_proof

tasks = [
    "P -> P",
    "P -> (P -> Q) -> Q",
    "P -> (P -> Q) -> (Q || R)",
    "((!Q -> !P) && P) -> ((!!P -> !!Q) || (Q -> R))",
    "(P -> Q -> R -> S -> T) -> P -> Q -> R -> S -> T",
    "(!P -> Q) -> (P || Q)",
    "(P -> Q) -> (!P || Q)",
    "P -> P -> P -> P -> P",
    "(P && !P) -> Q",
    "P || !P",
    "!(P || Q) -> (!P && !Q)",
]


@pytest.mark.parametrize("goal", tasks)
def test_synthesis(goal: str) -> None:
    goal_expr = parse_prop(goal)
    assert construct_proof(goal_expr) is not None


@pytest.mark.parametrize("goal", tasks)
def test_synthesis_remove_double_neg(goal: str) -> None:
    goal_expr = parse_prop(goal)
    assert construct_proof(goal_expr, remove_double_neg=True) is not None


@pytest.mark.parametrize("goal", tasks)
def test_synthesis_with_canonicalization(goal: str) -> None:
    goal_expr = parse_prop(goal)
    assert construct_proof(goal_expr, canonicalize=True) is not None


@pytest.mark.parametrize("goal", tasks)
def test_synthesis_with_canonicalization_and_neg_removal(goal: str) -> None:
    goal_expr = parse_prop(goal)
    assert (
        construct_proof(goal_expr, canonicalize=True, remove_double_neg=True)
        is not None
    )


@pytest.mark.parametrize("goal", tasks)
def test_rnn_synthesis(goal: str) -> None:
    goal_expr = parse_prop(goal)
    assert construct_proof(goal_expr, user_rnn=True) is not None


@pytest.mark.parametrize("goal", tasks)
def test_rnn_synthesis_remove_double_neg(goal: str) -> None:
    goal_expr = parse_prop(goal)
    assert construct_proof(goal_expr, remove_double_neg=True, use_rnn=True) is not None


@pytest.mark.parametrize("goal", tasks)
def test_rnn_synthesis_with_canonicalization(goal: str) -> None:
    goal_expr = parse_prop(goal)
    assert construct_proof(goal_expr, canonicalize=True, use_rnn=True) is not None


@pytest.mark.parametrize("goal", tasks)
def test_rnn_synthesis_with_canonicalization_and_neg_removal(goal: str) -> None:
    goal_expr = parse_prop(goal)
    assert (
        construct_proof(
            goal_expr, canonicalize=True, remove_double_neg=True, use_rnn=True
        )
        is not None
    )
