import pytest

from prop.prop_parser import parse_prop
from prop.lang import *

from synth.synthesis import SynthesisTask

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

heuristic_types = [
    "naive",
    "rnn",
    "rnn-small",
    "transformer",
]


@pytest.mark.parametrize("goal", tasks)
@pytest.mark.parametrize("heuristic", heuristic_types)
def test_synthesis(goal: str, heuristic: str) -> None:
    goal_expr = parse_prop(goal)
    task = SynthesisTask(goal_expr, heuristic=heuristic)
    assert task.construct_proof() is not None


@pytest.mark.parametrize("goal", tasks)
@pytest.mark.parametrize("heuristic", heuristic_types)
def test_synthesis_remove_double_neg(goal: str, heuristic: str) -> None:
    goal_expr = parse_prop(goal)
    task = SynthesisTask(goal_expr, heuristic=heuristic, remove_double_neg=True)
    assert task.construct_proof() is not None


@pytest.mark.parametrize("goal", tasks)
@pytest.mark.parametrize("heuristic", heuristic_types)
def test_synthesis_with_canonicalization(goal: str, heuristic: str) -> None:
    goal_expr = parse_prop(goal)
    task = SynthesisTask(goal_expr, heuristic=heuristic, should_canonicalize=True)
    assert task.construct_proof() is not None


@pytest.mark.parametrize("goal", tasks)
@pytest.mark.parametrize("heuristic", heuristic_types)
def test_synthesis_with_canonicalization_and_neg_removal(
    goal: str, heuristic: str
) -> None:
    goal_expr = parse_prop(goal)
    task = SynthesisTask(
        goal_expr, heuristic=heuristic, should_canonicalize=True, remove_double_neg=True
    )
    assert task.construct_proof() is not None
