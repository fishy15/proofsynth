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
    "((P || Q) -> Q) -> (P || Q) -> Q",
    "((P || Q) -> Q) -> !(!P && !Q) -> Q",
    "!(!P && !Q) -> (P || Q)",
    "!(!P && !Q) -> (!!P || !!Q)",
    "(P -> Q) -> (Q -> R) -> (P -> R)",
    "((P -> Q) -> (R -> S)) -> (P -> Q) -> (R -> S)",
    "P -> !!!!P",
    "((P && Q) && R) -> (P && (Q && R))",
    "(Q -> P) -> Q -> P",
    "(!P || Q) -> P -> Q",
]

heuristic_types = [
    "naive",
    "rnn",
    "rnn-small",
    "transformer",
]


@pytest.mark.parametrize("goal", tasks)
@pytest.mark.parametrize("heuristic", heuristic_types)
@pytest.mark.timeout(120)
def test_synthesis(goal: str, heuristic: str) -> None:
    goal_expr = parse_prop(goal)
    task = SynthesisTask(goal_expr, heuristic=heuristic)
    proof = task.construct_proof()
    print("\nIterations Used:", task.iterations_used)
    print("# of new generated terms:", task.successful_iterations)
    assert proof is not None


@pytest.mark.parametrize("goal", tasks)
@pytest.mark.parametrize("heuristic", heuristic_types)
@pytest.mark.timeout(120)
def test_synthesis_remove_double_neg(goal: str, heuristic: str) -> None:
    goal_expr = parse_prop(goal)
    task = SynthesisTask(goal_expr, heuristic=heuristic, remove_double_neg=True)
    proof = task.construct_proof()
    print("\nIterations Used:", task.iterations_used)
    print("# of new generated terms:", task.successful_iterations)
    assert proof is not None


@pytest.mark.parametrize("goal", tasks)
@pytest.mark.parametrize("heuristic", heuristic_types)
@pytest.mark.timeout(120)
def test_synthesis_with_canonicalization(goal: str, heuristic: str) -> None:
    goal_expr = parse_prop(goal)
    task = SynthesisTask(goal_expr, heuristic=heuristic, should_canonicalize=True)
    proof = task.construct_proof()
    print("\nIterations Used:", task.iterations_used)
    print("# of new generated terms:", task.successful_iterations)
    assert proof is not None


@pytest.mark.parametrize("goal", tasks)
@pytest.mark.parametrize("heuristic", heuristic_types)
@pytest.mark.timeout(120)
def test_synthesis_with_canonicalization_and_neg_removal(
    goal: str, heuristic: str
) -> None:
    goal_expr = parse_prop(goal)
    task = SynthesisTask(
        goal_expr, heuristic=heuristic, should_canonicalize=True, remove_double_neg=True
    )
    proof = task.construct_proof()
    print("\nIterations Used:", task.iterations_used)
    print("# of new generated terms:", task.successful_iterations)
    assert proof is not None
