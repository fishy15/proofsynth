import pytest

from typing import List

from synth.init_tactics import *
from synth.proofstate import ProofState


@pytest.mark.parametrize(
    "goal,exp_hyp,exp_goal",
    [
        ("P -> Q -> R", ["P", "Q"], "R"),
        ("(P -> Q) -> R", ["P -> Q"], "R"),
    ],
)
def test_intros(goal: str, exp_hyp: List[str], exp_goal: str) -> None:
    state = ProofState.new_goal(goal)
    expected = ProofState.new(exp_hyp, exp_goal)
    assert intros(state) == expected


@pytest.mark.parametrize(
    "init_hyp,init_goal,exp_hyp,exp_goal",
    [
        (["P && Q"], "R", ["P", "Q"], "R"),
        (["P || Q"], "R", ["P || Q"], "R"),
        (["A -> B", "P && Q", "!C"], "R", ["A -> B", "P", "Q", "!C"], "R"),
    ],
)
def test_intro_ands(
    init_hyp: List[str], init_goal: str, exp_hyp: List[str], exp_goal: str
) -> None:
    state = ProofState.new(init_hyp, init_goal)
    expected = ProofState.new(exp_hyp, exp_goal)
    assert intro_ands(state) == expected
