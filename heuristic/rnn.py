import random

import torch.nn.functional as F
import numpy as np

from typing import List

from heuristic.rnn_model import MyRNN, load_model
from heuristic.heuristic import Heuristic
from prop.lang import *
from prop.tactics import (
    SingleTactic,
    DoubleTactic,
    bottom_up_tactics_single,
    bottom_up_tactics_double,
    all_tactics,
)

NUM_SINGLE = len(bottom_up_tactics_single)
assert bottom_up_tactics_single == all_tactics[:NUM_SINGLE]
assert bottom_up_tactics_double == all_tactics[NUM_SINGLE:]

NUM_DOUBLE = len(bottom_up_tactics_double)


class RNNLoadException(Exception):
    pass


_rnn = load_model("checkpoint.pt")


def get_random_idx(relative_probs: list[float]) -> int:
    total_p = sum(relative_probs)
    target = random.random() * total_p

    prefix_sum = 0.0
    for i, prob in enumerate(relative_probs):
        prefix_sum += prob
        if target <= prefix_sum:
            return i
    return len(relative_probs) - 1


class RNNHeuristic(Heuristic):
    rnn: MyRNN

    def __init__(self):
        try:
            self.rnn = _rnn
        except:
            raise RNNLoadException(
                "checkpoint.pt missing from local dir, trying symlinking it?"
            )

    def pick_tactic_single(self, hypotheses: List[Expr], goal: Expr) -> SingleTactic:
        assert len(hypotheses) == 3

        output_values = self.rnn.evaluate(hypotheses, goal)[:NUM_SINGLE]
        output_probs = F.softmax(output_values, dim=0).tolist()
        idx = get_random_idx(output_probs)
        return bottom_up_tactics_single[idx]

    def pick_tactic_double(self, hypotheses: List[Expr], goal: Expr) -> DoubleTactic:
        assert len(hypotheses) == 3

        output_values = self.rnn.evaluate(hypotheses, goal)[NUM_SINGLE:]
        output_probs = F.softmax(output_values).tolist()
        idx = get_random_idx(output_probs)
        return bottom_up_tactics_double[idx]
