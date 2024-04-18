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


class RNNHeuristic(Heuristic):
    rnn: MyRNN

    def __init__(self):
        try:
            self.rnn = load_model("checkpoint.pt")
        except:
            raise RNNLoadException(
                "checkpoint.pt missing from local dir, trying symlinking it?"
            )

    def pick_tactic_single(self, hypotheses: List[Expr], goal: Expr) -> SingleTactic:
        assert len(hypotheses) == 3

        output_values = self.rnn.evaluate(hypotheses, goal)[:NUM_SINGLE]
        output_probs = F.softmax(output_values).tolist()
        idx = np.random.choice(np.arange(NUM_SINGLE), p=output_probs)
        return bottom_up_tactics_single[idx]

    def pick_tactic_double(self, hypotheses: List[Expr], goal: Expr) -> DoubleTactic:
        assert len(hypotheses) == 3

        output_values = self.rnn.evaluate(hypotheses, goal)[NUM_SINGLE:]
        output_probs = F.softmax(output_values).tolist()
        idx = np.random.choice(np.arange(NUM_DOUBLE), p=output_probs)
        return bottom_up_tactics_double[idx]
