from typing import List

from prop.lang import *
from synth.proofstate import ProofState


class EndTacticException(Exception):
    pass


def split_or(state: ProofState) -> List[ProofState]:
    match state.goal:
        case EOr(p, q):
            return [ProofState(state.hypotheses, p), ProofState(state.hypotheses, q)]
        case _:
            return [state]
