from prop.lang import *
from synth.proofstate import ProofState


class InitTacticException(Exception):
    pass


def intros(state: ProofState) -> ProofState:
    while True:
        match state.goal:
            case EImplies(p, q):
                state = ProofState(state.hypotheses + [p], q)
            case _:
                return state


def intro_ands(state: ProofState) -> ProofState:
    while True:
        for i, hyp in enumerate(state.hypotheses):
            match hyp:
                case EAnd(p, q):
                    new_hypotheses = (
                        state.hypotheses[:i] + [p, q] + state.hypotheses[i + 1 :]
                    )
                    state = ProofState(new_hypotheses, state.goal)
                    break
        else:
            return state
