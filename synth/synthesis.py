from prop.lang import Expr

from synth.proofstate import ProofState
from synth.init_tactics import apply_all_init
from synth.end_tactics import apply_all_end


def construct_proof(goal: Expr):
    state = ProofState([], goal)
    return solve_proofstate(state)


def solve_proofstate(state: ProofState) -> bool:
    state_after_inits = apply_all_init(state)
    state_after_both = apply_all_end(state_after_inits)

    if len(state_after_both) == 1 and state == state_after_both[0]:
        # we didn't make any progress
        return find_tactics(state)
    else:
        return any(map(solve_proofstate, state_after_both))


def find_tactics(state: ProofState) -> bool:
    print("hypothesis:", " ".join(map(str, state.hypotheses)))
    print("goal:", state.goal)
    return False
