import random

from itertools import product
from typing import Optional

from prop.lang import Expr
from prop.tactics import Tactic, THypothesis, EvalException
from synth.proofstate import ProofState
from synth.init_tactics import apply_all_init
from synth.end_tactics import apply_all_end
from heuristic.heuristic import Heuristic
from heuristic.naive import Naive

ITERATION_LIMIT = 10000


def construct_proof(goal: Expr) -> Optional[Tactic]:
    state = ProofState([], goal)
    return solve_proofstate(state)


def solve_proofstate(state: ProofState) -> Optional[Tactic]:
    state_after_inits = apply_all_init(state)
    state_after_both = apply_all_end(state_after_inits)

    if len(state_after_both) == 1 and state == state_after_both[0]:
        # we didn't make any progress
        return find_tactics(state, Naive())
    else:
        for proof in map(solve_proofstate, state_after_both):
            if proof is not None:
                return proof
        return None


def find_tactics(state: ProofState, heuristic: Heuristic) -> Optional[Tactic]:
    # print("hypothesis:", " ".join(map(str, state.hypotheses)))
    # print("goal:", state.goal)

    current_proofs: dict[Expr, Tactic] = {}
    for h in state.hypotheses:
        current_proofs[h] = THypothesis(h)

    def insert_if_valid(proof: Tactic):
        try:
            expr = proof.eval()
            if (
                expr not in current_proofs
                and expr.depth() < 2 * state.goal.depth() + 10
            ):
                current_proofs[expr] = proof
        except EvalException:
            # just means that it is invalid
            pass

    iterations = 0

    while state.goal not in current_proofs and iterations <= ITERATION_LIMIT:
        k = min(4, len(current_proofs))
        sample = random.sample(list(current_proofs.items()), k)
        sample_exprs = [s[0] for s in sample]
        sample_tactics = [s[1] for s in sample]

        if random.random() < 0.5:
            # double tactic
            double_tactic = heuristic.pick_tactic_double(sample_exprs, state.goal)
            for hyp1, hyp2 in product(sample_tactics, repeat=2):
                insert_if_valid(double_tactic(hyp1, hyp2))
        else:
            # single tactic
            single_tactic = heuristic.pick_tactic_single(sample_exprs, state.goal)
            for hyp in sample_tactics:
                insert_if_valid(single_tactic(hyp))

        iterations += 1

    # if state.goal in current_proofs:
    #     print(current_proofs[state.goal])
    # else:
    #     print("failed to find proof :(")

    return current_proofs.get(state.goal, None)
