import random

from itertools import product
from typing import Optional

from prop.canonicalize import canonicalize, remove_double_negation
from prop.lang import Expr
from prop.tactics import Tactic, THypothesis, EvalException
from synth.proofstate import ProofState
from synth.init_tactics import apply_all_init
from synth.end_tactics import apply_all_end
from heuristic.heuristic import Heuristic
from heuristic.naive import Naive
from heuristic.rnn import RNNHeuristic

ITERATION_LIMIT = 10000
SAMPLE_SIZE = 3


def construct_proof(goal: Expr, **kwargs: bool) -> Optional[Tactic]:
    if "canonicalize" in kwargs:
        should_canonicalize = kwargs["canonicalize"]
    else:
        should_canonicalize = False

    if "remove_double_neg" in kwargs:
        remove_double_neg = kwargs["remove_double_neg"]
    else:
        remove_double_neg = False

    if "rnn" in kwargs:
        use_rnn = kwargs["rnn"]
    else:
        use_rnn = False

    if should_canonicalize:
        canonical_goal = canonicalize(goal)
        # print('canonicalization:', goal, "TO", canonical_goal)
        goal = canonical_goal
    elif remove_double_neg:
        goal_remove_negs = remove_double_negation(goal)
        # print('remove double negs:', goal, "TO", goal_remove_negs)
        goal = goal_remove_negs

    state = ProofState([], goal)
    return solve_proofstate(state, remove_double_neg=remove_double_neg, use_rnn=use_rnn)


def solve_proofstate(
    state: ProofState, iterations_allowed: int = ITERATION_LIMIT, **kwargs: bool
) -> Optional[Tactic]:
    if "remove_double_neg" in kwargs:
        remove_double_neg = kwargs["remove_double_neg"]
    else:
        remove_double_neg = False

    if "use_rnn" in kwargs:
        use_rnn = kwargs["use_rnn"]
    else:
        use_rnn = False

    state_after_inits = apply_all_init(state)
    state_after_both = apply_all_end(state_after_inits)

    if len(state_after_both) == 1 and state == state_after_both[0]:
        # we didn't make any progress
        heuristic: Heuristic
        if use_rnn:
            heuristic = RNNHeuristic()
        else:
            heuristic = Naive()

        return find_tactics(state, heuristic, iterations_allowed, remove_double_neg)
    else:
        iterations_per_goal = iterations_allowed // len(state_after_both)
        proofs_per_goal = (
            solve_proofstate(s, iterations_per_goal) for s in state_after_both
        )
        for proof in proofs_per_goal:
            if proof is not None:
                return proof
        return None


def find_tactics(
    state: ProofState,
    heuristic: Heuristic,
    iterations_allowed: int,
    remove_double_neg: bool,
) -> Optional[Tactic]:
    # print("hypothesis:", " ".join(map(str, state.hypotheses)))
    # print("goal:", state.goal)

    current_proofs: dict[Expr, Tactic] = {}
    current_hypotheses: list[Expr] = []
    for h in state.hypotheses:
        current_proofs[h] = THypothesis(h)
        current_hypotheses.append(h)

    def insert_if_valid(proof: Tactic):
        try:
            expr = proof.eval()
            if remove_double_neg:
                expr = remove_double_negation(expr)

            if (
                expr not in current_proofs
                and expr.depth() < 2 * state.goal.depth() + 10
            ):
                current_proofs[expr] = proof
                current_hypotheses.append(expr)
        except EvalException:
            # just means that it is invalid
            pass

    iterations = 0

    while state.goal not in current_proofs and iterations <= iterations_allowed:
        sample_exprs = [random.choice(current_hypotheses) for _ in range(SAMPLE_SIZE)]
        sample_tactics = [current_proofs[h] for h in sample_exprs]

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
