import random

from itertools import product, chain
from typing import Optional

from prop.canonicalize import canonicalize, remove_double_negation
from prop.lang import Expr
from prop.tactics import Tactic, THypothesis, TSubcases, EvalException
from synth.proofstate import ProofState
from synth.init_tactics import apply_all_init
from synth.end_tactics import split_and, split_or
from heuristic.heuristic import Heuristic
from heuristic.naive import Naive
from heuristic.rnn import RNNHeuristic
from heuristic.transformer import CodeT5Heuristic

ITERATION_LIMIT = 10000
SAMPLE_SIZE = 3


class SynthesisTask:
    goal: Expr
    heuristic: Heuristic
    should_canonicalize: bool
    remove_double_neg: bool
    iterations_used: int
    successful_iterations: int

    def __init__(
        self,
        goal: Expr,
        should_canonicalize=False,
        remove_double_neg=False,
        heuristic: str = "naive",
    ):
        self.goal = goal

        if heuristic == "naive":
            self.heuristic = Naive()
        elif heuristic == "rnn":
            self.heuristic = RNNHeuristic()
        elif heuristic == "rnn-small":
            self.heuristic = RNNHeuristic(small=True)
        else:
            self.heuristic = CodeT5Heuristic()

        self.should_canonicalize = should_canonicalize
        self.remove_double_neg = remove_double_neg
        self.iterations_used = 0
        self.successful_iterations = 0

    def construct_proof(
        self,
    ) -> Optional[Tactic]:
        goal = self.goal

        if self.should_canonicalize:
            canonical_goal = canonicalize(goal)
            # print('canonicalization:', goal, "TO", canonical_goal)
            goal = canonical_goal
        elif self.remove_double_neg:
            goal_remove_negs = remove_double_negation(goal)
            # print('remove double negs:', goal, "TO", goal_remove_negs)
            goal = goal_remove_negs

        state = ProofState([], goal)
        return self.solve_proofstate(state)

    def solve_proofstate(
        self,
        state: ProofState,
        iterations_allowed: int = ITERATION_LIMIT,
    ) -> Optional[Tactic]:
        state_after_inits = apply_all_init(state)
        state_after_both = split_or(state_after_inits)

        if len(state_after_both) == 1 and state == state_after_both[0]:
            iters_half = iterations_allowed // 2
            proof_all = self.find_tactics(state, iters_half)
            if proof_all is None:
                state_goal_ands_split = split_and(state)
                iterations_per_goal = iters_half // len(state_goal_ands_split)
                proofs = [
                    self.find_tactics(s, iterations_per_goal)
                    for s in state_goal_ands_split
                ]
                if all(p is not None for p in proofs):
                    return TSubcases(proofs)  # type:ignore
            return proof_all
        else:
            iterations_per_goal = iterations_allowed // (len(state_after_both) + 1)

            solve_current = [self.find_tactics(state_after_inits, iterations_per_goal)]

            solve_split_gen = (
                self.solve_proofstate(
                    s,
                    iterations_per_goal,
                )
                for s in state_after_both
            )

            for proof in chain(solve_current, solve_split_gen):
                if proof is not None:
                    return proof
            return None

    def find_tactics(
        self,
        state: ProofState,
        iterations_allowed: int,
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
                if self.remove_double_neg:
                    expr = remove_double_negation(expr)

                if (
                    expr not in current_proofs
                    and expr.depth() < 2 * state.goal.depth() + 10
                ):
                    current_proofs[expr] = proof
                    current_hypotheses.append(expr)
                    self.successful_iterations += 1
            except EvalException:
                # just means that it is invalid
                pass

        iterations = 0

        while state.goal not in current_proofs and iterations <= iterations_allowed:
            sample_exprs = [
                random.choice(current_hypotheses) for _ in range(SAMPLE_SIZE)
            ]
            sample_tactics = [current_proofs[h] for h in sample_exprs]

            if random.random() < 0.5:
                # double tactic
                double_tactic = self.heuristic.pick_tactic_double(
                    sample_exprs, state.goal
                )
                for hyp1, hyp2 in product(sample_tactics, repeat=2):
                    insert_if_valid(double_tactic(hyp1, hyp2))
            else:
                # single tactic
                single_tactic = self.heuristic.pick_tactic_single(
                    sample_exprs, state.goal
                )
                for hyp in sample_tactics:
                    insert_if_valid(single_tactic(hyp))

            self.iterations_used += 1
            iterations += 1

        # if state.goal in current_proofs:
        #     print(current_proofs[state.goal])
        # else:
        #     print("failed to find proof :(")

        return current_proofs.get(state.goal, None)
