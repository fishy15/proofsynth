import random
import sys

from prop.lang import *
from prop.tactics import *

MAX_DEPTH = 5
SAMPLE_SIZE = 3

EXPR_LIMIT = 10_000
PROOF_LIMIT = 100_000
SAMPLE_LIMIT = 10_000_000


def generate_init_exprs(limit: int = EXPR_LIMIT) -> list[Expr]:
    current_exprs: list[Expr] = list(map(EVar, "PQRST"))
    current_exprs_set = set(current_exprs)

    for _ in range(limit):
        expr_cons = random.choice([ENeg, EImplies, EAnd, EOr])
        if expr_cons == ENeg:
            old_expr: Expr = random.choice(current_exprs)
            new_expr = expr_cons(old_expr)
        else:
            old_expr1 = random.choice(current_exprs)
            old_expr2 = random.choice(current_exprs)
            new_expr = expr_cons(old_expr1, old_expr2)

        if new_expr not in current_exprs_set and new_expr.depth() <= MAX_DEPTH:
            current_exprs.append(new_expr)
            current_exprs_set.add(new_expr)

    return current_exprs


def generate_random_programs(
    exprs: list[Expr], limit: int = PROOF_LIMIT
) -> list[Tactic]:
    current_proofs: list[Tactic] = list(map(THypothesis, exprs))
    current_proofs_set = set(current_proofs)

    for _ in range(limit):
        tactic: SingleTactic | DoubleTactic
        if random.random() < 0.5:
            # single program
            tactic = random.choice(bottom_up_tactics_single)
            random_proof = random.choice(current_proofs)
            new_proof = tactic(random_proof)
        else:
            tactic = random.choice(bottom_up_tactics_double)
            random_proof1 = random.choice(current_proofs)
            random_proof2 = random.choice(current_proofs)
            new_proof = tactic(random_proof1, random_proof2)

        try:
            new_proof.eval()  # check if this is a valid program
            if new_proof not in current_proofs_set:
                current_proofs.append(new_proof)
                current_proofs_set.add(new_proof)
        except EvalException:
            pass

    return current_proofs


def generate_training_examples(
    proofs: list[Tactic], limit: int = SAMPLE_LIMIT, file=sys.stdout
) -> None:
    extraction_and_goals = [(p.extract_examples(), p.eval()) for p in proofs]

    for _ in range(limit):
        while True:
            proof_extract, goal = random.choice(extraction_and_goals)
            if proof_extract:
                break

        hypotheses = list(proof_extract.keys())
        samples = [random.choice(hypotheses) for _ in range(SAMPLE_SIZE)]

        tactics_used: set[SingleTactic | DoubleTactic] = set()
        for s in samples:
            s1, s2 = proof_extract[s]
            tactics_used.update(s1)
            tactics_used.update(s2)

        random_tactic = random.choice(list(tactics_used))

        to_convert = samples + [goal]
        input_str = ",".join(map(str, to_convert))
        output_str = str(all_tactics.index(random_tactic))
        result = f"{input_str},{output_str}"

        # compact the output so each character is a token
        result = result.replace(" && ", "&")
        result = result.replace(" || ", "|")
        result = result.replace(" -> ", ">")

        print(result, file=file)
