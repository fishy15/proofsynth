import random
import sys

from prop.lang import *
from prop.tactics import *

MAX_DEPTH = 3
SAMPLE_SIZE = 3

EXPR_LIMIT = 10_000_000
PROOF_LIMIT = 100_000
SAMPLE_LIMIT = 10_000_000


def generate_init_exprs(limit: int = EXPR_LIMIT) -> list[Expr]:
    current_exprs_small: list[Expr] = list(map(EVar, "PQRST"))
    current_exprs_set = set(current_exprs_small)

    in_set_already = 0
    depth_too_big = 0

    for _ in range(limit):
        expr_cons = random.choice([ENeg, EImplies, EAnd, EOr])
        if expr_cons == ENeg:
            old_expr: Expr = random.choice(current_exprs_small)
            new_expr = expr_cons(old_expr)
        else:
            old_expr1 = random.choice(current_exprs_small)
            old_expr2 = random.choice(current_exprs_small)
            new_expr = expr_cons(old_expr1, old_expr2)

        new_depth = new_expr.depth()
        if new_expr not in current_exprs_set and new_depth <= MAX_DEPTH:
            if new_depth < MAX_DEPTH:
                current_exprs_small.append(new_expr)
            current_exprs_set.add(new_expr)
        elif new_expr in current_exprs_set:
            in_set_already += 1
        else:
            depth_too_big += 1

    current_exprs = list(current_exprs_set)

    print(len(current_exprs), in_set_already, depth_too_big)
    return current_exprs


def generate_random_programs(
    exprs: list[Expr], limit: int = PROOF_LIMIT
) -> list[Tactic]:
    current_proofs: list[Tactic] = list(map(THypothesis, exprs))
    current_proofs_set = set(current_proofs)

    while len(current_proofs) < limit:
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

    print(len(current_proofs))
    return current_proofs


def create_input(samples: list[Expr], goal: Expr) -> str:
    to_convert = samples + [goal]
    result = ",".join(map(str, to_convert))

    # compact the output so each character is a token
    result = result.replace(" && ", "&")
    result = result.replace(" || ", "|")
    result = result.replace(" -> ", ">")
    return result


def generate_training_examples(
    all_hypotheses: list[Expr],
    proofs: list[Tactic],
    limit: int = SAMPLE_LIMIT,
    file=sys.stdout,
) -> None:
    extraction_and_goals = [(p.extract_examples(), p.eval()) for p in proofs]

    for _ in range(limit):
        while True:
            proof_extract, goal = random.choice(extraction_and_goals)
            if proof_extract:
                break

        hypotheses = list(proof_extract.keys())
        samples_set = set(random.choice(hypotheses) for _ in range(SAMPLE_SIZE))

        tactics_used: set[SingleTactic | DoubleTactic] = set()
        for s in samples_set:
            s1, s2 = proof_extract[s]
            tactics_used.update(s1)
            tactics_used.update(s2)

        while len(samples_set) < SAMPLE_SIZE:
            samples_set.add(random.choice(all_hypotheses))

        samples = list(samples_set)
        random_tactic = random.choice(list(tactics_used))

        input_str = create_input(samples, goal)
        output_str = str(all_tactics.index(random_tactic))
        result = f"{input_str},{output_str}"

        print(result, file=file)
