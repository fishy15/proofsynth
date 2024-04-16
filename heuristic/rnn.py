import random

from prop.lang import *
from prop.tactics import *

MAX_DEPTH = 8
LIMIT = 100000


def generate_init_exprs(limit: int = LIMIT) -> list[Expr]:
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


def generate_random_programs(exprs: list[Expr], limit: int = LIMIT) -> list[Tactic]:
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
