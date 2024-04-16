import random

from prop.lang import *

MAX_DEPTH = 8


def generate_init_exprs(limit: int = 100000) -> list[Expr]:
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
