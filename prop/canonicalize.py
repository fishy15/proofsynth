from prop.lang import *

"""
Rewrites the expression to only use -> and !.
Also deletes double negations.
"""


def canonicalize(expr: Expr) -> Expr:
    match expr:
        case ENeg(body):
            body = canonicalize(body)
            match body:
                case ENeg(e):
                    return e
                case _:
                    return ENeg(body)
        case EImplies(left, right):
            left = canonicalize(left)
            right = canonicalize(right)
            return EImplies(left, right)
        case EOr(left, right):
            left = canonicalize(ENeg(left))
            right = canonicalize(right)
            return EImplies(left, right)
        case EAnd(left, right):
            left = canonicalize(left)
            right = canonicalize(ENeg(right))
            return ENeg(EImplies(left, right))
        case _:
            return expr
