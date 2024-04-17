from heuristic.rnn import *

exprs = generate_init_exprs()
proofs = generate_random_programs(exprs)
generate_training_examples(proofs, limit=100)
