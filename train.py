import torch

from heuristic.rnn_generate import *
from heuristic.rnn import *


def generate():
    limit = int(input("how much? "))

    exprs = generate_init_exprs()
    proofs = generate_random_programs(exprs)
    with open("training.txt", "w") as f:
        generate_training_examples(proofs, limit=limit, file=f)


def process_examples():
    examples = load_training_examples("training.txt", 64)
    torch.save(examples, "training.pt")


def train_model():
    examples = torch.load("training.pt")
    model = train(examples)
    torch.save(model, "model.pt")


def main():
    print("1 - generate")
    print("2 - process examples")
    print("3 - train model")

    typ = int(input("input type: "))
    if typ == 1:
        generate()
    elif typ == 2:
        process_examples()
    else:
        train_model()


if __name__ == "__main__":
    main()
