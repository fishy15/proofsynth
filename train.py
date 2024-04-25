import torch

from heuristic.rnn_generate import *
from heuristic.rnn_model import *


def generate():
    limit = int(input("how much? "))
    directory = input("which dir? ")

    exprs = generate_init_exprs()
    proofs = generate_random_programs(exprs)
    with open(f"{directory}/training.txt", "w") as f:
        generate_training_examples(exprs, proofs, limit=limit, file=f)


def train_model():
    save_dir = input("which dir? ")
    with open(f"{save_dir}/training.txt", "r") as f:
        examples = f.readlines()

    is_small = input("is small? (y/n) ")
    small = is_small.lower() == "y"

    should_load = input("load checkpoint? (y/n) ")
    if should_load.lower() == "y":
        checkpoint_file = input("checkpoint name? ")
        checkpoint_file = f"{save_dir}/{checkpoint_file}"
    else:
        checkpoint_file = None

    model = train(examples, save_dir, small, checkpoint_file)
    torch.save(model.state_dict(), f"{save_dir}/model.pt")


def main():
    print("1 - generate")
    print("2 - train model")

    typ = int(input("input type: "))
    if typ == 1:
        generate()
    else:
        train_model()


if __name__ == "__main__":
    main()
