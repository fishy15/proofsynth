import os
import torch

from heuristic.rnn_generate import *
from heuristic.rnn_model import *
from heuristic.transformer_finetune import *
from transformers import T5ForSequenceClassification


def generate() -> None:
    limit = int(input("how much? "))
    directory = input("which dir? ")

    exprs = generate_init_exprs()
    proofs = generate_random_programs(exprs)
    with open(f"{directory}/training.txt", "w") as f:
        generate_training_examples(exprs, proofs, limit=limit, file=f)


def train_model() -> None:
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


def finetune_model() -> None:
    base_model = T5ForSequenceClassification.from_pretrained(
        "Salesforce/codet5-small", num_labels=len(all_tactics)
    )

    save_dir = input("which dir? ")
    transformer_checkpoints = f"{save_dir}/transformer_checkpoints"
    os.makedirs(transformer_checkpoints, exist_ok=True)
    dataset = load_dataset(f"{save_dir}/training.txt")

    resume_str = input("want to resume? (y/n) ")
    resume = resume_str == "y"

    finetune(dataset, base_model, transformer_checkpoints, resume)


def main():
    print("1 - generate")
    print("2 - train RNN")
    print("3 - finetune transformer")

    typ = int(input("input type: "))
    if typ == 1:
        generate()
    elif typ == 2:
        train_model()
    else:
        finetune_model()


if __name__ == "__main__":
    main()
