import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import Tuple

from prop.tactics import (
    bottom_up_tactics_single,
    bottom_up_tactics_double,
    all_tactics,
    SingleTactic,
    DoubleTactic,
)

ALPHABET = "PQRST!&|>()_"
BATCH_SIZE = 2
TERM_SIZE = 64


class MyRnn(nn.Module):
    emb: nn.Embedding
    rnn: nn.RNN
    final: nn.Linear

    def __init__(self, seq_len: int, d_model: int, d_hidden: int, num_layers: int):
        super().__init__()
        self.emb = nn.Embedding(len(ALPHABET), d_model)
        self.rnn = nn.RNN(d_model, d_hidden, num_layers, batch_first=True)
        self.final = nn.Linear(seq_len * d_hidden, len(all_tactics))

    def forward(self, batch_indices):
        after_rnn = self.rnn(self.emb(batch_indices))[0]  # ignore hidden layers
        return self.final(
            torch.flatten(after_rnn, 1)
        )  # don't want to flatten batch dimension


def load_training_examples(
    file_name: str, max_term_len: int
) -> list[Tuple[torch.Tensor, torch.Tensor]]:
    def convert(s: str) -> torch.Tensor:
        if len(s) > max_term_len:
            s = s[:max_term_len]
        while len(s) < max_term_len:
            s += "_"

        lst = [ALPHABET.index(c) for c in s]
        return torch.Tensor(lst).long()

    examples = []
    with open(file_name, "r") as f:
        for line in f:
            terms = line.strip().split(",")
            t1, t2, t3, goal = map(convert, terms[:-1])
            cls = int(terms[-1])

            input_tensor = torch.cat((t1, t2, t3, goal))
            output_tensor = torch.Tensor([cls]).long()

            examples.append((input_tensor, output_tensor))

    return examples


def train(examples: list[Tuple[torch.Tensor, torch.Tensor]]) -> nn.RNN:
    model = MyRnn(TERM_SIZE * 4, 32, 32, 3)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    random.seed(1001)

    num_epochs = 10
    batch_size = BATCH_SIZE

    for t in range(0, num_epochs):
        print("Epoch", t)
        loss_this_epoch = 0.0
        ex_idxs = [i for i in range(0, len(examples))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.CrossEntropyLoss()

        for exs in batch(ex_idxs, batch_size):
            inp = torch.stack([examples[i][0] for i in exs])
            target = torch.stack([examples[i][1] for i in exs]).squeeze()
            output = model(inp)

            loss = loss_fcn(output, target)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            loss_this_epoch += loss.item()

        print("Loss:", loss_this_epoch)

    model.eval()
    return model


def batch(lst, sz):
    for i in range(0, len(lst), sz):
        yield lst[i : i + sz]
