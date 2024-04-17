import random

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Optional, Tuple

from prop.tactics import all_tactics

ALPHABET = "PQRST!&|>()_"
BATCH_SIZE = 20
TERM_SIZE = 64

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class MyRNN(nn.Module):
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    def convert(s: str) -> torch.Tensor:
        if len(s) > max_term_len:
            s = s[:max_term_len]
        while len(s) < max_term_len:
            s += "_"

        lst = [ALPHABET.index(c) for c in s]
        return torch.Tensor(lst).long()

    example_inps = []
    example_outs = []
    with open(file_name, "r") as f:
        for line in f:
            terms = line.strip().split(",")
            t1, t2, t3, goal = map(convert, terms[:-1])
            cls = int(terms[-1])

            input_tensor = torch.cat((t1, t2, t3, goal))
            output_tensor = torch.Tensor([cls]).long()

            example_inps.append(input_tensor)
            example_outs.append(output_tensor)

    return torch.stack(example_inps), torch.stack(example_outs)


def save_checkpoint(epoch, iter, model, optimizer, dir):
    file_name = f"{dir}/checkpoint_{epoch}_{iter}.pt"
    print("saving checkpoint at", file_name)

    checkpoint = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "epoch": epoch,
    }

    torch.save(checkpoint, file_name)


def train(
    examples: Tuple[torch.Tensor, torch.Tensor],
    checkpoint_dir: str,
    load_from: Optional[str] = None,
) -> MyRNN:
    example_inps, example_outs = examples
    model = MyRNN(TERM_SIZE * 4, 32, 32, 3)
    model.to(device)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    random.seed(1001)

    if load_from is not None:
        checkpoint = torch.load(load_from)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optim"])
        epoch_start = checkpoint["epoch"]
    else:
        epoch_start = 0

    num_epochs = 20
    batch_size = BATCH_SIZE

    for t in range(epoch_start, num_epochs):
        print("Epoch", t)
        loss_this_epoch = 0.0
        ex_idxs = [i for i in range(0, len(example_inps))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.CrossEntropyLoss()

        iteration_cnt = 0
        for exs in batch(ex_idxs, batch_size):
            inp = torch.stack([example_inps[i] for i in exs]).to(device)
            target = torch.stack([example_outs[i] for i in exs]).squeeze().to(device)
            output = model(inp)

            loss = loss_fcn(output, target)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            loss_this_epoch += loss.item()
            iteration_cnt += 1

            if iteration_cnt % 10000 == 0:
                save_checkpoint(t, iteration_cnt, model, optimizer, checkpoint_dir)

        print("Loss:", loss_this_epoch)

    model.eval()
    return model


def batch(lst, sz):
    for i in range(0, len(lst), sz):
        yield lst[i : i + sz]
