from typing import Optional
import torch
import torch.nn.functional as F

from datasets import Dataset
from transformers import (
    RobertaTokenizerFast,
    T5ForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from prop.lang import Expr
from prop.tactics import all_tactics
from heuristic.rnn_generate import create_input

BATCH_SIZE = 1
tokenizer = RobertaTokenizerFast.from_pretrained("Salesforce/codet5-small")


def tokenize(ex, **kwargs):
    return tokenizer(
        ex, truncation=True, max_length=512, padding="max_length", **kwargs
    )


def load_dataset(filepath: str) -> Dataset:
    def read_from_file():
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                last_comma = line.rfind(",")
                inp = line[:last_comma]
                out = int(line[last_comma + 1 :])
                yield {"input": inp, "label": out}

    def prepare(exs):
        tokenized_examples = tokenize(exs["input"])
        tokenized_examples["label"] = exs["label"]
        return tokenized_examples

    dataset = Dataset.from_generator(read_from_file)
    tokenized_dataset = dataset.map(prepare, batched=True, num_proc=8)

    return tokenized_dataset


def finetune(examples: Dataset, base_model, directory: str, resume: bool):
    dataset = examples.train_test_split(test_size=0.0001)

    training_args = TrainingArguments(
        output_dir=directory,
        learning_rate=1e-4,  # taken from https://huggingface.co/docs/transformers/model_doc/t5
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=900,
        save_strategy="steps",
        save_steps=900,
        load_best_model_at_end=True,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model()


_finetuned_model: Optional[T5ForSequenceClassification] = None
_base_model: Optional[T5ForSequenceClassification] = None


def load_model(base: bool = False) -> T5ForSequenceClassification:
    global _finetuned_model
    global _base_model

    if base:
        if _base_model is None:
            _base_model = T5ForSequenceClassification.from_pretrained(
                "Salesforce/codet5-small", num_labels=len(all_tactics)
            )
        return _base_model
    else:
        if _finetuned_model is None:
            _finetuned_model = T5ForSequenceClassification.from_pretrained(
                "hf_checkpoint", num_labels=len(all_tactics)
            )
        return _finetuned_model


def evaluate(
    model: T5ForSequenceClassification, hypotheses: list[Expr], goal: Expr
) -> torch.Tensor:
    input_str = create_input(hypotheses, goal)
    inputs = tokenizer(input_str, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze()
        return F.softmax(logits, dim=0)
