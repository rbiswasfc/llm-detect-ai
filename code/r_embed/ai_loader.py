
import os
import random
import time
from dataclasses import dataclass, field

import torch
from transformers import DataCollatorWithPadding


@dataclass
class AiCollator(DataCollatorWithPadding):
    """
    data collector for LLM Detect AI Generated Text task
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):
        labels = None
        if "generated" in features[0].keys():
            labels = [feature["generated"] for feature in features]

        features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            } for feature in features
        ]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        if labels is not None:
            batch["labels"] = labels

        tensor_keys = [
            "input_ids",
            "attention_mask",
        ]

        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        if labels is not None:
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        return batch


@dataclass
class AiCollatorTrain(DataCollatorWithPadding):
    """
    data collector for LLM Detect AI Generated Text task
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"
    kwargs: field(default_factory=dict) = None

    def __post_init__(self):
        [setattr(self, k, v) for k, v in self.kwargs.items()]

        # mappings
        example2idx = dict()
        example_ids = self.train_ds["id"]

        for idx in range(len(example_ids)):
            example2idx[example_ids[idx]] = idx
        self.example2idx = example2idx

        seed = seed = int(time.time() * 1000) + os.getpid()
        self.rng = random.Random(seed)

        print("=="*40)
        print(f"setting random seed in data collator as: {seed}")
        print("=="*40)

    def process_features(self, example_ids):
        updated_features = []
        for eid in example_ids:
            example = dict()

            example["id"] = eid
            ex_info = self.train_ds[self.example2idx[eid]]

            # use fields
            example["input_ids"] = ex_info["input_ids"]
            example["attention_mask"] = ex_info["attention_mask"]
            example["generated"] = ex_info["generated"]
            updated_features.append(example)

        return updated_features

    def __call__(self, features):
        bs = len(features)
        selected_prompt_id = self.rng.choice(self.prompt_ids)
        selected_example_ids_pos = self.rng.sample(self.prompt2ids_pos[selected_prompt_id], k=bs//2)
        selected_example_ids_neg = self.rng.sample(self.prompt2ids_neg[selected_prompt_id], k=bs//2)
        selected_example_ids = selected_example_ids_pos + selected_example_ids_neg
        features = self.process_features(selected_example_ids)

        labels = None
        if "generated" in features[0].keys():
            labels = [feature["generated"] for feature in features]

        features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            } for feature in features
        ]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        if labels is not None:
            batch["labels"] = labels

        tensor_keys = [
            "input_ids",
            "attention_mask",
        ]

        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        if labels is not None:
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        return batch

# ---


def show_batch(batch, tokenizer, n_examples=16, task='training', print_fn=print):
    print_fn("##"*40)
    bs = batch['input_ids'].size(0)
    print_fn(f"batch size: {bs}")

    print_fn(f"shape of input_ids: {batch['input_ids'].shape}")

    n_examples = min(n_examples, bs)
    print_fn(f"Showing {n_examples} from a {task} batch...")

    print_fn("\n\n")
    for idx in range(n_examples):
        print_fn(f"Example {idx+1}")
        print_fn(f"Input:\n\n{tokenizer.decode(batch['input_ids'][idx], skip_special_tokens=False)}")
        # print("\n\n")

        if "infer" not in task.lower():
            print_fn("--"*20)
            labels = batch['labels'][idx]
            print_fn(f"Label: {labels}")
        print_fn('=='*40)
    print_fn("##"*40)
