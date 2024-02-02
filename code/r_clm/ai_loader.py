from dataclasses import dataclass

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
        if "labels" in features[0].keys():
            labels = [feature["labels"] for feature in features]

        features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            } for feature in features
        ]

        batch = self.tokenizer.pad(
            features,
            padding='longest',
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        tensor_keys = [
            "input_ids",
            "attention_mask",
        ]

        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        seq_len = batch["input_ids"].size(1)

        if labels is not None:
            padded_labels = []
            for label in labels:
                padded_label = [-100] * (seq_len - len(label)) + label  # left pad
                padded_labels.append(padded_label)
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.int64)

        return batch


# ---


def show_batch(batch, tokenizer, n_examples=16, task='training', print_fn=print):
    bs = batch['input_ids'].size(0)
    print_fn(f"batch size: {bs}")

    print_fn(f"shape of input_ids: {batch['input_ids'].shape}")

    n_examples = min(n_examples, bs)
    print_fn(f"Showing {n_examples} from a {task} batch...")

    print_fn("\n\n")
    for idx in range(n_examples):
        print_fn(f"Example {idx+1}")
        print_fn(f"Input:\n\n{tokenizer.decode(batch['input_ids'][idx], skip_special_tokens=False)}")
        print_fn(f"Input ids:\n\n{batch['input_ids'][idx]}")
        if 'labels' in batch:
            print_fn(f"Labels:\n\n{batch['labels'][idx]}")
        print_fn('~~'*40)
