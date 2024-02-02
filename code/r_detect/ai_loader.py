
from copy import deepcopy
from dataclasses import dataclass, field

import torch
from transformers import DataCollatorWithPadding


def apply_mask_augmentation(input_ids, tokenizer, mask_prob=0.1):
    input_ids = deepcopy(input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    indices_mask = torch.bernoulli(torch.full(input_ids.shape, mask_prob)).bool()

    do_not_mask_tokens = list(set(tokenizer.all_special_ids))
    pass_gate = [
        [0 if token_id in do_not_mask_tokens else 1 for token_id in token_id_seq] for token_id_seq in input_ids
    ]
    pass_gate = torch.tensor(pass_gate, dtype=torch.bool)

    indices_mask = torch.logical_and(indices_mask, pass_gate)
    input_ids[indices_mask] = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    return input_ids


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

        buffer_dict = dict()
        buffer_keys = ["id"]

        for key in buffer_keys:
            if key in features[0].keys():
                value = [feature[key] for feature in features]
                buffer_dict[key] = value

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
            padding='longest',  # self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        # for key, value in buffer_dict.items():
        #     batch[key] = value

        if labels is not None:
            batch["labels"] = labels

        tensor_keys = [
            "input_ids",
            "attention_mask",
        ]

        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        if labels is not None:
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.float32)

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

    def __call__(self, features):

        buffer_dict = dict()
        buffer_keys = ["id"]

        for key in buffer_keys:
            if key in features[0].keys():
                value = [feature[key] for feature in features]
                buffer_dict[key] = value

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
            padding='longest',  # self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        # for key, value in buffer_dict.items():
        #     batch[key] = value

        if self.cfg.train_params.use_mask_aug:
            batch["input_ids"] = apply_mask_augmentation(
                batch["input_ids"], self.tokenizer, self.cfg.train_params.mask_aug_prob
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
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.float32)

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
        # print("\n\n")

        if "infer" not in task.lower():
            print_fn("--"*20)
            labels = batch['labels'][idx]
            print_fn(f"Label: {labels}")
        print_fn('~~'*40)
