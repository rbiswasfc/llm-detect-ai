import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

loss_fct = torch.nn.MarginRankingLoss(margin=0.7)
class BCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        human_outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        ai_outputs = model(input_ids=inputs["ai_input_ids"], attention_mask=inputs["ai_attention_mask"])

        human_outputs = human_outputs.get("logits").view(-1)
        ai_outputs = ai_outputs.get("logits").view(-1)

        loss = loss_fct(ai_outputs, human_outputs, torch.ones_like(ai_outputs))

        return (loss, ai_outputs) if return_outputs else loss

essay_df = pd.read_csv("train_essays_pos_neg.csv").sample(200_000)
train_df = essay_df.copy().reset_index(drop=True)
train_df["human"] = train_df["human"].str.strip()
train_df["ai"] = train_df["ai"].str.strip()

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-large",
    num_labels=1
)

train_ds = Dataset.from_pandas(train_df)

def preprocess_function(examples, max_length=1280):
    tokenized_samples = tokenizer(examples["human"], truncation=True, max_length=max_length)
    tokenized_samples_ai = tokenizer(examples["ai"], truncation=True, max_length=max_length)

    tokenized_samples["ai_input_ids"] = tokenized_samples_ai["input_ids"]
    tokenized_samples["ai_attention_mask"] = tokenized_samples_ai["attention_mask"]

    return tokenized_samples

train_tokenized_ds = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)

training_args = TrainingArguments(
    output_dir=f"checkpoint/deberta-v3-large-v18-margin",
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_grad_norm=1,
    optim='adamw_8bit',
    num_train_epochs=1,
    weight_decay=0.1,
    fp16=True,
    save_strategy="epoch",
    remove_unused_columns=False,
    warmup_steps=0.1,
    logging_steps=100,
    gradient_checkpointing=False,
    report_to='tensorboard'
)

trainer = BCETrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_ds,
    tokenizer=tokenizer,
)

trainer.train()