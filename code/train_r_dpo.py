# adapted from https://github.com/huggingface/alignment-handbook/blob/main/scripts/run_dpo.py

import argparse
import os

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf
from peft import LoraConfig, PeftConfig, PeftModel, TaskType
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import DPOTrainer


def get_datasets(cfg):
    """
    prepare training and test datasets for DPO
    """
    raw_datasets = DatasetDict()
    train_df = pd.read_parquet(cfg.train_path)  # prompt, chosen, rejected
    test_df = pd.read_parquet(cfg.test_path)

    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    train_ds = train_ds.remove_columns(["dpo_id", "diff"])
    test_ds = test_ds.remove_columns(["dpo_id", "diff"])

    raw_datasets["train"] = train_ds
    raw_datasets["test"] = test_ds

    return raw_datasets


def get_tokenizer(cfg):

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.sft_model_path,
        use_fast=True,
        padding_side='left',
        truncation_side='left',
    )

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seed for reproducibility
    set_seed(cfg.seed)

    # set up accelerator
    accelerator = Accelerator()

    # datasets ---
    raw_datasets = get_datasets(cfg)
    tokenizer = get_tokenizer(cfg)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # model ----
    accelerator.print(f"Merging peft adapters for {cfg.sft_model_path}")
    peft_config = PeftConfig.from_pretrained(cfg.sft_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=quantization_config,
    )

    model = PeftModel.from_pretrained(base_model, cfg.sft_model_path)
    model.eval()
    model = model.merge_and_unload()
    model_kwargs = None  # {"use_cache": False}

    peft_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=cfg_dict["lora"]["target_modules"],
    )

    ref_model = None
    ref_model_kwargs = None  # {"use_cache": False}

    # Training args ---
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        # lr_scheduler_type=cfg.lr_scheduler_type,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_grad_norm=cfg.max_grad_norm,
        optim=cfg.dpo.optim,
        num_train_epochs=cfg.num_train_epochs,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=None,
        warmup_steps=cfg.warmup_ratio,
        logging_steps=1,
        report_to='wandb',
        # gradient_checkpointing=True,
    )

    # DPO Trainer ---
    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=cfg.dpo.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=cfg.dpo.max_length,
        max_prompt_length=cfg.dpo.max_prompt_length,
        peft_config=peft_config,
    )

    # Training loop ---
    train_result = dpo_trainer.train()
    metrics = train_result.metrics
    # max_train_samples = int(0.25*len(raw_datasets["train"]))
    # metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
    dpo_trainer.log_metrics("train", metrics)
    dpo_trainer.save_metrics("train", metrics)
    dpo_trainer.save_state()

    accelerator.print("*** Training complete ***")

    # Evaluation loop ---
    accelerator.print("*** Evaluate ***")
    metrics = dpo_trainer.evaluate()
    dpo_trainer.log_metrics("eval", metrics)
    dpo_trainer.save_metrics("eval", metrics)

    # Save model ---
    dpo_trainer.save_model(cfg.output_dir)

    # Ensure we don't timeout on model save / push to Hub
    accelerator.print("*** Waiting for all processes to finish ***")
    accelerator.wait_for_everyone()

    accelerator.print("*** Run complete! ***")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_path', type=str, required=True)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config_path)

    os.makedirs(cfg.output_dir, exist_ok=True)
    main(cfg)
