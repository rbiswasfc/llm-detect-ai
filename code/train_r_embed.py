import json
import logging
import os
import random
import time
from copy import deepcopy

import datasets
import hydra
import numpy as np
import pandas as pd
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

try:
    from r_embed.ai_dataset import AiDataset
    from r_embed.ai_loader import AiCollator, AiCollatorTrain, show_batch
    from r_embed.ai_model import AiModel
    from r_embed.ai_optimizer import get_optimizer
    from utils.train_utils import (AverageMeter, as_minutes, get_lr,
                                   save_checkpoint)

except Exception as e:
    print(e)
    raise ImportError

logger = get_logger(__name__)


pd.options.display.max_colwidth = 1000

# -------- Evaluation -------------------------------------------------------------#


def run_evaluation(accelerator, model, valid_dl):
    model.eval()

    all_losses = []

    progress_bar = tqdm(range(len(valid_dl)), disable=not accelerator.is_local_main_process)

    for batch in valid_dl:
        with torch.no_grad():
            loss = model(**batch)

        batch_losses = accelerator.gather_for_metrics(loss)
        batch_losses = batch_losses.cpu().numpy().tolist()
        all_losses.extend(batch_losses)

        progress_bar.update(1)
    progress_bar.close()

    # compute metric
    eval_dict = dict()
    eval_dict['valid_loss'] = np.mean(all_losses)

    return eval_dict


# -------- Main Function ---------------------------------------------------------#


@hydra.main(version_base=None, config_path="../conf/r_embed", config_name="conf_r_embed")
def run_training(cfg):
    # ------- Accelerator ---------------------------------------------------------------#
    if cfg.use_wandb:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train_params.gradient_accumulation_steps,
            log_with="wandb",
        )

        accelerator.init_trackers(
            cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train_params.gradient_accumulation_steps,
        )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    def print_line():
        prefix, unit, suffix = "#", "~~", "#"
        accelerator.print(prefix + unit*50 + suffix)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # ------- Runtime Configs -----------------------------------------------------------#
    print_line()
    accelerator.print(f"setting seed: {cfg.seed}")
    set_seed(cfg.seed)

    if accelerator.is_main_process:
        os.makedirs(cfg.outputs.model_dir, exist_ok=True)
    print_line()

    # ------- load data ----------------------------------------------------------#
    print_line()
    data_dir = cfg.input_data_dir

    train_df = pd.read_parquet(os.path.join(data_dir, "train_essays.parquet"))
    train_df = train_df[~train_df['text'].isna()].copy()

    valid_df = pd.read_parquet(os.path.join(data_dir, "valid_essays.parquet"))
    valid_df = valid_df[~valid_df['text'].isna()].copy()

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    prompt_ids = train_df["prompt_id"].unique().tolist()
    prompt_ids = [p for p in prompt_ids if p <= 8]

    pos_df = train_df[train_df["generated"] == 1].copy()
    neg_df = train_df[train_df["generated"] == 0].copy()

    pos_gdf = pos_df.groupby("prompt_id")["id"].apply(list).reset_index()
    prompt2ids_pos = dict(zip(pos_gdf["prompt_id"], pos_gdf["id"]))

    neg_gdf = neg_df.groupby("prompt_id")["id"].apply(list).reset_index()
    prompt2ids_neg = dict(zip(neg_gdf["prompt_id"], neg_gdf["id"]))

    accelerator.print(f"shape of train data: {train_df.shape}")
    accelerator.print(f"{train_df.head()}")
    accelerator.print(f"shape of validation data: {valid_df.shape}")
    accelerator.print(f"Prompts: {prompt_ids}")

    with accelerator.main_process_first():
        dataset_creator = AiDataset(cfg)

        train_ds = dataset_creator.get_dataset(train_df)
        valid_ds = dataset_creator.get_dataset(valid_df)

    tokenizer = dataset_creator.tokenizer

    # ------- data loaders ----------------------------------------------------------------#
    train_ds.set_format(
        type=None,
        columns=[
            'id',
            'input_ids',
            'attention_mask',
            'generated'
        ]
    )

    valid_ds = valid_ds.sort("input_length")

    valid_ds.set_format(
        type=None,
        columns=[
            'id',
            'input_ids',
            'attention_mask',
            'generated'
        ]
    )
    valid_ids = valid_df["id"]

    # ---
    kwargs = dict(
        train_ds=train_ds,
        prompt_ids=prompt_ids,
        prompt2ids_pos=prompt2ids_pos,
        prompt2ids_neg=prompt2ids_neg,
    )

    data_collector_train = AiCollatorTrain(
        tokenizer=tokenizer,
        pad_to_multiple_of=64,
        kwargs=kwargs,
    )

    data_collector = AiCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=64
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train_params.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collector_train,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.train_params.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collector,
    )

    accelerator.print("data preparation done...")
    print_line()

    # --- show batch -------------------------------------------------------------------#
    print_line()

    for b in train_dl:
        break
    show_batch(b, tokenizer, task='training', print_fn=print, n_examples=4)

    print_line()

    for b in valid_dl:
        break
    show_batch(b, tokenizer, task='validation', print_fn=accelerator.print)

    print_line()

    # ------- Config -------------------------------------------------------------------#
    accelerator.print("config for the current run:")
    accelerator.print(json.dumps(cfg_dict, indent=4))
    print_line()

    # ------- Model --------------------------------------------------------------------#
    print_line()
    print("creating the LLM Detection model...")
    model = AiModel(cfg, accelerator.device)
    print_line()

    # ------- Optimizer ----------------------------------------------------------------#
    print_line()
    print("creating the optimizer...")
    optimizer = get_optimizer(model, cfg)
    # ------- Prepare -------------------------------------------------------------------#

    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )

    # ------- Scheduler -----------------------------------------------------------------#
    print_line()
    num_epochs = cfg.train_params.num_train_epochs
    grad_accumulation_steps = cfg.train_params.gradient_accumulation_steps
    warmup_pct = cfg.train_params.warmup_pct

    num_update_steps_per_epoch = len(train_dl)//grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct*num_training_steps)

    accelerator.print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"# training steps: {num_training_steps}")
    accelerator.print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # ------- training setup --------------------------------------------------------------#
    best_lb = 1e6  # track recall@1000

    patience_tracker = 0
    current_iteration = 0

    # ------- training  --------------------------------------------------------------------#
    start_time = time.time()
    accelerator.wait_for_everyone()

    for epoch in range(num_epochs):
        # close and reset progress bar
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch), disable=not accelerator.is_local_main_process)
        loss_meter = AverageMeter()

        # Training ------
        model.train()
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                loss = model(**batch)
                accelerator.backward(loss)

                if accelerator.sync_gradients:

                    accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)

                    optimizer.step()  # gradient_state.sync_gradients check is performed inside optimizer.step
                    scheduler.step()
                    optimizer.zero_grad()

                loss_meter.update(loss.item())

            if accelerator.sync_gradients:
                progress_bar.set_description(
                    f"STEP: {current_iteration+1:5}/{num_update_steps_per_epoch:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"Loss: {loss_meter.avg:.4f}. "
                )

                progress_bar.update(1)
                current_iteration += 1

                if cfg.use_wandb:
                    accelerator.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)
                    accelerator.log({"lr": get_lr(optimizer)}, step=current_iteration)

            # >--------------------------------------------------|
            # >-- evaluation ------------------------------------|
            # >--------------------------------------------------|

            if (accelerator.sync_gradients) & (current_iteration % cfg.train_params.eval_frequency == 0):
                # set model in eval mode
                model.eval()
                scores_dict = run_evaluation(accelerator, model, valid_dl)
                lb = scores_dict["valid_loss"]

                print_line()
                et = as_minutes(time.time()-start_time)
                accelerator.print(
                    f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}"
                )
                print_line()
                accelerator.print(f">>> Current LB (valid_loss) = {round(lb, 4)}")

                print_line()

                is_best = False
                if lb <= best_lb:
                    best_lb = lb
                    is_best = True
                    patience_tracker = 0

                    # -----
                    best_dict = dict()
                    for k, v in scores_dict.items():
                        best_dict[f"{k}_at_best"] = v
                else:
                    patience_tracker += 1

                # saving -----
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                model_state = {
                    'step': current_iteration,
                    'epoch': epoch + 1,
                    'state_dict': unwrapped_model.state_dict(),
                    'lb': lb,
                }

                if accelerator.is_main_process:
                    save_checkpoint(cfg, model_state, is_best=is_best)

                # -- post eval
                model.train()
                torch.cuda.empty_cache()
                print_line()

                # early stopping ----
                if patience_tracker >= cfg.train_params.patience:
                    print("stopping early")
                    model.eval()
                    accelerator.end_training()
                    return


if __name__ == "__main__":
    run_training()
