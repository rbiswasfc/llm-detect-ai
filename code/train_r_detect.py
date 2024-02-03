import logging
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import partial

import bitsandbytes as bnb
import datasets
import hydra
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_kbit_training)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification,
                          BitsAndBytesConfig, get_cosine_schedule_with_warmup)

try:
    from r_detect.ai_dataset import AiDataset
    from r_detect.ai_loader import AiCollator, AiCollatorTrain, show_batch
    from r_detect.ai_model import (LlamaForDetectAI, MistralForDetectAI,
                                   PhiForDetectAI)
    from r_detect.ai_optimizer import get_optimizer
    from utils.metric_utils import compute_metrics
    from utils.train_utils import AverageMeter, as_minutes, get_lr


except Exception as e:
    print(e)
    raise ImportError

logger = get_logger(__name__)


def run_evaluation(accelerator, model, valid_dl, valid_ids):
    model.eval()

    all_predictions = []
    all_truths = []

    progress_bar = tqdm(range(len(valid_dl)), disable=not accelerator.is_local_main_process)

    for step, batch in enumerate(valid_dl):
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits

        predictions = torch.sigmoid(logits)
        predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"].to(torch.long).reshape(-1)))
        predictions, references = predictions.cpu().numpy().tolist(), references.cpu().numpy().tolist()

        all_predictions.extend(predictions)
        all_truths.extend(references)

        progress_bar.update(1)
    progress_bar.close()

    # compute metric
    eval_dict = compute_metrics(all_predictions, all_truths)

    result_df = pd.DataFrame()
    result_df["id"] = valid_ids
    result_df["predictions"] = all_predictions
    result_df["truths"] = all_truths

    oof_df = deepcopy(result_df)
    oof_df = oof_df.rename(columns={"predictions": "generated"})
    oof_df = oof_df[["id", "generated"]].copy()

    to_return = {
        "scores": eval_dict,
        "result_df": result_df,
        "oof_df": oof_df,
    }

    return to_return


@hydra.main(version_base=None, config_path="../conf/r_detect", config_name="conf_r_detect")
def run_training(cfg):
    # ------- Accelerator ---------------------------------------------------------------#
    if cfg.use_wandb:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train_params.gradient_accumulation_steps,
            log_with="wandb",
            # mixed_precision='fp16',
        )

        accelerator.init_trackers(
            cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train_params.gradient_accumulation_steps,
            # mixed_precision='fp16',
        )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # print_line = partial(print_line, accelerator)

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

    # ------- load data -----------------------------------------------------------------#
    print_line()
    data_dir = cfg.input_data_dir

    try:
        essay_df = pd.read_csv(os.path.join(data_dir, "train_essays.csv"))
    except Exception as e:
        essay_df = pd.read_parquet(os.path.join(data_dir, "train_essays.parquet"))

    essay_df = essay_df[~essay_df['text'].isna()].copy()
    essay_df = essay_df.reset_index(drop=True)

    # train_df = pd.read_parquet(os.path.join(data_dir, "train_essays.parquet"))
    # train_df = train_df[~train_df['text'].isna()].copy()

    # valid_df = pd.read_parquet(os.path.join(data_dir, "valid_essays.parquet"))
    # valid_df = valid_df[~valid_df['text'].isna()].copy()

    rng = random.Random(cfg.seed)
    essay_df['fold'] = essay_df['text'].apply(lambda x: 'train' if rng.random() < 0.99 else 'valid')
    train_df = essay_df[essay_df['fold'] == 'train'].copy()
    valid_df = essay_df[essay_df['fold'] == 'valid'].copy()

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    accelerator.print(f"shape of train data: {train_df.shape}")
    accelerator.print(f"{train_df.head()}")
    accelerator.print(f"shape of validation data: {valid_df.shape}")

    with accelerator.main_process_first():
        dataset_creator = AiDataset(cfg)

        train_ds = dataset_creator.get_dataset(train_df)
        valid_ds = dataset_creator.get_dataset(valid_df)

    tokenizer = dataset_creator.tokenizer

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
    valid_ids = valid_df["id"]  # .tolist()

    data_collator = AiCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=64
    )
    data_collator_train = AiCollatorTrain(
        tokenizer=tokenizer,
        pad_to_multiple_of=64,
        kwargs=dict(cfg=cfg)
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train_params.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator_train,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.train_params.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    accelerator.print("data preparation done...")
    print_line()

    # --- show batch -------------------------------------------------------------------#
    print_line()

    for b in train_dl:
        break
    show_batch(b, tokenizer, task='training', print_fn=accelerator.print)

    print_line()

    for b in valid_dl:
        break
    show_batch(b, tokenizer, task='training', print_fn=accelerator.print)

    # --- model -------------------------------------------------------------------------#
    print_line()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    if 'solar' in cfg.model.backbone_path.lower():
        base_model = LlamaForDetectAI.from_pretrained(
            cfg.model.backbone_path,
            num_labels=cfg.model.num_labels,  # 2
            quantization_config=bnb_config,
        )
    elif 'phi' in cfg.model.backbone_path.lower():
        base_model = PhiForDetectAI.from_pretrained(
            cfg.model.backbone_path,
            num_labels=cfg.model.num_labels,  # 2
            quantization_config=bnb_config,
            trust_remote_code=True,  # IMP
        )
    else:
        base_model = MistralForDetectAI.from_pretrained(
            cfg.model.backbone_path,
            num_labels=cfg.model.num_labels,  # 2
            quantization_config=bnb_config,
        )
        # base_model.peft_config = dict()

    base_model.config.pretraining_tp = 1
    # base_model.config.pad_token_id = tokenizer.pad_token_id

    # # base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    # for param in base_model.parameters():
    #     if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
    #         param.data = param.data.to(torch.float32)

    # lora ---
    peft_config = LoraConfig(
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.lora_alpha,
        lora_dropout=cfg.model.lora.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules=cfg_dict["model"]["lora"]["target_modules"],
        modules_to_save=cfg_dict["model"]["lora"]["modules_to_save"],
    )

    model = get_peft_model(base_model, peft_config)
    print(model.device)
    model.print_trainable_parameters()
    accelerator.wait_for_everyone()

    # --- optimizer ---------------------------------------------------------------------#
    print_line()
    optimizer = get_optimizer(cfg, model, print_fn=accelerator.print)

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

    # scheduler = accelerator.prepare(scheduler)

    # ------- training setup --------------------------------------------------------------#
    best_lb = -1.
    save_trigger = cfg.train_params.save_trigger

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
            with accelerator.accumulate(model):  # gives sync vs no sync context manager
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # Q: why need this check?
                    # A: gradient_state.sync_gradients check is NOT performed inside clip_grad_norm_
                    accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)

                    optimizer.step()  # gradient_state.sync_gradients check is performed inside optimizer.step
                    scheduler.step()
                    optimizer.zero_grad()

                # check if loss.item() is okay for TPU
                # happening on all processes - values of loss meter in different processes are different
                loss_meter.update(loss.item())  # tracks loss in each batch, no accumulation

            if accelerator.sync_gradients:
                progress_bar.set_description(
                    f"STEP: {current_iteration+1:5}/{num_training_steps:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"Loss: {loss_meter.avg:.4f}. "
                )

                progress_bar.update(1)
                current_iteration += 1

                if cfg.use_wandb:
                    accelerator.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)  # only on main process
                    accelerator.log({"lr": get_lr(optimizer)}, step=current_iteration)

            # >--------------------------------------------------|
            # >-- evaluation ------------------------------------|
            # >--------------------------------------------------|

            if (accelerator.sync_gradients) & (current_iteration % cfg.train_params.eval_frequency == 0):
                # set model in eval mode
                model.eval()
                eval_response = run_evaluation(accelerator, model, valid_dl, valid_ids)

                scores_dict = eval_response["scores"]
                result_df = eval_response["result_df"]
                oof_df = eval_response["oof_df"]

                lb = scores_dict["lb"]

                print_line()
                et = as_minutes(time.time()-start_time)
                accelerator.print(
                    f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}"
                )
                print_line()
                accelerator.print(f">>> Current LB (AUC) = {round(lb, 4)}")

                print_line()

                is_best = False
                if lb >= best_lb:
                    best_lb = lb
                    is_best = True
                    patience_tracker = 0

                    # -----
                    best_dict = dict()
                    for k, v in scores_dict.items():
                        best_dict[f"{k}_at_best"] = v
                else:
                    patience_tracker += 1

                if is_best:  # do in main process
                    oof_df.to_csv(os.path.join(cfg.outputs.model_dir, f"oof_df_best.csv"), index=False)
                    result_df.to_csv(os.path.join(cfg.outputs.model_dir, f"result_df_best.csv"), index=False)
                else:
                    accelerator.print(f">>> patience reached {patience_tracker}/{cfg.train_params.patience}")
                    accelerator.print(f">>> current best score: {round(best_lb, 4)}")

                oof_df.to_csv(os.path.join(cfg.outputs.model_dir, f"oof_df_last.csv"), index=False)
                result_df.to_csv(os.path.join(cfg.outputs.model_dir, f"result_df_last.csv"), index=False)

                # saving -----
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)

                # # debug --
                # selected_adapters = list(unwrapped_model.peft_config.keys())
                # accelerator.print(f"selected adapters: {selected_adapters}")
                # for adapter_name in selected_adapters:
                #     peft_config = unwrapped_model.peft_config[adapter_name]
                #     peft_config = asdict(peft_config)
                #     accelerator.print(f"adapter: {adapter_name}")
                #     accelerator.print(peft_config)
                #     for k, v in peft_config.items():
                #         accelerator.print(f"{k}: {v} ({type(v)})")
                # # ------
                unwrapped_model.save_pretrained(
                    f"{cfg.outputs.model_dir}/last",
                    state_dict=accelerator.get_state_dict(model),
                    save_function=accelerator.save,
                )

                if accelerator.is_main_process:
                    tokenizer.save_pretrained(f"{cfg.outputs.model_dir}/last")

                if best_lb > save_trigger:
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(f"{cfg.outputs.model_dir}/best")
                    unwrapped_model.save_pretrained(
                        f"{cfg.outputs.model_dir}/best",
                        state_dict=accelerator.get_state_dict(model),
                        save_function=accelerator.save,
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(f"{cfg.outputs.model_dir}/best")

                # logging ----
                if cfg.use_wandb:
                    accelerator.log({"lb": lb}, step=current_iteration)
                    accelerator.log({"best_lb": best_lb}, step=current_iteration)

                    # -- log scores dict
                    for k, v in scores_dict.items():
                        accelerator.log({k: round(v, 4)}, step=current_iteration)

                    # --- log best scores dict
                    for k, v in best_dict.items():
                        accelerator.log({k: round(v, 4)}, step=current_iteration)

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

    # --- end training
    accelerator.end_training()


if __name__ == "__main__":
    run_training()
