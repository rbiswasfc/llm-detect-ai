import math
import os
import random
import shutil
import string
from copy import deepcopy

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlInit)


def generate_random_string():
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(6))


def get_desired_dtype(dtype):
    if dtype == 'fp16':
        return torch.float16
    elif dtype == 'bf16':
        return torch.bfloat16
    else:
        return torch.float32


def print_line(logger=None):
    prefix, unit, suffix = "#", "~~", "#"
    if logger is None:
        print(prefix + unit*50 + suffix)
    else:
        logger.print(prefix + unit*50 + suffix)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm%ds' % (m, s)


def execution_setup(cfg):
    print_line()
    if cfg.use_random_seed:
        seed = random.randint(401, 999)
        cfg.seed = seed

    print(f"setting seed: {cfg.seed}")
    seed_everything(cfg.seed)

    # folder ---
    os.makedirs(cfg.outputs.model_dir, exist_ok=True)

    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def init_wandb(cfg):
    project = cfg.wandb.project
    tags = cfg.wandb.tags

    if cfg.wandb.all_data_flag:
        run_id = f"{cfg.wandb.run_name}-all-data"
    else:
        run_id = f"{cfg.wandb.run_name}"

    run = wandb.init(
        project=project,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=tags,
        name=run_id,
        anonymous="must",
        job_type="Train",
    )

    return run


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']*1e6


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(cfg, state, is_best):
    os.makedirs(cfg.outputs.model_dir, exist_ok=True)
    name = f"detect_ai_model"

    filename = f'{cfg.outputs.model_dir}/{name}_last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)

    if is_best:
        shutil.copyfile(filename, f'{cfg.outputs.model_dir}/{name}_best.pth.tar')


class EMA():
    """
    credit: https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332567
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
