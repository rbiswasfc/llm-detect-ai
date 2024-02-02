import argparse
import json
import os
import random
import string

import pandas as pd
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig)


def generate_random_string():
    chars = string.ascii_lowercase + string.digits
    return 'e_' + ''.join(random.choice(chars) for _ in range(8))


def get_instruction(inputs):
    ret = f"""
Prompt: {inputs['prompt_name']}
Task: {inputs['task']}
Score: {inputs['holistic_essay_score']}
Student Grade Level: {inputs['grade_level']}
English Language Learner: {inputs['ell_status']}
Disability Status: {inputs['student_disability_status']}
    """.strip()
    n_chars = random.randint(64, 128)

    start = inputs['text'][:n_chars]

    ret = f"### Instruction:\n{ret}\n\n### Response: {start}"
    return ret


def get_inputs(prompt, tokenizer, n=1):
    return tokenizer([prompt]*n, return_tensors="pt")


def process_response(texts):
    ret = []

    for text in texts:
        if "</s>" in text:
            text = text.split("### Response:")[-1].split("</s>")[0].strip()
        else:
            text = text.split("### Response:")[-1].split("<|endoftext|>")[0].strip()
        text = text.replace("<unk>", "")
        ret.append(text)
    return ret


def pre_process_essay(essay_df):

    essay_df = essay_df[~essay_df['text'].isna()].copy()
    essay_df = essay_df.reset_index(drop=True)

    essay_df["student_disability_status"] = essay_df["student_disability_status"].fillna("Unknown")
    essay_df["ell_status"] = essay_df["ell_status"].fillna("Unknown")
    essay_df["grade_level"] = essay_df["grade_level"].fillna(-1)
    essay_df["holistic_essay_score"] = essay_df["holistic_essay_score"].fillna(-1)

    essay_df["prompt"] = essay_df.apply(get_instruction, axis=1)
    return essay_df


def generate(cfg):
    accelerator = Accelerator()

    essay_df = pd.read_csv(cfg.input_data_path).rename(columns={"full_text": "text"})
    essay_df = pre_process_essay(essay_df)

    prompts = essay_df["prompt"].values.tolist()
    # prompts = [p for p in prompts if "Task: Text dependent" in p]

    # prompts = [p for p in prompts if "car-free" not in p.lower()]
    # prompts = [p for p in prompts if "facial action" not in p.lower()]
    # prompts = [p for p in prompts if "electoral" not in p.lower()]

    print(f"Number of prompts: {len(prompts)}")

    # model & tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        use_fast=True,
        padding_side="left",
        truncation_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16,
    )

    model = accelerator.prepare(model)
    model.eval()

    n_examples = cfg.n_examples
    n_gen_per_prompt = cfg.n_gen_per_prompt
    output_dir = cfg.output_dir

    progress_bar = tqdm(range(n_examples))

    for i in range(n_examples):
        # print(f"---- Example {i+1}/{n_examples} ------")
        temperature = 1.5  # + 0.75 * random.random()
        top_k = 512  # random.randint(4, 8)
        penalty_alpha = 0.5  # random.random()
        guidance_scale = 1.1  # + 0.5 * random.random()
        eta_cutoff = 1e-4 + 5e-4 * random.random()
        repetition_penalty = 1.2  # 1.0 + 0.2 * random.random()

        try:
            generation_config = GenerationConfig.from_pretrained(
                cfg.model_path,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                penalty_alpha=penalty_alpha,
                guidance_scale=guidance_scale,
                max_new_tokens=cfg.max_num_tokens,
                pad_token_id=tokenizer.pad_token_id,
                # eta_cutoff=eta_cutoff,
                # repetition_penalty=repetition_penalty,
            )
        except Exception as e:
            print(e)
            generation_config = GenerationConfig(
                # cfg.base_model_path,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                penalty_alpha=penalty_alpha,
                guidance_scale=guidance_scale,
                max_new_tokens=cfg.max_num_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eta_cutoff=eta_cutoff,
            )

        try:
            prompt = random.choice(prompts)
            this_example = dict()
            this_id = generate_random_string()
            this_example['id'] = this_id
            this_example['prompt'] = prompt
            this_example['temperature'] = temperature
            this_example['top_k'] = top_k
            this_example['guidance_scale'] = guidance_scale
            this_example['penalty_alpha'] = penalty_alpha
            # this_example['typical_p'] = typical_p

            inputs = get_inputs(prompt, tokenizer, n=n_gen_per_prompt)
            device = accelerator.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.module.generate(**inputs, generation_config=generation_config)
            output = tokenizer.batch_decode(output)

            output = process_response(output)
            this_example['responses'] = output

            with open(f"{output_dir}/{this_id}.json", "w") as f:
                json.dump(this_example, f)

        except Exception as e:
            print(e)
        progress_bar.update(1)
    progress_bar.close()


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('--config_path', type=str, required=True)

    args = ap.parse_args()
    cfg = OmegaConf.load(args.config_path)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # execution
    generate(cfg)
