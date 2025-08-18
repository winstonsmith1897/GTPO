# train_model.py
# Single entrypoint: reads everything from YAML but allows CLI overrides (model/dataset/etc).

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastLanguageModel, PatchFastRL
import argparse
import importlib
import random
import re
import warnings
from typing import Dict

import numpy as np
import torch
import yaml
from datasets import concatenate_datasets, load_dataset
from trl import GRPOConfig

# =============== UTILITIES ===============

def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dget(cfg: Dict, path: str, default=None):
    """Nested dict get using dotted path (e.g., 'a.b.c')."""
    cur = cfg
    for p in path.split('.'):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def import_object(dotted_path: str):
    """
    Import an object by string path.
    Supports 'pkg.module:attr' and 'pkg.module.attr'.
    """
    if ":" in dotted_path:
        module_path, attr = dotted_path.split(":", 1)
    else:
        module_path, attr = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def parse_num_generations(val):
    """
    Accepts an int or strings like '8/12'. If multiple values are present,
    uses the first number and emits a warning.
    """
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        parts = re.split(r"[\/,; ]+", val.strip())
        for p in parts:
            if p.isdigit():
                if len(parts) > 1:
                    warnings.warn(f"num_generations '{val}' contains multiple values. Using {p}.")
                return int(p)
    raise ValueError(f"Invalid num_generations: {val}")


def ensure_list(x):
    """Return x if already a list; wrap into a single-item list if not; [] if None."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


# =============== DATASET FORMATTERS ===============

def format_gsm8k(system_instruction: str):
    """
    GSM8K: Prefer the final line after '####' as the answer; the rest is reasoning.
    Fallbacks: split on '###' or last token as last resort.
    """
    def _fmt(sample):
        question = sample["question"]
        full = sample["answer"]

        m = re.search(r"####\s*(.+)\s*$", full, flags=re.MULTILINE)
        if m:
            ans_only = m.group(1).strip()
            reasoning = full[: m.start()].strip()
        else:
            if "###" in full:
                reasoning, tail = full.rsplit("###", 1)
                ans_only = tail.strip()
            else:
                tokens = full.strip().split()
                ans_only = tokens[-1].strip(".")
                reasoning = full

        formatted_output = (
            f"<reasoning>\n{reasoning}\n</reasoning>\n"
            f"<answer>{ans_only}</answer>"
        )
        return {
            "prompt": [
                {"role": "system", "content": system_instruction},
                {"role": "user",   "content": question},
            ],
            "answer": formatted_output,
        }
    return _fmt


def format_hendrycks_math(system_instruction: str):
    """
    Hendrycks MATH: Use the last \\boxed{...} as the answer; everything before is reasoning.
    Fallback: last token as the answer.
    """
    def _fmt(sample):
        question = sample["problem"]
        solution_full = sample["solution"]

        boxed = None
        for m in re.finditer(r"\\boxed\{([^}]*)\}", solution_full, flags=re.DOTALL):
            boxed = m  # keep the last match
        if boxed:
            ans_only = boxed.group(1).strip()
            reasoning = solution_full[: boxed.start()].strip()
        else:
            ans_only = solution_full.strip().split()[-1].strip(".")
            reasoning = solution_full

        formatted_output = (
            f"<reasoning>\n{reasoning}\n</reasoning>\n"
            f"<answer>{ans_only}</answer>"
        )
        return {
            "prompt": [
                {"role": "system", "content": system_instruction},
                {"role": "user",   "content": question},
            ],
            "answer": formatted_output,
        }
    return _fmt


# =============== DATASET LOADER ===============

def load_and_prepare_dataset(cfg: Dict, tokenizer):
    """
    Load, format, and filter the dataset according to the YAML/CLI config.
    Returns a shuffled HuggingFace Dataset with 'prompt' and 'answer' columns.
    """
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"].lower()
    system_instruction = ds_cfg.get(
        "system_instruction",
        "You are a helpful assistant for solving problems. "
        "Given a question, first think step by step between <reasoning> and </reasoning>. "
        "Then, give the final answer between <answer> and </answer>."
    )
    max_prompt_length = cfg["training"]["max_prompt_length"]

    if name == "gsm8k":
        split = ds_cfg.get("split", "train")
        subset = ds_cfg.get("subset", "main")
        raw = load_dataset("gsm8k", subset)[split]
        fmt = format_gsm8k(system_instruction)
        formatted = raw.map(fmt, desc="Formatting GSM8K")
        keep_cols = {"prompt", "answer"}
        drop_cols = [c for c in formatted.column_names if c not in keep_cols]
        formatted = formatted.remove_columns(drop_cols)

    elif name in ("math", "hendrycks_math", "eleutherai/hendrycks_math"):
        subsets = ds_cfg.get("subsets", [
            "algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
        ])
        fmt = format_hendrycks_math(system_instruction)
        splits = []
        for sub in subsets:
            raw = load_dataset("EleutherAI/hendrycks_math", sub)["train"]
            f = raw.map(fmt, desc=f"Formatting {sub}")
            keep_cols = {"prompt", "answer"}
            drop_cols = [c for c in f.column_names if c not in keep_cols]
            f = f.remove_columns(drop_cols)
            splits.append(f)
        formatted = concatenate_datasets(splits)

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    def is_valid_prompt(example):
        full = "".join(msg["content"] for msg in example["prompt"])
        return len(tokenizer(full)["input_ids"]) <= max_prompt_length

    filtered = formatted.filter(is_valid_prompt, desc="Filtering long prompts")
    print(f"[INFO] DATASET: {name} — SIZE AFTER FILTER: {len(filtered)}")
    if len(filtered) > 0:
        print("[INFO] SAMPLE:\n", filtered[0])
    seed = int(cfg["model"].get("random_seed", 42))
    return filtered.shuffle(seed=seed)


# =============== MODEL LOADER ===============

def load_model_and_tokenizer(cfg: Dict):
    """
    Load base model and tokenizer with Unsloth + LoRA configuration.
    """
    mcfg = cfg["model"]
    model_name = mcfg["model_name"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = int(mcfg["max_seq_length"]),
        load_in_4bit   = bool(mcfg.get("load_in_4bit", True)),
        fast_inference = True,
        max_lora_rank  = int(mcfg["lora_rank"]),
        gpu_memory_utilization = float(mcfg.get("gpu_memory_utilization", 0.5)),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = int(mcfg["lora_rank"]),
        target_modules = mcfg["target_modules"],
        lora_alpha = int(mcfg["lora_rank"]),
        use_gradient_checkpointing = "unsloth",
        random_state = int(mcfg.get("random_seed", 42)),
    )
    return model, tokenizer


# =============== TRAINING ===============

def build_output_dir(cfg: Dict):
    """Construct a descriptive output directory name from config fields."""
    t = cfg["training"]
    base = cfg.get("run_name", "GTPO_RUN")
    parts = [
        base,
        f"adam1-{t['adam_beta1']}",
        f"adam2-{t['adam_beta2']}",
        f"wd-{t['weight_decay']}",
        f"gen{t['num_generations']}",
        f"acc-{t['gradient_accumulation_steps']}",
        f"wu{t.get('warmup_ratio', 0.05)}",
        f"len{cfg['model']['max_seq_length']}",
        f"lr{t['learning_rate']}",
        f"BetaKL{t.get('beta', 0.0)}",
    ]
    return "_".join(str(p) for p in parts)


def train(cfg: Dict):
    """
    End-to-end training:
    - set env vars
    - patch Unsloth RL
    - load model/tokenizer
    - load/format dataset
    - build trainer + args
    - train and save LoRA adapters
    """
    # Optional environment variables from YAML
    # if "env" in cfg:
    #     for k, v in cfg["env"].items():
    #         if v is not None:
    #             os.environ[str(k)] = str(v)

    # Patch RL for Unsloth
    PatchFastRL("GRPO", FastLanguageModel)

    # Seeds
    set_seeds(int(cfg["model"].get("random_seed", 42)))

    # Model & dataset
    model, tokenizer = load_model_and_tokenizer(cfg)
    dataset = load_and_prepare_dataset(cfg, tokenizer)

    # Trainer class
    tr_cfg = cfg["trainer"]
    trainer_cls_path = tr_cfg.get("class_path", "gtpo.gtpo_training.UnslothGTPOTrainer")
    GTPOTrainer = import_object(trainer_cls_path)

    # Reward functions (can be 1 or many)
    reward_paths = ensure_list(tr_cfg.get("reward_funcs", "reward_math.final_reward"))
    reward_funcs = [import_object(p) for p in reward_paths]

    # Training args
    t = cfg["training"]
    num_generations = parse_num_generations(t["num_generations"])
    max_prompt_length = int(t["max_prompt_length"])
    max_completion_length = int(cfg["model"]["max_seq_length"]) - max_prompt_length

    output_dir = build_output_dir(cfg)

    training_args = GRPOConfig(
        learning_rate = float(t["learning_rate"]),
        adam_beta1 = float(t["adam_beta1"]),
        adam_beta2 = float(t["adam_beta2"]),
        weight_decay = float(t["weight_decay"]),
        beta = float(t.get("beta", 0.0)),
        warmup_ratio = float(t.get("warmup_ratio", 0.05)),
        lr_scheduler_type = t["lr_scheduler_type"],
        optim = t["optimizer"],
        logging_steps = int(t["logging_steps"]),
        per_device_train_batch_size = int(t["per_device_train_batch_size"]),
        gradient_accumulation_steps = int(t["gradient_accumulation_steps"]),
        num_generations = int(num_generations),
        max_prompt_length = int(max_prompt_length),
        max_completion_length = int(max_completion_length),
        num_train_epochs = int(t["num_train_epochs"]),
        save_steps = int(t["save_steps"]),
        max_grad_norm = float(t["max_grad_norm"]),
        report_to = ensure_list(t.get("report_to", [])),
        output_dir = output_dir,
        log_completions = True,
    )

    # Init trainer
    trainer = GTPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = reward_funcs,
        args = training_args,
        train_dataset = dataset,
    )

    print("\n\n=========== START TRAINING =========\n\n")
    trainer.train()

    print(f"\n[INFO] Saving LoRA to: {output_dir}\n")
    model.save_lora(output_dir)


# =============== CLI ===============

def parse_args():
    """Parse CLI arguments for YAML path and lightweight overrides."""
    p = argparse.ArgumentParser(description="GTPO training from YAML with CLI overrides.")
    p.add_argument("config", nargs="?", default="config.yaml", help="Path to YAML file (default: config.yaml)")

    # Requested overrides
    p.add_argument("--model-name", type=str, help="Override model name (e.g., Qwen/Qwen2.5-3B-Instruct)")
    p.add_argument("--dataset-name", type=str, help="Override dataset (gsm8k | hendrycks_math | math)")

    # Useful optional overrides
    p.add_argument("--dataset-subset", action="append",
                   help="Dataset subset. Repeatable. (e.g., --dataset-subset algebra)")
    p.add_argument("--dataset-split", type=str, help="Dataset split (e.g., train, validation, test)")
    p.add_argument("--run-name", type=str, help="Run name for output_dir")
    p.add_argument("--cuda", type=str, help="Value for CUDA_VISIBLE_DEVICES")
    return p.parse_args()


def normalize_sections(cfg: Dict):
    """
    Ensure the YAML has the expected top-level sections:
    model / training / dataset / trainer.
    If keys are at root (legacy), migrate them into the right section.
    """
    flat_keys = set(cfg.keys())
    expected_sections = {"model", "training", "dataset", "trainer"}
    if not (expected_sections <= flat_keys):
        cfg.setdefault("model", {})
        cfg.setdefault("training", {})
        cfg.setdefault("dataset", {})
        cfg.setdefault("trainer", {})

        # Migrate known model keys
        for k in [
            "model_name","max_seq_length","max_prompt_length","lora_rank",
            "load_in_4bit","gpu_memory_utilization","target_modules","random_seed"
        ]:
            if k in cfg:
                cfg["model"][k] = cfg.pop(k)

        # Migrate known training keys
        for k in [
            "warmup_ratio","learning_rate","adam_beta1","adam_beta2","weight_decay",
            "beta","lr_scheduler_type","optimizer","logging_steps",
            "per_device_train_batch_size","gradient_accumulation_steps",
            "num_generations","num_train_epochs","num_iterations","save_steps",
            "max_grad_norm","report_to"
        ]:
            if k in cfg:
                cfg["training"][k] = cfg.pop(k)

        # Optional CUDA env migration
        if "CUDA_VISIBLE_DEVICES" in cfg:
            cfg.setdefault("env", {})["CUDA_VISIBLE_DEVICES"] = cfg.pop("CUDA_VISIBLE_DEVICES")
    return cfg


def apply_cli_overrides(cfg: Dict, args):
    """Apply model/dataset overrides from CLI on top of the YAML."""
    # Model override
    if args.model_name:
        cfg.setdefault("model", {})["model_name"] = args.model_name

    # Dataset overrides
    if args.dataset_name:
        name = args.dataset_name.strip().lower()
        # Common aliases
        if name in ("math", "hendrycks_math", "eleutherai/hendrycks_math"):
            name = "hendrycks_math"
        elif name in ("gsm8k",):
            name = "gsm8k"
        else:
            raise ValueError(f"Unrecognized dataset: {args.dataset_name}")
        cfg.setdefault("dataset", {})["name"] = name

    if args.dataset_subset:
        # For gsm8k: use the first subset as 'subset'
        # For hendrycks_math: use the full list as 'subsets'
        dsname = cfg.get("dataset", {}).get("name", "").lower()
        if dsname == "gsm8k":
            cfg["dataset"]["subset"] = args.dataset_subset[0]
        else:
            cfg["dataset"]["subsets"] = args.dataset_subset

    if args.dataset_split:
        cfg.setdefault("dataset", {})["split"] = args.dataset_split

    if args.run_name:
        cfg["run_name"] = args.run_name

    if args.cuda:
        cfg.setdefault("env", {})["CUDA_VISIBLE_DEVICES"] = args.cuda

    return cfg


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = normalize_sections(cfg)
    cfg = apply_cli_overrides(cfg, args)

    # Minimal sanity check
    for key in ("model", "training", "dataset", "trainer"):
        if key not in cfg:
            raise ValueError(f"Missing section in YAML: '{key}'")

    train(cfg)


if __name__ == "__main__":
    main()
