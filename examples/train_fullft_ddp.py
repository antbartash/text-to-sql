"""
SFT training script for multi-GPU DDP.

Launch with:
    torchrun --nproc_per_node=NUM_GPUS sft_ddp_train.py [--args]

Example:
    torchrun --nproc_per_node=4 train.py \
        --output_dir ./sft-output \
        --run_name my-run \
        --epochs 1 \
        --lr 1e-4

Hardware config flags (each defaults to the value from get_optimal_training_config()):
    --fp16          Force fp16 on  (mutually exclusive with --bf16)
    --no_fp16       Force fp16 off
    --bf16          Force bf16 on  (mutually exclusive with --fp16)
    --no_bf16       Force bf16 off
    --tf32          Force tf32 on
    --no_tf32       Force tf32 off
    --attn_impl     Override attention implementation (e.g. "flash_attention_2", "sdpa", "eager")
    --liger_kernel  Force Liger kernel on
    --no_liger_kernel Force Liger kernel off
"""

import argparse
import gc
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
import transformers
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer
import bitsandbytes as bnb
import _config
from utils import get_optimal_training_config, get_vm_usage_metrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT DDP")

    # Model
    parser.add_argument("--checkpoint", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable thinking mode in chat template (Qwen3 specific)")

    # Data
    parser.add_argument("--dataset", type=str, default="gretelai/synthetic_text_to_sql")
    parser.add_argument("--train_size", type=int, default=97500)
    parser.add_argument("--valid_size", type=int, default=2500)
    parser.add_argument("--valid_split_ratio", type=float, default=0.025)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=256)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--effective_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="nadam")
    parser.add_argument("--betas", type=float, nargs=2, default=[0.95, 0.999],
                        metavar=("BETA1", "BETA2"))
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--neftune_noise_alpha", type=float, default=0.0,
                        help="NEFTune noise alpha; 0 disables it")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--num_evals_per_run", type=int, default=8,
                        help="Target number of evals spread evenly across training")
    parser.add_argument("--seed", type=int, default=42)

    # Checkpointing & output
    parser.add_argument("--output_dir", type=str, default="./sft-output")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--resume_training", action="store_true",
                        help="Resume from the latest checkpoint in output_dir")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="W&B run ID to resume a previous run (requires --resume_training)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name; defaults to a timestamped name")

    # ---------------------------------------------------------------------------
    # Hardware config overrides
    # Each flag defaults to None, meaning "use whatever get_optimal_training_config()
    # returns".  Pass the flag explicitly to override that auto-detected value.
    # ---------------------------------------------------------------------------
    hw = parser.add_argument_group(
        "hardware config",
        "Override auto-detected hardware settings from get_optimal_training_config(). "
        "If a flag is not provided, the auto-detected default is used.",
    )
    # fp16: --fp16 forces True, --no_fp16 forces False, omitting leaves it to auto-detect
    hw.add_argument("--fp16", dest="fp16", action="store_true", default=None,
                    help="Force fp16 training on")
    hw.add_argument("--no_fp16", dest="fp16", action="store_false",
                    help="Force fp16 training off")
    # bf16
    hw.add_argument("--bf16", dest="bf16", action="store_true", default=None,
                    help="Force bf16 training on")
    hw.add_argument("--no_bf16", dest="bf16", action="store_false",
                    help="Force bf16 training off")
    # tf32
    hw.add_argument("--tf32", dest="tf32", action="store_true", default=None,
                    help="Force TF32 matmul on")
    hw.add_argument("--no_tf32", dest="tf32", action="store_false",
                    help="Force TF32 matmul off")
    # liger_kernel
    hw.add_argument("--liger_kernel", dest="liger_kernel", action="store_true", default=None,
                    help="Force Liger kernel on")
    hw.add_argument("--no_liger_kernel", dest="liger_kernel", action="store_false",
                    help="Force Liger kernel off")
    # attn_impl: string, None means "use whatever the auto-detect returns (or model default)"
    hw.add_argument("--attn_impl", type=str, default=None,
                    choices=["flash_attention_3", "flash_attention_2", "sdpa", "eager"],
                    help="Attention implementation; if omitted, uses auto-detected value")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hardware config resolution
# ---------------------------------------------------------------------------

def resolve_hw_config(args: argparse.Namespace) -> dict:
    """
    Merge auto-detected hardware config with any explicit CLI overrides.

    get_optimal_training_config() provides the baseline; any arg that was
    explicitly set on the command line (i.e. is not None) wins over it.
    """
    auto = get_optimal_training_config()

    def override(key: str, cli_value):
        """Return cli_value if it was explicitly set, else fall back to auto-detect."""
        return cli_value if cli_value is not None else auto.get(key, False)

    hw = {
        "fp16":         override("fp16",         args.fp16),
        "bf16":         override("bf16",         args.bf16),
        "tf32":         override("tf32",         args.tf32),
        "liger_kernel": override("liger_kernel", args.liger_kernel),
        # attn_impl: auto-detect may or may not include it; CLI always wins if set
        "attn_impl":    args.attn_impl if args.attn_impl is not None else auto.get("attn_impl", None),
    }

    if hw["fp16"] and hw["bf16"]:
        raise ValueError("--fp16 and --bf16 are mutually exclusive.")

    return hw


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(args: argparse.Namespace):
    ds = load_dataset(args.dataset, streaming=False)
    ds_train_raw, ds_test = ds["train"], ds["test"]

    split = ds_train_raw.train_test_split(test_size=args.valid_split_ratio, seed=args.seed)
    ds_train = split["train"].select(range(min(args.train_size, len(split["train"]))))
    ds_valid = split["test"].select(range(min(args.valid_size, len(split["test"]))))

    return ds_train, ds_valid, ds_test


def construct_message_with_assistant_content(example: dict) -> dict:
    """Format a dataset example into chat messages for SFT (includes assistant turn)."""
    messages = [
        {
            "role": "system",
            "content": (
                f"The user asks a question. Your task is to generate the SQL query to answer "
                f"that question. Return SQL query only. The context of the question is the "
                f"following: '{example['sql_context']}'"
            ),
        },
        {"role": "user", "content": example["sql_prompt"]},
        {"role": "assistant", "content": example["sql"]},
    ]
    return {"messages": messages}


def preprocess_datasets(ds_train, ds_valid):
    ds_train = ds_train.map(construct_message_with_assistant_content)
    ds_valid = ds_valid.map(construct_message_with_assistant_content)
    return ds_train, ds_valid


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    checkpoint: str,
    attn_impl: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # NOTE: do NOT use device_map="auto" with DDP.
    model_kwargs = {"dtype": dtype} # SHOULD NOT BE SPECIFIED WITH OLDER GPUS
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def make_formatting_func(tokenizer, enable_thinking: bool):
    """Return a formatting function closed over the tokenizer and thinking flag."""
    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,  # no generation prompt during training
            enable_thinking=enable_thinking,
        )
    return formatting_func


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

BNB_OPTIMIZER_MAP = {
    "adam":   bnb.optim.Adam8bit,
    "adamw":  bnb.optim.AdamW8bit,
}

# NAdam has no bnb 8-bit variant — fall back to paged AdamW
FALLBACK_MAP = {
    "nadam": bnb.optim.PagedAdamW8bit,
}

def build_optimizer(model, args: argparse.Namespace):
    name = args.optimizer.lower()

    if name in BNB_OPTIMIZER_MAP:
        optimizer_class = BNB_OPTIMIZER_MAP[name]
    elif name in FALLBACK_MAP:
        print(f"[warn] No 8-bit variant for '{name}', falling back to PagedAdamW8bit")
        optimizer_class = FALLBACK_MAP[name]
    else:
        raise ValueError(f"Unknown optimizer: '{name}'. Choose from: {list(BNB_OPTIMIZER_MAP) + list(FALLBACK_MAP)}")

    return optimizer_class(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        eps=args.epsilon,
    )


# ---------------------------------------------------------------------------
# Step / eval schedule helpers
# ---------------------------------------------------------------------------

def compute_steps(dataset_size: int, args: argparse.Namespace, num_gpus: int = 1) -> dict:
    """
    Compute gradient accumulation steps, eval cadence, etc.
    """
    grad_accum = max(1, args.effective_batch_size // (args.per_device_train_batch_size * num_gpus))
    steps_per_epoch = max(1, dataset_size // (args.per_device_train_batch_size * num_gpus * grad_accum))
    total_steps = steps_per_epoch * args.epochs
    eval_steps = max(1, min(total_steps, total_steps // args.num_evals_per_run))
    warmup_steps = int(total_steps * args.warmup_ratio)
    return {
        "grad_accum": grad_accum,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "eval_steps": eval_steps,
        "warmup_steps": warmup_steps,
    }


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def init_wandb(args: argparse.Namespace, run_name: str) -> None:
    """Initialize W&B. Only called on rank 0."""
    wandb_kwargs = dict(project=os.environ["WANDB_PROJECT"], name=run_name)
    if args.resume_training and args.wandb_run_id:
        wandb_kwargs.update(id=args.wandb_run_id, resume="allow")
    wandb.init(**wandb_kwargs)


def log_eval_metrics_to_wandb(trainer, args: argparse.Namespace) -> None:
    """Log extra hyperparameter context alongside eval metrics. Only called on rank 0."""
    for log in trainer.state.log_history:
        if "eval_loss" in log:
            wandb.log({
                "eval_loss":            log["eval_loss"],
                "eval_perplexity":      math.exp(log["eval_loss"]),
                "step":                 log["step"],
                "learning_rate":        args.lr,
                "lr_scheduler_type":    args.lr_scheduler,
                "weight_decay":         args.weight_decay,
                "neftune_noise_alpha":  args.neftune_noise_alpha,
                "betas":                tuple(args.betas),
                "warmup_ratio":         args.warmup_ratio,
                "effective_batch_size": args.effective_batch_size,
                "optimizer":            args.optimizer,
                "epsilon":              args.epsilon,
                "seed":                 args.seed,
            })


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_training_summary(args, steps: dict, hw: dict) -> None:
    print("===== Training Setup Summary =====")
    print(f"Num epochs:            {args.epochs}")
    print(f"Effective batch size:  {args.effective_batch_size}")
    print(f"Per-device train batch size: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {steps['grad_accum']}")
    print(f"LR scheduler:          {args.lr_scheduler}")
    print(f"Epsilon:               {args.epsilon}")
    print(f"Steps per epoch:       {steps['steps_per_epoch']}  (single-GPU estimate)")
    print(f"Total training steps:  {steps['total_steps']}  (single-GPU estimate)")
    print(f"Warmup steps:          {steps['warmup_steps']}")
    print(f"Eval steps:            {steps['eval_steps']}  (target {args.num_evals_per_run} evals/run)")
    print(f"Logging steps:         1")
    print("----- Hardware config -----")
    print(f"fp16:                  {hw['fp16']}")
    print(f"bf16:                  {hw['bf16']}")
    print(f"tf32:                  {hw['tf32']}")
    print(f"liger_kernel:          {hw['liger_kernel']}")
    print(f"attn_impl:             {hw['attn_impl'] or '(model default)'}")
    print("===================================")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- env setup ---
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["WANDB_API_KEY"] = _config.WANDB_API_KEY
    os.environ["WANDB_PROJECT"] = _config.WANDB_PROJECT

    # --- GPU / precision config (auto-detect, then apply CLI overrides) ---
    hw = resolve_hw_config(args)

    if hw["tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- seed ---
    set_seed(args.seed)

    # --- rank detection (set by torchrun) ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    # --- run name ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name or f"sft-ddp-{timestamp}"

    # --- W&B (main process only; Trainer suppresses it on other ranks automatically) ---
    if is_main:
        init_wandb(args, run_name)
        get_vm_usage_metrics()

    ds_train, ds_valid, _ = load_data(args)
    ds_train, ds_valid = preprocess_datasets(ds_train, ds_valid)
    dtype = torch.bfloat16 if hw["bf16"] else torch.float16 if hw["fp16"] else torch.float32
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, attn_impl=hw["attn_impl"], dtype=dtype)
    formatting_func = make_formatting_func(tokenizer, args.enable_thinking)
    steps = compute_steps(len(ds_train), args, num_gpus=num_gpus)

    if is_main:
        print_training_summary(args, steps, hw)

    # --- training args ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=steps["grad_accum"],
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        neftune_noise_alpha=args.neftune_noise_alpha if args.neftune_noise_alpha > 0 else None,
        max_grad_norm=args.max_grad_norm,

        use_liger_kernel=hw["liger_kernel"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        bf16=hw["bf16"],
        fp16=hw["fp16"],
        bf16_full_eval=hw["bf16"],
        fp16_full_eval=hw["fp16"],
        tf32=hw["tf32"],

        save_strategy="steps",
        save_steps=steps["eval_steps"],
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=steps["eval_steps"],
        logging_strategy="steps",
        logging_steps=1,

        report_to=["wandb"],
        run_name=run_name,

        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        seed=args.seed,
    )

    # --- trainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        formatting_func=formatting_func,
        args=training_args,
        optimizers=(build_optimizer(model, args), None),  # (optimizer, scheduler)
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    # --- train ---
    if is_main:
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    last_checkpoint = None
    if args.resume_training and os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)

    if last_checkpoint is not None:
        if is_main:
            print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        if is_main:
            print("Starting fresh training run")
        trainer.train()

    if is_main:
        print(f"End time: {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    # --- save best model (main process only) ---
    if is_main:
        model_path = os.path.join(args.output_dir, "best_model")
        trainer.save_model(model_path)
        print(f"Model saved to {model_path}")

    # --- extra W&B logging (main process only) ---
    if is_main:
        log_eval_metrics_to_wandb(trainer, args)
        wandb.finish()

    # --- cleanup ---
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()