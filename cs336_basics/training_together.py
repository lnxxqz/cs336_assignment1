"""训练入口：uv run python -m cs336_basics.training_together。"""

import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 同时支持直接运行脚本和 python -m。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_basics.adamw import adamw
from cs336_basics.config import CONFIG, Config
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.get_batch import get_batch
from cs336_basics.get_lr_cosine_schedule import get_lr_cosine_schedule
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.tokenizer import tokenizer
from cs336_basics.train_bpe import run_train_bpe
from cs336_basics.transformer_lm import transformer


def get_device(config: Config) -> str:
    if config.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return config.device


def validate_config(config: Config) -> None:
    if config.init_mode not in {"scratch", "resume", "weights"}:
        raise ValueError("init_mode 必须是 scratch、resume 或 weights")
    if config.init_mode != "scratch" and config.init_checkpoint is None:
        raise ValueError("加载模型时必须填写 init_checkpoint")
    for name in ("total_steps", "batch_size", "save_every", "eval_every", "eval_batches", "log_every"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} 必须大于 0")
    if not 0 <= config.warmup_steps < config.total_steps:
        raise ValueError("warmup_steps 必须满足 0 <= warmup_steps < total_steps")
    if config.num_heads <= 0 or config.d_model % config.num_heads or (config.d_model // config.num_heads) % 2:
        raise ValueError("d_model 必须能被 num_heads 整除，并且每个 head 的维度必须是偶数")


def initialize_training(config: Config, device: str):
    """兼容旧 checkpoint 的 model/optimizer/iteration 格式。"""
    model = transformer(**config.model_kwargs()).to(device)
    optimizer = adamw(
        model.parameters(), config.learning_rate, config.weight_decay,
        config.betas, config.eps,
    )
    start_step = 0
    if config.init_mode != "scratch":
        checkpoint_path = config.path(config.init_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        saved_model_config = checkpoint.get("model_config")
        if saved_model_config is not None and saved_model_config != config.model_kwargs():
            raise ValueError("配置中的模型结构与 checkpoint 不一致，请检查 config.py 的模型参数")
        model.load_state_dict(checkpoint.get("model", checkpoint))
        if config.init_mode == "resume":
            if "optimizer" not in checkpoint or "iteration" not in checkpoint:
                raise ValueError("resume 需要包含 optimizer 和 iteration 的 checkpoint；纯权重请使用 weights")
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = checkpoint["iteration"] + 1
        print(f"loaded {config.init_mode}: {checkpoint_path}")
    return model, optimizer, start_step


def prepare_tokenizer(config: Config) -> tokenizer:
    vocab_path = config.path(config.vocab_path)
    merges_path = config.path(config.merges_path)
    if vocab_path.exists() != merges_path.exists():
        raise FileNotFoundError("分词器文件不完整：vocab_path 和 merges_path 必须同时存在")
    if not vocab_path.exists():
        vocab, merges = run_train_bpe(
            config.path(config.train_data), config.vocab_size, list(config.special_tokens)
        )
        for path, value in ((vocab_path, vocab), (merges_path, merges)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                pickle.dump(value, f)
    result = tokenizer.from_files(vocab_path, merges_path, special_tokens=list(config.special_tokens))
    if len(result.vocab) != config.vocab_size:
        raise ValueError("分词器词表大小与 vocab_size 不一致，请使用匹配的分词器和 token 缓存")
    return result


def getdata(data_path: Path, token_path: Path, tokeniz: tokenizer, config: Config):
    if not token_path.exists():
        print(f"encoding {data_path} -> {token_path}")
        with data_path.open("r", encoding="utf-8") as f:
            ids = tokeniz.encode(f.read())
        dtype = np.uint16 if config.vocab_size <= 65536 else np.uint32
        token_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(token_path, np.asarray(ids, dtype=dtype))
    data = np.load(token_path, mmap_mode="r")
    if data.ndim != 1 or len(data) <= config.context_length:
        raise ValueError(f"{token_path} 必须是一维 token 数组，长度必须大于 context_length")
    return data


def get_loss(model, x, label, loss):
    logits = model(x)
    return loss(logits.reshape(-1, logits.shape[-1]), label.reshape(-1))


@torch.no_grad()
def evaluate(model, dataset, config: Config, device: str, loss) -> float:
    model.eval()
    try:
        losses = [
            get_loss(model, *get_batch(dataset, config.batch_size, config.context_length, device), loss).item()
            for _ in range(config.eval_batches)
        ]
        return sum(losses) / len(losses)
    finally:
        model.train()


def train(config: Config = CONFIG) -> Path:
    validate_config(config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = get_device(config)
    model, optimizer, start_step = initialize_training(config, device)
    if start_step >= config.total_steps:
        raise ValueError(f"checkpoint 已完成 {start_step} 步，total_steps 必须大于这个值")

    tokeniz = prepare_tokenizer(config)
    dataset = getdata(config.path(config.train_data), config.path(config.train_tokens), tokeniz, config)
    valid_dataset = getdata(config.path(config.valid_data), config.path(config.valid_tokens), tokeniz, config)
    run_dir = config.path(config.run_dir)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"device: {device}; mode: {config.init_mode}; completed steps: {start_step}")
    print(f"checkpoints: {config.checkpoint_dir}")
    print(f"final model: {config.final_model_path}")
    print(f"log: {run_dir / 'train.jsonl'}")
    model.train()
    loss_fn = cross_entropy()
    start_time = time.perf_counter()
    with (run_dir / "train.jsonl").open("a", encoding="utf-8") as log:
        for i in range(start_step, config.total_steps):
            now_lr = get_lr_cosine_schedule(
                i, config.learning_rate, config.min_learning_rate,
                config.warmup_steps, config.total_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = now_lr
            optimizer.zero_grad()
            x, y = get_batch(dataset, config.batch_size, config.context_length, device)
            entropy = get_loss(model, x, y, loss_fn)
            entropy.backward()
            gradient_clipping(model.parameters(), config.max_grad_norm)
            optimizer.step()
            step = i + 1
            is_final = step == config.total_steps
            record = {"step": step, "lr": now_lr}
            if step % config.eval_every == 0 or is_final:
                record["valid_loss"] = evaluate(model, valid_dataset, config, device, loss_fn)
            if step % config.log_every == 0 or "valid_loss" in record or is_final:
                record.update(loss=entropy.item(), elapsed_seconds=time.perf_counter() - start_time)
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(record)
            if step % config.save_every == 0 or is_final:
                checkpoint_path = config.checkpoint_dir / f"step_{step:08d}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": i,  # 保持旧 checkpoint 的下标约定，恢复时 +1。
                    "model_config": config.model_kwargs(),
                    "config": config.to_dict(),
                }, checkpoint_path)

    torch.save({"model": model.state_dict(), "model_config": config.model_kwargs()}, config.final_model_path)
    print(f"train ok! model at {config.final_model_path}")
    return config.final_model_path


if __name__ == "__main__":
    train()
