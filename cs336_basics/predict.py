"""预测入口：uv run python -m cs336_basics.predict。"""

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_basics.config import CONFIG, Config
from cs336_basics.decoding import decoding
from cs336_basics.tokenizer import tokenizer
from cs336_basics.training_together import get_device
from cs336_basics.transformer_lm import transformer


def load_model(config: Config = CONFIG):
    device = get_device(config)
    path = config.final_model_path if config.predict_checkpoint is None else config.path(config.predict_checkpoint)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    # 新格式自动读取模型结构；旧 checkpoint 使用 config.py 中的模型结构。
    model_config = checkpoint.get("model_config", config.model_kwargs())
    model = transformer(**model_config).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    return model, model_config["context_length"]


def main(config: Config = CONFIG):
    tokeniz = tokenizer.from_files(
        config.path(config.vocab_path), config.path(config.merges_path),
        special_tokens=list(config.special_tokens),
    )
    model, context_length = load_model(config)
    prompt = input("请输入提示词：\n")
    if not prompt:
        raise ValueError("提示词不能为空")
    with torch.inference_mode():
        ids = decoding(
            model, prompt, tokeniz, context_length, config.max_new_tokens,
            tau=config.temperature, top_p=config.top_p,
        )
    print(tokeniz.decode(ids))


if __name__ == "__main__":
    main()
