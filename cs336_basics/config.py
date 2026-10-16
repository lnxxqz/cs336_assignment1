"""训练与预测的统一配置"""

from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # 数据和分词器：复用已有文件，相对路径统一相对于项目根目录。
    train_data: Path = Path("TinyStoriesV2-GPT4-train.txt")
    valid_data: Path = Path("TinyStoriesV2-GPT4-valid.txt")
    vocab_path: Path = Path("train_model/rule_vocab.pkl")
    merges_path: Path = Path("train_model/rule_merges.pkl")
    train_tokens: Path = Path("train_model/tokens.npy")
    valid_tokens: Path = Path("train_model/valid_tokens.npy")
    special_tokens: tuple[str, ...] = ("<|endoftext|>",)

    # 模型结构：加载旧模型时必须与训练该模型的结构一致。
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    d_ff: int = 1344
    num_layers: int = 4
    num_heads: int = 16
    rope_theta: float = 10000.0

    # total_steps 是目标总步数，不是本次额外训练的步数。
    device: str = "auto"  # auto / cpu / cuda / cuda:0
    seed: int = 42
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    warmup_steps: int = 1000
    total_steps: int = 25600
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    batch_size: int = 50
    max_grad_norm: float = 1.0
    save_every: int = 500
    eval_every: int = 500
    eval_batches: int = 1
    log_every: int = 1

    # scratch：从头训练，忽略 init_checkpoint。
    # resume：恢复模型、优化器和步数；优化器参数以 checkpoint 为准，学习率按上面的计划更新。
    # weights：只加载权重，优化器与步数重新开始，使用上面的训练超参数。
    init_mode: str = "resume"
    init_checkpoint: Path | None = Path("train_model/lr_1e-3_full/model4499")

    # 换一个 run_dir 就能把不同实验分开。输出结构：
    # checkpoints/step_00005000.pt：完整训练状态，文件名是已完成的步数。
    # model_final.pt：最终权重 + 模型结构，不含优化器。
    # train.jsonl：训练日志；config.json：本次启动的配置快照。
    run_dir: Path = Path("train_model/lr_1e-3_full")

    # None 表示使用上述最终模型；也可以填写任意已有 checkpoint。
    predict_checkpoint: Path | None = Path("train_model/lr_1e-3_full/model7999")
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0

    def path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def checkpoint_dir(self) -> Path:
        return self.path(self.run_dir) / "checkpoints"

    @property
    def final_model_path(self) -> Path:
        return self.path(self.run_dir) / "model_final.pt"

    def model_kwargs(self) -> dict:
        names = (
            "vocab_size", "context_length", "d_model", "num_layers",
            "num_heads", "d_ff", "rope_theta",
        )
        return {name: getattr(self, name) for name in names}

    def to_dict(self) -> dict:
        return {
            name: str(self.path(value)) if isinstance(value, Path) else value
            for name, value in asdict(self).items()
        }


CONFIG = Config()

