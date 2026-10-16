# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

## Training and prediction

训练和预测统一在 [`cs336_basics/config.py`](./cs336_basics/config.py) 配置。
文件内按数据路径、模型结构、训练超参数、加载模式、实验输出和预测参数分组。
相对路径始终相对于项目根目录，不受启动时所在目录影响。

```sh
uv run python -m cs336_basics.training_together
uv run python -m cs336_basics.predict
```

也支持直接运行 `python cs336_basics/training_together.py` 和 `python cs336_basics/predict.py`。
导入这两个模块不会启动训练、读取数据或弹出输入提示。

修改配置中的 `init_mode` 来选择训练方式：

| 模式 | 行为 |
| --- | --- |
| `"scratch"` | 从第 0 步训练，忽略 `init_checkpoint` |
| `"resume"` | 从 `init_checkpoint` 恢复模型、优化器和步数 |
| `"weights"` | 从 `init_checkpoint` 只加载权重，使用新优化器，从第 0 步开始 |

默认保持原入口的选择：从 `train_model/lr_1e-3_full/model4499` 续训，
预测读取 `train_model/lr_1e-3_full/model7999`。改成自己的 checkpoint 路径即可。
`resume` 会恢复 checkpoint 中的 AdamW 参数和动量；学习率仍按当前配置的计划更新。
需要使用新的 weight decay、betas 等超参数重新开始训练时，使用 `weights`。
`total_steps` 表示目标总步数；例如从已完成 4500 步的 checkpoint 训练到 25600 步，
填写 `total_steps = 25600`。当前恢复功能不恢复随机数状态，不保证 batch 序列与连续训练完全一致。

每次新实验设置不同的 `run_dir`。训练会创建以下文件：

```text
run_dir/
├── config.json                  # 本次启动配置（同目录再次启动时更新）
├── train.jsonl                  # loss、valid_loss、学习率、步数和本次耗时，追加写入
├── checkpoints/
│   └── step_00005000.pt          # 权重、优化器、迭代下标、模型结构和配置
└── model_final.pt               # 完成训练后保存，含权重和模型结构
```

`save_every`、`eval_every`、`log_every` 分别控制保存、验证和日志间隔；
最后一步始终验证并保存。`eval_batches` 控制验证损失平均的 batch 数。
checkpoint 文件名中的数字是已完成的步数；内部 `iteration` 仍采用原来的 0 起始下标，
因此旧 checkpoint 可以继续使用。`model_final.pt` 不含优化器，适合预测或 `weights` 模式。

预测时，将 `predict_checkpoint = None` 即可读取当前 `run_dir/model_final.pt`，
或者填写某个 checkpoint 的路径。新模型文件自动提供模型结构；旧文件需要配置匹配的模型参数。
分词器路径仍需指向该模型训练时使用的词表与 merges。
数据和 token 缓存存在时会复用；更换数据或分词器时，也应更换对应的 token 缓存路径。

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

