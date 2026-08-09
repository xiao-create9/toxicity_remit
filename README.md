# toxicity_remit

REMIT（Reliable Endpoint-conditioned Motif Interaction Rationales for Toxicity
Prediction）的全新实验工程。本仓库当前完成 Stage A 的数据协议与预测基线：

- 可审计的分子标准化、去重和标签冲突处理；
- 三组固定的 80/10/10 Bemis–Murcko scaffold splits；
- 分层 YAML 配置、命令行覆盖和配置哈希；
- 固定随机性、运行清单、数据哈希、日志和失败记录；
- 阻止训练/选模阶段访问测试集的数据访问门禁。
- ECFP4 + Random Forest、GINE、AttentiveFP 三种预测模型；
- 3 scaffold splits × 3 model seeds 的双 GPU 运行与自动结果聚合。

旧工程 `Toxicity_prediction` 仅作为历史参考，本项目没有也不会 import 旧工程模块。

## 环境

推荐 Python 3.11 和 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync --extra dev
uv run remit config show
uv run pytest
```

CUDA 12.8 服务器使用仓库提供的 Conda 环境入口。Conda 创建并管理完整环境，YAML
中的 `pip` 部分负责安装官方 PyTorch CUDA 12.8 wheel 和锁定的 Python 依赖；服务器
不需要安装 `uv`，也不会额外创建 `.venv`：

```bash
conda env create -f environment.server.yml
conda activate toxicity-remit
python -m remit.system_check --require-cuda --expected-cuda 12.8
```

## 准备数据

将原始 CSV 放到 `data/raw/`。默认 Tox21 配置要求：

- SMILES 列：`smiles`；
- 样本 ID 列：`mol_id`，缺失时回退到原始行号；
- 12 个 Tox21 endpoint 标签列，取值为 `0`、`1` 或缺失。

若列名或输入文件不同，通过命令行覆盖，无需修改代码：

```bash
uv run remit data prepare \
  --set data.input_path=data/raw/tox21.csv \
  --set data.smiles_column=smiles
```

该命令依次生成：

```text
data/processed/tox21/
├── molecules.parquet
├── invalid_molecules.csv
├── duplicate_conflicts.csv
├── source_rows.csv
├── standardization_report.json
└── manifest.json

data/splits/tox21/scaffold/
├── split_0.csv
├── split_0.json
├── split_1.csv
├── split_1.json
├── split_2.csv
└── split_2.json
```

所有 split 文件都包含显式 `sample_id,partition,scaffold`，并通过以下门禁：

1. 每个处理后分子恰好出现一次；
2. canonical molecule 不跨 train/validation/test；
3. scaffold group 不跨 train/validation/test；
4. split 引用的处理数据 SHA-256 与当前文件一致。

## 预测基线

单次运行：

```bash
uv run remit train --config configs/train_ecfp_rf.yaml --split-id 0 --seed 2026
uv run remit train --config configs/train_gine.yaml --split-id 0 --seed 2026
uv run remit train --config configs/train_attentivefp.yaml --split-id 0 --seed 2026
```

双 A800 标准矩阵共 27 个 runs：RF 在 CPU 顺序运行，18 个 GNN runs 在 GPU 0/1
之间分配。全部完成后脚本自动生成聚合报告。

```bash
uv run python scripts/run_standard_matrix.py --gpu-ids 0 1
```

只打印计划而不运行：

```bash
uv run python scripts/run_standard_matrix.py --gpu-ids 0 1 --dry-run
```

手动重新聚合：

```bash
uv run remit report prediction
```

完整模型、指标、产物和服务器运行说明见
[`docs/experiments/prediction_baselines.md`](docs/experiments/prediction_baselines.md)。

## 配置系统

入口为 `configs/default.yaml`，其中的 `defaults` 依次组合数据、实验和运行时配置。
点分路径覆盖示例：

```bash
uv run remit config show \
  --set split.seeds='[13,37,73]' \
  --set data.standardization.uncharge=false
```

最终解析配置会保存到每个 run 的 `config.yaml`，并写入稳定的 `config_hash`。

## 可复现运行协议

以下命令创建一次不训练模型的协议 smoke run：

```bash
uv run remit protocol smoke --split-id 0 --seed 2026
```

目录符合论文实验计划约定：

```text
runs/{experiment}/{dataset}/{split_id}/{seed}/{run_id}/
├── config.yaml
├── manifest.json
├── train.log
├── thresholds.json
├── predictions_validation.parquet
├── predictions_test.parquet
├── metrics_validation.json
├── metrics_test.json
├── training_history.csv
└── checkpoint.pt / checkpoint.joblib
```

训练代码必须通过 `SplitAccessGuard` 读取分区。在 `training` 或 `model_selection`
阶段读取 `test` 会直接失败；最终测试只能在 `final_evaluation` 阶段进行。

## 常用检查

```bash
uv run ruff check .
uv run pytest --cov=remit --cov-report=term-missing
uv run remit data verify
uv run remit data summary
```

The frozen Tox21 source, processing statistics, split counts, and artifact hashes are recorded in
[`docs/data/tox21.md`](docs/data/tox21.md).

## 当前范围

本提交已实现 ECFP4 + RF、GINE 和 AttentiveFP。尚未实现 XGBoost、D-MPNN、
endpoint embedding 或 REMIT 动态门控；这些内容将在预测基线结果稳定后继续实现。
