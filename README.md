# toxicity_remit

REMIT（Reliable Endpoint-conditioned Motif Interaction Rationales for Toxicity
Prediction）的全新实验工程。本仓库当前完成 Stage A 的基础设施部分：

- 可审计的分子标准化、去重和标签冲突处理；
- 三组固定的 80/10/10 Bemis–Murcko scaffold splits；
- 分层 YAML 配置、命令行覆盖和配置哈希；
- 固定随机性、运行清单、数据哈希、日志和失败记录；
- 阻止训练/选模阶段访问测试集的数据访问门禁。

旧工程 `Toxicity_prediction` 仅作为历史参考，本项目没有也不会 import 旧工程模块。

## 环境

推荐 Python 3.11 和 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync --extra dev
uv run remit config show
uv run pytest
```

## 准备数据

将原始 CSV 放到 `data/raw/`。默认 Tox21 配置要求：

- SMILES 列：`smiles`；
- 可选样本 ID 列：`sample_id`；
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
├── metrics_validation.json
└── metrics_test.json
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

本提交尚未实现 ECFP、RF、XGBoost 或 GNN 基线。它们属于 Stage A 的下一批工作，
将在当前不可变 split 和运行协议之上实现。
