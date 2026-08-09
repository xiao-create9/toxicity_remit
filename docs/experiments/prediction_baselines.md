# Stage A prediction baselines

## Scope

This stage implements three models under exactly the same Tox21 molecule table and fixed scaffold
indices:

1. ECFP4 (radius 2, 2,048 bits, chirality enabled) with one Random Forest per endpoint;
2. a shared multi-task GINE with categorical atom and bond embeddings;
3. a shared multi-task AttentiveFP with the same atom and bond embeddings.

The purpose is to establish a trustworthy prediction reference before adding endpoint-conditioned
REMIT gates. The one-epoch CPU smoke results generated during development are engineering checks
only and must never be copied into the paper.

## Shared chemical features

GINE and AttentiveFP receive identical categorical inputs. Atom fields are atomic number,
chirality, total degree, formal charge, total hydrogens, radical electrons, hybridization,
aromaticity, and ring membership. Bond fields are bond type, stereo, conjugation, and ring
membership. Every chemical bond is represented in both directions.

GINE uses edge-aware `GINEConv`, residual updates, LayerNorm, dropout, and add-plus-mean graph
pooling. It has no virtual node, because a virtual communication shortcut would make later edge
rationale evaluation ambiguous. AttentiveFP uses the PyTorch Geometric implementation after the
shared categorical encoders.

## Leakage-safe model selection

- Unknown endpoint labels are masked from the loss and all metrics.
- Per-endpoint positive class weights are calculated from the training partition only and capped at
  20.
- GNN checkpoints are selected only by validation macro PR-AUC.
- Endpoint decision thresholds maximize MCC on validation only.
- Test is loaded once, after checkpoint and thresholds are frozen.
- Every data access is appended to `data_access.jsonl`.
- Random Forest models use fixed hyperparameters and never fit on validation.

Reported metrics are PR-AUC, ROC-AUC, MCC, balanced accuracy, recall, specificity, ECE, and Brier
score. Every prediction is stored in long form as one molecule-endpoint row.

## Standard configuration

| Model | Main configuration |
|---|---|
| ECFP4 + RF | 500 trees, balanced class weights, all CPU cores |
| GINE | 5 layers, hidden size 256, dropout 0.2, trainable epsilon |
| AttentiveFP | 3 layers, hidden size 256, 2 readout timesteps, dropout 0.2 |
| GNN training | AdamW, batch 128, maximum 100 epochs, patience 20 |
| Precision | BF16 autocast on CUDA; FP32 on CPU |
| Repetitions | split IDs 0/1/2 and model seeds 2026/2027/2028 |

## A800 server setup (Conda + CUDA 12.8)

The Linux dependency source is the official PyTorch CUDA 12.8 wheel index. Conda creates and owns
the complete Python 3.11 environment. The `pip` section inside `environment.server.yml` installs the
official CUDA wheel and the locked Python dependencies as part of `conda env create`; the server
does not need uv and no nested `.venv` is created:

```bash
git clone https://github.com/xiao-create9/toxicity_remit.git
cd toxicity_remit

conda env create -f environment.server.yml
conda activate toxicity-remit

python -m remit.system_check --require-cuda --expected-cuda 12.8
nvidia-smi
```

`requirements-server-cu128.txt` contains the resolved versions, so the server does not resolve a
fresh Python environment. Do not install another package after environment creation. If the
environment must change, update the definitions and recreate it. The project records the actual
PyTorch, CUDA runtime, cuDNN, GPU, and driver values in the preflight output.

If environment creation fails partway through, remove only this named environment and recreate it
after fixing the reported network, driver, or package error:

```bash
conda env remove -n toxicity-remit
conda env create -f environment.server.yml
```

Download and verify the dataset if `data/processed/tox21/molecules.parquet` was not copied to the
server:

```bash
curl --fail --location \
  --output data/raw/tox21.csv.gz \
  https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz
gzip --decompress --keep data/raw/tox21.csv.gz
remit data prepare
remit data verify
git diff --exit-code -- data/splits/tox21/scaffold
```

The final command proves that server-side standardization reproduces the versioned split indices
and metadata exactly.

## Execution

Validate the run matrix first:

```bash
python scripts/run_standard_matrix.py --gpu-ids 0 1 --dry-run
```

Run all 27 jobs:

```bash
python scripts/run_standard_matrix.py --gpu-ids 0 1
```

RF runs sequentially on CPU because each forest already uses all CPU cores. GINE and AttentiveFP
runs are divided into two per-GPU queues; each GPU executes one run at a time. Standard runs do not
use DDP.

To run only GNN jobs:

```bash
python scripts/run_standard_matrix.py --models gine attentivefp --gpu-ids 0 1
```

## Outputs

Each run contains:

```text
config.yaml
manifest.json
train.log
data_access.jsonl
thresholds.json
metrics_validation.json
metrics_test.json
predictions_validation.parquet
predictions_test.parquet
training_history.csv          # GNN only
checkpoint.pt                 # GNN
checkpoint.joblib             # RF
failure.json                  # failed runs only
```

After the complete matrix, `reports/stage_a_prediction/` contains:

```text
protocol.json
included_runs.csv
excluded_runs.csv
prediction_summary.csv
endpoint_summary.csv
report.md
```

The aggregator accepts only completed runs whose config hashes match the three frozen model config
files. Smoke runs and manually overridden configurations are listed as excluded, preventing them
from contaminating the paper table.

## Interpretation gate

- If GINE is within roughly 1–2 macro PR-AUC points of the strongest GNN, retain it as the primary
  REMIT encoder and explanation-comparison backbone.
- If GINE is consistently more than 2–3 points behind AttentiveFP across scaffold splits, retain
  GINE only for controlled explainer comparisons and implement the main REMIT system on
  AttentiveFP.
- Do not choose the backbone from test results. The decision must use validation aggregates and be
  frozen before the final test comparison.
