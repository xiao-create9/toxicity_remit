# Tox21 Stage A dataset and split record

This record freezes the provenance and aggregate statistics of the Tox21 artifacts generated on
2026-08-09. Raw and generated data are intentionally excluded from Git; they can be regenerated
from the source and configuration recorded here.

## Source

- Distribution: MoleculeNet Tox21 from DeepChem
- URL: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz`
- Raw rows: 7,831
- Endpoints: 12 binary tasks with missing labels
- Compressed SHA-256: `45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360`
- Decompressed SHA-256: `7d7e7facd853a63e79ddce4e9c3fcb7a0d83a1c300b603031c0f2c64fbe77761`

## Standardization result

- Processing config SHA-256: `9387a3c22fce5e90e6d6ee2fefa5c6589049b72af4ca3c306320d274b54a4650`
- Processed Parquet SHA-256: `7163425e5cf47f110befc32cfcf33d2e351bbb554d3b8c0df5d1f0b4d639c885`
- Valid source rows: 7,822
- Invalid source rows: 9
- Unique canonical molecules: 7,586
- Duplicate source rows merged: 236
- Conflicting molecule-endpoint labels set to missing: 103

Eight invalid rows contain hypervalent aluminium forms rejected by the pinned RDKit parser; one
contains a hypervalent antimony form. Exact rows and errors are preserved in
`data/processed/tox21/invalid_molecules.csv`. No structure is silently repaired.

| Endpoint | Known | Positive | Negative | Positive rate |
|---|---:|---:|---:|---:|
| NR-AR | 7,018 | 299 | 6,719 | 4.26% |
| NR-AR-LBD | 6,546 | 230 | 6,316 | 3.51% |
| NR-AhR | 6,336 | 750 | 5,586 | 11.84% |
| NR-Aromatase | 5,626 | 283 | 5,343 | 5.03% |
| NR-ER | 5,993 | 764 | 5,229 | 12.75% |
| NR-ER-LBD | 6,741 | 336 | 6,405 | 4.98% |
| NR-PPAR-gamma | 6,252 | 175 | 6,077 | 2.80% |
| SR-ARE | 5,649 | 897 | 4,752 | 15.88% |
| SR-ATAD5 | 6,847 | 256 | 6,591 | 3.74% |
| SR-HSE | 6,266 | 348 | 5,918 | 5.55% |
| SR-MMP | 5,628 | 891 | 4,737 | 15.83% |
| SR-p53 | 6,568 | 412 | 6,156 | 6.27% |

## Fixed scaffold splits

- Strategy: Bemis–Murcko scaffold group split
- Fractions: 80% train, 10% validation, 10% test
- Seeds: 13, 37, 73
- Split config SHA-256: `d87f5facef5f0943ab44095f239f6c6c47603ebec35ff2636a3e28ed4307652d`
- Canonical molecule leakage: none
- Scaffold leakage: none
- Every endpoint has positive examples in every partition across all three splits

| Split | Seed | Train | Validation | Test | Train scaffolds | Validation scaffolds | Test scaffolds | Index SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 13 | 6,068 | 759 | 759 | 3,772 | 96 | 95 | `bd10c977...59988d` |
| 1 | 37 | 6,068 | 759 | 759 | 3,772 | 95 | 96 | `5b1a67ca...0d80fd` |
| 2 | 73 | 6,068 | 759 | 759 | 3,772 | 96 | 95 | `29d76881...ef7e3` |

The three seeds do not produce duplicate partitions. Pairwise changed-partition fractions are
16.24% for splits 0/1, 7.59% for splits 0/2, and 18.24% for splits 1/2.

## Reproduction and verification

```bash
curl --fail --location \
  --output data/raw/tox21.csv.gz \
  https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz
gzip --decompress --keep data/raw/tox21.csv.gz
uv run remit data prepare
uv run remit data verify
uv run remit data summary
```

`data prepare` records the exact source digest, processing configuration, RDKit version, output
digests, duplicate conflicts, and invalid rows. `data verify` fails if any processed artifact or
split index changes, or if a canonical molecule/scaffold crosses partitions.
