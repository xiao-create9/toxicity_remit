# Tox21 raw data provenance

- Dataset: MoleculeNet Tox21, as distributed by DeepChem
- Download URL: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz`
- Download date: 2026-08-09
- Compressed file: `tox21.csv.gz`
- Compressed SHA-256: `45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360`
- Decompressed file: `tox21.csv`
- Decompressed SHA-256: `7d7e7facd853a63e79ddce4e9c3fcb7a0d83a1c300b603031c0f2c64fbe77761`
- Raw rows: 7,831 molecules plus one header row
- Tasks: 12 binary endpoints with missing labels

The raw data files are intentionally ignored by Git. The processing manifest records the
decompressed input digest and exact standardization configuration. Invalid structures are retained
in the audit trail at `data/processed/tox21/invalid_molecules.csv`; they are never silently dropped.
