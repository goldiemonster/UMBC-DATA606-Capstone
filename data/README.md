# Data

Bridge inventory data from the FHWA [National Bridge Inventory](https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm)
(NBI) for the BridgeWatch AI capstone project.

## Contents

- **`{STATE}{YY}.txt`** — 42 raw files: 7 states (CA, FL, MD, MI, NY, TX, WA) × 6 years (2020–2025, encoded
  as `20`–`25`). Comma-delimited, one row per bridge, 123 columns, identical schema across every file. Column
  names embed their official NBI item number from FHWA's *Recording and Coding Guide for the Structure
  Inventory and Appraisal of the Nation's Bridges* (e.g. `YEAR_BUILT_027` = Item 27).
- **`processed/bridge_deterioration_dataset.csv.gz`** — the model-ready dataset built in
  [`../notebooks/eda.ipynb`](../notebooks/eda.ipynb): each row pairs one bridge's record in year *A* (features)
  with whether its condition rating dropped by year *A+1* (`deteriorated_next_period` label). Gzip-compressed
  to stay under GitHub's 100 MB per-file limit; read it with `pd.read_csv(path, compression="gzip")` (pandas
  also infers this automatically from the `.gz` extension).

See [`../docs/proposal.md`](../docs/proposal.md) for the full data dictionary, target/feature definitions, and
data-quality findings.
