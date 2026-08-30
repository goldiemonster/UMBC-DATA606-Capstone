# BidIntel Data

## Authoritative source

The BidIntel capstone uses **U.S. federal contract transaction data from
USAspending.gov**, operated by the U.S. Department of the Treasury, Bureau of
the Fiscal Service.

Source archive:

https://www.usaspending.gov/download_center/award_data_archive

USAspending's `Contracts_Full` files contain major agencies' prime award
transaction data for full fiscal years. These are the authoritative bulk files
used as the starting point for this project.

## Why the raw data are not committed to GitHub

The complete contract archives are very large. GitHub blocks ordinary Git
objects larger than 100 MiB and recommends keeping repositories small.

For reproducibility, this repository contains the download code rather than
duplicating the full federal dataset in Git history.

## Download the complete dataset

From the repository root:

```bash
python data/download_usaspending_contracts.py
```

The script discovers the currently published USAspending archive files and
downloads the latest `All_Contracts_Full` snapshot for **every available fiscal
year** into:

```text
data/raw/usaspending/contracts/
```

It also creates:

```text
data/raw/usaspending/contracts/manifest.csv
```

with source URLs, fiscal years, file sizes, SHA-256 hashes, and download
timestamps.

### Inspect first without downloading

Because the full archive can require substantial disk space:

```bash
python data/download_usaspending_contracts.py --list-only
```

### Download the initial BidIntel modeling window

For the first modeling iteration, a more practical scope is FY2020-FY2025:

```bash
python data/download_usaspending_contracts.py --years 2020-2025
```

The full archive can still be obtained later with the default command.

## Planned modeling target

The initial supervised-learning target is derived from:

```text
number_of_offers_received
```

Proposed binary target:

```text
single_offer = 1 when number_of_offers_received == 1
single_offer = 0 when number_of_offers_received > 1
```

Only attributes that would reasonably be available before the competition
outcome will be considered as model predictors. Potential leakage variables
will be identified during EDA and excluded.

## Data lineage

```text
USAspending.gov / FPDS procurement reporting
                |
                v
USAspending Award Data Archive
                |
                v
FY####_All_Contracts_Full_*.zip
                |
                v
data/raw/usaspending/contracts/
                |
                v
cleaning / EDA / feature engineering
                |
                v
data/processed/
```
