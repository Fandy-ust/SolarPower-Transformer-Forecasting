# Data layout

Processed training artifacts used to live on the git branch `Dataset`. They are expected under `data/processed/`:

| File | Purpose |
|------|---------|
| `campus_scale.txt` | Per-campus JSON z-score statistics (tracked in-repo) |
| `train_selected_sites.zip` | Archive containing `train_selected_sites.csv` |
| `train_selected_sites.csv` | Training split (extract from ZIP) |
| `validation_enhanced.csv` | Validation split |
| `test_enhanced.csv` | Held-out evaluation split |

## Setup

1. Copy **`campus_scale.txt`** - already tracked at [`data/processed/campus_scale.txt`](processed/campus_scale.txt).
2. Obtain the CSV/ZIP splits (from your historical Dataset branch archive, a release artifact, or re-generate locally):
   ```bash
   unzip -p path/to/train_selected_sites.zip > data/processed/train_selected_sites.csv
   cp path/to/validation_enhanced.csv data/processed/
   cp path/to/test_enhanced.csv data/processed/
   ```
3. Confirm paths match [`configs/default.yaml`](../configs/default.yaml).

> **Large files:** CSV/ZIP blobs are intentionally **gitignored**. Use GitHub Releases or external storage rather than committing ~50 MB binaries to git history.

## License / attribution

Upstream solar records come from **[UNISOLAR](https://github.com/CDAC-lab/UNISOLAR)** ([Harsha Kumara](https://github.com/CDAC-lab/Harshakumara) & Dilantha Haputhanthri). Keep their licensing terms wherever you redistribute the underlying records.
