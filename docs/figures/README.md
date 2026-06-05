# Result figures

| Path | Notes |
|------|------|
| `r2_vs_steps.png` | \(R^2\) vs horizon steps from historical experiments |
| `rmse_vs_steps.png` | RMSE curve vs horizon |
| `samples/sample_*_*.png` | Representative prediction traces |

Re-run **`python src/evaluate.py`** after training Phase 3 to regenerate diagnostic curves declared in [`configs/default.yaml`](../configs/default.yaml) (`evaluation.plots.*`).
