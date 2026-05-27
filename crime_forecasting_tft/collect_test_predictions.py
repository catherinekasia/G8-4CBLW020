import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TemporalFusionTransformer

from dataset import training, validation, val_dataloader
from preprocessing import df
from config import max_prediction_length, latest_checkpoint

CKPT_PATH = latest_checkpoint()

#claude! training was done on snellius with cuda, but inference is on a mac without cuda, so we need to make sure all tensors 
#load onto cpu
_orig_zeros = torch.zeros
def _safe_zeros(*args, **kwargs):
    if "device" in kwargs and kwargs["device"] is not None and "cuda" in str(kwargs["device"]):
        kwargs["device"] = torch.device("cpu")
    return _orig_zeros(*args, **kwargs)

torch.zeros = _safe_zeros
best_tft = TemporalFusionTransformer.load_from_checkpoint(CKPT_PATH, map_location="cpu")
torch.zeros = _orig_zeros

best_tft = best_tft.cpu()
best_tft.eval()

# ── Determine which months are in the test window ───────────────────────────
# val_dataloader uses from_dataset(training, df, predict=True), which generates
# exactly one prediction window per (lsoa_code, crime_type) group: the encoder
# ends at max_t - max_prediction_length and the decoder covers the last
# max_prediction_length months of df.  Those last months ARE the test set.

#we could also have just looked at the last max_prediction_length rows of df, but this way we get the exact month labels that TFT is using for each time_idx.
max_t = df["time_idx"].max()
test_time_idxs = list(range(max_t - max_prediction_length + 1, max_t + 1))

#map time_idx to month labels using the original df, so we get the exact month strings that TFT is using for each time_idx, like before
time_idx_to_month = (
    df[["time_idx", "month"]].drop_duplicates()
    .set_index("time_idx")["month"]
)
test_months = [time_idx_to_month[t].strftime("%Y-%m") for t in test_time_idxs]
print(f'month tested: {test_months}')

#running predictions in raw mode to get all quantiles
raw_preds = best_tft.predict(
    val_dataloader,
    mode="raw",
    return_x=True,
    trainer_kwargs=dict(accelerator="cpu"),
)

all_preds = raw_preds.output["prediction"]
n_samples, pred_len, _ = all_preds.shape
QUANTILE_COLS = ["q0.02", "q0.10", "q0.25", "q0.50", "q0.75", "q0.90", "q0.98"]

#mapping from group tensors to lsoa_code and crime_type labels, using the categorical encoders that TFT's dataset created internally
val_dataset = val_dataloader.dataset
group_tensors = raw_preds.x["groups"]  # (n_samples, 2) — [lsoa_idx, crime_idx]

lsoa_enc = val_dataset.categorical_encoders["__group_id__lsoa_code"]
idx_to_lsoa = {v: k for k, v in lsoa_enc.classes_.items()}
lsoa_indices = group_tensors[:, 0].numpy()

crime_enc = val_dataset.categorical_encoders["__group_id__crime_type"]
idx_to_crime = {v: k for k, v in crime_enc.classes_.items()}
crime_indices = group_tensors[:, 1].numpy()

#building output df
records = []
for i in range(n_samples):
    lsoa = idx_to_lsoa.get(int(lsoa_indices[i]), "UNKNOWN")
    crime = idx_to_crime.get(int(crime_indices[i]), "UNKNOWN")
    for step in range(pred_len):
        row = {
            "lsoa_code": lsoa,
            "crime_type": crime,
            "month": test_months[step],
            "horizon_step": step + 1,
        }
        for q_idx, col in enumerate(QUANTILE_COLS):
            row[col] = round(float(all_preds[i, step, q_idx].item()), 4)
        records.append(row)

out_df = pd.DataFrame(records)

out_path = os.path.join(".", "data", "test_quantile_predictions.csv")
out_df.to_csv(out_path, index=False)

n_groups = n_samples

#manual checks
print(f"\nSaved {len(out_df):,} rows → {os.path.abspath(out_path)}")
print(f"  Groups (lsoa × crime_type): {n_groups:,}")
print(f"  Prediction steps per group: {pred_len}")
print(f"  Quantile columns: {QUANTILE_COLS}")
print()
print(out_df.head(12).to_string(index=False))
