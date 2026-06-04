import os
import csv
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch

from pytorch_forecasting import TemporalFusionTransformer

from dataset import validation
from preprocessing import df
from config import max_prediction_length, latest_checkpoint


#not necessarily relevant anymore bc workers changed to 0, but keeping guard in case that changes
if __name__ == '__main__':

    #get latest checkpoint path from config.py
    CKPT_PATH = latest_checkpoint()

    #this is a workaround to the checkpoint being trained on CUDA, 
    #redirects any cuda tensor allocations to CPU during loading to avoid crashing
    _orig_zeros = torch.zeros
    def _safe_zeros(*args, **kwargs):
        if "device" in kwargs and kwargs["device"] is not None and "cuda" in str(kwargs["device"]):
            kwargs["device"] = torch.device("cpu")
        return _orig_zeros(*args, **kwargs)
    
    torch.zeros = _safe_zeros
    best_tft = TemporalFusionTransformer.load_from_checkpoint(CKPT_PATH, map_location="cpu")
    torch.zeros = _orig_zeros

    #moving it back to cpu
    best_tft = best_tft.cpu()
    best_tft.eval()

    #not necessary but takes month strings from original df rather than hardcoding
    max_t = df["time_idx"].max()
    test_time_idxs = list(range(max_t - max_prediction_length + 1, max_t + 1))
    time_idx_to_month = (
        df[["time_idx", "month"]].drop_duplicates().set_index("time_idx")["month"]
    )
    test_months = [time_idx_to_month[t].strftime("%Y-%m") for t in test_time_idxs]
    print(f'months tested: {test_months}')

    #convert the integers back to lsoa codes and crime type strings for the csv output
    val_dataset = validation
    lsoa_enc = val_dataset.categorical_encoders["__group_id__lsoa_code"]
    idx_to_lsoa = {v: k for k, v in lsoa_enc.classes_.items()}
    crime_enc = val_dataset.categorical_encoders["__group_id__crime_type"]
    idx_to_crime = {v: k for k, v in crime_enc.classes_.items()}

    #these are the same 7 quantiles used in training
    QUANTILE_COLS = ["q0.02", "q0.10", "q0.25", "q0.50", "q0.75", "q0.90", "q0.98"]
    fieldnames = ["lsoa_code", "crime_type", "month", "horizon_step"] + QUANTILE_COLS

    #small batch dataloader; use a smaller batch and number of works to 0 to avoid OOM
    small_dl = validation.to_dataloader(train=False, batch_size=128, num_workers=0)

    #each batch from the dataloader is a tuple (x,y) where x is the input dict w/ encoder sequences,
    #decoder sequences, static features, groups, targets scaled, and encoder/decoder lengths while
    #y is a tuple of (target, weight) w the actual cirme counts the model is supposed to predict

    #define output path + ensure it exists
    out_path = os.path.join(".", "data", "test_quantile_predictions.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    #pass 1 loops through entire dataloader and pulls out only the groups tensor from each
    #batch; done separately from inference to avoid OOM
    all_groups = []
    with torch.no_grad():
        for x, _ in small_dl:
            all_groups.append(x["groups"].clone())
    all_groups = torch.cat(all_groups, dim=0)  # (n_samples, 2)
    lsoa_indices = all_groups[:, 0].numpy()
    crime_indices = all_groups[:, 1].numpy()
    n_groups = len(lsoa_indices)

    #pass 2 runs the model and feeds every batch through the TFT and collects the predictions
    raw_preds = best_tft.predict(
        small_dl,
        #mode raw returns all quantiles rather than collapsing them
        mode="raw",
        #return_x false is a memory fix to not store the entire x dict for every batch; throws x
        #away afer each batch and only keeps prediction numbers 
        return_x=False,
        trainer_kwargs=dict(accelerator="cpu"),
    )
    #raw_preds is a tensor of shape (n_samples, max_prediction_length, n_quantiles) with the predicted crime counts for each
    # quantile at each horizon step for each group
    all_preds = raw_preds["prediction"]

    #write csv directly rather than building in-memory lists of dicts to avoid OOM
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_groups):
            lsoa = idx_to_lsoa.get(int(lsoa_indices[i]), "UNKNOWN")
            crime = idx_to_crime.get(int(crime_indices[i]), "UNKNOWN")
            for step in range(max_prediction_length):
                row = {
                    "lsoa_code": lsoa,
                    "crime_type": crime,
                    "month": test_months[step],
                    "horizon_step": step + 1,
                }
                for q_idx, col in enumerate(QUANTILE_COLS):
                    row[col] = round(float(all_preds[i, step, q_idx].item()), 4)
                writer.writerow(row)

    preview = pd.read_csv(out_path, nrows=12)
    print()
    print(preview.to_string(index=False))
