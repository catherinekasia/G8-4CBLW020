import os
import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from config import max_prediction_length

_MAX_ENCODER_LENGTH = 36


#hardcoded paths to the two datas parquet files, the two checkpoints, and the two output csvs; these are set up to be easily switched out for different runs but can be edited as needed
#sorry for hardcode but no other way to do this bc there wasn't differentiation in naming convention while training
RUNS = [
    (
        "data/panel_full.parquet",
        "lightning_logs/tft_crime/version_4/checkpoints/epoch=60-step=39284.ckpt",
        "data/english_predictions.csv",
    ),
    (
        "data/panel_wales_data.parquet",
        "lightning_logs/tft_crime/version_9/checkpoints/epoch=17-step=168786.ckpt",
        "data/wales_predictions.csv",
    ),
]


##to be able to inference on two separate files, theres not way to call dataset.py/preprocessing.py at the same time
#build_validation is the helper to do the necessary preprocessing to be able to have both

def _build_validation(parquet_path):
    df = pd.read_parquet(parquet_path)
    df["month"] = pd.to_datetime(df["month"].astype(str).str[:7], format="%Y-%m")
    df["time_idx"] = (
        (df["month"].dt.year - df["month"].dt.year.min()) * 12
        + (df["month"].dt.month - 1)
    )
    df["time_idx"] -= df["time_idx"].min()
    df["time_idx"] = df["time_idx"].astype(int)
    if "month_of_year" not in df.columns:
        df["month_of_year"] = df["month"].dt.month.astype(str)
    df["month_of_year"] = df["month_of_year"].astype(str).astype("category")
    df["log_crime_count"] = np.log1p(df["crime_count"]).astype("float32")
    df["crime_count"] = df["crime_count"].astype("float32")
    for col in ["lsoa_code", "crime_type", "pfa_code", "loc_auth_code", "season"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in ["ra_3mo", "ra_6mo", "diff_12mo", "spatial_lag", "tmax", "tmin", "rain", "af"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0).astype("float32")
    df = df.sort_values(["lsoa_code", "crime_type", "time_idx"]).reset_index(drop=True)

    max_t = df["time_idx"].max()
    test_cutoff     = max_t - max_prediction_length
    training_cutoff = test_cutoff - max_prediction_length
    training_ds = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="crime_count",
        group_ids=["lsoa_code", "crime_type"],
        min_encoder_length=_MAX_ENCODER_LENGTH // 2,
        max_encoder_length=_MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["crime_type", "pfa_code", "loc_auth_code"],
        static_reals=[
            "population", "econ_score", "infrastructure_score", "health_score",
            "percent_working", "percent_child", "percent_old", "police_station_count",
        ],
        time_varying_known_categoricals=["season", "month_of_year"],
        time_varying_known_reals=["time_idx", "days_in_month", "holiday_count"],
        time_varying_unknown_reals=[
            "crime_count", "log_crime_count", "ra_3mo", "ra_6mo", "diff_12mo",
            "spatial_lag", "tmax", "tmin", "rain", "af",
        ],
        target_normalizer=GroupNormalizer(
            groups=["lsoa_code", "crime_type"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )
    validation = TimeSeriesDataSet.from_dataset(training_ds, df, predict=True, stop_randomization=True)
    return df, validation


#not necessarily relevant anymore bc workers changed to 0, but keeping guard in case that changes
if __name__ == '__main__':

    for parquet_path, CKPT_PATH, out_path in RUNS:
        df, validation = _build_validation(parquet_path)


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
