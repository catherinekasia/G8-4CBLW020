import pandas as pd
import numpy as np
from pathlib import Path
from config import prototype, prototype_pfa, db_path as _db_path
from data_loader import df 

_db_tag = Path(_db_path).stem
_pfa_tag = prototype_pfa if prototype else _db_tag
_CACHE = Path(f"data/panel_{_pfa_tag}.parquet")

if _CACHE.exists():
    print(f"loading panel from cache ({_CACHE})")
    df = pd.read_parquet(_CACHE)
else:

    #if month is already a string, pd.to_datetime will still work, and if it's already a Period, it will also still work, so this is safe either way
    df["month"] = pd.to_datetime(df["month"].astype(str), format="%Y-%m")
    #time_idx is months since the earliest month in the dataset
    df["time_idx"] = ((df["month"].dt.year - df["month"].dt.year.min()) * 12
                     + (df["month"].dt.month - 1))
    #making sure it starts at 0
    df["time_idx"] -= df["time_idx"].min()
    #take it to an int to ensure not a float
    df["time_idx"] = df["time_idx"].astype(int)

    #adding month of year to add more seasonality info
    df["month_of_year"] = df["month"].dt.month.astype(str).astype("category")

    # log1p maps 0 → 0 cleanly; avoids the large negative values log(x + 1e-8) gives for zero-crime months
    df["log_crime_count"] = np.log1p(df["crime_count"]).astype("float32")

    #PyTorch normalizers require continuous/floating-point data, not integers
    df["crime_count"] = df["crime_count"].astype("float32")

    # data_loader already converted these to category; astype("category") is a no-op if already set
    for c in ["lsoa_code", "crime_type", "pfa_code", "loc_auth_code", "season"]:
        if df[c].dtype.name != "category":
            df[c] = df[c].astype("category")

    #sorting by lsoa_code, crime_type, time_idx to make sure all rows for a given series are together and in order
    df = df.sort_values(["lsoa_code", "crime_type", "time_idx"]).reset_index(drop=True)

    #sanity check to make sure all series have same length, which is important for zero-padding in TFT (was done during aggregation)
    series_lens = df.groupby(["lsoa_code", "crime_type"]).size()
    assert series_lens.nunique() == 1, "Series have unequal lengths — check zero-padding"

    #this should be eventually deleted as the data from before rolling averages should be dropped, but not ready to do that yet
    #(Keeping this despite not needing this anymore)
    df["ra_3mo"]   = df["ra_3mo"].fillna(0.0)
    df["ra_6mo"]   = df["ra_6mo"].fillna(0.0)
    df["diff_12mo"] = df["diff_12mo"].fillna(0.0)
    df["spatial_lag"] = df["spatial_lag"].fillna(0.0)

    #setting up weather to ensure no nulls
    n_weather_nan = df[["tmax", "tmin", "rain", "af"]].isna().sum().sum()
    df["tmax"] = df["tmax"].fillna(0.0)
    df["tmin"] = df["tmin"].fillna(0.0)
    df["rain"] = df["rain"].fillna(0.0)
    df["af"]   = df["af"].fillna(0.0)
    print(f'weather nulls: {n_weather_nan}')

    #confirm no nulls
    feature_cols = [
        "crime_count", "log_crime_count", "ra_3mo", "ra_6mo", "diff_12mo", "spatial_lag",
        "population", "econ_score", "infrastructure_score", "health_score",
        "percent_working", "percent_child", "percent_old", "police_station_count",
        "days_in_month", "holiday_count",
        "tmax", "tmin", "rain", "af",
    ]
    assert df[feature_cols].isna().sum().sum() == 0

    df.to_parquet(_CACHE)
    print(f"saved preprocessed panel to {_CACHE}")

