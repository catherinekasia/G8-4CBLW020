import pandas as pd
import numpy as np
from pathlib import Path
from config import prototype, prototype_pfa

_pfa_tag = prototype_pfa if prototype else "full"
_CACHE = Path(f"data/panel_{_pfa_tag}.parquet")

if _CACHE.exists():
    print(f"Loading preprocessed panel from cache ({_CACHE}) — delete file to rebuild from DB")
    df = pd.read_parquet(_CACHE)
else:
    from data_loader import df  # runs the SQL query only when cache is missing

    # month is "YYYY-MM" string -> a pandas Period -> integer offset from minimum
    # .astype(str) breaks the Categorical dtype before pd.to_datetime, otherwise pandas 2.x
    # preserves Categorical and the column can't round-trip through parquet correctly
    #thanks claude
    df["month"] = pd.to_datetime(df["month"].astype(str), format="%Y-%m")
    df["time_idx"] = ((df["month"].dt.year - df["month"].dt.year.min()) * 12
                     + (df["month"].dt.month - 1))
    df["time_idx"] -= df["time_idx"].min()  # start at 0
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
    df["ra_3mo"]   = df["ra_3mo"].fillna(0.0)
    df["ra_6mo"]   = df["ra_6mo"].fillna(0.0)
    df["diff_12mo"] = df["diff_12mo"].fillna(0.0)
    df["spatial_lag"] = df["spatial_lag"].fillna(0.0)

    n_weather_nan = df[["tmax", "tmin", "rain", "af"]].isna().sum().sum()
    if n_weather_nan > 0:
        print(f"WARNING: {n_weather_nan} NULL weather values (likely NULL-pfa_code LSOAs) — filling with 0")
    df["tmax"] = df["tmax"].fillna(0.0)
    df["tmin"] = df["tmin"].fillna(0.0)
    df["rain"] = df["rain"].fillna(0.0)
    df["af"]   = df["af"].fillna(0.0)

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
    print(f"Saved preprocessed panel to {_CACHE}")

