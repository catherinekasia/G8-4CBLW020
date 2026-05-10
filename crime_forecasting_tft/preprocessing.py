import pandas as pd
import numpy as np
from data_loader import df

# month is "YYYY-MM" string -> a pandas Period -> integer offset from minimum
#thanks claude
df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
df["time_idx"] = ((df["month"].dt.year - df["month"].dt.year.min()) * 12
                 + (df["month"].dt.month - 1))
df["time_idx"] -= df["time_idx"].min()  # start at 0
df["time_idx"] = df["time_idx"].astype(int)

#adding month of year to add more seasonality info
df["month_of_year"] = df["month"].dt.month.astype(str).astype("category")

#no clue what this does but in tutorial they log-transform the target for the encoder
df["log_crime_count"] = np.log(df["crime_count"] + 1e-8)

#PyTorch normalizers require continuous/floating-point data, not integers
df["crime_count"] = df["crime_count"].astype(float)

#changing to string format then to category
for c in ["lsoa_code", "crime_type", "pfa_code", "loc_auth_code", "season"]:
    df[c] = df[c].astype(str).astype("category")

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

#confirm no nulls
feature_cols = [
    "crime_count", "log_crime_count", "ra_3mo", "ra_6mo", "diff_12mo", "spatial_lag",
    "population", "econ_score", "infrastructure_score", "health_score",
    "percent_working", "percent_child", "percent_old", "police_station_count",
    "days_in_month", "holiday_count",
]
assert df[feature_cols].isna().sum().sum() == 0

