import sqlite3
import pandas as pd
from config import db_path, prototype, prototype_pfa



_DTYPE_MAP: dict[str, str] = {
    # integers only safe to downcast when the column is guaranteed non-null after the JOINs
    # crime_count already safe b/c of zero-padding done on earlier joins
    "crime_count":          "int16",   # 0–1627, non-null (main fact column)
    "population":           "int16",   # 940–9512, non-null (inner-joined from demographics)
    "days_in_month":        "int8",    # 28–31, non-null (inner-joined from month_info)
    "holiday_count":        "int8",    # 0–4, non-null (inner-joined from month_info)
    "police_station_count": "int8",    # 0–5, non-null (COALESCE fills missing with 0)
    # floats — all nullable columns stay float32 so NaN survives
    # diff_12mo is INTEGER in SQLite but NULL for the first 12 months of each series,
    # so pandas loads it as float64; keep as float32 rather than trying to cast to int
    "diff_12mo":            "float32",
    "ra_3mo":               "float32",
    "ra_6mo":               "float32",
    "spatial_lag":          "float32",
    "econ_score":           "float32",
    "infrastructure_score": "float32",
    "health_score":         "float32",
    "percent_working":      "float32",
    "percent_child":        "float32",
    "percent_old":          "float32",
    "tmax":                 "float32",
    "tmin":                 "float32",
    "rain":                 "float32",
    "af":                   "float32",
}


def load_panel(db_path: str, prototype_pfa: str | None = None) -> pd.DataFrame:
    '''
    :param db_path: path to SQLite database containing the joined data
    :param prototype_pfa: if not None, only load data for this PFA

    Following tutorial, need to load the data into a single panel dataframe with one row per unit of analysis

    Join notes (verified):
    - lsoa_demographics and lsoa_info each have exactly one row per lsoa_code — no fan-out.
    - pfa_weather has unique (pfa_code, month) pairs — no fan-out.
    - lsoa_infrastructure covers only 1369 LSOAs (police stations); the LEFT JOIN + COALESCE
      fills the rest with 0.
    - pfa_weather: LEFT JOIN because some months/PFAs may have no weather record.

    :returns: panel dataframe with one row per LSOA × crime_type × month, with appropriate dtypes for memory efficiency
    '''
    con = sqlite3.connect(db_path)

    pfa_clause = ""
    params = ()
    if prototype_pfa is not None:
        pfa_clause = "WHERE i.pfa_code = ?"
        params = (prototype_pfa,)

    query = f"""
    SELECT
        lm.lsoa_code,
        lm.month,
        lm.crime_type,
        lm.crime_count,
        lm."3mo_ra"          AS ra_3mo,
        lm."6mo_ra"          AS ra_6mo,
        lm."12mo_diff_safe"  AS diff_12mo,
        lm.spatial_lag,
        d.pop AS population,
        d.econ_score,
        d.infrastructure_score,
        d.health_score,
        d.percent_working,
        d.percent_child,
        d.percent_old,
        i.pfa_code,
        i.loc_auth_code,
        COALESCE(inf.police_station_count, 0) AS police_station_count,
        m.days_in_month,
        m.season,
        m.holiday_count,
        w.tmax,
        w.tmin,
        w.rain,
        w.af
    FROM lsoa_month            lm
    JOIN lsoa_demographics     d   USING (lsoa_code)
    JOIN lsoa_info             i   USING (lsoa_code)
    LEFT JOIN lsoa_infrastructure inf USING (lsoa_code)
    JOIN month_info            m   USING (month)
    LEFT JOIN pfa_weather      w   ON i.pfa_code = w.pfa_code AND lm.month = w.month
    {pfa_clause}
    """
    df = pd.read_sql(query, con, params=params)
    con.close()

    # assert no cross-join fan-out, every (lsoa_code, crime_type, month) must be unique
    assert df.duplicated(["lsoa_code", "crime_type", "month"]).sum() == 0, \
        "Duplicate (lsoa_code, crime_type, month) rows — a JOIN is producing a fan-out"

    df = df.astype({k: v for k, v in _DTYPE_MAP.items() if k in df.columns})

    # instead of doing it in preprocessing, converting strings to category here saves memory
    for col in ("lsoa_code", "crime_type", "month", "pfa_code", "loc_auth_code", "season"):
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df

df = load_panel(db_path, prototype_pfa=prototype_pfa if prototype else None)

#manual inspection to ensure everything is fine
print(
    f"{len(df)} rows;"
    f"{df['lsoa_code'].nunique()} LSOAs;"
    f"{df['crime_type'].nunique()} crime types; "
    f"{df['month'].nunique()} months; "
    f"~{df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
