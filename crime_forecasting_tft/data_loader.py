import sqlite3
import pandas as pd
from config import db_path, prototype, prototype_pfa

def load_panel(db_path: str, prototype_pfa: str | None = None) -> pd.DataFrame:
    '''
    Input: path to database, optional PFA name for prototyping
    Output: panel dataframe with one row per LSOA × crime_type × month

    Following tutorial, need to load the data into a single panel dataframe with one row per 
    LSOA × crime_type × month. This is the format expected by TFT.
    '''
    con = sqlite3.connect(db_path)

    #pfa filter for protyping 
    pfa_clause = ""
    params = ()
    if prototype_pfa is not None:
        pfa_clause = "WHERE i.pfa_code = ?"
        params = (prototype_pfa,)

    #query to pull together all relevant data; awkward query checked w/claude, not sure if the joins are right
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
    return df

df = load_panel(db_path, prototype_pfa=prototype_pfa if prototype else None)

#manual check to make sure data looks right
print(f'{len(df)}: rows'
      f'{df["lsoa_code"].nunique()}: LSOAs, '
      f'{df["crime_type"].nunique()}: crime types, '
      f'{df["month"].nunique()}: months')