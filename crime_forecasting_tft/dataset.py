from preprocessing import df
from config import max_prediction_length, max_encoder_length, batch_size
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

#checking for span of data for confirmation
print(df["month"].min(), "→", df["month"].max())

max_t = df["time_idx"].max()
#last 6 months for test, the 6 before that for validation, everything before that for training
test_cutoff      = max_t - max_prediction_length
training_cutoff  = test_cutoff - max_prediction_length

training = TimeSeriesDataSet(
    df[df["time_idx"] <= training_cutoff],
    time_idx="time_idx",
    target="crime_count",
    group_ids=["lsoa_code", "crime_type"],

    # encoder length is how many months the model can look back to make predictions; prediction length is how many months into the future it predicts
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,

    # static real features 
    static_categoricals=["crime_type", "pfa_code", "loc_auth_code"],
    static_reals=[
        "population",
        "econ_score",
        "infrastructure_score",
        "health_score",
        "percent_working",
        "percent_child",
        "percent_old",
        "police_station_count",
    ],

    #available in both encoder and decoder; for decoder, we assume we know seasonality and month length info in advance, but not the weather info (which is only in encoder)
    time_varying_known_categoricals=["season", "month_of_year"],
    time_varying_known_reals=[
        "time_idx",
        "days_in_month",
        "holiday_count",
    ],

    #time varying unknown reals are the features that the model can only see for the encoder period, not the decoder period; this is where the target goes, along with any features that wouldn't be known in advance for the future months
    time_varying_unknown_reals=[
        "crime_count",
        "log_crime_count",
        "ra_3mo",
        "ra_6mo",
        "diff_12mo",
        "spatial_lag",
        "tmax",
        "tmin",
        "rain",
        "af",
    ],

    #scale each time series separately and indicate that target always positive (following tutorial)
    target_normalizer=GroupNormalizer(
        groups=["lsoa_code", "crime_type"],
        transformation="softplus",
    ),

    # helpful extras the tutorial uses
    # add_relative_time_idx adds a feature that counts how many months into the series we are, which helps the model learn position within the series and can improve extrapolation
    add_relative_time_idx=True,
    # add_target_scales adds features for the overall min and max of the target within each series, which can help the model learn the scale of the target and improve performance when different series have different scales
    add_target_scales=True,
    # add_encoder_length adds a feature that indicates how many months the model can see in the encoder, which can help it learn how much historical context it has when making predictions
    add_encoder_length=True,
    #already zero-padded so we can set this to false to avoid pytorch-forecasting trying to be smarter about it and messing with our data
    allow_missing_timesteps=False,
)

#validation set is the same as training but with predict=True to indicate that the model should try to predict the target in the decoder period, which allows us to evaluate its performance on unseen data
validation = TimeSeriesDataSet.from_dataset(
    training, df, predict=True, stop_randomization=True,
)

#batch size should eventually be tuned; using smaller batch size for training and larger for validation/test is a common practice to balance memory constraints with evaluation speed
#num_workers does parallel data loading; persistent_workers keeps the worker processes alive between epochs, which can speed up training if the dataset is large and loading is a bottleneck;; 
#Don't use multiple workers on personal computer unless it's beefy
train_dataloader = training.to_dataloader(train=True,  batch_size=batch_size,     num_workers=4, persistent_workers=True)
val_dataloader   = validation.to_dataloader(train=False, batch_size=batch_size*4, num_workers=4, persistent_workers=True)