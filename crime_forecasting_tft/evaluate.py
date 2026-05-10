from pytorch_forecasting import Baseline, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, SMAPE
from dataset import val_dataloader
from train import trainer
import numpy as np
from data_loader import df

import os
import glob

#baseline here is just naive prediction of the last value observed in encoder period
baseline_predictions = Baseline().predict(val_dataloader, return_y=True)

#MAE is mean absolute error, which measures the average absolute difference between the predicted and actual values; lower values indicate better performance
baseline_mae = MAE()(baseline_predictions.output, baseline_predictions.y)
print(f"Baseline MAE: {baseline_mae:.4f}")

#SMAPE is symmetric mean absolute percentage error, which measures the average percentage difference between the predicted and actual values, normalized by the average of 
#the predicted and actual values; lower values indicate better performance, and it is symmetric because it treats overestimates and underestimates equally
baseline_smape = SMAPE()(baseline_predictions.output, baseline_predictions.y)
print(f"Baseline SMAPE: {baseline_smape:.4f}")


#best_path is the path to the best model checkpoint saved during training, which is determined by the EarlyStopping callback based on the validation loss; 
#loading this checkpoint allows us to evaluate the performance of the best model on the validation set
checkpoint_dir = "lightning_logs/crime_forecasting_tft/"
latest_version = sorted(os.listdir(checkpoint_dir))[-1]
checkpoint_path = glob.glob(f"{checkpoint_dir}/{latest_version}/checkpoints/*.ckpt")[0]

#best_tft is the Temporal Fusion Transformer model loaded from the best checkpoint, which contains the weights and architecture of the model that achieved the best performance on the validation set during training
best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)

#predictions is the output of the best model when making predictions on the validation dataloader; 
#return_y=True indicates that we also want to return the actual target values for evaluation, 
#and trainer_kwargs=dict(accelerator="auto") ensures that the predictions are made using the appropriate hardware (CPU or GPU) based on availability
predictions = best_tft.predict(
    val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="auto"),
)
print("MAE :", MAE()(predictions.output, predictions.y).item())
print("SMAPE:", SMAPE()(predictions.output, predictions.y).item())

#sample predictions with attention overlays; mode="raw" returns the raw output of the model, including attention weights and other intermediate 
#values, which can be used for visualization; return_x=True also returns the input data, which is needed for plotting the predictions against the actual values and visualizing the attention overlays
raw = best_tft.predict(
    val_dataloader, mode="raw", return_x=True,
    trainer_kwargs=dict(accelerator="auto"),
)
for i in range(10):
    best_tft.plot_prediction(raw.x, raw.output, idx=i, add_loss_to_title=True)


#to find the worst predictions, we can calculate the loss for each individual prediction in the validation set and then sort them to identify which ones had the highest loss;
#this can help us understand where the model is struggling and potentially identify patterns or specific cases that are difficult for the model to predict accurately
losses = SMAPE(reduction="none").loss(predictions.output, predictions.y[0]).mean(1)
worst = losses.argsort(descending=True)
for i in range(10):
    best_tft.plot_prediction(
        raw.x, raw.output,
        idx=worst[i],
        add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles),
    )

#interpretation of the model's predictions using interpret_output method, gives which features and time steps were most influential in the model's predictions;
interp = best_tft.interpret_output(raw.output, reduction="sum")
best_tft.plot_interpretation(interp)


dep = best_tft.predict_dependency(
    val_dataloader.dataset,
    #edit when needed
    "econ_score",
    np.linspace(0, df["econ_score"].max(), 30),
    show_progress_bar=True, mode="dataframe",
    trainer_kwargs=dict(accelerator="auto"),
)
agg = dep.groupby("econ_score").normalized_prediction.agg(
    median="median",
    q25=lambda x: x.quantile(.25),
    q75=lambda x: x.quantile(.75),
)
ax = agg.plot(y="median")
ax.fill_between(agg.index, agg.q25, agg.q75, alpha=0.3)