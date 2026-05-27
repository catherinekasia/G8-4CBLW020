import torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from dataset import training, train_dataloader, val_dataloader
from config import learning_rate

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

# TF32 on Ampere/Hopper GPUs (Snellius): same accuracy as FP32, up to 8x faster matmuls
torch.set_float32_matmul_precision("high")

early_stop = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, mode="min", verbose=False,
)
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("lightning_logs", name="tft_crime")

trainer = pl.Trainer(
    max_epochs=50,
    #this is for apple silicon gpu, edit to match hardware as necessary
    accelerator="mps",
    #when on super computer bump up devices
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[lr_logger, early_stop],
    logger=logger,
    # limit_train_batches=50,  #uncomment for fast iteration during dev
)

# trainer = pl.Trainer(
#     max_epochs=50,
#     accelerator="gpu",
#     devices=4,
#     strategy="ddp_find_unused_parameters_true",
#     precision="bf16-mixed",
#     gradient_clip_val=0.1,
#     callbacks=[lr_logger, early_stop],
#     logger=logger,
# )

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=learning_rate,
    #hidden_size is the size of the hidden layers in the model; larger values can capture more complex patterns, we have more data so more
    hidden_size=64,
    #attention_head_size is the number of attention heads in the multi-head attention mechanism; more heads can allow the model to focus on different parts of the input, but also increases computational cost, so we can start with a smaller number and tune up if needed
    attention_head_size=4,
    #droput is the dropout rate for regularization; higher values can help prevent overfitting but may also make training more difficult, so we can start with a moderate value and tune as needed
    dropout=0.15,
    #hidden_continuous_size is the size of the hidden layers for continuous variables; this can be smaller than hidden_size since continuous variables may require less capacity to model effectively, so we can start with a smaller value and tune up if needed
    hidden_continuous_size=32, #≤ hidden_size
    #loss function is the part of the model that measures how well the model's predictions match the actual values
    loss=QuantileLoss(),
    #log_interval controls how often the training loss is logged; setting it to a smaller value can provide more granular feedback on training progress,adjust as needed
    log_interval=10,
    #optimizer is the optimization algorithm used to update the model's weights during training; "ranger" is a combination of RAdam and Lookahead optimizers that can provide faster convergence and better performance compared to traditional optimizers like Adam, so we can start with it and tune if needed
    optimizer="ranger",
    #reduce_on_plateau_patience controls how many epochs to wait for an improvement in validation loss before reducing the learning rate; setting it to a smaller value can help the model converge faster, but may also lead to premature learning rate reductions if the validation loss is noisy, so we can start with a moderate value and tune as needed
    reduce_on_plateau_patience=4,
)

##tensorboard --logdir=lightning_logs

if __name__ == "__main__":
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )