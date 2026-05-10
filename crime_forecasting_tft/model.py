import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from dataset import training, train_dataloader, val_dataloader

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

trainer_lr = pl.Trainer(
    accelerator="mps",
    gradient_clip_val=0.1,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    #edit to match lr
    learning_rate=6.918309709189363e-06,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=QuantileLoss(),
    optimizer="ranger",
    log_interval=10,
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters: {tft.size() / 1e3:.1f}k")



#tutorial shown; find optimal learning rate by running a learning rate finder, which runs short training runs with different learning rates and
#tracks the loss to find the learning rate that leads to the steepest decline in loss, which can help speed up training and improve convergence compared to using a default learning rate
if __name__ == "__main__":
    res = Tuner(trainer_lr).lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0, min_lr=1e-6,
    )
    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()