# Pytorch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
import torchmetrics

# Hydra and WB
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb

# Custom
from src.architectures.ConvRNN import ConvRNNCell
from src.predictors.RainPredictor import RainPredictor
from src.data_modules.SEVIR import SEVIRDataset
from src.utils.Logger import ImagePredictionLogger


wandb_key = "40c5c10f4b03fb955b2343280005f183c6e39d70"
wandb.login(key=wandb_key)

dm =  SEVIRDataset(file_path=".")
dm.prepare_data()
dm.setup()

val_samples = next(iter(dm.val_dataloader()))
val_imgs, val_labels = val_samples[0], val_samples[1]
val_imgs.shape, val_labels.shape
wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   patience=3,
   verbose=False,
   mode='min'
)

MODEL_CKPT_PATH = './model/'
MODEL_CKPT = 'model-{epoch:02d}-{val_loss:.2f}'

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=MODEL_CKPT_PATH,
    filename=MODEL_CKPT,
    save_top_k=3,
    mode='min')

RNN_model = ConvRNNCell(
    input_channels=1, 
    hidden_channels=3, 
    kernel_size=5, 
    depth=1, 
    activation=nn.ReLU
    )

main_model = RainPredictor(
    model=RNN_model, 
    learning_rate=0.01, 
    loss_metrics=torchmetrics.MeanSquaredError,
    quality_metrics=torchmetrics.JaccardIndex(task="binary"),#IOU
    scheduler_step=7,
    scheduler_gamma=0.9
    )

trainer = pl.Trainer(
    callbacks=[
        checkpoint_callback, 
        early_stop_callback, 
        ImagePredictionLogger(val_samples)
        ], 
    logger=wandb_logger, 
    accelerator="gpu", 
    devices=[0], 
    max_epochs=50
    )

trainer.fit(main_model, dm)

trainer.test(model=main_model, datamodule=dm)

wandb.finish()

run = wandb.init(project='GSN_rain_prediction', job_type='producer')

artifact = wandb.Artifact('model', type='model')
artifact.add_dir(MODEL_CKPT_PATH)

run.log_artifact(artifact)
run.join()