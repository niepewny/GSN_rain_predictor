# Pytorch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

# utils
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
import os

# Custom
from src.architectures.ConvRNN import ConvRNNCell
from src.predictors.RainPredictor import RainPredictor
from src.data_modules.SEVIR import ConvLSTMSevirDataModule
from src.utils.Logger import ImagePredictionLogger

if __name__ == "__main__":
    wandb_key = "40c5c10f4b03fb955b2343280005f183c6e39d70"
    wandb.login(key=wandb_key)

    dm = ConvLSTMSevirDataModule(
        train_files=[os.path.join("D:\\gsn_dataset\\2018", f) for f in os.listdir("D:\\gsn_dataset\\2018")][:1],
        val_files=[os.path.join("D:\\gsn_dataset\\2019", f) for f in os.listdir("D:\\gsn_dataset\\2019")][:1],
        test_files=[os.path.join("D:\\gsn_dataset\\2018", f) for f in os.listdir("D:\\gsn_dataset\\2018")][-1:],
        batch_size=3,
        num_workers=2
        )

    dm.setup()

    val_samples = next(iter(dm.val_dataloader()))
    val_imgs, val_labels = val_samples[0], val_samples[1]
    val_imgs.shape, val_labels.shape
    wandb_logger = WandbLogger(project='wandb-lightning', job_type='train', )

    early_stop_callback = EarlyStopping(
       monitor='validation_loss',
       patience=3,
       verbose=False,
       mode='min'
    )

    MODEL_CKPT_PATH = './model/'
    MODEL_CKPT = 'model-{epoch:02d}-{validation_loss:.2f}'

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        dirpath=MODEL_CKPT_PATH,
        filename=MODEL_CKPT,
        save_top_k=3,
        mode='min')

    RNN_model = ConvRNNCell(
        input_channels=1,
        hidden_channels=4,
        kernel_size=5,
        depth=1,
        activation=F.relu
        )

    main_model = RainPredictor(
        model=RNN_model,
        learning_rate=0.01,
        loss_metrics=torchmetrics.MeanSquaredError(squared=True),
        quality_metrics=torchmetrics.MeanSquaredError(squared=True), # JaccardIndex(num_classes=None, task="binary", threshold=0.5),#IOU
        scheduler_step=7,
        scheduler_gamma=0.9
        )

    trainer = pl.Trainer(
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            # ImagePredictionLogger(val_samples)
            ],
        logger=wandb_logger,
        accelerator="cpu",
        # devices=[0],
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
