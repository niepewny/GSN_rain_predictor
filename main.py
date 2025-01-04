# Pytorch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

# utils
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import wandb
import os
from datetime import datetime
from omegaconf import OmegaConf

# Custom
from src.predictors.RainPredictor import RainPredictor
from src.data_modules.SEVIR_data_loader import SEVIR_dataset
from src.utils.Logger import ImagePredictionLogger


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    wandb.login(key=cfg.wandb.key)
    rnn_cell_name = cfg.main_model.RNN_cell["_target_"].split('.')[-1]
    timestamp = datetime.now().strftime("%d.%H.%M.%S")
    experiment_id = f"{rnn_cell_name}" \
                    f"_D{cfg.main_model.RNN_cell.depth}" \
                    f"_K{cfg.main_model.RNN_cell.kernel_size}" \
                    f"_H{cfg.main_model.RNN_cell.hidden_channels}" \
                    f"_SS{cfg.dataset.sequence_length}.{cfg.dataset.step}" \
                    f"_{timestamp}"

    dataset = SEVIR_dataset(
        file_paths=[os.path.join(cfg.dataset.dir, f) for f in os.listdir(cfg.dataset.dir)][:3],
        step=cfg.dataset.step,
        width=cfg.dataset.width,
        height=cfg.dataset.height
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_workers
    )

    early_stop_callback = EarlyStopping(
        monitor=cfg.early_stopping.monitor,
        patience=cfg.early_stopping.patience,
        mode=cfg.early_stopping.mode,
        verbose=cfg.early_stopping.verbose,
    )

    model_dir = os.path.join(cfg.checkpoint.dirpath, experiment_id)
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        dirpath=model_dir,
        filename=cfg.checkpoint.filename,
        save_top_k=cfg.checkpoint.save_top_k,
        mode=cfg.ckeckpoint.mode)

    main_model = RainPredictor(
        model=instantiate(cfg.main_model.RNN_cell),
        learning_rate=cfg.model.learning_rate,
        loss_metrics=instantiate(cfg.main_model.loss_metrics),
        scheduler_step=cfg.model.scheduler_step,
        scheduler_gamma=cfg.model.scheduler_gamma
        )

    trainer = pl.Trainer(
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            ImagePredictionLogger(val_samples)
            ],
        logger=WandbLogger(
            project=cfg.wandb.project_name,
            job_type='train',
            name=experiment_id,
            config=OmegaConf.to_container(cfg, resolve=True),
    ),
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs
        )

    trainer.fit(main_model, dataloader)

    trainer.test(model=main_model, datamodule=dataloader)

    wandb.finish()

    run = wandb.init(
        project=cfg.wandb.project_name,
        job_type='producer',
        config=OmegaConf.to_container(cfg, resolve=True),
        name=experiment_id
    )

    artifact = wandb.Artifact(f'model_{experiment_id}', type='model')
    artifact.add_dir(model_dir)

    run.log_artifact(artifact)
    run.join()
