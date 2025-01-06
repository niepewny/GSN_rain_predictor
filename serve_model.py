import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.predictors.RainPredictor import RainPredictor
from src.data_modules.SEVIR_data_loader import ConvLSTMSevirDataModule
import torch

from omegaconf import OmegaConf


def load_model(checkpoint_path: str, config_path: str) -> RainPredictor:
    cfg = OmegaConf.load(config_path)

    # model z checkpointu
    model = RainPredictor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=cfg.model.RNN_cell,  
        mapping_activation=cfg.model.mapper_activation,
        learning_rate=cfg.model.learning_rate,
        loss_metrics=cfg.model.loss_metrics,
        scheduler_step=cfg.model.scheduler_step,
        scheduler_gamma=cfg.model.scheduler_gamma
    )

    model.eval()  # Ustaw model w tryb ewaluacji
    return model



if __name__ == "__main__":
    file_path_h5_dir = "data"
    # model_weights_path = "outputs/ConvRNN_D3_K7_H8_SS4_2_04_18-37-04/checkpoints/model-epoch=01-validation_loss=1.53.ckpt"
    # yaml_config_hydra = "outputs/ConvRNN_D3_K7_H8_SS4_2_04_18-37-04/.hydra/config.yaml"

    model_weights_path = "outputs/ConvRNN_D1_K5_H4_SS3_1_04_17-54-4-serve/checkpoints/model-epoch=03-validation_loss=0.09.ckpt"
    yaml_config_hydra = "outputs/ConvRNN_D1_K5_H4_SS3_1_04_17-54-4-serve/.hydra/config.yaml"
    dm = ConvLSTMSevirDataModule(
        step=2,
        width=192,
        height=192,
        batch_size=4,
        num_workers=1,
        sequence_length=9,
        train_files_percent=0.7,
        val_files_percent=0.15,
        test_files_percent=0.15,
        files_dir=file_path_h5_dir)

    dm.setup('test')
    test_loader = dm.test_dataloader()
    try:
        batch = next(iter(test_loader))
    except exception as e:
        print("No data in test_loader")
        sys.exit(1)

    pre_trained_model = load_model(model_weights_path, yaml_config_hydra)

    with torch.no_grad():
        outputs = pre_trained_model(batch)

    print(outputs.shape)

    # visualize_batch_tensor_interactive(batch, 0, "SEVIR dataset")
