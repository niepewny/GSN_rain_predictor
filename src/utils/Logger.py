from pytorch_lightning.callbacks import Callback
import wandb
import numpy as np


class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=1):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer, model):
        val_imgs = self.val_imgs.to(device=model.device)
        output = model(val_imgs)

        val_imgs = val_imgs[:self.num_samples].cpu().numpy()
        predictions = output[:self.num_samples].cpu().detach().numpy()

        logged_images = []
        for true_img, pred_img in zip(val_imgs, predictions):
            combined_img = np.concatenate([true_img, pred_img], axis=2)
            normalized_img = (combined_img - combined_img.min()) / (combined_img.max() - combined_img.min())
            logged_images.append(
                wandb.Image(normalized_img, caption="Ground Truth | Prediction")
            )

        trainer.logger.experiment.log({
            "examples": logged_images
        })
