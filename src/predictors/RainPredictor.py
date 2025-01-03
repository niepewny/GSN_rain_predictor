import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class RainPredictor(pl.LightningModule):

    def __init__(self, model, learning_rate, loss_metrics, quality_metrics, scheduler_step, scheduler_gamma):
        super().__init__()

        self.model = model
        self.lr = learning_rate
        self.loss = loss_metrics
        self.quality = quality_metrics
        self.quality_metric = type(self.quality).__name__
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma

        self.mapping_layer = nn.Conv2d(
                in_channels=model.out_channels,
                out_channels=1,
                kernel_size=1,
                padding=0
            )

        self.current_epoch_training_loss = torch.tensor(0.0)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        batch_size, sequence_length, channels, height, width = x.size()
        self.model.initialize_hidden_state(batch_size, height, width, x.device)
        for i in range(sequence_length):
            outputs = self.model(x[:, i], gen_output=(i == sequence_length-1))
        
        outputs = self.mapping_layer(outputs)

        return outputs

    def compute_loss(self, y_pred, y):
        return self.loss(y_pred, y)

    def common_step(self, batch, batch_idx):
        ### temp
        batch = batch.permute(0, 3, 1, 2)
        batch = batch.unsqueeze(2)
        ###
        x = batch[:, :-1]
        y = batch[:, -1]
        outputs = self(x)
        loss = self.compute_loss(outputs, y)
        return loss, outputs, y

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        # preds = torch.argmax(outputs, dim=1) # add some custom metrics (for instance IOU, but threshold would be neccesary. Another way is just to give it up)
        #quality = self.quality(preds, y)
        quality = 0
        return loss, quality

    def training_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        self.training_step_outputs.append(loss)
        quality = self.quality(outputs, y)
        self.log_dict(
            {
                "train_loss": loss,
                f"train_{self.quality_metric}": quality
            },
            on_step = False,
            on_epoch = True,
            prog_bar = True
        )
        # if batch_idx % 100 == 0:
        #     x = batch[0][:8]
        #     grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
        #     self.logger.experiment.add_image("fashion_images", grid) #TODO

        return {'loss':loss}

    def on_train_epoch_end(self):
        outs = torch.stack(self.training_step_outputs)
        self.current_epoch_training_loss = outs.mean()
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        loss, quality = self.common_test_valid_step(batch, batch_idx)
        self.validation_step_outputs.append(loss)
        self.log_dict(
            {
                "validation_loss": loss,
                f"validation_{self.quality_metric}": quality
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        return {'validation_loss': loss, self.quality_metric: quality}

    def on_validation_epoch_end(self):
        outs = torch.stack(self.validation_step_outputs)
        avg_loss = outs.mean()

        # self.logger.experiment.log('train and vall losses', {'train': self.current_epoch_training_loss.item(), 'val': avg_loss.item()}, self.current_epoch)
        #
        # self.logger.experiment.log({
        #     'train_loss': self.current_epoch_training_loss.item(),
        #     'val_loss': avg_loss.item(),
        #     'epoch': self.current_epoch
        # })
        self.log('validation_loss', avg_loss, on_epoch=True)
        #todo upper or lower - one from chat, other from .ipynb
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        loss, quality = self.common_test_valid_step(batch, batch_idx)
        self.validation_step_outputs.append(loss)
        self.test_step_outputs.append(loss) # Not sure if it's ok.
        self.log_dict(
            {
                "test_loss": loss,
                f"test_{self.quality_metric}": quality
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        return {'test_loss':loss, self.quality_metric: quality}

    # Not sure if it should stay. Chat says it's good, because model is tested in batches.
    def on_test_epoch_end(self):
        outs = torch.stack(self.test_step_outputs)
        avg_loss = outs.mean()
        self.log('test_loss_epoch', avg_loss, on_epoch=True)
        self.test_step_outputs.clear()

    # We should decide if we should stick to Adam/scheduler or dump it to hydra
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma)
        return [optimizer], [lr_scheduler]