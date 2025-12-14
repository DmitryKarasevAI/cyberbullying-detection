import lightning as L  # noqa: N812
import torch
import torchmetrics
from omegaconf import DictConfig
from torch import nn


class CyberbullyingModule(L.LightningModule):
    def __init__(self, model: nn.Module, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.num_classes)
        self.label_key = cfg.label_key
        self.lr = cfg.lr

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        return self.model(input_ids, pad_mask=pad_mask)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch.get("attention_mask"))
        target = batch[self.label_key].long()
        loss = self.criterion(logits, target)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch.get("attention_mask"))
        target = batch[self.label_key].long()
        loss = self.criterion(logits, target)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        preds = logits.argmax(dim=1)
        self.val_accuracy(preds, target)
        self.log(
            "val_accuracy",
            self.val_accuracy,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch.get("attention_mask"))
        target = batch[self.label_key].long()
        loss = self.criterion(logits, target)
        self.log("test_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        preds = logits.argmax(dim=1)
        self.test_accuracy(preds, target)
        self.log(
            "test_accuracy",
            self.test_accuracy,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
