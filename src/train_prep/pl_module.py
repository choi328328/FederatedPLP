import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import AveragePrecision
from torchmetrics import AUROC

class PL(pl.LightningModule):
    def __init__(
        self,
        network,
        batch_size,
        loss,
        train_loader,
        val_loader,
        test_loader,
        binary="binary"
    ):
        super().__init__()
        self.network = network
        self.loss = loss
        self.batch_size=batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.binary = binary
        self.auroc = AUROC(pos_label=1)
        self.AP = AveragePrecision(pos_label=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, batch, network):
        x, y = batch
        x= x.view(y.size()[0],1,-1) # batch_size is stricted to 256
        y= y.type(torch.FloatTensor)
        y_hat = self.network(x).view(-1)
        y_hat = self.sigmoid(y_hat)
        loss = self.loss(y_hat, y)
        return y_hat, y, loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        y_hat, y, loss = self.forward(batch, self.network)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y, loss = self.forward(batch, self.network)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        y_hat, y, loss = self.forward(batch, self.network)
        return {"test_y_hat": y_hat, "test_y": y, "test_loss": loss}

    def test_epoch_end(self, outputs):
        y_hat_cat = torch.cat([x["test_y_hat"] for x in outputs])
        y_cat = torch.cat([x["test_y"] for x in outputs])
        loss_stack = torch.stack([x["test_loss"] for x in outputs])

        avg_loss = loss_stack.mean()
        auroc = self.auroc(y_hat_cat, y_cat.int())
        ap = self.AP(y_hat_cat, y_cat.int())

        mean_ap = ap #if self.binary == "binary" else (sum(ap) / len(ap)).item()
        self.log("test_loss", avg_loss)
        self.log("test_auroc", auroc)
        self.log("test_ap", mean_ap)
    
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader