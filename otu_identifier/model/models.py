from torch import nn, unsqueeze
from torch.optim import Adam
import pytorch_lightning as pl


class OtuIdentifyNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(256, 84)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lin2 = nn.Linear(84, 84)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.lin3 = nn.Linear(84, 1)

    def forward(self, x):
        out = self.act1(self.lin1(x))
        out = self.dropout1(out)
        out = self.act2(self.lin2(out))
        out = self.lin3(out)
        return out

    def bcewithlogitsloss(self, y_val, y_label):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(y_val, y_label)

    def training_step(self, train_batch, batch_idx):
        x_train, y_train = train_batch
        y_pred = self.forward(x_train)
        y_train = unsqueeze(y_train, 1)
        loss = self.bcewithlogitsloss(y_pred, y_train)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        y = unsqueeze(y, 1)
        loss = self.bcewithlogitsloss(logits, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
