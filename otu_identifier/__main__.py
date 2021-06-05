import torch
import pytorch_lightning as pl

from dataset.data import SeqDataset
from model.models import OtuIdentifyNet


dataset = SeqDataset(filename="../domain_test.fasta")
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)
train_set = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True)
test_set = torch.utils.data.DataLoader(val_set, batch_size=50, shuffle=True)
model = OtuIdentifyNet()
trainer = pl.Trainer(gpus=-1)

trainer.fit(model, train_set, test_set)
