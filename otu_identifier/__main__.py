import torch
import pytorch_lightning as pl

from dataset.data import SeqDataset
from model.models import OtuIdentifyNet


dataset = SeqDataset(filename="../domain_test.fasta")
training = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)
validating = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)
model = OtuIdentifyNet()
trainer = pl.Trainer(gpus=1)

trainer.fit(model, training, validating)
