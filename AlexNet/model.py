import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import pytorch_lightning as pl
from config import Config
from PIL import Image

from dataloader import STL10DataModule


config = Config('.env')

AVAIL_GPUS = min(1, torch.cuda.device_count())
EPOCHS = config.get('EPOCHS', 10)
LR = config.get('LR', 1e-3)
OUTPUT_DIR = config.get('OUTPUT_DIR', 'output')

# Output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seed for reproducibility
torch.manual_seed(config.get('SEED', 42))

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.LRN = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)
        self.init_bias()

    def init_bias(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
    
        nn.init.constant_(self.conv2.bias, 1)
        nn.init.constant_(self.conv4.bias, 1)
        nn.init.constant_(self.conv5.bias, 1)

    def forward(self, x):
        # Convolutional layers

        # 1st Convolutional Layer
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.LRN(x)
        # 2nd Convolutional Layer
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.LRN(x)
        # 3rd Convolutional Layer
        x = self.relu(self.conv3(x))
        # 4th Convolutional Layer
        x = self.relu(self.conv4(x))
        # 5th Convolutional Layer
        x = self.relu(self.conv5(x))
        x = self.maxpool(x)

        # flatten the output of the last convolutional layer
        x = x.view(-1, 256*6*6)

        # Fully connected layers (classifier)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class AlexNetModel(pl.LightningModule):
    def __init__(self, num_classes=1000):
        super(AlexNetModel, self).__init__()
        self.model = AlexNet(num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=LR)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.test_step_outputs.append({'loss': loss, 'acc': acc})
        return {'loss': loss, 'acc': acc}
    
    def on_test_epoch_end(self):
        avg_acc = torch.stack([x['acc'] for x in self.test_step_outputs]).mean()
        avg_loss = torch.stack([x['loss'] for x in self.test_step_outputs]).mean()
        self.log_dict({'test_loss': avg_loss, 'test_acc': avg_acc})
        self.test_step_outputs.clear()

if __name__ == '__main__':
    dm = STL10DataModule()
    model = AlexNetModel()

    trainer = pl.Trainer(
        num_nodes=AVAIL_GPUS,
        max_epochs=EPOCHS,
        default_root_dir='AlexNet_STL10_logs',
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)

    # Save the model
    trainer.save_checkpoint(os.path.join(OUTPUT_DIR, 'alexnet_model.ckpt'))