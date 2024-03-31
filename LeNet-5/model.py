import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import pytorch_lightning as pl
from dataloader import MNISTDataModule
from config import Config

config = Config('.env')

AVAIL_GPUS = min(1, torch.cuda.device_count())
EPOCHS = config.get('EPOCHS', 10)
LR = config.get('LR', 1e-3)

torch.manual_seed(config.get('SEED', 42))

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.tanh = nn.Tanh()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool(x)
        x = self.tanh(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*5*5)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
class LeNet5Classifier(pl.LightningModule):
    def __init__(self, lr, n_classes=10):
        super(LeNet5Classifier, self).__init__()
        self.lr = lr
        self.model = LeNet5(num_classes=n_classes)
        self.loss = nn.CrossEntropyLoss()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.training_step_outputs.append(loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_outputs).mean()
        self.log('train_loss', avg_loss)
        self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.validation_step_outputs.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('val_loss', avg_loss)
        self.validation_step_outputs.clear()

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
    dm = MNISTDataModule()
    model = LeNet5Classifier(lr=LR)

    trainer = pl.Trainer(
        num_nodes=AVAIL_GPUS,
        max_epochs=EPOCHS,
        default_root_dir='LeNet5_MNIST_logs',
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    trainer.save_checkpoint('./checkpoints/lenet5_final_mnist.ckpt')