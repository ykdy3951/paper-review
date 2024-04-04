from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
from config import Config
import os

config = Config('.env')

DATA_DIR = config.get('DATA_DIR', 'data')
BATCH_SIZE = config.get('BATCH_SIZE', 32)
NUM_WORKERS = config.get('NUM_WORKERS', 4)

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir = DATA_DIR, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS):
        super(ImageNetDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.val_dir = os.path.join(self.data_dir, 'val')
        self.test_dir = os.path.join(self.data_dir, 'test')

        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.imagenet_train = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
            self.imagenet_val = datasets.ImageFolder(self.val_dir, transform=self.train_transform)
        if stage == 'test' or stage is None:
            self.imagenet_test = datasets.ImageFolder(self.test_dir, transform=self.test_transform)
    
    def train_dataloader(self):
        return DataLoader(self.imagenet_train, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.imagenet_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.imagenet_test, batch_size=self.batch_size, num_workers=self.num_workers)

class STL10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir = DATA_DIR, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS):
        super(STL10DataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def prepare_data(self):
        datasets.STL10(self.data_dir, split='train', download=True)
        datasets.STL10(self.data_dir, split='test', download=True)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.stl10_full = datasets.STL10(self.data_dir, split='train', transform=self.train_transform)
            self.stl10_train, self.stl10_val = random_split(self.stl10_full, [4500, 500])
        if stage == 'test' or stage is None:
            self.stl10_test = datasets.STL10(self.data_dir, split='test', transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.stl10_train, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.stl10_val, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.stl10_test, batch_size=self.batch_size, num_workers=self.num_workers)