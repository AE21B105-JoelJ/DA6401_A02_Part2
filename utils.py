# DA6401 - Assignment 02 (AE21B105) Source Code #

# Importing the necessary libraries #
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as F
import lightning as L
from typing import List
from lightning.pytorch import Trainer
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import Precision
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import os

# Function to give the activation function #
def return_activation_function(activation : str = "ReLU"):
    possible_activations = ["ReLU", "Mish", "GELU", "SELU", "SiLU", "LeakyReLU" ]
    # Assertion to be made for the activations possible #
    assert activation in possible_activations, f"activation not in {possible_activations}"

    if activation == "ReLU":
        return nn.ReLU()
    elif activation == "GELU":
        return nn.GELU()
    elif activation == "SiLU":
        return nn.SiLU()
    elif activation == "SELU":
        return nn.SELU()
    elif activation == "Mish":
        return nn.Mish()
    else:
        return nn.LeakyReLU()

# Lightning module for fast training #
class Lightning_CNN(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters(ignore = ['model'])

        # Define the model
        self.model = model

        # Defining the loss and optimizers
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 5e-4)

        # Defining the metrics
        self.prec_metric = Precision(task="multiclass", num_classes=10, average="weighted")

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        input_, target_ = batch
        output_ = self(input_)
        # Finding the loss to backprop #
        loss = self.loss_fn(output_, target_)
        # Logging the metrics #
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_, target_ = batch
        output_ = self(input_)
        # Finding the loss to backprop #
        loss = self.loss_fn(output_, target_)

        output_pred = torch.argmax(output_, dim=1) 
        precision = self.prec_metric(output_pred, target_)
        # Logging the metrics #
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_acc", precision, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        input_, target_ = batch
        output_ = self(input_)
        # Finding the loss to backprop #
        loss = self.loss_fn(output_, target_)
        
        output_pred = torch.argmax(output_, dim=1) 
        precision = self.prec_metric(output_pred, target_)
        # Logging the metrics #
        self.log("test_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_acc", precision, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        return self.optimizer

# class to orient (all images to landscape)
class OrientReshape:
    def __init__(self, size = (224, 224)):
        self.size = size
    
    def __call__(self, img):
        # rotate the image to landscape if potrait #
        if img.height > img.width:
            img = img.rotate(90, expand = True)
        # Reshape to target dimension #
        img = F.resize(img, size = self.size)

        return img
    

# Data augementation and transforms
def create_data_augment_compose(input_size = (224, 224)):
    data_transforms = {
        "orient_" : transforms.Compose([
            OrientReshape(size=input_size),
            transforms.ToTensor()
        ]),
        "train_" : transforms.Compose([
            transforms.RandomHorizontalFlip(p = 0.3),
            transforms.RandomVerticalFlip(p = 0.3),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.RandomErasing(p = 0.2, scale=(0.02, 0.075)),
        ])
    }

    return data_transforms

# Create a dataset with the image folders
def create_dataset_image_folder(path_, input_size = (224,224)):
    # Getting the transform
    data_transforms = create_data_augment_compose(input_size)
    # Path to dataset
    data_dir = path_ #os.path.join(os.path.abspath(""), "nature_12K/inaturalist_12K/train/") 
    # Creating dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms["orient_"])
    # Getting the labels for stratified split
    labels = [sample[1] for sample in full_dataset.samples]

    # Stratified split
    train_indices, val_indices = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42)

    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    return train_dataset, val_dataset, data_transforms

def create_dataloaders(batch_size, num_workers, train_dataset, val_dataset, is_data_aug, data_transforms):
    # Transforming the dataset with transforms
    if is_data_aug:
        train_dataset.dataset.transform = data_transforms['train_']
    else:
        train_dataset.dataset.transform = data_transforms['orient_']

    val_dataset.dataset.transform = data_transforms['orient_']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def get_test_dataloader(path_, data_transforms):
    # Path to dataset #
    data_dir = path_ #os.path.join(os.path.abspath(""), "nature_12K/inaturalist_12K/val/")
    test_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms["orient_"])

    # Applying transforms to datasets #
    test_dataset.transform = data_transforms["orient_"] 

    batch_size = 20
    num_workers = 4 # Adaptive number of workers

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return test_loader