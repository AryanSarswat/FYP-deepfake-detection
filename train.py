from DatasetLoader import DataLoaderWrapper
from Baseline import create_model

import numpy as np
import pandas as pd

from tqdm import tqdm
import wandb

import torch
import torch.optim
import torch.nn as nn
import albumentations
from skearn.model_selection import train_test_split

args = {
    "epochs": 10,
    "batch_size": 32,
    "lr": 0.001,
    "architecture": "EfficientNetV2_s",
    "optimizer": "Adam",
}

wandb.init(project="deepfake-baseline", config=args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transforms = albumentations.Compose([
    albumentations.Resize(256, 256),
    albumentations.Normalize(),
    albumentations.HorizontalFlip(p=0.5),
])

test_transforms = albumentations.Compose([
    albumentations.Resize(256, 256),
    albumentations.Normalize(),
])

PATH = 'ast_dataset/data_final.csv'

PATH = 'ast_dataset/data_final.csv'

df = pd.read_csv(PATH)
X = df['image_path'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

train = DataLoaderWrapper(X_train, y_train, transforms=train_transforms, batch_size=args['batch_size'])
test = DataLoaderWrapper(X_test, y_test, transforms=test_transforms, batch_size=args['batch_size'])

def train(model, data_loader, optimizer, criteria, epoch):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for idx, (X, y) in tqdm(enumerate(data_loader), desc=f"Epoch {epoch} Training", total=len(data_loader)):
        X = X.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        y_pred = model(X)
        
        loss = criteria(y_pred, y)
        epoch_loss += loss.item()
        
        # Append Statistics
        y_pred = torch.round(y_pred, decimals=0)
        epoch_acc += (y_pred == y).sum().item()
        
        loss.backward()
        optimizer.step()
    
    train_loss = epoch_loss / len(data_loader.dataset)
    train_acc = epoch_acc / len(data_loader.dataset)

    print(f"Epoch {epoch} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}")

    return train_loss, train_acc

def validate(model, data_loader, criteria, epoch):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for idx, (X, y) in tqdm(enumerate(data_loader), desc=f"Epoch {epoch} Validation", total=len(data_loader)):
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)

            loss = criteria(y_pred, y)
            epoch_loss += loss.item()

            # Append Statistics
            y_pred = torch.round(y_pred, decimals=0)
            epoch_acc += (y_pred == y).sum().item()

    val_loss = epoch_loss / len(data_loader.dataset)
    val_acc = epoch_acc / len(data_loader.dataset)

    print(f"Epoch {epoch} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

    return val_loss, val_acc

model = create_model()
wandb.watch(model)
criteria = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

for epoch in range(args['epochs']):
    train_loss, train_acc = train(model, train, optimizer, criteria, epoch)
    val_loss, val_acc = validate(model, test, criteria, epoch)

    wandb.log({
        "Train Loss": train_loss,
        "Train Acc": train_acc,
        "Val Loss": val_loss,
        "Val Acc": val_acc,
    })
    
print("Training Complete")
print("Saving Model")

torch.save(model.state_dict(), "efficient_net_baseline.pth")







