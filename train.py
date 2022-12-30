from DatasetLoader.VideoDataset import DataLoaderWrapper
from CvT.CvT import create_model

import numpy as np
import pandas as pd

from tqdm import tqdm
import wandb

import torch
import torch.optim
import torch.nn as nn
import albumentations

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

args = {
    "epochs": 50,
    "batch_size": 1, 
    "num_frames" : 32,
    "architecture": "CvT",
    "optimizer": "Adam",
    "patience" : 5,
    "lr" : 1e-4,
    "weight_decay": 1e-5,
    "min_delta" : 1e-3
}

wandb.init(project="deepfake-baseline", config=args, name="CvT")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

train_transforms = albumentations.Compose([
    albumentations.Normalize(),
])

test_transforms = albumentations.Compose([
    albumentations.Normalize(),
])

PATH = 'videos/data_video.csv'

df = pd.read_csv(PATH)
X = df['video_path'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

train_loader = DataLoaderWrapper(X_train, y_train, transforms=None, batch_size=args['batch_size'], shuffle=True)
test_loader = DataLoaderWrapper(X_test, y_test, transforms=None, batch_size=args['batch_size'])

def train(model, data_loader, optimizer, criteria, epoch):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1 = 0
    idx = 0
    for idx, (X, y) in tqdm(enumerate(data_loader), desc=f"Epoch {epoch} Training", total=len(data_loader)):
        X = X.to(device)
        y = y.to(device)
        
        y_pred = model(X)
        
        loss = criteria(y_pred, y)
        epoch_loss += loss.item()
        
        # Append Statistics
        y_pred = torch.where(y_pred >= 0.5, torch.ones_like(y_pred), torch.zeros_like(y_pred))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_pred_cpu = y_pred.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()
        
        y_pred_cpu = y_pred_cpu.reshape(-1)
        y_cpu = y_cpu.reshape(-1)
            
        epoch_acc += accuracy_score(y_cpu, y_pred_cpu)
        epoch_precision += precision_score(y_cpu, y_pred_cpu, zero_division=0)
        epoch_recall += recall_score(y_cpu, y_pred_cpu, zero_division=0)
        epoch_f1 += f1_score(y_cpu, y_pred_cpu, zero_division=0)
    
    train_loss = epoch_loss / (idx + 1)
    train_acc = epoch_acc / (idx + 1)
    train_precision = epoch_precision / (idx + 1)
    train_recall = epoch_recall / (idx + 1)
    train_f1 = epoch_f1 / (idx + 1)

    print(f"Epoch {epoch} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Train F1: {train_f1:.4f} Train Precision: {train_precision:.4f} Train recall: {train_recall:.4f}")

    return train_loss, train_acc, train_f1, train_precision, train_recall

def validate(model, data_loader, criteria, epoch):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1 = 0
    idx = 0
    with torch.no_grad():
        for idx, (X, y) in tqdm(enumerate(data_loader), desc=f"Epoch {epoch} Validation", total=len(data_loader)):
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)

            loss = criteria(y_pred, y)
            epoch_loss += loss.item()

            y_pred = torch.where(y_pred > 0.5, torch.ones_like(y_pred), torch.zeros_like(y_pred))
        
            y_pred_cpu = y_pred.detach().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()
            
            y_pred_cpu = y_pred_cpu.reshape(-1)
            y_cpu = y_cpu.reshape(-1)
            
            epoch_acc += accuracy_score(y_cpu, y_pred_cpu)
            epoch_precision += precision_score(y_cpu, y_pred_cpu, zero_division=0)
            epoch_recall += recall_score(y_cpu, y_pred_cpu, zero_division=0)
            epoch_f1 += f1_score(y_cpu, y_pred_cpu, zero_division=0)

    val_loss = epoch_loss / (idx + 1)
    val_acc = epoch_acc / (idx + 1)
    val_precision = epoch_precision / (idx + 1)
    val_recall = epoch_recall / (idx + 1)
    val_f1 = epoch_f1 / (idx + 1)

    print(f"Epoch {epoch} Validation Loss: {val_loss:.4f} Validation Acc: {val_acc:.4f} Validation F1: {val_f1:.4f} Validation Precision: {val_precision:.4f} Validation recall: {val_recall:.4f}")

    return val_loss, val_acc, val_f1, val_precision, val_recall

model = create_model(num_frames=args["num_frames"], dim=512, depth=6, heads=6, head_dims=128, dropout=0.2, scale_dim=4)
model = model.to(device)

num_parameters = sum(p.numel() for p in model.parameters())
print(f"[INFO] Number of parameters in model : {num_parameters:,}")

#wandb.watch(model)

def weighted_binary_cross_entropy(output, target):
    loss = 0.3 * (target * torch.log(output)) + 0.7 * ((1 - target) * torch.log(1 - output))
    return torch.neg(torch.mean(loss))

criteria = weighted_binary_cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

previous_loss = np.inf
patience = 0

try:
    for epoch in range(args['epochs']):
        train_loss, train_acc, train_f1, train_precision, train_recall = train(model, train_loader, optimizer, criteria, epoch)
        val_loss, val_acc, val_f1, val_precision, val_recall = validate(model, test_loader, criteria, epoch)
        
        try:
            wandb.log({
                "Train Loss": train_loss,
                "Train Acc": train_acc,
                "Train F1": train_f1,
                "Train Precision": train_precision,
                "Train Recall": train_recall,
                "Val Loss": val_loss,
                "Val Acc": val_acc,
                "Val F1": val_f1,
                "Val Precision": val_precision,
                "Val Recall": val_recall
            })
        except:
            print("[ERROR] Could not upload data, temporary connection")
        
        if abs(val_loss - previous_loss) < args['min_delta']:
            patience += 1
        else:
            patience = 0
        
        if patience > args["patience"]:
            print("[INFO] No improvement in Validation loss Early Stopping")
            break
        
        previous_loss = val_loss
        print(f"[INFO] Number of times loss has not improved : {patience}")
except KeyboardInterrupt:
    print("[ERROR] Training Interrupted")
    print("[INFO] Saving Model")
    torch.save(model.state_dict(), "CvT.pth")
    
    
print("[INFO] Training Complete")
print("[INFO] Saving Model")

torch.save(model.state_dict(), "CvT.pth")







