import typing

import albumentations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
import wandb
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from DatasetLoader.VideoDataset import DataLoaderWrapper
from models.ViViT import create_model
from contextlib import nullcontext

# Optimisations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

args = {
    "epochs": 50,
    "batch_size": 16,
    "num_frames" : 16,
    "architecture": "ViViT_0.45_weighted_lsa_fft",
    "save_path": "checkpoints/ViViT_0.45_weighted_lsa_fft",
    "optimizer": "AdamW",
    "patience" : 5,
    "lr" : 2e-6,
    "weight_decay": 1e-1,
    "min_delta" : 1e-3,
    "dtype": 'float32',
    'patch_size' : 16,
    'dim': 256,
    'head_dims' : 256,
    'depth': 6,
    'weight' : 0.3,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[args['dtype']]
ctx = nullcontext() if (not torch.cuda.is_available()) else torch.amp.autocast(device_type='cuda', dtype=ptdtype)

args["experiment_name"] = f"{args['architecture']}_frames_{args['num_frames']}_batch_{args['batch_size']}_lr_{args['lr']}_weighted_loss"

#wandb.init(project="deepfake-baseline", config=args, name=args["experiment_name"])


print(f"Using device: {device}")

train_transforms = albumentations.Compose([
    albumentations.Normalize(),
])

test_transforms = albumentations.Compose([
    albumentations.Normalize(),
])

PATH = 'videos_16/data_video.csv'

df = pd.read_csv(PATH)
X = df['video_path'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

train_loader = DataLoaderWrapper(X_train, y_train, num_frames=16, height=224, width=224, transforms=None, batch_size=args['batch_size'] ,shuffle=True, fft=True)
test_loader = DataLoaderWrapper(X_test, y_test, num_frames=16, height=224, width=224, transforms=None, batch_size=args['batch_size'], fft=True)

def train_epoch(model, data_loader, optimizer, criteria, epoch):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1 = 0
    epoch_auc = 0
    idx = 0
    scaler = torch.cuda.amp.GradScaler()
    
    pbar = tqdm(enumerate(data_loader), desc=f"Epoch {epoch} Training - Loss: {epoch_loss:.3f} - Running accuracy: {epoch_acc:.3f}", total=len(data_loader))
    
    for idx, (X, y) in pbar:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        y_pred = model(X)
        loss = criteria(y_pred, y)
            
            
        epoch_loss += loss.item()
        
        # Append Statistics
        y_pred = torch.where(y_pred >= 0.5, torch.ones_like(y_pred), torch.zeros_like(y_pred))
        
        # Update every other batch simulating gradient accumulation
        if idx % 2 == 0 or idx == len(data_loader) - 1:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        y_pred_cpu = y_pred.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()
        
        y_pred_cpu = y_pred_cpu.reshape(-1)
        y_cpu = y_cpu.reshape(-1)
            
        epoch_acc += accuracy_score(y_cpu, y_pred_cpu)
        epoch_precision += precision_score(y_cpu, y_pred_cpu, zero_division=0)
        epoch_recall += recall_score(y_cpu, y_pred_cpu, zero_division=0)
        epoch_f1 += f1_score(y_cpu, y_pred_cpu, zero_division=0)
        epoch_auc += roc_auc_score(y_cpu, y_pred_cpu, zero_division=0)
        
        
        pbar.set_description(f"Epoch {epoch} Train - Loss: {epoch_loss / (idx + 1):.3f} - Running accuracy: {epoch_acc / (idx + 1):.3f}")
    
    train_loss = epoch_loss / (idx + 1)
    train_acc = epoch_acc / (idx + 1)
    train_precision = epoch_precision / (idx + 1)
    train_recall = epoch_recall / (idx + 1)
    train_f1 = epoch_f1 / (idx + 1)
    train_auc = epoch_auc / (idx + 1)

    print(f"Epoch {epoch} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Train F1: {train_f1:.4f} Train Precision: {train_precision:.4f} Train recall: {train_recall:.4f}, Train AUC: {train_aouc:.4f}")

    return train_loss, train_acc, train_f1, train_precision, train_recall, train_auc

@torch.no_grad()
def validate_epoch(model, data_loader, criteria, epoch):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1 = 0
    epoch_auc = 0
    idx = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader), desc=f"Epoch {epoch} Validation - Loss: {epoch_loss:.3f} - Running accuracy: {epoch_acc:.3f}", total=len(data_loader))
        
        for idx, (X, y) in pbar:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

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
            epoch_auc += roc_auc_score(y_cpu, y_pred_cpu, zero_division=0)
            
            pbar.set_description(f"Epoch {epoch} Validation - Loss: {epoch_loss / (idx + 1):.3f} - Running accuracy: {epoch_acc / (idx + 1):.3f}")

    val_loss = epoch_loss / (idx + 1)
    val_acc = epoch_acc / (idx + 1)
    val_precision = epoch_precision / (idx + 1)
    val_recall = epoch_recall / (idx + 1)
    val_f1 = epoch_f1 / (idx + 1)
    val_auc = epoch_auc / (idx + 1)

    print(f"Epoch {epoch} Validation Loss: {val_loss:.4f} Validation Acc: {val_acc:.4f} Validation F1: {val_f1:.4f} Validation Precision: {val_precision:.4f} Validation recall: {val_recall:.4f}, Validation AUC: {val_auc:.4f}")

    return val_loss, val_acc, val_f1, val_precision, val_recall, val_auc

model = create_model(num_frames=args["num_frames"], patch_size=16, in_channels=9, height=224, width=224, dim=args['dim'], depth=args['depth'], heads=6, head_dims=args['head_dims'], dropout=0.25, scale_dim=4, lsa=True)
model = model.to(device)
#model = torch.compile(model)

num_parameters = sum(p.numel() for p in model.parameters())
print(f"[INFO] Number of parameters in model : {num_parameters:,}")

#wandb.watch(model)

class weighted_binary_cross_entropy(nn.Module):
    def __init__(self, weight=None):
        super(weighted_binary_cross_entropy, self).__init__()
    
    def forward(self, output, target):
        loss = (args['weight'] * (target * torch.log(output))) + (1 * ((1 - target) * torch.log(1 - output)))
        return torch.neg(torch.mean(loss))

criteria = weighted_binary_cross_entropy()
optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

previous_loss = np.inf
patience = 0

SAVED_ONCE = False
LOWEST_LOSS = np.inf

try:
    for epoch in range(args['epochs']):
        train_loss, train_acc, train_f1, train_precision, train_recall, train_auc = train_epoch(model, train_loader, optimizer, criteria, epoch)
        val_loss, val_acc, val_f1, val_precision, val_recall, val_auc = validate_epoch(model, test_loader, criteria, epoch)
        
        try:
            wandb.log({
                "Train Loss": train_loss,
                "Train Acc": train_acc,
                "Train F1": train_f1,
                "Train Precision": train_precision,
                "Train Recall": train_recall,
                "Train AUC": train_auc,
                "Val Loss": val_loss,
                "Val Acc": val_acc,
                "Val F1": val_f1,
                "Val Precision": val_precision,
                "Val Recall": val_recall,
                "Val AUC": val_auc,
            })
        except:
            print("[ERROR] Could not upload data, temporary connection")
        
        delta = val_loss - previous_loss
        
        if val_loss < LOWEST_LOSS:
            torch.save(model.state_dict(), args["save_path"] + ".pt")
            LOWEST_LOSS = val_loss
            SAVED_ONCE = True

        if abs(delta) < args['min_delta'] or val_loss > LOWEST_LOSS:
            patience += 1
            if val_loss < previous_loss:
                print(f"[INFO] Validation Loss improved by {delta:.2e}")
            else:
                print(f"[INFO] Validation Loss worsened by {delta:.2e}")
        else:
            patience = 0
        
        if patience >= args["patience"]:
            print("[INFO] No improvement in Validation loss Early Stopping")
            break
        
        previous_loss = val_loss
        print(f"[INFO] Number of times loss has not improved : {patience}")
except KeyboardInterrupt:
    print("[ERROR] Training Interrupted")
    print("[INFO] Saving Model")
    torch.save(model.state_dict(), args["save_path"] + "_interrupted.pt")
    
print("[INFO] Training Complete")
wandb.finish()

if not SAVED_ONCE:
    print("[INFO] Saving Model")
    torch.save(model.state_dict(), args["save_path"] + ".pt")
