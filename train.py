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
from models.util import DataAugmentation

from DatasetLoader.VideoDataset import DataLoaderWrapper
from models.CvT import create_model
from contextlib import nullcontext

# Optimisations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

weight = 0.4
args = {
    "epochs": 50,
    "batch_size": 8,
    "num_frames" : 16,
    "architecture": f"CvT_{weight}_weighted_RGB",
    "save_path": f"checkpoints/CvT_{weight}_weighted_RGB",
    "optimizer": "SGD",
    "patience" : 5,
    "lr" : 2e-5,
    "weight_decay": 1e-3,
    "min_delta" : 1e-3,
    "dtype": 'float32',
    'patch_size' : None,
    'dim': 256,
    'heads' : 8,
    'head_dims' : 128,
    'depth': 8,
    'weight' : weight,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args["experiment_name"] = f"{args['architecture']}_frames_{args['num_frames']}_batch_{args['batch_size']}_lr_{args['lr']}_weighted_loss_{args['weight']}"

wandb.init(project="deepfake-baseline", config=args, name=args["experiment_name"])


print(f"Using device: {device}")

aug = DataAugmentation(size=224)

PATH = 'videos_16/data_video.csv'

df = pd.read_csv(PATH)
X = df['video_path'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

train_loader = DataLoaderWrapper(X_train, y_train, num_frames=16, height=224, width=224, transforms=aug, batch_size=args['batch_size'] ,shuffle=True)
test_loader = DataLoaderWrapper(X_test, y_test, num_frames=16, height=224, width=224, transforms=None, batch_size=args['batch_size'])

def train_epoch(model, data_loader, optimizer, criteria, epoch):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1 = 0
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
        if idx % 4 == 0 or idx == len(data_loader) - 1:
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
        
        
        pbar.set_description(f"Epoch {epoch} Train - Loss: {epoch_loss / (idx + 1):.3f} - Running accuracy: {epoch_acc / (idx + 1):.3f}")
    
    train_loss = epoch_loss / (idx + 1)
    train_acc = epoch_acc / (idx + 1)
    train_precision = epoch_precision / (idx + 1)
    train_recall = epoch_recall / (idx + 1)
    train_f1 = epoch_f1 / (idx + 1)

    print(f"Epoch {epoch} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Train F1: {train_f1:.4f} Train Precision: {train_precision:.4f} Train recall: {train_recall:.4f}")

    return train_loss, train_acc, train_f1, train_precision, train_recall

@torch.no_grad()
def validate_epoch(model, data_loader, criteria, epoch):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1 = 0
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
            
            pbar.set_description(f"Epoch {epoch} Validation - Loss: {epoch_loss / (idx + 1):.3f} - Running accuracy: {epoch_acc / (idx + 1):.3f}")

    val_loss = epoch_loss / (idx + 1)
    val_acc = epoch_acc / (idx + 1)
    val_precision = epoch_precision / (idx + 1)
    val_recall = epoch_recall / (idx + 1)
    val_f1 = epoch_f1 / (idx + 1)

    print(f"Epoch {epoch} Validation Loss: {val_loss:.4f} Validation Acc: {val_acc:.4f} Validation F1: {val_f1:.4f} Validation Precision: {val_precision:.4f} Validation recall: {val_recall:.4f}")

    return val_loss, val_acc, val_f1, val_precision, val_recall


@torch.no_grad()
def test(model, data_loader, dataset_name):
    model.eval()

    preds = []
    actual = []

    for idx, (X, y) in tqdm(enumerate(data_loader), desc=f"Inference", total=len(data_loader)):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        y_pred = model(X)
        y_pred = torch.where(y_pred > 0.5, torch.ones_like(y_pred), torch.zeros_like(y_pred))

        y_pred_cpu = y_pred.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()

        y_pred_cpu = y_pred_cpu.reshape(-1)
        y_cpu = y_cpu.reshape(-1)

        preds.extend(y_pred_cpu)
        actual.extend(y_cpu)

    val_acc = accuracy_score(actual, preds)
    val_precision = precision_score(actual, preds)
    val_recall = recall_score(actual, preds)
    val_f1 = f1_score(actual, preds)
    val_auc = roc_auc_score(actual, preds)

    print(f"Test Acc:{val_acc:.4f} Test F1:{val_f1:.4f} Test Precision:{val_precision:.4f} Test Recall:{val_recall:.4f} Test AUC:{val_auc:.4f}")

    wandb.run.summary[f"{dataset_name}_test_acc"] = val_acc
    wandb.run.summary[f'{dataset_name}_test_precision'] = val_precision
    wandb.run.summary[f'{dataset_name}_test_recall'] = val_recall
    wandb.run.summary[f'{dataset_name}_test_f1'] = val_f1
    wandb.run.summary[f'{dataset_name}_test_auc'] = val_auc



model = create_model(num_frames=args["num_frames"], in_channels=3, lsa=True, dropout=0.3, head_dims=args['head_dims'], heads=args['heads'], depth=args['heads'], dim=args['dim'])
model = model.to(device)
#model = torch.compile(model)

num_parameters = sum(p.numel() for p in model.parameters())
print(f"[INFO] Number of parameters in model : {num_parameters:,}")

wandb.watch(model)

class weighted_binary_cross_entropy(nn.Module):
    def __init__(self, weight=None):
        super(weighted_binary_cross_entropy, self).__init__()
    
    def forward(self, output, target):
        loss = (args['weight'] * (target * torch.log(output))) + (1 * ((1 - target) * torch.log(1 - output)))
        return torch.neg(torch.mean(loss))

criteria = weighted_binary_cross_entropy()
optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'], momentum=0.7)

previous_loss = np.inf
patience = 0

SAVED_ONCE = False
LOWEST_LOSS = np.inf

try:
    for epoch in range(args['epochs']):
        train_loss, train_acc, train_f1, train_precision, train_recall = train_epoch(model, train_loader, optimizer, criteria, epoch)
        val_loss, val_acc, val_f1, val_precision, val_recall = validate_epoch(model, test_loader, criteria, epoch)
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
                "Val Recall": val_recall,
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

if not SAVED_ONCE:
    print("[INFO] Saving Model")
    torch.save(model.state_dict(), args["save_path"] + ".pt")

# Inference

DFDC_PATH = './dfdc/videos_16/test_videos.csv'
CELEB_PATH = './celeb-df/videos_16/data_video.csv'

def test_dataset(test_path):
    print(f"[INFO] Testing on {test_path}")
    df = pd.read_csv(test_path)
    X = df['filename'].values
    y = df['label'].values

    inference_data = DataLoaderWrapper(X, y, transforms=None, batch_size=4, shuffle=False)
    
    dataset_name = test_path.split('/')[1]
    test(model, inference_data, dataset_name)

test_dataset(DFDC_PATH)
test_dataset(CELEB_PATH)

wandb.finish()

