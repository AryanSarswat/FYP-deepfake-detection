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
from models.CvT import create_model
from models.util import DataAugmentationImage

# Optimisations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#! Hyper parameters

# Dataset
PATH_KODF = 'videos_16/data_video.csv'
PATH_FACEFORENSICS = None
PATH_DFDC = './dfdc/videos_16/test_videos.csv'
PATH_CELEB_DF = './celeb-df/videos_16/data_video.csv'
IMAGE_SIZE = 224
NUM_FRAMES = 16
AUGMENTATIONS = DataAugmentationImage(size=IMAGE_SIZE)

RGB = False
FFT = True
DCT = False
IN_CHANNELS = 3 # Default

assert not (RGB ^ FFT ^ DCT), "Only one of RGB, FFT or DCT can be True"

if FFT:
    IN_CHANNELS = 9
elif DCT:
    IN_CHANNELS = 3

# Model
MODEL_NAME = 'CvT'
DEPTH = 6
DIM = 256
HEADS = 6
HEAD_DIM = 128
PATCH_SIZE = None
LSA = True
DROPOUT = 0.3


# Training 
WEIGHT = 0.45
BATCH_SIZE = 8
LR = 2e-4
WEIGHT_DECAY = 1e-3
MOMENTUM = 0.7
PATIENCE = 5
MIN_DELTA = 1e-3
NUM_EPOCHS = 50
LOG_WANDB = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Create Save path and Model name
MODEL_NAME = f"{MODEL_NAME}_{WEIGHT}_weighted"

if FFT:
    MODEL_NAME += "_fft"
elif DCT:
    MODEL_NAME += "_dct"
else:
    MODEL_NAME += "_rgb"
    
if LSA:
    MODEL_NAME += "_lsa"

if AUGMENTATIONS != None:
    MODEL_NAME += "_aug"

SAVE_PATH = f"checkpoints/{MODEL_NAME}"

class weighted_binary_cross_entropy(nn.Module):
    def __init__(self, weight=None):
        super(weighted_binary_cross_entropy, self).__init__()
        if weight is not None:
            self.loss = lambda output, target: (weight * (target * torch.log(output))) + (1 * ((1 - target) * torch.log(1 - output)))
    
    def forward(self, output, target):
        loss = self.loss(output, target)
        return torch.neg(torch.mean(loss))

class SCLoss(nn.Module):
    def __init__(self, device, dim, weight=None) -> None:
        super().__init__()
        if weight is not None:
            self.softmax = lambda output, target: (weight * (target * torch.log(output))) + ((1 - target) * torch.log(1 - output))
        else:
            self.softmax = nn.BCELoss()
        self.real_center = nn.Parameter(torch.randn(1, dim), requires_grad=True).to(device)
        self.euc_dist = lambda x, y: torch.sqrt(torch.sum((x - y) ** 2, dim=1))
        self.dim = dim
        self.lamb = 0.5
        self.m = 0.3
    
    def forward(self, output, target, vectors):
        loss = self.softmax(output, target)
        loss = torch.neg(torch.mean(loss))
        
        real_indexes = torch.where(target == 0)[0]
        fake_indexes = torch.where(target == 1)[0]
        
        real_vectors = vectors[real_indexes]
        fake_vectors = vectors[fake_indexes]
        
        # Calculate Center Loss
        
        m_real = torch.mean(real_vectors - self.real_center, dim=0)
        m_fake = torch.mean(fake_vectors - self.real_center, dim=0)
        L = m_real - m_fake + self.m * np.sqrt(self.dim)
        center_loss = m_real + max(L, 0)
        
        return loss + self.lamb * center_loss

def get_dataset(path, training=False):
    df = pd.read_csv(path)
    X = df['video_path'].values
    y = df['label'].values
    
    if training:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

        train_loader = DataLoaderWrapper(X_train, y_train, num_frames=NUM_FRAMES, height=IMAGE_SIZE, width=IMAGE_SIZE, transforms=AUGMENTATIONS, batch_size=BATCH_SIZE, shuffle=True, fft=FFT, dct=DCT)
        test_loader = DataLoaderWrapper(X_test, y_test, num_frames=NUM_FRAMES, height=IMAGE_SIZE, width=IMAGE_SIZE, transforms=None, batch_size=BATCH_SIZE, fft=FFT, dct=DCT)
        
        return train_loader, test_loader
    else:
        inference_data = DataLoaderWrapper(X, y, transforms=None, batch_size=BATCH_SIZE, shuffle=False, fft=FFT, dct=DCT)
        return inference_data

def train_epoch(model, data_loader, optimizer, criteria, epoch, device):
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
        
        y_pred, vectors = model(X)
        loss = criteria(y_pred, y, vectors)
            
            
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
def validate_epoch(model, data_loader, criteria, epoch, device):
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

def train_model(model, train_loader, test_loader, optimizer, criteria, epochs, patience, min_delta, save_path, device):
    previous_loss = np.inf
    patience_counter = 0

    saved_once = False
    lowest_loss = np.inf
    
    try:
        for epoch in range(epochs):
            train_loss, train_acc, train_f1, train_precision, train_recall = train_epoch(model, train_loader, optimizer, criteria, epoch, device)
            val_loss, val_acc, val_f1, val_precision, val_recall = validate_epoch(model, test_loader, criteria, epoch, device)
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
            except Exception as e:
                print("[ERROR] Could not upload data, temporary connection loss or something else")
                print(e)
            
            delta = val_loss - previous_loss
            
            if val_loss < lowest_loss:
                torch.save(model.state_dict(), save_path + ".pt")
                lowest_loss = val_loss
                saved_once = True

            if abs(delta) < min_delta or val_loss > lowest_loss:
                patience_counter += 1
                if val_loss < previous_loss:
                    print(f"[INFO] Validation Loss improved by {delta:.2e}")
                else:
                    print(f"[INFO] Validation Loss worsened by {delta:.2e}")
            else:
                patience_counter = 0
            
            if patience_counter >= patience:
                print("[INFO] No improvement in Validation loss Early Stopping")
                break
            
            previous_loss = val_loss
            print(f"[INFO] Number of times loss has not improved : {patience_counter}")
    except KeyboardInterrupt:
        print("[ERROR] Training Interrupted")
        print("[INFO] Saving Model as Interrupted")
        torch.save(model.state_dict(), save_path + "_interrupted.pt")
        
    print("[INFO] Training Complete")

    if not saved_once:
        print("[INFO] Saving Model")
        torch.save(model.state_dict(), save_path + ".pt")
        
    return model

@torch.no_grad()
def inference(model, data_loader, dataset_name, device):
    model.eval()

    preds = []
    actual = []
    
    val_acc = 0
    val_precision = 0
    val_recall = 0
    val_f1 = 0
    val_auc = 0

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

# Initialise datasets
train_loader, test_loader = get_dataset(PATH_DFDC, training=True)

dfdc_loader = get_dataset(PATH_DFDC, training=False)
celeb_df_loader = get_dataset(PATH_CELEB_DF, training=False)
# faceforensics_loader = get_dataset(PATH_FACEFORENSICS, training=False)

# Initialise model
MODEL = create_model(num_frames=NUM_FRAMES, in_channels=IN_CHANNELS, lsa=LSA, dropout=DROPOUT, head_dims=HEAD_DIM, heads=HEADS, depth=DEPTH, dim=DIM)
MODEL = MODEL.to(DEVICE)
num_parameters = sum(p.numel() for p in MODEL.parameters())
print(f"[INFO] Number of parameters in model : {num_parameters:,}")

OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)

# Initialise wandb
wandb_configs = {
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "num_frames" : NUM_FRAMES,
    "architecture": MODEL_NAME,
    "save_path": SAVE_PATH,
    "optimizer": str(OPTIMIZER),
    "patience" : PATIENCE,
    "lr" : LR,
    "weight_decay": WEIGHT_DECAY,
    "min_delta" : MIN_DELTA,
    'patch_size' : PATCH_SIZE,
    'dim': DIM,
    'heads' : HEADS,
    'head_dims' : HEAD_DIM,
    'depth': DEPTH,
    'weight' : WEIGHT,
}

wandb_configs["experiment_name"] = f"{wandb_configs['architecture']}_frames_{wandb_configs['num_frames']}_batch_{wandb_configs['batch_size']}_lr_{wandb_configs['lr']}"

if LOG_WANDB:
    wandb.init(project="deepfake-baseline", config=wandb_configs, name=wandb_configs["experiment_name"])
    wandb.watch(MODEL)

# Initialise loss function
CRITERIA = SCLoss(DEVICE, DIM)

# Train model
MODEL = train_model(MODEL, train_loader, test_loader, OPTIMIZER, CRITERIA, NUM_EPOCHS, PATIENCE, MIN_DELTA, SAVE_PATH, DEVICE)

# Inference
inference(model=MODEL, data_loader=dfdc_loader, dataset_name="DFDC", device=DEVICE)
inference(model=MODEL, data_loader=celeb_df_loader, dataset_name="Celeb_DF", device=DEVICE)
#inference(model=MODEL, data_loader=faceforenics_loader, dataset_name="FaceForenics++", device=DEVICE)

# Finish wandb
wandb.finish()

