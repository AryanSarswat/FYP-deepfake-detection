import gc

import numpy as np
import pandas as pd
import torch
import torchmetrics.classification as metrics
import torch.nn as nn
import torch.optim
from DatasetLoader.VideoDataset import DataLoaderWrapper
from models.EfficientNetV2 import create_model
from models.losses import *
from models.util import DataAugmentationImage, fix_random_seed
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

import wandb

# Fix seed for reproducibility
fix_random_seed(seed=16)

# Optimisations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#! Hyper parameters
# Debugging
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# Dataset
PATH_KODF = './KoDF/videos_16/data_video.csv'
PATH_FACEFORENSICS = './faceforensics/videos_16/data_video.csv' 
PATH_DFDC = './dfdc/videos_16/data_video.csv'
PATH_CELEB_DF = './celeb-df/videos_16/data_video.csv'
IMAGE_SIZE = 224
NUM_FRAMES = 16
AUGMENTATIONS = DataAugmentationImage(size=IMAGE_SIZE)

RGB = True
FFT = False
DCT = False
WAVELET = False
IN_CHANNELS = 3 # Default

assert (RGB ^ FFT ^ DCT ^ WAVELET), "Only one of RGB, FFT or DCT can be True"

if FFT:
    IN_CHANNELS = 6
elif DCT:
    IN_CHANNELS = 3
elif WAVELET:
    IN_CHANNELS = 3

# Model
MODEL_NAME = 'Baseline'
DEPTH = 8
DIM = 512
HEADS = 8
HEAD_DIM = 64
PATCH_SIZE = None
LSA = True
DROPOUT = 0.4

# Training 
WEIGHT = 0.4
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-2
MOMENTUM = 0.7
PATIENCE = 5
MIN_DELTA = 1e-3
NUM_EPOCHS = 50
LOG_WANDB = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Create Save path and Model name
MODEL_NAME = f"{MODEL_NAME}_{WEIGHT}"

if FFT:
    MODEL_NAME += "_fft_only"
elif DCT:
    MODEL_NAME += "_dct_only"
elif WAVELET:
    MODEL_NAME += "_wavelet_only"
else:
    MODEL_NAME += "_rgb"
    
if LSA:
    MODEL_NAME += "_lsa"

if AUGMENTATIONS != None:
    MODEL_NAME += "_aug"

SAVE_PATH = f"checkpoints/{MODEL_NAME}"

def get_dataset(path, training=False):
    df = pd.read_csv(path)
    X = df['filename'].values
    y = df['label'].values
    
    if training:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

        train_loader = DataLoaderWrapper(X_train, y_train, num_frames=NUM_FRAMES, height=IMAGE_SIZE, width=IMAGE_SIZE, transforms=AUGMENTATIONS, batch_size=BATCH_SIZE, shuffle=True, fft=FFT, dct=DCT, wavelet=WAVELET)
        test_loader = DataLoaderWrapper(X_test, y_test, num_frames=NUM_FRAMES, height=IMAGE_SIZE, width=IMAGE_SIZE, transforms=None, batch_size=BATCH_SIZE, fft=FFT, dct=DCT, wavelet=WAVELET)
        
        return train_loader, test_loader
    else:
        inference_data = DataLoaderWrapper(X, y, transforms=None, batch_size=BATCH_SIZE, shuffle=False, fft=FFT, dct=DCT, wavelet=WAVELET)
        return inference_data

def train_epoch(model, data_loader, optimizer, criteria, epoch, device, acc_metric, f1_metric, recall_metric, precision_metric, auc_metric):
    # Reset metrics
    acc_metric.reset()
    f1_metric.reset()
    recall_metric.reset()
    precision_metric.reset()
    auc_metric.reset()
    
    torch.cuda.empty_cache()
    gc.collect()
    
    model.train()
    epoch_loss = 0
    epoch_center_loss = 0
    epoch_acc = 0
    idx = 0
    scaler = torch.cuda.amp.GradScaler()
    
    pbar = tqdm(enumerate(data_loader), desc=f"Epoch {epoch} Training - Loss: {epoch_loss:.3f} - Running accuracy: {epoch_acc:.3f}", total=len(data_loader))
    
    preds = []
    targets = []
    
    
    for idx, (X, y_t, y_s) in pbar:
        X = X.to(device, non_blocking=True)
        y_t = y_t.to(device, non_blocking=True)
        y_s = y_s.to(device, non_blocking=True)
        
        
        #y_pred_t, y_pred_s, vectors = model(X)
        y_pred_t = model(X)
        
        #loss, center_loss = criteria(y_pred_t, y_pred_s, y_t, y_s, vectors)
        loss = criteria(y_pred_t, y_s)
        total_loss = loss
        total_loss = torch.clamp(total_loss, -5, 5)

        if total_loss.isnan().any():
            print("Nan loss, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue
        else:
            epoch_loss += loss.detach().cpu().item()
            #epoch_center_loss += center_loss.item()
            
            # Update every other batch simulating gradient accumulation
            if idx % 2 == 0 or idx == len(data_loader) - 1:
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
            y_pred_t = y_pred_t.detach().cpu()
            y_t = y_s.detach().cpu().int()
            
            preds.extend(y_pred_t)
            targets.extend(y_t)
            
            epoch_acc += acc_metric(y_pred_t, y_t)
                
        pbar.set_description(f"Epoch {epoch} Train - Loss: {epoch_loss / (idx + 1):.3f} - Running accuracy: {epoch_acc / (idx + 1):.3f}")
    
    preds = torch.stack(preds)
    targets = torch.stack(targets)
    
    train_loss = epoch_loss / (idx + 1)
    train_center_loss = epoch_center_loss / (idx + 1)
    train_acc = acc_metric(preds, targets)
    train_precision = precision_metric(preds, targets)
    train_recall = recall_metric(preds, targets)
    train_f1 = f1_metric(preds, targets)
    train_auc = auc_metric(preds, targets)
    
    print(f"Epoch {epoch} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Train F1: {train_f1:.4f} Train Precision: {train_precision:.4f} Train recall: {train_recall:.4f} Train AUC: {train_auc:.4f}")

    return train_loss, train_acc, train_f1, train_precision, train_recall, train_auc, train_center_loss

@torch.no_grad()
def validate_epoch(model, data_loader, criteria, epoch, device, acc_metric, f1_metric, recall_metric, precision_metric, auc_metric):
    # Reset metrics
    acc_metric.reset()
    f1_metric.reset()
    recall_metric.reset()
    precision_metric.reset()
    auc_metric.reset()
    
    preds = []
    targets = []
    
    model.eval()
    epoch_loss = 0
    epoch_center_loss = 0
    epoch_acc = 0
    idx = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader), desc=f"Epoch {epoch} Validation - Loss: {epoch_loss:.3f} - Running accuracy: {epoch_acc:.3f}", total=len(data_loader))
        
        for idx, (X, y_t, y_s) in pbar:
            X = X.to(device, non_blocking=True)
            y_t = y_t.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)

            #y_pred_t, y_pred_s, vectors = model(X)
            y_pred_t = model(X)
            
            #loss, center_loss = criteria(y_pred_t, y_pred_s, y_t, y_s, vectors)
            loss = criteria(y_pred_t, y_s)
            
            epoch_loss += loss.item()
            #epoch_center_loss += center_loss.item()

            y_pred_t = y_pred_t.detach().cpu()
            y_t = y_s.detach().cpu().int()
            
            epoch_acc += acc_metric(y_pred_t, y_t)
            
            preds.extend(y_pred_t)
            targets.extend(y_t)
            
            pbar.set_description(f"Epoch {epoch} Validation - Loss: {epoch_loss / (idx + 1):.3f} - Running accuracy: {epoch_acc / (idx + 1):.3f}")

    preds = torch.stack(preds)
    targets = torch.stack(targets)
    
    val_loss = epoch_loss / (idx + 1)
    val_acc = acc_metric(preds, targets)
    val_precision = precision_metric(preds, targets)
    val_recall = recall_metric(preds, targets)
    val_f1 = f1_metric(preds, targets)
    val_auc = auc_metric(preds, targets)
    val_center_loss = epoch_center_loss / (idx + 1)

    print(f"Epoch {epoch} Validation Loss: {val_loss:.4f} Validation Acc: {val_acc:.4f} Validation F1: {val_f1:.4f} Validation Precision: {val_precision:.4f} Validation recall: {val_recall:.4f}, Validation AUC: {val_auc:.4f}")

    return val_loss, val_acc, val_f1, val_precision, val_recall, val_auc, val_center_loss

def train_model(model, train_loader, test_loader, optimizer, scheduler, criteria, epochs, patience, min_delta, save_path, device, acc_metric, precision_metric, recall_metric, f1_metric, auc_metric):
    previous_loss = np.inf
    patience_counter = 0

    saved_once = False
    lowest_loss = np.inf
    
    try:
        for epoch in range(epochs):
            train_loss, train_acc, train_f1, train_precision, train_recall, train_auc, train_center_loss = train_epoch(model, train_loader, optimizer, criteria, epoch, device, acc_metric, precision_metric, recall_metric, f1_metric, auc_metric)
            val_loss, val_acc, val_f1, val_precision, val_recall, val_auc, val_center_loss = validate_epoch(model, test_loader, criteria, epoch, device, acc_metric, precision_metric, recall_metric, f1_metric, auc_metric)
            scheduler.step(val_loss)
            try:
                wandb.log({
                    "Train Loss": train_loss,
                    "Train Acc": train_acc,
                    "Train F1": train_f1,
                    "Train Precision": train_precision,
                    "Train Recall": train_recall,
                    "Train Center Loss": train_center_loss,
                    "Train AUC": train_auc,
                    "Val Loss": val_loss,
                    "Val Acc": val_acc,
                    "Val F1": val_f1,
                    "Val Precision": val_precision,
                    "Val Recall": val_recall,
                    "Val Center Loss": val_center_loss,
                    "Val AUC": val_auc,
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
def inference(model, data_loader, dataset_name, device, acc_metric, precision_metric, recall_metric, f1_metric, auc_metric):
    model.eval()

    # Reset metrics
    acc_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    auc_metric.reset()
    
    preds = []
    targets = []
    

    for idx, (X, y_t, y_s) in tqdm(enumerate(data_loader), desc=f"Inference", total=len(data_loader)):
        X = X.to(device, non_blocking=True)
        y_t = y_t.to(device, non_blocking=True)
        y_s = y_s.to(device, non_blocking=True)

        #y_pred_t, y_pred_s, vectors = model(X)
        y_pred_t = model(X)
        
        y_pred_t = y_pred_t.detach().cpu()
        y_t = y_s.detach().cpu().int()

        
        preds.extend(y_pred_t)
        targets.extend(y_t)
        
    preds = torch.stack(preds)
    targets = torch.stack(targets)        

    val_acc = acc_metric(preds, targets)
    val_precision = precision_metric(preds, targets)
    val_recall = recall_metric(preds, targets)
    val_f1 = f1_metric(preds, targets)
    val_auc = auc_metric(preds, targets)

    print(f"Test Acc:{val_acc:.4f} Test F1:{val_f1:.4f} Test Precision:{val_precision:.4f} Test Recall:{val_recall:.4f} Test AUC:{val_auc:.4f}")

    wandb.run.summary[f"{dataset_name}_test_acc"] = val_acc
    wandb.run.summary[f'{dataset_name}_test_precision'] = val_precision
    wandb.run.summary[f'{dataset_name}_test_recall'] = val_recall
    wandb.run.summary[f'{dataset_name}_test_f1'] = val_f1
    wandb.run.summary[f'{dataset_name}_test_auc'] = val_auc

if __name__ == "__main__":
    # Initialise datasets
    train_loader, test_loader = get_dataset(PATH_KODF, training=True)
    dfdc_loader = get_dataset(PATH_DFDC, training=False)
    celeb_df_loader = get_dataset(PATH_CELEB_DF, training=False)
    faceforensics_loader = get_dataset(PATH_FACEFORENSICS, training=False)

    # Initialise model
    MODEL = create_model(#num_frames=NUM_FRAMES, 
                         #in_channels=IN_CHANNELS, 
                         #dim=DIM, 
                         #lsa=LSA, 
                         #depth=DEPTH,
                         #heads=HEADS,
                         #head_dims=HEAD_DIM,
    )#dropout=DROPOUT)
    
    MODEL = MODEL.to(DEVICE)
    num_parameters = sum(p.numel() for p in MODEL.parameters())
    print(f"[INFO] Number of parameters in model : {num_parameters:,}")

    OPTIMIZER = torch.optim.AdamW(MODEL.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, mode='min', factor=0.1, patience=3, verbose=True)
    
    # Initialise Metrics

    accuracy = metrics.BinaryAccuracy()
    auroc = metrics.BinaryAUROC()
    f1 = metrics.BinaryF1Score()
    recall = metrics.BinaryRecall()
    precision = metrics.BinaryPrecision()

    # Initialise loss function
    CRITERIA = nn.BCEWithLogitsLoss()
    
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

    wandb_configs["experiment_name"] = f"{wandb_configs['architecture']}_frames_{wandb_configs['num_frames']}_batch_{wandb_configs['batch_size']}_lr_{wandb_configs['lr']}_loss_{CRITERIA}"

    if LOG_WANDB:
        wandb.init(project="deepfake-models-face-abalation", config=wandb_configs, name=wandb_configs["experiment_name"])
        wandb.watch(MODEL)

    # Train model
    MODEL = train_model(MODEL, train_loader, test_loader, OPTIMIZER, SCHEDULER, CRITERIA, NUM_EPOCHS, PATIENCE, MIN_DELTA, SAVE_PATH, DEVICE, acc_metric=accuracy, auc_metric=auroc, f1_metric=f1, recall_metric=recall, precision_metric=precision)

    # Inference
    inference(model=MODEL, data_loader=dfdc_loader, dataset_name="DFDC", device=DEVICE, acc_metric=accuracy, precision_metric=precision, recall_metric=recall, f1_metric=f1, auc_metric=auroc)
    inference(model=MODEL, data_loader=celeb_df_loader, dataset_name="Celeb_DF", device=DEVICE, acc_metric=accuracy, precision_metric=precision, recall_metric=recall, f1_metric=f1, auc_metric=auroc)
    inference(model=MODEL, data_loader=faceforensics_loader, dataset_name="FaceForenics++", device=DEVICE, acc_metric=accuracy, precision_metric=precision, recall_metric=recall, f1_metric=f1, auc_metric=auroc)

    # Finish wandb
    wandb.finish()

