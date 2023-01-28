import albumentations
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from tqdm import tqdm


from DatasetLoader.VideoDataset import DataLoaderWrapper
from models.ViViT import create_model, load_model

@torch.no_grad()
def validate(model, data_loader):
    model.eval()
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1 = 0
    idx = 0
    
    preds = []
    actual = []
    
    with torch.no_grad():
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
            
            epoch_acc += accuracy_score(y_cpu, y_pred_cpu)
            epoch_precision += precision_score(y_cpu, y_pred_cpu, zero_division=0)
            epoch_recall += recall_score(y_cpu, y_pred_cpu, zero_division=0)
            epoch_f1 += f1_score(y_cpu, y_pred_cpu, zero_division=0)

    val_acc = epoch_acc / (idx + 1)
    val_precision = epoch_precision / (idx + 1)
    val_recall = epoch_recall / (idx + 1)
    val_f1 = epoch_f1 / (idx + 1)

    print(f"Test Acc: {val_acc:.4f} Test F1: {val_f1:.4f} Test Precision: {val_precision:.4f} Test recall: {val_recall:.4f}")
    print(classification_report(actual, preds))

    return val_acc, val_f1, val_precision, val_recall

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model = create_model(num_frames=16, height=224, width=224, patch_size=16, dim=256, depth=6, heads=6, head_dims=256, dropout=0, scale_dim=4, lsa=True, in_channels=6) 
    model = load_model(base_model, 'checkpoints/ViViT_weighted_spt_lsa.pt')
    model = model.to(device)
    model.eval()
    
    PATH = './dfdc/videos_16/test_videos.csv' 

    df = pd.read_csv(PATH)
    print(df.head())
    X = df['filename'].values
    y = df['label'].values

    print(f"X: {X.shape}, y: {y.shape}")
    
    test_transforms = albumentations.Compose([
        albumentations.Normalize(),
    ])

    train_loader = DataLoaderWrapper(X, y, transforms=test_transforms, batch_size=8, shuffle=False)
    
    validate(model, train_loader)
    
    
    
