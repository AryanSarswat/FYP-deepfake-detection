from models.CvT import load_model
from DatasetLoader.ImageDataset import DataLoaderWrapper

from tqdm import tqdm

import torch
import albumentations
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

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
            X = X.to(device)
            y = y.to(device)

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
    
    model = load_model('efficient_net_baseline.pth')
    model = model.to(device)
    model.eval()
    print(model)
    
    PATH = './dfdc/images/data.csv' 

    df = pd.read_csv(PATH)
    X = df['image_path'].values
    y = df['label'].values

    print(f"X: {X.shape}, y: {y.shape}")
    
    test_transforms = albumentations.Compose([
        albumentations.Resize(256, 256),
        albumentations.Normalize(),
    ])

    train_loader = DataLoaderWrapper(X, y, transforms=test_transforms, batch_size=32, shuffle=False)
    
    validate(model, train_loader)
    
    
    