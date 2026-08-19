import os
import glob
import json
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GeosteeringDataset
from baseline_model import NaiveLinearBaseline

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_baseline():
    SEED = 42
    BATCH_SIZE = 128
    EPOCHS = 30
    LEARNING_RATE = 0.001
    WINDOW_SIZE = 50

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Naive Baseline on device: {device}")

    # 1. Load wells & split (80/20)
    data_dir = os.path.normpath("../data/train")
    csv_files = glob.glob(os.path.join(data_dir, "*__horizontal_well.csv"))
    all_wells = [os.path.basename(f).split("__")[0] for f in csv_files]
    train_wells, val_wells = train_test_split(all_wells, test_size=0.2, random_state=SEED)

    # 2. Datasets
    train_dataset = GeosteeringDataset(well_ids=train_wells, data_dir=data_dir, window_size=WINDOW_SIZE)
    val_dataset = GeosteeringDataset(
        well_ids=val_wells,
        data_dir=data_dir,
        window_size=WINDOW_SIZE,
        gr_mean=train_dataset.gr_mean, gr_std=train_dataset.gr_std,
        z_mean=train_dataset.z_mean, z_std=train_dataset.z_std,
        tvt_mean=train_dataset.tvt_mean, tvt_std=train_dataset.tvt_std
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Model, Loss, Optimizer
    model = NaiveLinearBaseline(num_features=2, window_size=WINDOW_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    tvt_std = train_dataset.tvt_std
    best_val_rmse = float('inf')

    # 4. Main Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_rmse = np.sqrt(train_loss / len(train_loader)) * tvt_std

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                preds = model(x_b)
                val_loss += criterion(preds, y_b).item()

        val_rmse = np.sqrt(val_loss / len(val_loader)) * tvt_std
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train RMSE: {train_rmse:.2f} ft | Val RMSE: {val_rmse:.2f} ft")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), "../src/models/baseline_geosteering_model.pth")

    print(f"\nTraining Complete. Best Baseline Val RMSE: {best_val_rmse:.2f} ft")

if __name__ == "__main__":
    train_baseline()