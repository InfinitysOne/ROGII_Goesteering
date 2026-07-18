import os
import glob
import json
import csv
import random

import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.multivariate import factor
from torch.ao.pruning import scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

# Unsere eigenen Module
from dataset import GeosteeringDataset
from model import GeosteeringHybridModel

def set_seed(seed: int):
    """Seed every source of randomness we control, so re-running this script
    reproduces the same split, initialization, and training trajectory."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_full_model():
    # ---------------------------------------------------------
    # 1. Setup & Hyperparameter
    # ---------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 256  # Größerer Batch für schnelleres Training
    EPOCHS = 30  # Mehr Epochen, da wir jetzt echte Validierung haben
    LEARNING_RATE = 0.001
    WINDOW_SIZE = 50
    EARLY_STOP_PATIENCE = 7 # stop if val RMSE doesn't improve for this many epochs

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starte vollständiges Training auf Gerät: {device}")

    # ---------------------------------------------------------
    # 2. Alle Bohrloch-IDs automatisch finden
    # ---------------------------------------------------------
    # os.path.normpath behebt unsere alten Windows-Slash-Probleme
    data_dir = os.path.normpath("../data/train")

    # Sucht alle CSVs, die auf __horizontal_well.csv enden
    csv_files = glob.glob(os.path.join(data_dir, "*__horizontal_well.csv"))

    all_wells = []
    for file in csv_files:
        # Extrahiert z.B. "fc0d20b2" aus dem ganzen Dateipfad
        filename = os.path.basename(file)
        well_id = filename.split("__")[0]
        all_wells.append(well_id)

    print(f"Insgesamt {len(all_wells)} Bohrlöcher im Trainingsordner gefunden.")

    # ---------------------------------------------------------
    # 3. Train / Validation Split (80% / 20%)
    # ---------------------------------------------------------
    # WICHTIG: Wir splitten nach Bohrloch-IDs, NICHT nach einzelnen Zeilen!
    # So garantieren wir, dass das Modell im Val-Set komplett neue Geologie sieht.
    train_wells, val_wells = train_test_split(all_wells, test_size=0.2, random_state=SEED)

    print(f"Trainings-Set: {len(train_wells)} Bohrlöcher")
    print(f"Validation-Set: {len(val_wells)} Bohrlöcher")

    # ---------------------------------------------------------
    # 4. Datasets & DataLoaders erstellen
    # ---------------------------------------------------------
    # tvt_mean/tvt_std are computed from the TRAINING wells only
    # We then pass those exact values into the validation dataset so both splits
    # are scaled identically and val statistics never leak into normalization
    train_dataset = GeosteeringDataset(well_ids=train_wells, data_dir=data_dir, window_size=WINDOW_SIZE)
    val_dataset = GeosteeringDataset(
        well_ids=val_wells, data_dir=data_dir, window_size=WINDOW_SIZE,
        gr_norm=train_dataset.gr_norm, z_norm=train_dataset.z_norm,
        tvt_std=train_dataset.tvt_std, tvt_mean=train_dataset.tvt_mean,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---------------------------------------------------------
    # 5. Modell, Loss und Optimizer
    # ---------------------------------------------------------
    model = GeosteeringHybridModel(num_features=2, window_size=WINDOW_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_rmse = float('inf')  # Startet unendlich hoch
    epochs_no_improvement = 0
    tvt_std = train_dataset.tvt_std  # Wir brauchen das, um den Fehler zurückzurechnen

    model_dir = "../src/models"
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, "best_geosteering_model.pth")
    norm_stats_path = os.path.join(model_dir, "normalization_stats.json")
    log_path = os.path.join(model_dir, "training_log.csv")

    # Persist per-epoch metrics to disk, so training curves
    # and epoch-by-epoch numbers survive past the console/tqdm scrollback
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_rmse_ft", "val_rmse_ft", "lr"])

    # ---------------------------------------------------------
    # 6. Die Haupt-Schleife
    # ---------------------------------------------------------
    for epoch in range(EPOCHS):
        # --- TRAINING ---
        model.train()
        train_running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [TRAIN]", colour='green')

        for x_batch, y_batch in train_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

        train_mse = train_running_loss / len(train_loader)

        # RÜCKRECHNUNG: Wurzel ziehen und mit Standardabweichung multiplizieren,
        # damit wir den echten Fehler in Fuß (ft) sehen!
        train_rmse = np.sqrt(train_mse) * tvt_std

        # --- VALIDATION ---
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():  # Keine Gradienten berechnen
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [VAL]", colour='blue')
            for x_batch, y_batch in val_bar:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                val_running_loss += loss.item()

        val_mse = val_running_loss / len(val_loader)
        val_rmse = np.sqrt(val_mse) * tvt_std

        current_lr = optimizer.param_groups[0]['lr']
        print(f"--> Epoche {epoch + 1} | Train RMSE: {train_rmse:.2f} ft | Val RMSE: {val_rmse:.2f} ft")

        scheduler.step(val_rmse)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch +1, f"{train_rmse:.4f}", f"{val_rmse:.4f}", f"{current_lr:.6f}"])
        # ---------------------------------------------------------
        # 7. Model Checkpointing (Bestes Modell speichern)
        # ---------------------------------------------------------
        # Wenn der Fehler auf den unbekannten Daten kleiner geworden ist, speichern wir!
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)

            with open(norm_stats_path, "w", newline="") as f:
                json.dump({
                    "gr_norm": train_dataset.gr_norm,
                    "z_norm": train_dataset.z_norm,
                    "tvt_mean": train_dataset.tvt_mean,
                    "tvt_std": train_dataset.tvt_std,
                    "window_size": WINDOW_SIZE,
                }, f, indent=2)
            print(f"Neues bestes Modell gespeichert! (Val RMSE verbessert)")
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= EARLY_STOP_PATIENCE:
                print(f"\n Keine Verbesserung seit {EARLY_STOP_PATIENCE} Epochen - stoppe frühzeitig, "
                      f"um Overfitting/Rechenzeit zu vermeiden")
                break

    print(f"\nTraining komplett! Bestes Modell hatte einen Val RMSE von {best_val_rmse:.2f} ft.")


if __name__ == "__main__":
    train_full_model()