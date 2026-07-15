import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

# Unsere eigenen Module
from dataset import GeosteeringDataset
from model import GeosteeringHybridModel


def train_full_model():
    # ---------------------------------------------------------
    # 1. Setup & Hyperparameter
    # ---------------------------------------------------------
    BATCH_SIZE = 256  # Größerer Batch für schnelleres Training
    EPOCHS = 15  # Mehr Epochen, da wir jetzt echte Validierung haben
    LEARNING_RATE = 0.001
    WINDOW_SIZE = 50

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
    train_wells, val_wells = train_test_split(all_wells, test_size=0.2, random_state=42)

    print(f"Trainings-Set: {len(train_wells)} Bohrlöcher")
    print(f"Validation-Set: {len(val_wells)} Bohrlöcher")

    # ---------------------------------------------------------
    # 4. Datasets & DataLoaders erstellen
    # ---------------------------------------------------------
    train_dataset = GeosteeringDataset(well_ids=train_wells, data_dir=data_dir, window_size=WINDOW_SIZE)
    val_dataset = GeosteeringDataset(well_ids=val_wells, data_dir=data_dir, window_size=WINDOW_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---------------------------------------------------------
    # 5. Modell, Loss und Optimizer
    # ---------------------------------------------------------
    model = GeosteeringHybridModel(num_features=2, window_size=WINDOW_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_rmse = float('inf')  # Startet unendlich hoch
    tvt_std = train_dataset.tvt_std  # Wir brauchen das, um den Fehler zurückzurechnen

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

        print(f"--> Epoche {epoch + 1} | Train RMSE: {train_rmse:.2f} ft | Val RMSE: {val_rmse:.2f} ft")

        # ---------------------------------------------------------
        # 7. Model Checkpointing (Bestes Modell speichern)
        # ---------------------------------------------------------
        # Wenn der Fehler auf den unbekannten Daten kleiner geworden ist, speichern wir!
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), "../models/best_geosteering_model.pth")
            print(f"    🌟 Neues bestes Modell gespeichert! (Val RMSE verbessert)")

    print(f"\nTraining komplett! Bestes Modell hatte einen Val RMSE von {best_val_rmse:.2f} ft.")


if __name__ == "__main__":
    train_full_model()