import argparse
import csv
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

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

def train_full_model(resume: bool = False):
    # ---------------------------------------------------------
    # 1. Setup & Hyperparameter
    # ---------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 128
    EPOCHS = 30
    LEARNING_RATE = 0.0005
    WINDOW_SIZE = 50
    SCHEDULER_PATIENCE = 5
    EARLY_STOP_PATIENCE = 15 # stop if val RMSE doesn't improve for this many epochs
    EARYLY_STOP_MIN_DELTA = 0.05 # ft, ignore improvements smaller than this as noise

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting full training on device: {device}")

    # ---------------------------------------------------------
    # 2. Automatically find all well IDs
    # ---------------------------------------------------------
    # os.path.normpath fixes our old Windows slash issues
    data_dir = os.path.normpath("../data/train")

    # Searches for all CSVs ending in __horizontal_well.csv
    csv_files = glob.glob(os.path.join(data_dir, "*__horizontal_well.csv"))

    all_wells = []
    for file in csv_files:
        # Extracts e.g. "fc0d20b2" from the full file path
        filename = os.path.basename(file)
        well_id = filename.split("__")[0]
        all_wells.append(well_id)

    print(f"Found a total of {len(all_wells)} wells in the training folder.")

    # ---------------------------------------------------------
    # 3. Train / Validation Split (80% / 20%)
    # ---------------------------------------------------------
    # IMPORTANT: We split by well IDs, NOT by individual rows!
    # This guarantees that the model sees completely new geology in the val set.
    train_wells, val_wells = train_test_split(all_wells, test_size=0.2, random_state=SEED)

    print(f"Training set: {len(train_wells)} wells")
    print(f"Validation set: {len(val_wells)} wells")

    # ---------------------------------------------------------
    # 4. Create Datasets & DataLoaders
    # ---------------------------------------------------------
    # tvt_mean/tvt_std are computed from the TRAINING wells only
    # We then pass those exact values into the validation dataset so both splits
    # are scaled identically and val statistics never leak into normalization
    train_dataset = GeosteeringDataset(well_ids=train_wells, data_dir=data_dir, window_size=WINDOW_SIZE)
    val_dataset = GeosteeringDataset(well_ids=val_wells,
        data_dir=data_dir,
        window_size=WINDOW_SIZE,
        gr_mean=train_dataset.gr_mean, gr_std=train_dataset.gr_std,
        z_mean=train_dataset.z_mean, z_std=train_dataset.z_std,
        tvt_mean=train_dataset.tvt_mean, tvt_std=train_dataset.tvt_std
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---------------------------------------------------------
    # 5. Model, Loss, and Optimizer
    # ---------------------------------------------------------
    model = GeosteeringHybridModel(num_features=2, window_size=WINDOW_SIZE).to(device)
    criterion = nn.HuberLoss()  # to train with MSE, use nn.MSELoss()
    metric_mse = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE)

    best_val_rmse = float('inf')  # Starts infinitely high
    epochs_no_improvement = 0
    start_epoch = 0
    tvt_std = train_dataset.tvt_std  # We need this to back-calculate the error

    model_dir = "../src/models"
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, "best_geosteering_model.pth")
    resume_path    = os.path.join(model_dir, "resume_checkpoint.pth")
    norm_stats_path = os.path.join(model_dir, "normalization_stats.json")
    log_path = os.path.join(model_dir, "training_log.csv")

    # ---------------------------------------------------------
    # 5b. Resume from checkpoint if --resume flag was passed
    # ---------------------------------------------------------
    if resume:
        if os.path.exists(resume_path):
            print(f"\nResume checkpoint found — loading '{resume_path}'...")
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch           = ckpt["epoch"]          # next epoch to run
            best_val_rmse         = ckpt["best_val_rmse"]
            epochs_no_improvement = ckpt["epochs_no_improvement"]
            print(f"Resuming from epoch {start_epoch + 1}/{EPOCHS}  "
                  f"(best val RMSE so far: {best_val_rmse:.2f} ft)\n")
            # Append to the existing log instead of overwriting it
            log_mode = "a"
        else:
            print(f"\n[WARNING] --resume was set but no checkpoint found at '{resume_path}'. "
                  f"Starting from scratch.\n")
            log_mode = "w"
    else:
        log_mode = "w"

    # Persist per-epoch metrics to disk, so training curves
    # and epoch-by-epoch numbers survive past the console/tqdm scrollback
    with open(log_path, log_mode, newline="") as f:
        if log_mode == "w":  # only write header for a fresh run
            csv.writer(f).writerow(["epoch", "train_rmse_ft", "val_rmse_ft", "lr"])

    # ---------------------------------------------------------
    # 6. The Main Loop
    # ---------------------------------------------------------
    for epoch in range(start_epoch, EPOCHS):
        # --- TRAINING ---
        model.train()
        train_running_mse = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [TRAIN] ", colour='green')

        for x_batch, y_batch in train_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                mse = metric_mse(predictions, y_batch)
                train_running_mse += mse.item()

        train_mse = train_running_mse / len(train_loader)
        # BACK-CALCULATION: Take the square root and multiply by the standard deviation,
        # so we can see the real error in feet (ft)!
        train_rmse = np.sqrt(train_mse) * tvt_std

        # --- VALIDATION ---
        model.eval()
        val_running_mse = 0.0

        with torch.no_grad():  # Do not compute gradients
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [VAL]", colour='blue')
            for x_batch, y_batch in val_bar:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                predictions = model(x_batch)

                mse = metric_mse(predictions, y_batch)
                val_running_mse += mse.item()

        val_mse = val_running_mse / len(val_loader)
        val_rmse = np.sqrt(val_mse) * tvt_std

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n --> Epoch {epoch + 1} | Train RMSE: {train_rmse:.2f} ft | Val RMSE: {val_rmse:.2f} ft")

        # Step the LR scheduler on validation RMSE: once val RMSE plateaus,
        # shrink the LR instead of continuing to overshoot with a fixed step size.
        scheduler.step(val_rmse)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{train_rmse:.4f}", f"{val_rmse:.4f}", f"{current_lr:.6f}"])

        # Save full training state so we can resume from this exact epoch later.
        # This overwrites the previous resume checkpoint each epoch.
        torch.save({
            "epoch":                epoch + 1,      # next epoch to run on resume
            "model_state":          model.state_dict(),
            "optimizer_state":      optimizer.state_dict(),
            "scheduler_state":      scheduler.state_dict(),
            "best_val_rmse":        best_val_rmse,
            "epochs_no_improvement": epochs_no_improvement,
        }, resume_path)
        # ---------------------------------------------------------
        # 7. Model Checkpointing & Early Stopping
        # ---------------------------------------------------------
        improvement = best_val_rmse - val_rmse
        
        # If the error on unseen data has decreased, we save it!
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), checkpoint_path)

            with open(norm_stats_path, "w", newline="") as f:
                json.dump({
                    "gr_mean": train_dataset.gr_mean,
                    "gr_std": train_dataset.gr_std,
                    "z_mean": train_dataset.z_mean,
                    "z_std": train_dataset.z_std,
                    "tvt_mean": train_dataset.tvt_mean,
                    "tvt_std": train_dataset.tvt_std,
                    "window_size": WINDOW_SIZE,
                    "use_relative_z": train_dataset.use_relative_z
                }, f, indent=2)
            print(f"New best model saved! (Val RMSE improved)")

        if improvement >= EARYLY_STOP_MIN_DELTA:
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= EARLY_STOP_PATIENCE:
            print(f"\nNo significant improvement for {EARLY_STOP_PATIENCE} epochs — stopping early, "
                  f"to avoid overfitting and save computation time.")
            break

    print(f"\nTraining complete! Best model had a Val RMSE of {best_val_rmse:.2f} ft.")

    # Clean up the resume checkpoint so a future plain run always starts fresh.
    # The file is intentionally kept on disk only when training was interrupted.
    if os.path.exists(resume_path):
        os.remove(resume_path)
        print("Resume checkpoint deleted — next run will start from scratch.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the full Geosteering model.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last saved resume checkpoint. "
             "If omitted, training always starts from scratch.",
    )
    args = parser.parse_args()
    train_full_model(resume=args.resume)