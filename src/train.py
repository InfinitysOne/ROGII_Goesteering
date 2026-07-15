import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import GeosteeringDataset
from model import GeosteeringHybridModel

def train_model():
    # Hyperparameter tuning
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001
    WINODOW_SIZE = 50

    # Identify the right device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start Training on {device}")

    # Load the dataset
    train_wells = ["fc0d20b2"]

    train_dataset = GeosteeringDataset(well_ids=train_wells, data_dir="../data/train", window_size=WINODOW_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model loss function and optimizer
    model = GeosteeringHybridModel(num_features=2, window_size=WINODOW_SIZE).to(device)

    # MSE Loss function
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # tqdm shows a progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", colour="green")

        for batch_idx, (x_batch, y_batch) in enumerate(progress_bar):
            # Move data to the device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward pass
            predictions = model(x_batch)

            # Calculate loss
            loss = criterion(predictions, y_batch)

            # Backward pass and optimization of weights
            optimizer.zero_grad() # Reset gradients
            loss.backward()       # Calculate gradients
            optimizer.step()      # Update weights

            # Loss tracking (Square root of MSE = RMSE)
            running_loss += loss.item()
            current_rmse = np.sqrt(loss.item())

            # Progress bar tracking
            progress_bar.set_postfix({'Batch_RMSE': f"{current_rmse:.2f}"})

        # Calculate the average RMSE for the epoch
        epoch_mse = running_loss / len(train_loader)
        epoch_rmse = np.sqrt(epoch_mse)
        print(f"--> Epoch {epoch+1} | Average RMSE: {epoch_rmse:.2f} ft\n")

        torch.save(model.state_dict(), "../src/models/baseline_geosteering_model.pth")
        print("Training completed! Model saved in 'models' ")

if __name__ == "__main__":
    train_model()