from torch.utils.data import DataLoader
from src.dataset import GeosteeringDataset # Setzt voraus, dass src/ ein Python-Modul ist

# Teste das Dataset mit unserem bekannten Bohrloch
dataset = GeosteeringDataset(well_ids=["fc0d20b2"], data_dir="../data/train", window_size=50)

# Werfe es in einen PyTorch DataLoader (bündelt die Fenster in Batches)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Lade einen Batch
x_batch, y_batch = next(iter(dataloader))
print(f"X Tensor Shape: {x_batch.shape} -> (Batch_Size, Features, Window_Size)")
print(f"Y Tensor Shape: {y_batch.shape} -> (Batch_Size)")


import torch
print("CUDA verfügbar:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Gefundene GPU:", torch.cuda.get_device_name(0))
else:
    print("Leider immer noch CPU :(")