import os
import torch
import numpy as np
import pandas as pd
from pyexpat import features

from sympy.core.random import sample
from torch.utils.data import Dataset

class GeosteeringDataset(Dataset):
    def __init__(self, well_ids, data_dir, window_size=50, is_test=False):
        """
        Pytorch Dataset for ROGII Geosteering dataset.

        Args:
        well_ids (list): list of well IDs (e.g ['fc0d20b2])
        data_dir (string): directory to load data from
        window_size (int): how many feet (MD) the model can look at simultaneously
        is_test (bool): true if we make predictions for submission
        """
        self.window_size = window_size
        self.is_test = is_test
        self.samples = []

        print(f"Loading {len(well_ids)} borehole for {'Test' if is_test else 'Training'}-Set . . .")

        for well_id in well_ids:
            horiz_path = os.path.join(data_dir, f"{well_id}__horizontal_well.csv")

            # skip if datai do not exist
            if not os.path.exists(horiz_path):
                continue

            df = pd.read_csv(horiz_path)

            # 1. pre-processing: Fill holes in Gamma Ray (Forward Fill & Backward Fill)
            df['GR'] = df['GR'].ffill().bfill()

            # 2. define variables
            if not self.is_test:
                # column 'TVT' is Ground Truth in our Training
                # filter rows/columns if no TVT is there
                df = df.dropna(subset=['TVT']).reset_index(drop=True)
                targets = df['TVT'].values
            else:
                # generate empty targets so we can predict them in the test-set
                targets = np.zeros(len(df))

            # 3. choose features (start with GR and depth of Z)
            features = df[['GR', 'Z']].values

            features[:, 0] = features[:, 0] / 150.0  # GR 0 to 150
            features[:, 1] = features[:, 1] / 10000.0  # -8000 to -10,000

            # 4. Sliding Windows generation
            for i in range(len(features) - window_size):
                # X is sequenz of the last 'windows_size' Meter
                X_window = features[i : i + window_size]

                # y is the TVT the end of this exact window
                y_target = targets[i + window_size - 1]

                self.samples.append({
                    'x': X_window,
                    'y': y_target,
                })

    def __len__(self):
        """ Returns how many training samples we have """
        return len(self.samples)

    def __getitem__(self, idx):
        """ Returns one sample """
        sample = self.samples[idx]

        # Convert Array to Pytorch Tensor
        x_tensor = torch.tensor(sample['x'], dtype=torch.float32)
        y_tensor = torch.tensor(sample['y'], dtype=torch.float32)

        # transpose array to (Channel, Sequence_Length)
        x_tensor = x_tensor.transpose(0,1)

        return x_tensor, y_tensor