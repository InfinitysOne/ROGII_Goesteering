import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class GeosteeringDataset(Dataset):
    def __init__(self, well_ids, data_dir, window_size=50, is_test=False,
                 gr_norm = 150.0, z_norm = 10000.0, tvt_mean = None, tvt_std = None):
        """
        Pytorch Dataset for ROGII Geosteering dataset.

        Args:
        well_ids (list): list of well IDs (e.g ['fc0d20b2])
        data_dir (string): directory to load data from
        window_size (int): how many feet (MD) the model can look at simultaneously
        is_test (bool): true if we make predictions for submission
        gr_norm (float): GR scaling divisor (API units convention, ~0-150)
        z_norm (float): Z scaling divisor (depth convention, ft)
        tvt_mean (float or None): TVT mean used for target scaling. If None, it is computed from the wells
        passed in here (should only be left as None for the TRAINING dataset).
        tvt_std (float or None): see tvt_mean
        """
        self.window_size = window_size
        self.is_test = is_test
        self.samples = []

        self.gr_norm = gr_norm
        self.z_norm = z_norm

        print(f"Loading {len(well_ids)} borehole for {'Test' if is_test else 'Training'}-Set . . .")

        if tvt_mean is None or tvt_std is None:
            if is_test:
                raise ValueError(
                    "tvt_mean/tvt_std must be passed explicitly for a test dataset "
                    "(reuse the values computed on the training set)."
                )
            tvt_mean, tvt_std = self._compute_tvt_stats(well_ids, data_dir)
            print(f"Computed TVT stats from training wells: mean={tvt_mean}, std={tvt_std}")

        self.tvt_mean = tvt_mean
        self.tvt_std = tvt_std

        for well_id in well_ids:
            horiz_path = os.path.join(data_dir, f"{well_id}__horizontal_well.csv")

            horiz_path = os.path.normpath(horiz_path)

            # skip if datai do not exist
            if not os.path.exists(horiz_path):
                print(f"No data for {well_id} found. -> {horiz_path}")
                continue

            df = pd.read_csv(horiz_path)

            # 1. pre-processing: Fill holes in Gamma Ray (Forward Fill & Backward Fill)
            df['GR'] = df['GR'].ffill().bfill()

            # 2. Drop rows without ground truth before slicing features, so that
            #    'features' and 'targets' are always build from the exact same rows.
            if not self.is_test:
                df = df.dropna(subset=['TVT']).reset_index(drop=True)

            # 3. choose features (start with GR and depth of Z)
            features = df[['GR', 'Z']].values

            features[:, 0] = features[:, 0] / self.gr_norm  # GR 0 to 150
            features[:, 1] = features[:, 1] / self.z_norm  # -8000 to -10,000

            # 4. define targets
            if not self.is_test:
                # column 'TVT' is Ground Truth in our Training
                raw_targets = df["TVT"].values
                targets = (raw_targets - self.tvt_mean) / self.tvt_std
            else:
                # generate empty targets so we can predict them in the test-set
                targets = np.zeros(len(df))

            # Guard rail: catch any future regression bug
            assert len(features) == len(targets), (
                f"features/targets length mismatch for well {well_id}. "
                f"{len(features)} vs {len(targets)}."
            )

            # 5. Sliding Windows generation
            for i in range(len(features) - window_size):
                # X is sequenz of the last 'windows_size' Meter
                X_window = features[i : i + window_size]

                # y is the TVT the end of this exact window
                y_target = targets[i + window_size - 1]

                self.samples.append({
                    'x': X_window,
                    'y': y_target,
                })

    @staticmethod
    def _compute_tvt_stats(well_ids, data_dir):
        """ First-pass scan over the given wells' TVT columns to compute the actual mean/std to normalize
        against"""
        all_tvt = []
        for well_id in well_ids:
            horizon_path = os.path.normpath(os.path.join(data_dir, f"{well_id}__horizontal_well.csv"))
            if not os.path.exists(horizon_path):
                continue
            df = pd.read_csv(horizon_path, usecols=lambda c: c == "TVT")
            all_tvt.append(df['TVT'].dropna().values)
        all_tvt = np.concatenate(all_tvt) if all_tvt else np.array([0.0])
        mean = float(np.mean(all_tvt))
        std = float(np.std(all_tvt))
        if std == 0:
            std = 1.0 # avoid division by zero on degenerate inputs
        return mean, std

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