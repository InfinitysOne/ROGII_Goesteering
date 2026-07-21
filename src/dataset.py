import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class GeosteeringDataset(Dataset):
    def __init__(self, well_ids, data_dir, window_size=50, is_test=False, use_relative_z=False,
                 gr_mean = None, gr_std = None, z_mean = None, z_std = None, tvt_mean = None, tvt_std = None):
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
        self.use_relative_z = use_relative_z
        self.samples = []

        print(f"Loading {len(well_ids)} borehole for {'Test' if is_test else 'Training'}-Set . . .")

        if None in (tvt_mean, tvt_std, gr_mean, gr_std, z_mean, z_std):
            if is_test:
                raise ValueError(
                    "All mean/std parameters must be passed explicitly for a validation/test dataset "
                    "(reuse the values computed on the training set)."
                )
            stats = self._compute_global_stats(well_ids, data_dir, use_relative_z)
            tvt_mean, tvt_std, gr_mean, gr_std, z_mean, z_std = stats

            print("Computed global stats from training wells:")
            print(f"  TVT: mean={tvt_mean:.2f}, std={tvt_std:.2f}")
            print(f"  GR:  mean={gr_mean:.2f}, std={gr_std:.2f}")
            print(f"  Z:   mean={z_mean:.2f}, std={z_std:.2f}")

        self.tvt_mean, self.tvt_std = tvt_mean, tvt_std
        self.gr_mean, self.gr_std = gr_mean, gr_std
        self.z_mean, self.z_std = z_mean, z_std

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
            df['Z'] = df['Z'].ffill().bfill()

            # Relative Tiefe anwenden (Startwert auf 0 setzen)
            if self.use_relative_z:
                df['Z'] = df['Z'] - df['Z'].iloc[0]

            # 2. Drop rows without ground truth before slicing features, so that
            #    'features' and 'targets' are always build from the exact same rows.
            if not self.is_test:
                df = df.dropna(subset=['TVT']).reset_index(drop=True)

            # 3. choose features (start with GR and depth of Z)
            features = df[['GR', 'Z']].values.astype(np.float32)

            features[:, 0] = (features[:, 0] - self.gr_mean) / self.gr_std  # GR 0 to 150
            features[:, 1] = (features[:, 1] - self.z_mean) / self.z_std  # -8000 to -10,000

            # 4. define targets
            if not self.is_test:
                # column 'TVT' is Ground Truth in our Training
                raw_targets = df["TVT"].values.astype(np.float32)
                targets = (raw_targets - self.tvt_mean) / self.tvt_std
            else:
                # generate empty targets so we can predict them in the test-set
                targets = np.zeros(len(df), dtype=np.float32)

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
    def _compute_global_stats(well_ids, data_dir, use_relative_z):
        """Scans all training wells to compute the true mean and std for TVT, GR, and Z."""
        all_tvt, all_gr, all_z = [], [], []

        for well_id in well_ids:
            horizon_path = os.path.normpath(os.path.join(data_dir, f"{well_id}__horizontal_well.csv"))
            if not os.path.exists(horizon_path):
                continue

            df = pd.read_csv(horizon_path)
            df['GR'] = df['GR'].ffill().bfill()
            df['Z'] = df['Z'].ffill().bfill()

            if use_relative_z:
                df['Z'] = df['Z'] - df['Z'].iloc[0]

            if 'TVT' in df.columns:
                all_tvt.append(df['TVT'].dropna().values)
            all_gr.append(df['GR'].values)
            all_z.append(df['Z'].values)
        all_tvt = np.concatenate(all_tvt) if all_tvt else np.array([0.0])
        all_gr = np.concatenate(all_gr) if all_gr else np.array([0.0])
        all_z = np.concatenate(all_z) if all_z else np.array([0.0])

        def get_mean_std(arr):
            std = float(np.std(arr))
            return float(np.mean(arr)), std if std > 0 else 1.0
        tvt_mean, tvt_std = get_mean_std(all_tvt)
        gr_mean, gr_std = get_mean_std(all_gr)
        z_mean, z_std = get_mean_std(all_z)

        return tvt_mean, tvt_std, gr_mean, gr_std, z_mean, z_std

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