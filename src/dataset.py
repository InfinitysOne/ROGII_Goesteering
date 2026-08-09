import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# =====================================================================
# Shared preprocessing helpers (used by training AND inference)
# =====================================================================

def prepare_well_dataframe(df):
    """Fill sensor holes and derive dZ. Works for train and test wells."""
    df = df.copy()
    df['GR'] = pd.to_numeric(df['GR'], errors='coerce').ffill().bfill()
    df['Z'] = pd.to_numeric(df['Z'], errors='coerce').ffill().bfill()
    df['dZ'] = df['Z'].diff().fillna(0.0)
    return df


def robust_norm_stats(arr):
    """Robust location/scale (median, IQR/1.349) for per-signal GR
    standardization. Falls back to std, then 1.0, if the IQR degenerates.

    WHY per-signal instead of global GR stats: real wells showed that
    horizontal log and typewell can sit on entirely different GR
    calibrations (observed: horizontal mean 53.6 API vs typewell 95.8 API
    for the SAME well pair — a 1.5-sigma offset under global stats), and
    even for normal pairs the horizontal log is systematically smoother
    (std ratio ~0.6-0.7). Standardizing every signal by its OWN robust
    stats makes the correlation task purely shape-based and
    calibration-invariant — exactly how a human geosteerer rescales logs
    before matching — and structurally removes the per-well GR fingerprint.
    """
    arr = np.asarray(arr, dtype=np.float64)
    med = np.nanmedian(arr)
    q75, q25 = np.nanpercentile(arr, [75, 25])
    scale = (q75 - q25) / 1.349
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(arr))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    if not np.isfinite(med):
        med = 0.0
    return float(med), float(scale)


def load_typewell(path):
    """Load a typewell CSV and return (tvt, gr_normalized), sorted by TVT.

    The typewell GR is standardized by ITS OWN robust stats (median,
    IQR/1.349) — see robust_norm_stats for why.
    """
    df = pd.read_csv(path)
    df['TVT'] = pd.to_numeric(df['TVT'], errors='coerce')
    df['GR'] = pd.to_numeric(df['GR'], errors='coerce')
    df = df.dropna(subset=['TVT', 'GR']).sort_values('TVT')
    # collapse duplicate TVT entries (np.interp needs increasing x)
    df = df.groupby('TVT', as_index=False)['GR'].mean()
    tvt = df['TVT'].to_numpy(np.float64)
    gr = df['GR'].to_numpy(np.float64)
    med, scale = robust_norm_stats(gr)
    return tvt, (gr - med) / scale


def make_type_excerpt(type_tvt, type_gr_n, center, offsets, pos_channel):
    """
    Extract the typewell excerpt around `center` (in TVT-ft).

    Channels:
      0: typewell GR (normalized), linearly interpolated onto the fixed grid
         center + offsets. Outside the typewell coverage, np.interp clamps
         to the edge value.
      1: position channel (offsets / half_range, in [-1, 1]) — tells the
         CNN where in the excerpt each sample sits relative to the center.

    Returns float32 (2, type_len).
    """
    gr = np.interp(center + offsets, type_tvt, type_gr_n)
    return np.stack([gr, pos_channel]).astype(np.float32)


class GeosteeringDataset(Dataset):
    """
    Typewell-correlation (registration) dataset.

    THE TASK: instead of regressing TVT directly (which proved unlearnable
    from GR + trajectory alone), the model now solves the task a human
    geosteerer solves: match the current horizontal GR log against the
    typewell GR signature.

    For each window ending at index i (with known true TVT t_i):
      * Horizontal branch input: (GR, Z_rel, dZ) window as before.
      * Typewell branch input: a GR excerpt of the typewell, interpolated on
        a fixed grid of `type_len` samples spanning +-`type_half_range` ft
        around a DELIBERATELY MIS-CENTERED position
            center = t_i + jitter,  jitter ~ U(-jitter_ft, +jitter_ft).
      * Target: the normalized registration correction
            y = (t_i - center) / jitter_ft  = -jitter / jitter_ft  in [-1, 1].

    The model can only solve this by correlating the two GR signals — the
    trajectory alone contains zero information about the jitter. At
    inference, the excerpt is centered on the current dead-reckoned TVT
    estimate and the predicted correction re-registers the estimate against
    the typewell at every step (self-correcting navigation).

    Args:
      augment: enables (train split only)
        - fresh random jitter each __getitem__ (otherwise a FIXED per-sample
          jitter drawn once with a constant seed, so val loss is comparable
          across epochs)
        - light Gaussian noise (gr_noise_std) on both GR channels and a
          random gain (1 +- gr_gain_range) on the horizontal GR. Since GR is
          standardized PER SIGNAL (robust median/IQR), level and amplitude
          no longer carry well identity; a constant-offset augmentation is
          therefore obsolete, and the mild gain merely hardens the matching
          against imperfect robust-scale estimates.
      zrel_mean/zrel_std, dz_mean/dz_std: leave as None ONLY for the
        training dataset; pass training values for val explicitly. (GR has
        no global stats anymore — every horizontal log and every typewell is
        standardized by its own robust median/IQR, see robust_norm_stats.)
    """

    def __init__(self, well_ids, data_dir, window_size=100, is_test=False,
                 augment=False, gr_noise_std=0.05, gr_gain_range=0.1,
                 jitter_ft=25.0, type_half_range=50.0, type_len=128,
                 zrel_mean=None, zrel_std=None,
                 dz_mean=None, dz_std=None):
        self.window_size = window_size
        self.is_test = is_test
        self.augment = augment
        self.gr_noise_std = gr_noise_std
        self.gr_gain_range = gr_gain_range
        self.jitter_ft = jitter_ft
        self.type_half_range = type_half_range
        self.type_len = type_len

        self.offsets = np.linspace(-type_half_range, type_half_range,
                                   type_len).astype(np.float64)
        self.pos_channel = (self.offsets / type_half_range).astype(np.float32)

        print(f"Loading {len(well_ids)} borehole(s) for "
              f"{'Test' if is_test else 'Training'}-Set . . .")

        if None in (zrel_mean, zrel_std, dz_mean, dz_std):
            if is_test:
                raise ValueError(
                    "All mean/std parameters must be passed explicitly for a "
                    "validation/test dataset (reuse the training values)."
                )
            (zrel_mean, zrel_std,
             dz_mean, dz_std) = self._compute_global_stats(
                well_ids, data_dir, window_size)
            print("Computed global stats from training wells:")
            print(f"  Zrel: mean={zrel_mean:.2f}, std={zrel_std:.2f}")
            print(f"  dZ:   mean={dz_mean:.4f}, std={dz_std:.4f}")
            print("  GR:   per-signal robust normalization "
                  "(median/IQR of each log)")

        self.zrel_mean, self.zrel_std = zrel_mean, zrel_std
        self.dz_mean, self.dz_std = dz_mean, dz_std

        self.wells = []
        self.index = []

        for well_id in well_ids:
            horiz_path = os.path.normpath(
                os.path.join(data_dir, f"{well_id}__horizontal_well.csv"))
            type_path = os.path.normpath(
                os.path.join(data_dir, f"{well_id}__typewell.csv"))
            if not os.path.exists(horiz_path):
                print(f"No horizontal data for {well_id} — skipped.")
                continue
            if not os.path.exists(type_path):
                print(f"No TYPEWELL for {well_id} — skipped "
                      f"(required for the correlation task).")
                continue

            df = prepare_well_dataframe(pd.read_csv(horiz_path))
            n = len(df)
            if n < window_size:
                print(f"Well {well_id} shorter than window_size — skipped.")
                continue

            type_tvt, type_gr_n = load_typewell(type_path)
            if len(type_tvt) < 2:
                print(f"Typewell of {well_id} unusable — skipped.")
                continue

            gr_raw = df['GR'].to_numpy(np.float64)
            gr_med, gr_scale = robust_norm_stats(gr_raw)
            gr_n = ((gr_raw - gr_med) / gr_scale).astype(np.float32)
            dz_n = ((df['dZ'].to_numpy(np.float64) - dz_mean) / dz_std
                    ).astype(np.float32)
            z_raw = df['Z'].to_numpy(np.float64).astype(np.float32)

            if not self.is_test:
                tvt_true = df['TVT'].to_numpy(np.float64)
            else:
                tvt_true = np.full(n, np.nan)

            well_idx = len(self.wells)
            self.wells.append({
                'gr_n': gr_n, 'z_raw': z_raw, 'dz_n': dz_n,
                'tvt_true': tvt_true,
                'type_tvt': type_tvt, 'type_gr_n': type_gr_n,
            })

            for i in range(n - window_size + 1):
                if not self.is_test and np.isnan(tvt_true[i + window_size - 1]):
                    continue
                self.index.append((well_idx, i))

        # Fixed per-sample jitter for deterministic (val) datasets, so the
        # val loss measures the same registration problems every epoch.
        rng = np.random.default_rng(1234)
        self.fixed_jitter = rng.uniform(-self.jitter_ft, self.jitter_ft,
                                        size=len(self.index))

    @staticmethod
    def _compute_global_stats(well_ids, data_dir, window_size):
        """Streaming mean/std for per-window relative Z and dZ over all
        training wells. GR intentionally has NO global stats — every log is
        standardized by its own robust median/IQR (see robust_norm_stats)."""

        class Acc:
            def __init__(self):
                self.n, self.s, self.s2 = 0, 0.0, 0.0

            def add(self, arr):
                arr = np.asarray(arr, dtype=np.float64)
                arr = arr[~np.isnan(arr)]
                self.n += arr.size
                self.s += float(arr.sum())
                self.s2 += float((arr ** 2).sum())

            def stats(self):
                if self.n == 0:
                    return 0.0, 1.0
                mean = self.s / self.n
                var = max(self.s2 / self.n - mean ** 2, 0.0)
                std = float(np.sqrt(var))
                return float(mean), std if std > 0 else 1.0

        acc_zrel, acc_dz = Acc(), Acc()

        for well_id in well_ids:
            path = os.path.normpath(
                os.path.join(data_dir, f"{well_id}__horizontal_well.csv"))
            if not os.path.exists(path):
                continue
            df = prepare_well_dataframe(pd.read_csv(path))
            if len(df) < window_size:
                continue

            z = df['Z'].to_numpy(np.float64)
            acc_dz.add(df['dZ'].to_numpy(np.float64))
            z_w = np.lib.stride_tricks.sliding_window_view(z, window_size)
            acc_zrel.add(z_w - z_w[:, :1])

        return (*acc_zrel.stats(), *acc_dz.stats())

    def stats_dict(self):
        """Normalization stats (Zrel/dZ only — GR is per-signal)."""
        return {
            'zrel_mean': self.zrel_mean, 'zrel_std': self.zrel_std,
            'dz_mean': self.dz_mean, 'dz_std': self.dz_std,
        }

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        well_idx, i = self.index[idx]
        d = self.wells[well_idx]
        W = self.window_size
        sl = slice(i, i + W)

        # ---- Horizontal branch (3, W) ----
        z_win = d['z_raw'][sl]
        zrel = ((z_win - z_win[0] - self.zrel_mean) / self.zrel_std
                ).astype(np.float32)
        x_h = np.stack([d['gr_n'][sl], zrel, d['dz_n'][sl]])

        if self.augment:
            gain = np.float32(1.0 + np.random.uniform(-self.gr_gain_range,
                                                      self.gr_gain_range))
            noise = (np.random.randn(W) * self.gr_noise_std
                     ).astype(np.float32)
            x_h[0] = x_h[0] * gain + noise

        # ---- Typewell branch (2, type_len) + registration target ----
        t_true = d['tvt_true'][i + W - 1]
        if self.augment:
            jitter = np.random.uniform(-self.jitter_ft, self.jitter_ft)
        else:
            jitter = self.fixed_jitter[idx]

        center = t_true + jitter
        x_t = make_type_excerpt(d['type_tvt'], d['type_gr_n'], center,
                                self.offsets, self.pos_channel)
        if self.augment:
            x_t[0] += (np.random.randn(self.type_len) * self.gr_noise_std
                       ).astype(np.float32)

        # Target: the correction that moves the excerpt center onto the
        # true TVT, normalized to [-1, 1].
        y = np.float32((t_true - center) / self.jitter_ft)  # = -jitter/J

        return (torch.from_numpy(x_h), torch.from_numpy(x_t),
                torch.tensor(y, dtype=torch.float32))