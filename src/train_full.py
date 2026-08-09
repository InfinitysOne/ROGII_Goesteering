import argparse
import csv
import glob
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (GeosteeringDataset, prepare_well_dataframe,
                     load_typewell, make_type_excerpt, robust_norm_stats)
from model import GeosteeringCorrelationModel


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


def _prepare_nav_arrays(df, stats):
    """Precompute normalized per-well arrays used during navigation.
    GR is standardized by the well's OWN robust stats (median/IQR),
    matching the training-time per-signal normalization."""
    gr_raw = df['GR'].to_numpy(np.float64)
    gr_med, gr_scale = robust_norm_stats(gr_raw)
    return {
        'gr_n': ((gr_raw - gr_med) / gr_scale).astype(np.float32),
        'dz_n': ((df['dZ'].to_numpy(np.float64) - stats['dz_mean'])
                 / stats['dz_std']).astype(np.float32),
        'z': df['Z'].to_numpy(np.float64),
    }


@torch.no_grad()
def navigate_wells_batched(model, wells, stats, cfg, device, stride=1,
                           progress_desc=None):
    """Sequential navigation of MANY wells in lockstep — the single code
    path used by BOTH the anchored validation and (via navigate_well) the
    inference notebook.

    All wells advance one sample per iteration; the model corrections of all
    wells currently in a blind position are computed in ONE batched forward
    pass. This is what makes per-epoch validation feasible: with N wells the
    number of model calls drops from sum(blind lengths) to max(blind length)
    — about two orders of magnitude for a typical validation split.

    Args:
      model: correlation model, or None for the pure dead-reckoning baseline.
      wells: list of dicts, one per well:
        {'df': prepared DataFrame, 'typewell': (type_tvt, type_gr_n),
         'tvt_known': array with known TVT values and NaN where blind}
      stride: apply the model correction only every `stride` blind samples
        (speed knob for validation; keep 1 for final inference).
      progress_desc: if not None, show a tqdm bar with this description.

    Per-well behavior (identical to the previous single-well version):
      * At every known point the estimate SNAPS to the known value.
      * Blind steps: dead-reckoning prior est(i-1) - dZ plus the model
        correction, CLIPPED to [-jitter_ft, +jitter_ft].
      * A leading blind stretch is filled by REVERSE dead reckoning from the
        first known point.

    Returns: list of np.ndarray TVT estimates, in the order of `wells`.
    """
    W = cfg['window_size']
    J = cfg['jitter_ft']
    offsets = np.linspace(-cfg['type_half_range'], cfg['type_half_range'],
                          cfg['type_len']).astype(np.float64)
    pos_channel = (offsets / cfg['type_half_range']).astype(np.float32)

    states = []
    for w in wells:
        arrs = _prepare_nav_arrays(w['df'], stats)
        tvt_known = np.asarray(w['tvt_known'], dtype=np.float64)
        n = len(w['df'])
        states.append({
            **arrs,
            'type_tvt': w['typewell'][0], 'type_gr_n': w['typewell'][1],
            'known': ~np.isnan(tvt_known), 'tvt_known': tvt_known,
            'est': np.full(n, np.nan, dtype=np.float64),
            'n': n, 'i': 0, 'blind_step': 0,
        })

    active = [k for k, s in enumerate(states) if s['n'] > 0]
    bar = tqdm(desc=progress_desc, colour='cyan') if progress_desc else None

    while active:
        pending = []          # wells waiting for a batched model correction
        x_h_list, x_t_list = [], []
        still_active = []

        for k in active:
            s = states[k]
            # advance instantly through consecutive KNOWN samples (snap)
            while s['i'] < s['n'] and s['known'][s['i']]:
                s['est'][s['i']] = s['tvt_known'][s['i']]
                s['blind_step'] = 0
                s['i'] += 1
            i = s['i']
            if i >= s['n']:
                continue  # well finished

            still_active.append(k)

            if i == 0 or np.isnan(s['est'][i - 1]):
                s['i'] += 1  # leading gap: filled in reverse afterwards
                continue

            z = s['z']
            prior = s['est'][i - 1] - (z[i] - z[i - 1])  # dead reckoning

            use_model = (model is not None and i >= W - 1
                         and s['blind_step'] % stride == 0)
            if use_model:
                sl = slice(i - W + 1, i + 1)
                z_win = z[sl]
                zrel = ((z_win - z_win[0] - stats['zrel_mean'])
                        / stats['zrel_std']).astype(np.float32)
                x_h_list.append(np.stack([s['gr_n'][sl], zrel, s['dz_n'][sl]]))
                x_t_list.append(make_type_excerpt(
                    s['type_tvt'], s['type_gr_n'], prior, offsets, pos_channel))
                pending.append((k, prior))
            else:
                s['est'][i] = prior
                s['i'] += 1
                s['blind_step'] += 1

        if pending:
            x_h = torch.from_numpy(np.stack(x_h_list)).to(device)
            x_t = torch.from_numpy(np.stack(x_t_list)).to(device)
            out = model(x_h, x_t)
            corr_n = np.clip(np.asarray(out.cpu().numpy(),
                                        dtype=np.float64).reshape(-1),
                             -1.0, 1.0)
            for (k, prior), c in zip(pending, corr_n):
                s = states[k]
                s['est'][s['i']] = prior + c * J
                s['i'] += 1
                s['blind_step'] += 1

        active = still_active
        if bar is not None:
            bar.update(1)

    if bar is not None:
        bar.close()

    # Leading blind stretches: reverse dead reckoning from first known point
    results = []
    for s in states:
        est, z, known = s['est'], s['z'], s['known']
        if known.any():
            first_known = int(np.argmax(known))
            for i in range(first_known - 1, -1, -1):
                if np.isnan(est[i]):
                    est[i] = est[i + 1] + (z[i + 1] - z[i])
        results.append(est)
    return results


@torch.no_grad()
def navigate_well(model, df, typewell, stats, cfg, device, tvt_known,
                  stride=1):
    """Single-well convenience wrapper around navigate_wells_batched
    (used by the inference notebook). Identical behavior by construction."""
    return navigate_wells_batched(
        model, [{'df': df, 'typewell': typewell, 'tvt_known': tvt_known}],
        stats, cfg, device, stride=stride)[0]


# Cache for validation wells so the CSVs are read only once, not every epoch.
_VAL_CACHE = {}


def _load_val_well(data_dir, well_id):
    """Load (and cache) the prepared horizontal df and the self-normalized
    typewell for one validation well. Returns None if unusable."""
    key = (data_dir, well_id)
    if key in _VAL_CACHE:
        return _VAL_CACHE[key]

    horiz_path = os.path.normpath(
        os.path.join(data_dir, f"{well_id}__horizontal_well.csv"))
    type_path = os.path.normpath(
        os.path.join(data_dir, f"{well_id}__typewell.csv"))
    entry = None
    if os.path.exists(horiz_path) and os.path.exists(type_path):
        df = prepare_well_dataframe(pd.read_csv(horiz_path))
        type_tvt, type_gr_n = load_typewell(type_path)
        if len(type_tvt) >= 2:
            entry = {'df': df, 'typewell': (type_tvt, type_gr_n)}
    _VAL_CACHE[key] = entry
    return entry


@torch.no_grad()
def evaluate_anchored(model, val_wells, data_dir, stats, cfg, device,
                      blind_fraction=0.3, stride=1, return_per_well=False):
    """Anchored validation matching the Kaggle setting.

    For each validation well the blind zone is defined by the well's OWN
    TVT_input mask if present (this reproduces the competition scenario
    exactly, including realistically long blind stretches). Only if a well
    has no usable TVT_input mask, the last `blind_fraction` of the well is
    used as a simulated blind zone instead.

    All wells are then navigated in lockstep by navigate_wells_batched
    (known zone = ground truth, blind zone = dead reckoning + batched model
    re-registration), and the RMSE is computed over the blind samples only —
    the same samples the leaderboard scores. Pass model=None for the
    dead-reckoning baseline.

    Returns:
      rmse                            if return_per_well is False
      (rmse, per_well)                if return_per_well is True, where
        per_well is a list of (well_id, rmse_ft, n_blind_samples) — the
        diagnostic that shows whether the global RMSE is a uniform error or
        dominated by a few derailed wells.
    """
    if model is not None:
        model.eval()

    nav_wells, truths, masks, ids = [], [], [], []
    for well_id in val_wells:
        entry = _load_val_well(data_dir, well_id)
        if entry is None or len(entry['df']) < cfg['window_size'] + 10:
            continue
        df = entry['df']

        tvt_true = df['TVT'].to_numpy(np.float64)

        tvt_known = None
        if 'TVT_input' in df.columns:
            tvt_input = pd.to_numeric(df['TVT_input'],
                                      errors='coerce').to_numpy(np.float64)
            if np.isnan(tvt_input).any() and (~np.isnan(tvt_input)).any():
                tvt_known = tvt_input

        if tvt_known is None:  # fallback: simulate the last blind_fraction
            blind_start = max(int(len(df) * (1.0 - blind_fraction)),
                              cfg['window_size'])
            tvt_known = tvt_true.copy()
            tvt_known[blind_start:] = np.nan

        if np.all(np.isnan(tvt_known)) or np.all(np.isnan(tvt_true)):
            continue

        type_gr_n = None  # typewell already self-normalized in the cache
        nav_wells.append({'df': df,
                          'typewell': entry['typewell'],
                          'tvt_known': tvt_known})
        truths.append(tvt_true)
        masks.append(np.isnan(tvt_known))
        ids.append(well_id)

    if not nav_wells:
        return (float('inf'), []) if return_per_well else float('inf')

    desc = "[ANCHORED VAL]" if model is not None else "[BASELINE NAV]"
    ests = navigate_wells_batched(model, nav_wells, stats, cfg, device,
                                  stride=stride, progress_desc=desc)

    sq_err_sum, count = 0.0, 0
    per_well = []
    for well_id, est, tvt_true, blind in zip(ids, ests, truths, masks):
        m = blind & ~np.isnan(tvt_true) & ~np.isnan(est)
        if not m.any():
            continue
        err = est[m] - tvt_true[m]
        sq = float((err ** 2).sum())
        n = int(m.sum())
        sq_err_sum += sq
        count += n
        per_well.append((well_id, float(np.sqrt(sq / n)), n))

    rmse = np.sqrt(sq_err_sum / count) if count > 0 else float('inf')
    if return_per_well:
        return rmse, per_well
    return rmse


def per_well_summary(per_well, k_worst=3):
    """Compact one-line summary of per-well anchored RMSEs."""
    if not per_well:
        return "no wells evaluated"
    rmses = np.array([r for _, r, _ in per_well])
    worst = sorted(per_well, key=lambda t: -t[1])[:k_worst]
    ws = ", ".join(f"{wid}({r:.0f})" for wid, r, _ in worst)
    return (f"median={np.median(rmses):.1f} ft, p90="
            f"{np.percentile(rmses, 90):.1f} ft, "
            f"max={rmses.max():.1f} ft | worst: {ws}")


def save_per_well_csv(per_well, path):
    """Persist the per-well diagnostic (sorted worst-first) to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["well_id", "anchored_rmse_ft", "n_blind_samples"])
        for wid, rmse, n in sorted(per_well, key=lambda t: -t[1]):
            w.writerow([wid, f"{rmse:.4f}", n])


def train_full_model(resume: bool = False):
    # ---------------------------------------------------------
    # 1. Setup & Hyperparameter
    # ---------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 128
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.01
    WINDOW_SIZE = 100
    SCHEDULER_PATIENCE = 5
    EARLY_STOP_PATIENCE = 15     # on anchored val RMSE
    EARLY_STOP_MIN_DELTA = 0.05  # ft
    GR_NOISE_STD = 0.05          # augmentation: Gaussian noise on GR
    GR_GAIN_RANGE = 0.1          # augmentation: random gain +-10%
    BLIND_FRACTION = 0.3         # fallback blind zone if no TVT_input mask

    # Registration task config
    JITTER_FT = 25.0             # max mis-centering of the typewell excerpt
    TYPE_HALF_RANGE = 50.0       # excerpt spans +- this many TVT-ft
    TYPE_LEN = 128               # excerpt grid resolution
    EVAL_STRIDE = 1              # correction stride for the anchored
    # validation. Since validation navigates all
    # wells in lockstep with batched forwards,
    # stride 1 is normally affordable; raise to
    # 2-4 only if val is still too slow.

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting full training on device: {device}")

    # ---------------------------------------------------------
    # 2. Automatically find all well IDs (need horizontal + typewell)
    # ---------------------------------------------------------
    data_dir = os.path.normpath("../data/train")
    csv_files = glob.glob(os.path.join(data_dir, "*__horizontal_well.csv"))

    all_wells = sorted({os.path.basename(f).split("__")[0] for f in csv_files})
    all_wells = [w for w in all_wells if os.path.exists(
        os.path.join(data_dir, f"{w}__typewell.csv"))]
    print(f"Found {len(all_wells)} wells with horizontal + typewell data.")

    # ---------------------------------------------------------
    # 3. Train / Validation Split (80% / 20%) — split by WELLS, not rows
    # ---------------------------------------------------------
    train_wells, val_wells = train_test_split(all_wells, test_size=0.2,
                                              random_state=SEED)
    print(f"Training set: {len(train_wells)} wells")
    print(f"Validation set: {len(val_wells)} wells")

    # ---------------------------------------------------------
    # 4. Datasets & DataLoaders
    # ---------------------------------------------------------
    train_dataset = GeosteeringDataset(
        well_ids=train_wells, data_dir=data_dir, window_size=WINDOW_SIZE,
        augment=True, gr_noise_std=GR_NOISE_STD,
        gr_gain_range=GR_GAIN_RANGE,
        jitter_ft=JITTER_FT, type_half_range=TYPE_HALF_RANGE,
        type_len=TYPE_LEN)
    stats = train_dataset.stats_dict()

    val_dataset = GeosteeringDataset(
        well_ids=val_wells, data_dir=data_dir, window_size=WINDOW_SIZE,
        augment=False,
        jitter_ft=JITTER_FT, type_half_range=TYPE_HALF_RANGE,
        type_len=TYPE_LEN, **stats)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    cfg = {'window_size': WINDOW_SIZE, 'jitter_ft': JITTER_FT,
           'type_half_range': TYPE_HALF_RANGE, 'type_len': TYPE_LEN}

    # ---------------------------------------------------------
    # 4b. Baselines
    # ---------------------------------------------------------
    # (a) Anchored: pure dead reckoning (correction = 0 at every step)
    baseline_rmse, baseline_per_well = evaluate_anchored(
        None, val_wells, data_dir, stats, cfg, device,
        blind_fraction=BLIND_FRACTION, return_per_well=True)
    # (b) Window-level: always predicting correction 0 against uniform
    #     jitter has RMSE = JITTER_FT / sqrt(3)
    window_baseline = JITTER_FT / np.sqrt(3.0)
    print(f"\n[BASELINE] Anchored val RMSE, pure dead reckoning "
          f"(dTVT = -dZ): {baseline_rmse:.2f} ft")
    print(f"[BASELINE] Per well: {per_well_summary(baseline_per_well)}")
    print(f"[BASELINE] Window val RMSE of 'always predict 0 correction': "
          f"{window_baseline:.2f} ft")
    print("           -> The model must beat BOTH numbers to add value.\n")

    # ---------------------------------------------------------
    # 5. Model, Loss, Optimizer
    # ---------------------------------------------------------
    model = GeosteeringCorrelationModel(type_len=TYPE_LEN).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE)

    best_val_rmse = float('inf')
    epochs_no_improvement = 0
    start_epoch = 0

    model_dir = "../src/models"
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, "best_geosteering_model.pth")
    resume_path = os.path.join(model_dir, "resume_checkpoint.pth")
    norm_stats_path = os.path.join(model_dir, "normalization_stats.json")
    log_path = os.path.join(model_dir, "training_log.csv")

    # ---------------------------------------------------------
    # 5b. Resume support
    # ---------------------------------------------------------
    if resume:
        if os.path.exists(resume_path):
            print(f"\nResume checkpoint found — loading '{resume_path}'...")
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"]
            best_val_rmse = ckpt["best_val_rmse"]
            epochs_no_improvement = ckpt["epochs_no_improvement"]
            print(f"Resuming from epoch {start_epoch + 1}/{EPOCHS}  "
                  f"(best anchored val RMSE so far: {best_val_rmse:.2f} ft)\n")
            log_mode = "a"
        else:
            print(f"\n[WARNING] --resume was set but no checkpoint found at "
                  f"'{resume_path}'. Starting from scratch.\n")
            log_mode = "w"
    else:
        log_mode = "w"

    with open(log_path, log_mode, newline="") as f:
        if log_mode == "w":
            csv.writer(f).writerow(
                ["epoch", "train_corr_rmse_ft", "val_corr_rmse_ft",
                 "val_anchored_rmse_ft", "lr"])

    # ---------------------------------------------------------
    # 6. Main Loop
    # ---------------------------------------------------------
    for epoch in range(start_epoch, EPOCHS):
        # --- TRAINING ---
        model.train()
        train_running_loss = 0.0

        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch + 1}/{EPOCHS} [TRAIN] ",
                         colour='green')
        for x_h, x_t, y in train_bar:
            x_h, x_t, y = x_h.to(device), x_t.to(device), y.to(device)

            optimizer.zero_grad()
            predictions = model(x_h, x_t)
            loss = criterion(predictions, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_running_loss += loss.item()

        train_mse = train_running_loss / len(train_loader)
        # correction error in real ft (targets are normalized by JITTER_FT)
        train_rmse = np.sqrt(train_mse) * JITTER_FT

        # --- VALIDATION (window-level correction error) ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader,
                           desc=f"Epoch {epoch + 1}/{EPOCHS} [VAL]",
                           colour='blue')
            for x_h, x_t, y in val_bar:
                x_h, x_t, y = x_h.to(device), x_t.to(device), y.to(device)
                loss = criterion(model(x_h, x_t), y)
                val_running_loss += loss.item()

        val_mse = val_running_loss / len(val_loader)
        val_corr_rmse = np.sqrt(val_mse) * JITTER_FT

        # --- VALIDATION (anchored sequential navigation, matches Kaggle) ---
        val_anchored_rmse, val_per_well = evaluate_anchored(
            model, val_wells, data_dir, stats, cfg, device,
            blind_fraction=BLIND_FRACTION, stride=EVAL_STRIDE,
            return_per_well=True)

        current_lr = optimizer.param_groups[0]['lr']
        vs_base = val_anchored_rmse - baseline_rmse
        print(f"\n --> Epoch {epoch + 1} | Train corr RMSE: {train_rmse:.2f} ft"
              f" | Val corr RMSE: {val_corr_rmse:.2f} ft"
              f" (0-baseline {window_baseline:.2f})"
              f" | Val ANCHORED RMSE: {val_anchored_rmse:.2f} ft "
              f"({'+' if vs_base >= 0 else ''}{vs_base:.2f} ft vs baseline)")
        print(f"     Per well: {per_well_summary(val_per_well)}")

        scheduler.step(val_anchored_rmse)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch + 1, f"{train_rmse:.4f}", f"{val_corr_rmse:.4f}",
                 f"{val_anchored_rmse:.4f}", f"{current_lr:.6f}"])

        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_rmse": best_val_rmse,
            "epochs_no_improvement": epochs_no_improvement,
        }, resume_path)

        # ---------------------------------------------------------
        # 7. Checkpointing & Early Stopping (on anchored RMSE)
        # ---------------------------------------------------------
        improvement = best_val_rmse - val_anchored_rmse

        if val_anchored_rmse < best_val_rmse:
            best_val_rmse = val_anchored_rmse
            torch.save(model.state_dict(), checkpoint_path)
            save_per_well_csv(val_per_well,
                              os.path.join(model_dir, "val_per_well.csv"))
            with open(norm_stats_path, "w", newline="") as f:
                json.dump({
                    **stats,
                    **cfg,
                    "target": "typewell_registration_v2_per_signal_norm",
                    "gr_normalization": "per_signal_robust_median_iqr",
                    "features_horizontal": ["GR", "Z_rel_window_start", "dZ"],
                    "features_typewell": ["GR_excerpt", "position"],
                    "baseline_anchored_rmse_ft": baseline_rmse,
                }, f, indent=2)
            print("New best model saved! (Anchored val RMSE improved)")

        if improvement >= EARLY_STOP_MIN_DELTA:
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= EARLY_STOP_PATIENCE:
            print(f"\nNo significant improvement for {EARLY_STOP_PATIENCE} "
                  f"epochs — stopping early.")
            break

    print(f"\nTraining complete! Best anchored val RMSE: {best_val_rmse:.2f} ft"
          f" (dead-reckoning baseline: {baseline_rmse:.2f} ft).")

    if os.path.exists(resume_path):
        os.remove(resume_path)
        print("Resume checkpoint deleted — next run will start from scratch.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Geosteering typewell-correlation model.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last saved resume checkpoint. "
             "If omitted, training always starts from scratch.",
    )
    args = parser.parse_args()
    train_full_model(resume=args.resume)