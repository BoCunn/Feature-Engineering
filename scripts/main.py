

import os
import numpy as np
import pandas as pd
import librosa

# ─────────────────────────────────────────────
#  CONFIGURATION  ← edit these paths as needed
# ─────────────────────────────────────────────
AUDIO_DIR = "./data"   # folder containing  .wav / .mp3 files
OUTPUT_CSV = "features.csv"   # output file name
N_MFCC = 9                    # 9 MFCCs × (mean + std) = 18 MFCC features
SAMPLE_RATE = None            # None = use file's native sample rate
# ──────────────────────────────────────────────


def extract_features(file_path: str) -> dict:
   
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # ── MFCCs ───────────────────────────────────────────────────────────────
    # Shape: (N_MFCC, T) where T = number of frames
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    mfcc_means = np.mean(mfccs, axis=1)   # mean across time → shape (N_MFCC,)
    mfcc_stds  = np.std(mfccs,  axis=1)   # std  across time → shape (N_MFCC,)

    # ── RMS Energy ──────────────────────────────────────────────────────────
    # Shape: (1, T) → flatten to (T,)
    rms = librosa.feature.rms(y=y)[0]

    rms_mean = float(np.mean(rms))
    rms_std  = float(np.std(rms))
    rms_max  = float(np.max(rms))
    rms_min  = float(np.min(rms))

    # ── Zero-Crossing Rate ───────────────────────────────────────────────────
    # Shape: (1, T) → flatten to (T,)
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]

    zcr_mean = float(np.mean(zcr))
    zcr_std  = float(np.std(zcr))
    zcr_max  = float(np.max(zcr))
    zcr_min  = float(np.min(zcr))

    # ── Assemble feature dict ────────────────────────────────────────────────
    features = {"filename": os.path.basename(file_path)}

    for i, (m, s) in enumerate(zip(mfcc_means, mfcc_stds), start=1):
        features[f"mfcc_mean_{i:02d}"] = float(m)
        features[f"mfcc_std_{i:02d}"]  = float(s)

    features.update({
        "rms_mean": rms_mean,
        "rms_std":  rms_std,
        "rms_max":  rms_max,
        "rms_min":  rms_min,
        "zcr_mean": zcr_mean,
        "zcr_std":  zcr_std,
        "zcr_max":  zcr_max,
        "zcr_min":  zcr_min,
    })

    return features


def build_feature_matrix(audio_dir: str) -> pd.DataFrame:
    """Walk audio_dir, extract features from every audio file, return DataFrame."""
    supported = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    audio_files = sorted([
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.lower().endswith(supported)
    ])

    if not audio_files:
        raise FileNotFoundError(
            f"No audio files found in '{audio_dir}'.\n"
            "Update AUDIO_DIR at the top of this script."
        )

    print(f"Found {len(audio_files)} audio file(s) in '{audio_dir}'.\n")

    rows = []
    for i, path in enumerate(audio_files, start=1):
        print(f"  [{i:>3}/{len(audio_files)}] Processing: {os.path.basename(path)}")
        try:
            rows.append(extract_features(path))
        except Exception as exc:
            print(f"  Skipped (error: {exc})")

    return pd.DataFrame(rows)


def explain_shape(df: pd.DataFrame) -> None:
    """Print a human-readable explanation of the feature matrix shape."""
    n_samples, n_cols = df.shape
    n_features = n_cols - 1  # subtract 'filename' column

    mfcc_features = N_MFCC * 2        # mean + std per coefficient
    rms_features  = 4                  # mean, std, max, min
    zcr_features  = 4                  # mean, std, max, min
    total_features = mfcc_features + rms_features + zcr_features

    print("\n" + "═" * 60)
    print("  FEATURE MATRIX  –  Shape Explanation")
    print("═" * 60)
    print(f"  Rows    (samples)  : {n_samples}")
    print(f"  Columns (features) : {n_features}")
    print(f"\n  Feature breakdown:")
    print(f"    MFCC mean × {N_MFCC}                   = {N_MFCC:>2} features")
    print(f"    MFCC std  × {N_MFCC}                   = {N_MFCC:>2} features")
    print(f"    RMS  (mean, std, max, min)       =  4 features")
    print(f"    ZCR  (mean, std, max, min)       =  4 features")
    print(f"    {'─'*38}")
    print(f"    Total                            = {total_features:>2} features")
    print(f"\n  Matrix shape: {n_samples} × {n_features}")
    print("═" * 60 + "\n")


def main():
    df = build_feature_matrix(AUDIO_DIR)

    explain_shape(df)

    # Save CSV (filename column first, then all features)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅  CSV saved → {os.path.abspath(OUTPUT_CSV)}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1] - 1} feature columns\n")

    # Preview first 3 rows (feature columns only)
    print("Preview (first 3 rows, first 8 feature columns):")
    feature_cols = [c for c in df.columns if c != "filename"]
    print(df[feature_cols[:8]].head(3).to_string(index=False))
    print("  …")


if __name__ == "__main__":
    main()