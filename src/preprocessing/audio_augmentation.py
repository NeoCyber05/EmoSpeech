"""
Transformation Functions:
1. Noise Injection: random_factor ∈ [0, 0.035]
2. Time Stretching: rate_factor = 0.7
3. Pitch Shifting: pitch_factor = 0.7
4. Time Shifting: interval = random_integer × 1000 (random_integer ∈ [-5, 5])
"""

import os
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random

# Config
SR = 16000
TARGET_LEN = 32000  # 2 seconds * 16kHz
MERGED_DIR = Path("d:/AI/EmoSpeech/dataset/merged")
AUGMENTED_DIR = Path("d:/AI/EmoSpeech/dataset/augmented")


# ===================== TRANSFORMATION FUNCTIONS =====================

def noise_injection(y: np.ndarray) -> np.ndarray:
    """Add random noise: amplitude = random_factor * max(abs(y)), random_factor ∈ [0, 0.035]"""
    random_factor = np.random.uniform(0, 0.035)
    noise_amplitude = random_factor * np.max(np.abs(y))
    noise = np.random.normal(0, noise_amplitude, len(y))
    return y + noise


def time_stretching(y: np.ndarray, rate_factor: float = 0.7) -> np.ndarray:
    """Slow down playback with rate_factor = 0.7, then normalize to original length"""
    y_stretched = librosa.effects.time_stretch(y, rate=rate_factor)
    # Normalize to target length
    if len(y_stretched) > TARGET_LEN:
        start = (len(y_stretched) - TARGET_LEN) // 2
        y_stretched = y_stretched[start:start + TARGET_LEN]
    elif len(y_stretched) < TARGET_LEN:
        pad_left = (TARGET_LEN - len(y_stretched)) // 2
        pad_right = TARGET_LEN - len(y_stretched) - pad_left
        y_stretched = np.pad(y_stretched, (pad_left, pad_right), mode='constant')
    return y_stretched


def pitch_shifting(y: np.ndarray, pitch_factor: float = 0.7) -> np.ndarray:
    """Lower pitch with pitch_factor = 0.7 → n_steps = 12 * log2(0.7) ≈ -5.1 semitones"""
    n_steps = 12 * np.log2(pitch_factor)  # ≈ -5.1
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps)


def time_shifting(y: np.ndarray) -> np.ndarray:
    """Cyclic shift (like numpy.roll): interval = random_integer * 1000, random_integer ∈ [-5, 5]"""
    random_integer = np.random.randint(-5, 6)  # [-5, 5] inclusive
    interval = random_integer * 1000
    return np.roll(y, interval)


# ===================== AUGMENTATION PIPELINE =====================

AUGMENTATION_FUNCS = {
    'noise': noise_injection,
    'stretch': time_stretching,
    'pitch': pitch_shifting,
    'shift': time_shifting
}


def augment_audio(y: np.ndarray, aug_name: str) -> np.ndarray:
    """Apply single augmentation by name"""
    return AUGMENTATION_FUNCS[aug_name](y)


def process_dataset():
    """Apply 2 random augmentations per sample → 3x dataset (1 original + 2 augmented)"""
    AUGMENTED_DIR.mkdir(exist_ok=True)
    
    # Load metadata
    metadata_path = MERGED_DIR / "metadata.csv"
    df = pd.read_csv(metadata_path)
    
    all_data = []
    aug_names = list(AUGMENTATION_FUNCS.keys())
    
    print("=" * 60)
    print("  AUDIO DATA AUGMENTATION")
    print(f"  Input: {len(df)} files from {MERGED_DIR}")
    print(f"  Output: {AUGMENTED_DIR}")
    print("=" * 60)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        src_path = Path(row['path'])
        if not src_path.exists():
            continue
            
        # Load audio
        y, _ = librosa.load(src_path, sr=SR)
        
        # Keep original
        orig_name = src_path.name
        orig_out = AUGMENTED_DIR / orig_name
        sf.write(orig_out, y, SR)
        all_data.append({
            'filename': orig_name,
            'path': str(orig_out),
            'emotion': row['emotion'],
            'source': row['source'],
            'augmentation': 'original'
        })
        
        # Apply 2 random augmentations
        selected_augs = random.sample(aug_names, 2)
        
        for i, aug_name in enumerate(selected_augs, 1):
            y_aug = augment_audio(y.copy(), aug_name)
            aug_filename = f"{src_path.stem}_aug{i}_{aug_name}.wav"
            aug_out = AUGMENTED_DIR / aug_filename
            sf.write(aug_out, y_aug, SR)
            
            all_data.append({
                'filename': aug_filename,
                'path': str(aug_out),
                'emotion': row['emotion'],
                'source': row['source'],
                'augmentation': aug_name
            })
    
    # Save new metadata
    df_aug = pd.DataFrame(all_data)
    csv_path = AUGMENTED_DIR / "metadata.csv"
    df_aug.to_csv(csv_path, index=False)
    
    # Statistics
    print("\n" + "=" * 60)
    print("  AUGMENTATION COMPLETE")
    print("=" * 60)
    print(f"\nOriginal samples: {len(df)}")
    print(f"Total samples after augmentation: {len(df_aug)}")
    print(f"Increase: {len(df_aug) / len(df):.1f}x")
    
    print(f"\nBy augmentation type:")
    print(df_aug['augmentation'].value_counts().to_string())
    
    print(f"\nBy emotion:")
    print(df_aug['emotion'].value_counts().to_string())
    
    print(f"\nSaved to: {AUGMENTED_DIR}")
    print(f"Metadata: {csv_path}")


if __name__ == "__main__":
    process_dataset()
