"""
Audio Data Augmentation - Train Set Only
=========================================
Transformation Functions:
1. Noise Injection: random_factor ∈ [0, 0.035]
2. Time Stretching: rate_factor = 0.7
3. Pitch Shifting: pitch_factor = 0.7
4. Time Shifting: interval = random_integer × 1000 (random_integer ∈ [-5, 5])

Pipeline: Only augment TRAIN set. Copy val/test as-is to prevent data leakage.
"""

import os
import shutil
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

# Input: split folders
SPLIT_DIR = Path("d:/AI/EmoSpeech/dataset/split")
# Output: augmented folders
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
    n_steps = 12 * np.log2(pitch_factor)
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps)


def time_shifting(y: np.ndarray) -> np.ndarray:
    """Cyclic shift: interval = random_integer * 1000, random_integer ∈ [-5, 5]"""
    random_integer = np.random.randint(-5, 6)
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


def augment_train_set():
    """Augment TRAIN set only with balanced frequency per class"""
    
    train_in = SPLIT_DIR / "train"
    train_out = AUGMENTED_DIR / "train"
    train_out.mkdir(parents=True, exist_ok=True)
    
    # Load train metadata
    df = pd.read_csv(train_in / "metadata.csv")
    
    # Số augmentation cho mỗi nhãn (cùng loại, khác frequency)
    AUGMENTATION_COUNTS = {
        'anger': 2,
        'disgust': 2,
        'fear': 2,
        'happy': 2,
        'sad': 2,
        'neutral': 2,
        'surprise': 8  # Tăng để cân bằng
    }
    
    all_data = []
    aug_names = list(AUGMENTATION_FUNCS.keys())
    
    print("=" * 60)
    print("  AUGMENTING TRAIN SET (BALANCED)")
    print(f"  Input: {len(df)} files from {train_in}")
    print(f"  Output: {train_out}")
    print("=" * 60)
    print("\nAugmentation counts per emotion:")
    for emo, count in AUGMENTATION_COUNTS.items():
        print(f"  {emo}: {count} augmentations -> {count+1}x")
    print()
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting train"):
        src_path = Path(row['path'])
        if not src_path.exists():
            continue
            
        y, _ = librosa.load(src_path, sr=SR)
        emotion = row['emotion']
        
        # Keep original
        orig_name = src_path.name
        orig_out = train_out / orig_name
        sf.write(orig_out, y, SR)
        all_data.append({
            'filename': orig_name,
            'path': str(orig_out),
            'emotion': emotion,
            'source': row['source'],
            'augmentation': 'original'
        })
        
        # Apply augmentations
        n_aug = AUGMENTATION_COUNTS.get(emotion, 2)
        for i in range(n_aug):
            aug_name = random.choice(aug_names)
            y_aug = augment_audio(y.copy(), aug_name)
            aug_filename = f"{src_path.stem}_aug{i+1}_{aug_name}.wav"
            aug_out = train_out / aug_filename
            sf.write(aug_out, y_aug, SR)
            
            all_data.append({
                'filename': aug_filename,
                'path': str(aug_out),
                'emotion': emotion,
                'source': row['source'],
                'augmentation': aug_name
            })
    
    # Save metadata
    df_aug = pd.DataFrame(all_data)
    csv_path = train_out / "metadata.csv"
    df_aug.to_csv(csv_path, index=False)
    
    print(f"\nTrain: {len(df)} -> {len(df_aug)} samples")
    print(f"By emotion:\n{df_aug['emotion'].value_counts().to_string()}")
    
    return len(df_aug)


def copy_val_test():
    """Copy val and test sets without augmentation"""
    
    for split in ['val', 'test']:
        src_dir = SPLIT_DIR / split
        dst_dir = AUGMENTED_DIR / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.read_csv(src_dir / "metadata.csv")
        print(f"\nCopying {split} set ({len(df)} files)...")
        
        new_data = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split}"):
            src_path = Path(row['path'])
            if src_path.exists():
                dst_path = dst_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                new_data.append({
                    'filename': src_path.name,
                    'path': str(dst_path),
                    'emotion': row['emotion'],
                    'source': row['source'],
                    'augmentation': 'original'
                })
        
        # Save metadata
        new_df = pd.DataFrame(new_data)
        csv_path = dst_dir / "metadata.csv"
        new_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")


def process_dataset():
    """Main pipeline: augment train, copy val/test"""
    
    print("\n" + "=" * 60)
    print("  AUDIO DATA AUGMENTATION PIPELINE")
    print("  Train: augmented | Val/Test: original (no augmentation)")
    print("=" * 60 + "\n")
    
    # Step 1: Augment train set
    train_count = augment_train_set()
    
    # Step 2: Copy val and test (no augmentation)
    copy_val_test()
    
    # Summary
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        csv = AUGMENTED_DIR / split / "metadata.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            print(f"  {split}: {len(df)} samples")


if __name__ == "__main__":
    process_dataset()
