"""
Preprocess and merge TESS, CREMA-D, RAVDESS datasets
=====================================================
- Resample to 16kHz
- Normalize to 2 seconds (trim/zero-pad)
- Map labels to 7 unified emotions: anger, disgust, fear, happy, neutral, surprise, sad
"""

import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Config
SR = 16000           # Target sample rate
DURATION = 2.0       # Target duration in seconds
TARGET_LEN = int(SR * DURATION)  # 32000 samples

BASE_DIR = Path("d:/AI/EmoSpeech/dataset")
OUTPUT_DIR = Path("d:/AI/EmoSpeech/dataset/merged")
OUTPUT_DIR.mkdir(exist_ok=True)

# Label mappings to 7 unified emotions
UNIFIED_LABELS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'surprise', 'sad']

CREMAD_MAP = {
    'ANG': 'anger', 'DIS': 'disgust', 'FEA': 'fear',
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
}

RAVDESS_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'anger', '06': 'fear', '07': 'disgust', '08': 'surprise'
}

TESS_MAP = {
    'angry': 'anger', 'disgust': 'disgust', 'fear': 'fear',
    'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad',
    'pleasant_surprise': 'surprise', 'pleasant_surprised': 'surprise'
}


def normalize_audio(y: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    """Normalize audio to target length by trimming or zero-padding"""
    if len(y) > target_len:
        # Trim from center
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]
    elif len(y) < target_len:
        # Zero-pad
        pad_left = (target_len - len(y)) // 2
        pad_right = target_len - len(y) - pad_left
        y = np.pad(y, (pad_left, pad_right), mode='constant')
    return y


def process_cremad():
    """Process CREMA-D dataset"""
    cremad_dir = BASE_DIR / "CREMA-D"
    data = []
    
    for f in tqdm(list(cremad_dir.glob("*.wav")), desc="CREMA-D"):
        parts = f.stem.split("_")
        if len(parts) >= 3:
            emo_code = parts[2]
            if emo_code in CREMAD_MAP:
                label = CREMAD_MAP[emo_code]
                # Load and process
                y, _ = librosa.load(f, sr=SR)
                y = normalize_audio(y)
                
                # Save
                out_name = f"cremad_{f.stem}.wav"
                out_path = OUTPUT_DIR / out_name
                sf.write(out_path, y, SR)
                
                data.append({
                    'filename': out_name,
                    'path': str(out_path),
                    'emotion': label,
                    'source': 'CREMA-D',
                    'original': str(f)
                })
    return data


def process_ravdess():
    """Process RAVDESS dataset"""
    ravdess_dir = BASE_DIR / "RAVDESS"
    data = []
    
    all_files = []
    for actor_dir in ravdess_dir.iterdir():
        if actor_dir.is_dir():
            all_files.extend(list(actor_dir.glob("*.wav")))
    
    for f in tqdm(all_files, desc="RAVDESS"):
        parts = f.stem.split("-")
        if len(parts) >= 3:
            emo_code = parts[2]
            label = RAVDESS_MAP.get(emo_code)
            
            # Skip 'calm' as it's not in our 7 labels
            if label and label != 'calm' and label in UNIFIED_LABELS:
                y, _ = librosa.load(f, sr=SR)
                y = normalize_audio(y)
                
                out_name = f"ravdess_{f.stem}.wav"
                out_path = OUTPUT_DIR / out_name
                sf.write(out_path, y, SR)
                
                data.append({
                    'filename': out_name,
                    'path': str(out_path),
                    'emotion': label,
                    'source': 'RAVDESS',
                    'original': str(f)
                })
    return data


def process_tess():
    """Process TESS dataset"""
    tess_dir = BASE_DIR / "TESS"
    data = []
    
    all_files = []
    for folder in tess_dir.iterdir():
        if folder.is_dir():
            emo = "_".join(folder.name.split("_")[1:]).lower()
            for f in folder.glob("*.wav"):
                all_files.append((f, emo))
    
    for f, emo in tqdm(all_files, desc="TESS"):
        label = TESS_MAP.get(emo)
        if label and label in UNIFIED_LABELS:
            y, _ = librosa.load(f, sr=SR)
            y = normalize_audio(y)
            
            out_name = f"tess_{f.stem}.wav"
            out_path = OUTPUT_DIR / out_name
            sf.write(out_path, y, SR)
            
            data.append({
                'filename': out_name,
                'path': str(out_path),
                'emotion': label,
                'source': 'TESS',
                'original': str(f)
            })
    return data


def main():
    print("=" * 60)
    print("  MERGING DATASETS: TESS + CREMA-D + RAVDESS")
    print(f"  Target: {SR}Hz, {DURATION}s")
    print("=" * 60)
    
    all_data = []
    
    # Process each dataset
    all_data.extend(process_cremad())
    all_data.extend(process_ravdess())
    all_data.extend(process_tess())
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save metadata
    csv_path = OUTPUT_DIR / "metadata.csv"
    df.to_csv(csv_path, index=False)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("  MERGED DATASET STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal files: {len(df)}")
    print(f"\nBy source:")
    print(df['source'].value_counts().to_string())
    
    print(f"\nBy emotion:")
    print(df['emotion'].value_counts().to_string())
    
    print(f"\nSaved to: {OUTPUT_DIR}")
    print(f"Metadata: {csv_path}")


if __name__ == "__main__":
    main()
