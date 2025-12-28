"""
Split merged audio dataset into train/val/test (70/15/15)
=========================================================
- Stratified split by emotion to ensure balanced distribution
- Copy audio files to respective folders
- Create metadata.csv for each split
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Config
MERGED_DIR = Path("d:/AI/EmoSpeech/dataset/merged")
SPLIT_DIR = Path("d:/AI/EmoSpeech/dataset/split")
RANDOM_STATE = 42

def split_dataset():
    """Split merged dataset into train/val/test with stratification"""
    
    # Load metadata
    metadata_path = MERGED_DIR / "metadata.csv"
    df = pd.read_csv(metadata_path)
    
    print("=" * 60)
    print("  SPLITTING AUDIO DATASET")
    print(f"  Input: {len(df)} files from {MERGED_DIR}")
    print(f"  Output: {SPLIT_DIR}")
    print("=" * 60)
    
    # First split: train (70%) vs temp (30%)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.30, 
        random_state=RANDOM_STATE, 
        stratify=df['emotion']
    )
    
    # Second split: temp → val (50%) + test (50%) = 15% + 15%
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=RANDOM_STATE, 
        stratify=temp_df['emotion']
    )
    
    print(f"\nSplit distribution:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Create directories and copy files
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    for split_name, split_df in splits.items():
        split_path = SPLIT_DIR / split_name
        split_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying {split_name} files...")
        new_data = []
        
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
            src_path = Path(row['path'])
            if src_path.exists():
                dst_path = split_path / src_path.name
                shutil.copy2(src_path, dst_path)
                
                new_data.append({
                    'filename': src_path.name,
                    'path': str(dst_path),
                    'emotion': row['emotion'],
                    'source': row['source']
                })
        
        # Save metadata for this split
        new_df = pd.DataFrame(new_data)
        csv_path = split_path / "metadata.csv"
        new_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
    
    # Print emotion distribution per split
    print("\n" + "=" * 60)
    print("  EMOTION DISTRIBUTION PER SPLIT")
    print("=" * 60)
    
    for split_name, split_df in splits.items():
        print(f"\n{split_name.upper()}:")
        print(split_df['emotion'].value_counts().to_string())
    
    print("\n" + "=" * 60)
    print("  SPLIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    split_dataset()
