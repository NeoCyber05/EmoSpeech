# -*- coding: utf-8 -*-
"""
Module for splitting data into 70/15/15 (train/validation/test)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def load_features(csv_path: str) -> pd.DataFrame:
    print(f"[INFO] Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[OK] Loaded {len(df)} samples with {len(df.columns)} columns")
    return df


def split_data_70_15_15(
    df: pd.DataFrame,
    target_col: str = 'emotion',
    random_state: int = 42
) -> tuple:
    # Remove metadata columns
    metadata_cols = ['filename', 'source', 'augmentation']
    cols_to_drop = [col for col in metadata_cols if col in df.columns]
    df_clean = df.drop(columns=cols_to_drop)
    
    # Separate features and labels
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print(f"[WARN] Found {X.isnull().sum().sum()} missing values, filling...")
        X = X.fillna(X.mean())
    
    # Step 1: Split into train (70%) and temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.30, 
        random_state=random_state, 
        stratify=y
    )
    
    # Step 2: Split temp into validation (50% of 30% = 15%) and test (50% of 30% = 15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.50, 
        random_state=random_state, 
        stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_split_data(
    X_train, X_val, X_test, y_train, y_val, y_test,
    output_dir: str
):
    """
    Save split datasets to separate CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine features and labels
    train_df = X_train.copy()
    train_df['emotion'] = y_train.values
    
    val_df = X_val.copy()
    val_df['emotion'] = y_val.values
    
    test_df = X_test.copy()
    test_df['emotion'] = y_test.values
    
    # Save files
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'validation.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n[SAVED DATA]")
    print(f"  - Train: {train_path}")
    print(f"  - Validation: {val_path}")
    print(f"  - Test: {test_path}")
    
    return train_path, val_path, test_path


if __name__ == "__main__":

    features_path = r"D:\AI\EmoSpeech\data\features\features.csv"
    output_dir = r"D:\AI\EmoSpeech\data\split"
    
    df = load_features(features_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_70_15_15(df)
    
    # Save data
    save_split_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir)
