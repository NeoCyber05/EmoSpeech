"""
Feature Processing Pipeline
Xử lý feature selection và lưu CSV vào data/processed_features/
"""
import pandas as pd
import os
import sys

# Add paths
sys.path.append(r"D:\AI\EmoSpeech\src\utils")
sys.path.append(r"D:\AI\EmoSpeech\src\feature_selection")
sys.path.append(r"D:\AI\EmoSpeech\src\feature_selection\filter_method")

from data_splitter import load_features, split_data_70_15_15
from feature_selection_manager import show_menu, apply_feature_selection, METHODS



# Paths
RAW_FEATURES_PATH = r"D:\AI\EmoSpeech\data\features\features.csv"
OUTPUT_BASE_DIR = r"D:\AI\EmoSpeech\data\processed_features"


def save_processed_data(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    output_dir: str
):
    """Lưu data đã xử lý vào CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Gộp X và y lại
    train_df = X_train.copy()
    train_df['emotion'] = y_train.values
    
    val_df = X_val.copy()
    val_df['emotion'] = y_val.values
    
    test_df = X_test.copy()
    test_df['emotion'] = y_test.values
    
    # Lưu CSV
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"[OK] Saved train.csv: {train_df.shape}")
    print(f"[OK] Saved val.csv: {val_df.shape}")
    print(f"[OK] Saved test.csv: {test_df.shape}")


def main():
    print("\n" + "=" * 60)
    print("   FEATURE PROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and split data
    print("\n[STEP 1] Loading and splitting data...")
    df = load_features(RAW_FEATURES_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_70_15_15(df)
    
    # Step 2: Feature Selection (menu)
    print("\n[STEP 2] Feature Selection")
    method_choice = show_menu()
    method_name = METHODS[method_choice][0].lower().replace(" ", "_")
    
    # Nếu là CFS, đặt tên folder là cfs_selected
    if method_choice == 1:
        method_name = "cfs_selected"
    
    X_train_sel, X_val_sel, X_test_sel, selected_features = apply_feature_selection(
        method_choice,
        X_train, y_train, X_val, X_test
    )
    
    # Step 3: Save processed data
    print("\n[STEP 3] Saving processed data...")
    output_dir = os.path.join(OUTPUT_BASE_DIR, method_name)
    save_processed_data(
        X_train_sel, X_val_sel, X_test_sel,
        y_train, y_val, y_test,
        output_dir
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("   PROCESSING COMPLETED!")
    print("=" * 60)
    print(f"  Method: {METHODS[method_choice][0]}")
    print(f"  Features: {len(selected_features)}")
    print(f"  Output: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
