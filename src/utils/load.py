import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from tess import extract_feature

def load_tess_dataset(dataset_path):
    """
    Load danh sách các file audio từ dataset TESS
    
    Args:
        dataset_path: Đường dẫn đến thư mục dataset TESS
    
    Returns:
        DataFrame chứa thông tin về các file audio
    """
    file_paths = []
    emotions = []
    actors = []

    # Duyệt qua tất cả các thư mục con
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)

        # Kiểm tra xem có phải là thư mục không (tránh file hệ thống)
        if os.path.isdir(folder_path):

            parts = folder_name.split('_')
            actor_prefix = parts[0]  # OAF hoặc YAF
            emotion_label = "_".join(parts[1:]).lower()  # Ghép phần còn lại thành emotion (vd: pleasant_surprise)

            # Duyệt qua các file wav trong thư mục
            for filename in os.listdir(folder_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(folder_path, filename)

                    file_paths.append(file_path)
                    emotions.append(emotion_label)
                    actors.append(actor_prefix)

    # Tạo DataFrame để quản lý
    df = pd.DataFrame({
        'path': file_paths,
        'emotion': emotions,
        'actor': actors
    })

    return df

def process_audio_dataset(dataset_path, output_csv='audio_features.csv'):
    """
    Pipeline hoàn chỉnh: Load dataset và trích xuất features từ tất cả các file audio
    
    Args:
        dataset_path: Đường dẫn đến thư mục dataset
        output_csv: Tên file CSV đầu ra
    """
    print("=" * 60)
    print("BẮT ĐẦU PIPELINE TIỀN XỬ LÝ DỮ LIỆU ÂM THANH")
    print("=" * 60)
    
    # Bước 1: Load danh sách các file
    print("\n[1/3] Đang load danh sách các file audio...")
    df_files = load_tess_dataset(dataset_path)
    print(f"✓ Đã tìm thấy {len(df_files)} file âm thanh.")
    print("\nPhân bố các nhãn emotion:")
    print(df_files['emotion'].value_counts())
    
    # Bước 2: Trích xuất features từ mỗi file
    print(f"\n[2/3] Đang trích xuất features từ {len(df_files)} file audio...")
    features_list = []
    
    for idx, row in tqdm(df_files.iterrows(), total=len(df_files), desc="Đang xử lý"):
        features = extract_feature(
            file_path=row['path'],
            emotion=row['emotion']
        )
        
        if features is not None:
            features_list.append(features)
    
    # Bước 3: Tạo DataFrame và lưu vào CSV
    print(f"\n[3/3] Đang lưu kết quả vào file {output_csv}...")
    df_features = pd.DataFrame(features_list)
    
    # Sắp xếp các cột: filename, emotion trước, sau đó là các features
    priority_cols = ['filename', 'emotion', 'actor']
    existing_priority = [col for col in priority_cols if col in df_features.columns]
    other_cols = [col for col in df_features.columns if col not in priority_cols]
    df_features = df_features[existing_priority + other_cols]
    
    # Lưu vào CSV
    df_features.to_csv(output_csv, index=False)
    
    print("=" * 60)
    print(f"✓ HOÀN THÀNH! Đã xử lý {len(df_features)}/{len(df_files)} file.")
    print(f"✓ Kết quả đã được lưu vào: {output_csv}")
    # Đếm số features (loại trừ các cột metadata)
    metadata_cols = ['filename', 'emotion', 'actor']
    num_features = len([col for col in df_features.columns if col not in metadata_cols])
    print(f"✓ Tổng số features: {num_features} features")
    print("=" * 60)
    
    print("\nMẫu dữ liệu đầu ra:")
    print(df_features.head())
    
    return df_features

if __name__ == "__main__":
    # Đường dẫn dataset
    dataset_path = r"D:\AI\EmoSpeech\dataset\TESS"
    
    # Chạy pipeline
    df_result = process_audio_dataset(
        dataset_path=dataset_path,
        output_csv='tess_audio_features.csv'
    )