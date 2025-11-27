import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def load_and_prepare_data(csv_file):
    """
    Load dữ liệu từ CSV và chuẩn bị cho training
    
    Args:
        csv_file: Đường dẫn đến file CSV
    
    Returns:
        X, y, label_encoder
    """
    print("=" * 60)
    print("CHUẨN BỊ DỮ LIỆU")
    print("=" * 60)
    
    # Load CSV
    print(f"\n[1/4] Đang load dữ liệu từ {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"✓ Đã load {len(df)} mẫu dữ liệu")
    print(f"✓ Số features: {len(df.columns) - 2} features")  # trừ filename và emotion
    
    # Hiển thị thông tin về dataset
    print("\nPhân bố các nhãn emotion:")
    print(df['emotion'].value_counts())
    
    # Bỏ cột filename
    print("\n[2/4] Đang xử lý features...")
    if 'filename' in df.columns:
        df = df.drop('filename', axis=1)
        print("✓ Đã loại bỏ cột 'filename'")
    
    # Kiểm tra missing values
    if df.isnull().sum().sum() > 0:
        print(f"⚠ Phát hiện {df.isnull().sum().sum()} giá trị missing, đang xử lý...")
        df = df.fillna(df.mean())
        print("✓ Đã xử lý missing values")
    
    # Tách features và labels
    print("\n[3/4] Đang mã hóa nhãn...")
    X = df.drop('emotion', axis=1)
    y = df['emotion']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"✓ Số lượng classes: {len(label_encoder.classes_)}")
    print(f"✓ Các classes: {list(label_encoder.classes_)}")
    
    print("\n[4/4] Hoàn thành chuẩn bị dữ liệu!")
    print(f"✓ Shape của X: {X.shape}")
    print(f"✓ Shape của y: {y_encoded.shape}")
    
    return X, y_encoded, label_encoder

def train_xgboost_model(X, y, label_encoder, test_size=0.2, random_state=42):
    """
    Huấn luyện XGBoost model
    
    Args:
        X: Features
        y: Labels (đã encode)
        label_encoder: LabelEncoder để decode labels
        test_size: Tỷ lệ test set
        random_state: Random seed
    
    Returns:
        model, X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 60)
    print("HUẤN LUYỆN MODEL XGBOOST")
    print("=" * 60)
    
    # Chia train/test
    print(f"\n[1/3] Đang chia dữ liệu (train: {int((1-test_size)*100)}%, test: {int(test_size*100)}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"✓ Train set: {len(X_train)} mẫu")
    print(f"✓ Test set: {len(X_test)} mẫu")
    
    # Tạo và huấn luyện model
    print(f"\n[2/3] Đang huấn luyện XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    
    # Training với evaluation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    print("✓ Hoàn thành huấn luyện!")
    
    # Đánh giá model
    print(f"\n[3/3] Đang đánh giá model...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"✓ Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"✓ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Đánh giá chi tiết model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        label_encoder: LabelEncoder
    """
    print("\n" + "=" * 60)
    print("ĐÁNH GIÁ CHI TIẾT MODEL")
    print("=" * 60)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Classification Report
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        digits=4
    )
    print(report)
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_test, y_pred)
    
    # Vẽ confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title('Confusion Matrix - XGBoost Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Đã lưu confusion matrix vào 'confusion_matrix.png'")
    plt.close()
    
    # Feature Importance
    plot_feature_importance(model)

def plot_feature_importance(model, top_n=20):
    """
    Vẽ biểu đồ feature importance
    
    Args:
        model: Trained XGBoost model
        top_n: Số lượng features quan trọng nhất để hiển thị
    """
    print(f"\nVẽ biểu đồ {top_n} features quan trọng nhất...")
    
    # Lấy feature importance
    importance = model.feature_importances_
    feature_names = model.get_booster().feature_names
    
    # Tạo DataFrame và sắp xếp
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df, x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Đã lưu feature importance vào 'feature_importance.png'")
    plt.close()

def save_model(model, label_encoder, model_path='xgboost_model.pkl'):
    """
    Lưu model và label encoder
    
    Args:
        model: Trained model
        label_encoder: LabelEncoder
        model_path: Đường dẫn lưu model
    """
    print("\n" + "=" * 60)
    print("LƯU MODEL")
    print("=" * 60)
    
    # Lưu model
    model.save_model('xgboost_model.json')
    print(f"✓ Đã lưu XGBoost model vào 'xgboost_model.json'")
    
    # Lưu label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✓ Đã lưu label encoder vào 'label_encoder.pkl'")
    
    print("\nModel đã sẵn sàng để sử dụng!")

def main():
    """
    Pipeline chính để huấn luyện model
    """
    print("\n" + "=" * 60)
    print("PIPELINE HUẤN LUYỆN XGBOOST - NHẬN DIỆN CẢM XÚC")
    print("=" * 60)
    
    # Đường dẫn đến file CSV
    csv_file = 'tess_audio_features.csv'
    
    # Kiểm tra file tồn tại
    if not os.path.exists(csv_file):
        print(f"\n❌ LỖI: Không tìm thấy file '{csv_file}'")
        print("Vui lòng chạy load.py trước để tạo file dữ liệu!")
        return
    
    # 1. Load và chuẩn bị dữ liệu
    X, y, label_encoder = load_and_prepare_data(csv_file)
    
    # 2. Huấn luyện model
    model, X_train, X_test, y_train, y_test = train_xgboost_model(
        X, y, label_encoder,
        test_size=0.2,
        random_state=42
    )
    
    # 3. Đánh giá model
    evaluate_model(model, X_test, y_test, label_encoder)
    
    # 4. Lưu model
    save_model(model, label_encoder)
    
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH PIPELINE HUẤN LUYỆN!")
    print("=" * 60)
    print("\nCác file đã được tạo:")
    print("  📊 confusion_matrix.png - Ma trận nhầm lẫn")
    print("  📈 feature_importance.png - Độ quan trọng của features")
    print("  🤖 xgboost_model.json - Model đã huấn luyện")
    print("  🏷️ label_encoder.pkl - Label encoder")
    print("=" * 60)

if __name__ == "__main__":
    main()

