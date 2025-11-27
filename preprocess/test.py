import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("tess_audio_features.csv")
numeric_df = df.select_dtypes(include=['number'])

# Tính ma trận tương quan
corr_matrix = numeric_df.corr()

# # Hiển thị heatmap
# plt.figure(figsize=(12,10))
# sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
# plt.title("Correlation Heatmap of Audio Features")
# plt.tight_layout()
# plt.show()

# Liệt kê các feature có tương quan >= 0.75
print("\nCác cặp feature có tương quan >= 0.75:")
print("="*50)

# Lấy chỉ số tam giác trên để tránh trùng lặp
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
high_corr = corr_matrix.where(mask)

# Tìm các cặp có tương quan >= 0.75
for i in range(len(high_corr.columns)):
    for j in range(len(high_corr.columns)):
        if abs(high_corr.iloc[i, j]) >= 0.75:
            print(f"{high_corr.columns[i]} - {high_corr.columns[j]}: {high_corr.iloc[i, j]:.3f}")

# Kiểm tra phương sai của các feature
print("\n\nPhương sai (Variance) của các features:")
print("="*50)
variances = numeric_df.var()
variances_sorted = variances.sort_values(ascending=False)

for feature, variance in variances_sorted.items():
    print(f"{feature}: {variance:.6f}")

# Tìm features có phương sai thấp (có thể loại bỏ)
print(f"\n\nFeatures có phương sai thấp (< 0.01):")
low_variance_features = variances[variances < 0.01]
for feature, variance in low_variance_features.items():
    print(f"{feature}: {variance:.6f}")

# Thống kê tổng quan về phương sai
print(f"\n\nThống kê phương sai:")
print(f"Phương sai cao nhất: {variances.max():.6f}")
print(f"Phương sai thấp nhất: {variances.min():.6f}")
print(f"Phương sai trung bình: {variances.mean():.6f}")
print(f"Số features có phương sai < 0.01: {len(low_variance_features)}")