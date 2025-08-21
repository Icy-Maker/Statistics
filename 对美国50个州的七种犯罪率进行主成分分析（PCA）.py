import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 加载数据（示例用前5行）
data = {
    'State': ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California'],
    'x1': [14.2, 10.8, 9.5, 8.8, 11.5],
    'x2': [25.2, 51.6, 34.2, 27.6, 49.4],
    'x3': [96.8, 96.8, 138.2, 83.2, 287.0],
    'x4': [278.3, 284.0, 312.3, 203.4, 358.0],
    'x5': [1135.5, 1331.7, 2346.1, 972.6, 2139.4],
    'x6': [1881.9, 3369.8, 4467.4, 1862.1, 3499.8],
    'x7': [280.7, 753.3, 439.5, 183.4, 663.5]
}
df = pd.DataFrame(data).set_index('State')

# 1. 数据标准化
scaler = StandardScaler()
Z = scaler.fit_transform(df)

# 2. 主成分分析
pca = PCA()
pca.fit(Z)  # 拟合PCA模型

# 输出结果
print("特征值（解释方差）:", pca.explained_variance_)
print("特征向量（主成分方向）:\n", pca.components_)
print("各主成分贡献率:", pca.explained_variance_ratio_)
print("累计贡献率:", np.cumsum(pca.explained_variance_ratio_))

# 提取前两个主成分的载荷（示例）
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
print("前两个主成分的载荷矩阵（前2列）:\n", loadings[:, :2])