import pandas as pd
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler

# 示例数据（前5行）
data = {
    'x1': [5700, 1000, 3400, 3800, 4000],
    'x2': [12.8, 10.9, 8.8, 13.6, 12.8],
    'x3': [2500, 600, 1000, 1700, 1600],
    'x4': [270, 10, 10, 140, 140],
    'x5': [25000, 10000, 9000, 25000, 25000]
}
df = pd.DataFrame(data)

# 1. 数据标准化
scaler = StandardScaler()
Z = scaler.fit_transform(df)

# 2. 因子分析（假设提取2个公因子）
fa = FactorAnalyzer(n_factors=2, rotation='varimax')
fa.fit(Z)

# 输出结果
print("因子载荷矩阵：\n", fa.loadings_)
print("\n共性度：", fa.get_communalities())
print("\n特殊方差：", fa.get_uniquenesses())
print("\n因子得分（前5个样本）：\n", fa.transform(Z)[:5])

# 方差解释率
ev, v = fa.get_eigenvalues()
print("\n特征值：", ev)