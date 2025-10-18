import numpy as np
import pandas as pd

# 设置随机种子
np.random.seed(42)

# === 参数定义 ===
# 投入量（吨）
input_mean = 20
input_lower = 18
input_upper = 22

# 碳足迹系数（kg CO₂e / 吨）
ef_mean = 56.89
ef_lower = 34.63
ef_upper = 78.86

# 模拟次数
N = 10000

# === 基于95%置信区间估算标准差 ===
input_std = (input_upper - input_lower) / (2 * 1.96)
ef_std = (ef_upper - ef_lower) / (2 * 1.96)

# === 正态分布抽样 ===
input_samples = np.random.normal(loc=input_mean, scale=input_std, size=N)
ef_samples = np.random.normal(loc=ef_mean, scale=ef_std, size=N)

# 过滤负值
input_samples = np.clip(input_samples, 0, None)
ef_samples = np.clip(ef_samples, 0, None)

# 模拟总排放
emission_samples = input_samples * ef_samples  # 单位 kg CO2e

# 计算置信区间
ci_lower = np.percentile(emission_samples, 2.5)
ci_upper = np.percentile(emission_samples, 97.5)
error_bar = (ci_upper - ci_lower) / 2

# 原始中心值
original_value = input_mean * ef_mean

# 输出 DataFrame
result_df = pd.DataFrame({
    'Total Emissions (kg CO2e)': [original_value],
    'Error Bar (±)': [error_bar],
    'Lower Bound': [original_value - error_bar],
    'Upper Bound': [original_value + error_bar]
})

# 保存为 Excel
result_df.to_excel("Fig_1_uncertainty.xlsx", index=False)

print("✅ 结果已保存为 'Fig_1_uncertainty.xlsx'")
