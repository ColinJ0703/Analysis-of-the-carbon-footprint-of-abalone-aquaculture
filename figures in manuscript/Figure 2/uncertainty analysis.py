import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

# === 参数设定 ===
np.random.seed(42)

intake_mean = 20
intake_lower = 18
intake_upper = 22
intake_sd = (intake_upper - intake_lower) / (2 * 1.96)

ef_mean = 56.89
ef_lower = 34.63
ef_upper = 78.86
ef_sd = (ef_upper - ef_lower) / (2 * 1.96)

N = 10000
color_cap = 2000

# === 文件路径 ===
input_file = "uncertainty analyse.xlsx"
output_dir = "output_net_emission_uncertainty"
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# === 数据读取（显式保留行名） ===
df = pd.read_excel(input_file, index_col=0)
df_long = df.reset_index().melt(id_vars=df.index.name or "index",
                                var_name="Col_Name", value_name="CF")

if df.index.name is None:
    df_long.rename(columns={"index": "Row_Name"}, inplace=True)
else:
    df_long.rename(columns={df.index.name: "Row_Name"}, inplace=True)

df_long["Case"] = df_long["Col_Name"].astype(str) + "-" + df_long["Row_Name"].astype(str)

# === 颜色映射函数 ===
color_list = ["#1f77b4", "#ffffff", "#FFB400", "#d62728"]
cmap = LinearSegmentedColormap.from_list("custom_map", color_list, N=100)

# === 全局 min 计算 ===
all_values = []
for _, row in df_long.iterrows():
    original_cf = row["CF"]
    intake_samples = np.clip(np.random.normal(intake_mean, intake_sd, N), intake_lower, intake_upper)
    ef_samples = np.maximum(np.random.normal(ef_mean, ef_sd, N), 0)
    net_emissions = original_cf - intake_samples * ef_samples
    all_values.extend(net_emissions)

global_min = np.percentile(all_values, 2.5)
print(f"🎨 全局颜色映射范围: min={global_min:.2f}, center=0, max gradient at {color_cap}")

results = []
proportions = []
density_rows = []   # <<< 新增：收集每个子图的密度曲线数据

# === 模拟（不再绘图） ===
for _, row in df_long.iterrows():
    case_name = str(row["Case"])
    row_name = str(row["Row_Name"])
    col_name = str(row["Col_Name"])
    original_cf = row["CF"]

    # 采样
    intake_samples = np.clip(np.random.normal(intake_mean, intake_sd, N), intake_lower, intake_upper)
    ef_samples = np.maximum(np.random.normal(ef_mean, ef_sd, N), 0)
    net_emissions = original_cf - intake_samples * ef_samples

    # 截尾 95%
    p2_5, p97_5 = np.percentile(net_emissions, [2.5, 97.5])
    net_trim = net_emissions[(net_emissions >= p2_5) & (net_emissions <= p97_5)]

    # 核密度估计
    kde = gaussian_kde(net_trim)
    x_vals = np.linspace(net_trim.min(), net_trim.max(), 512)
    y_vals = kde(x_vals)

    # === 收集密度数据 ===
    density_rows.extend([
        {"Col_Name": col_name, "Row_Name": row_name, "Emission": float(x), "Density": float(y)}
        for x, y in zip(x_vals, y_vals)
    ])

    # 保存统计结果
    results.append({
        "Case": case_name,
        "Row_Name": row_name,
        "Col_Name": col_name,
        "Original_CF": original_cf,
        "Mean_Net_Emission": np.mean(net_emissions),
        "Lower_95CI": np.percentile(net_emissions, 2.5),
        "Upper_95CI": np.percentile(net_emissions, 97.5),
        "Prob_Net_Emission_lt0": np.mean(net_emissions < 0)
    })
    proportions.append({
        "Case": case_name,
        "Row_Name": row_name,
        "Col_Name": col_name,
        "Prob_Net_Emission_lt0": np.mean(net_emissions < 0),
        "Prob_Net_Emission_ge0": np.mean(net_emissions >= 0)
    })

# === 输出 Excel（含密度数据） ===
summary_df = pd.DataFrame(results)
proportion_df = pd.DataFrame(proportions)
density_df = pd.DataFrame(density_rows)  # Col_Name, Row_Name, Emission, Density

excel_path = os.path.join(output_dir, "net_emission_uncertainty_result.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    proportion_df.to_excel(writer, sheet_name="Probability_Stats", index=False)
    density_df.to_excel(writer, sheet_name="Density_Data", index=False)

print("✅ 所有模拟完成")
print(f"📄 结果表格: {excel_path}")
