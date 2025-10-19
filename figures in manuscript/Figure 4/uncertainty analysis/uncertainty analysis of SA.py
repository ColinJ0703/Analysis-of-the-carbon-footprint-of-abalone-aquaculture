import numpy as np
import pandas as pd

# ===================== 配置 =====================
np.random.seed(42)

# ---- Feed 参数（不随年份变化）----
# feed_mean 的单位通常是“kg 饲料 / kg 产量”（无量纲比例）
feed_mean  = 1.5
feed_lower = 1.25
feed_upper = 1.75
feed_std   = (feed_upper - feed_lower) / (2 * 1.645)  # 90%: 1.645

# feed_cf_* 的单位是 **kg CO2 / 吨产量**
feed_cf_mean  = 1736.476
feed_cf_lower = feed_cf_mean * (1 - 0.25)
feed_cf_upper = feed_cf_mean * (1 + 0.25)
feed_cf_std   = (feed_cf_upper - feed_cf_lower) / (2 * 1.645)  # 90%: 1.645

# ---- Energy 不确定性：±25%（单位已是 tCO2e）----
ENERGY_PCT = 0.25

# ---- Monte Carlo 次数 ----
N_SIMS = 10_000

# ---- 输入 / 输出 ----
ENERGY_XLSX   = "energy_data.xlsx"      # 与脚本同目录
PROD_XLSX     = "production.xlsx"       # 产量文件
SHEET_HINT    = "South Africa"          # 目标 sheet 名（支持模糊大小写/包含匹配）

OUTPUT_XLSX   = "result of SA.xlsx"     # 输出文件名保持不变
SHEET_SUM     = "CI95_Width_Sum"        # 宽度之和（原始年份）
SHEET_TREAT   = "treat"                 # 2020–2050 线性插值
INTERP_START  = 2020
INTERP_END    = 2050
FIVE_YEARS    = list(range(2020, 2051, 5))

# ===================== 工具函数 =====================
def clipped_normal_by_bounds(mean, lower, upper, size):
    """均值 mean，设 std 使 ~90% 落在 [lower, upper]，并裁剪到界内。"""
    if lower > upper:
        lower, upper = upper, lower
    std = (upper - lower) / (2 * 1.645)  # 90%: 1.645
    samples = np.random.normal(mean, std, size)
    return np.clip(samples, lower, upper)

def ci95_width(x):
    """90% 区间长度（P95 - P5，返回正数）"""
    l = float(np.percentile(x, 5.0))     # 90%: 下限 5%
    u = float(np.percentile(x, 95.0))    # 90%: 上限 95%
    return abs(u - l)

def resolve_sheet_name(xlsx_path, hint):
    """根据 hint 选择 sheet：精确 > 忽略大小写精确 > 忽略大小写包含。"""
    xf = pd.ExcelFile(xlsx_path)
    sheets = xf.sheet_names
    if hint in sheets:
        return hint
    low_map = {s.lower(): s for s in sheets}
    if hint.lower() in low_map:
        return low_map[hint.lower()]
    for s in sheets:
        if hint.lower() in s.lower():
            return s
    raise ValueError(f"找不到匹配的 sheet: {hint}。可用: {sheets}")

def read_production_5y(xlsx_path, hint, target_years):
    """读取产量（year, production），线性插值到 target_years（如 2020/2025…/2050）。"""
    sheet = resolve_sheet_name(xlsx_path, hint)
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    ycol = lower_map.get("year")
    pcol = (lower_map.get("production") or lower_map.get("prod") or
            lower_map.get("output") or lower_map.get("quantity") or lower_map.get("qty"))
    if not ycol or not pcol:
        raise ValueError(f"{xlsx_path}/{sheet} 需要包含 Year 与 Production 列")

    df = df[[ycol, pcol]].copy()
    df.columns = ["Year", "Production"]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Production"] = pd.to_numeric(df["Production"], errors="coerce")
    df = df.dropna().astype({"Year": int})
    df = df[(df["Year"] >= INTERP_START) & (df["Year"] <= INTERP_END)]
    if df.empty:
        raise ValueError("产量数据在 2020–2050 区间为空")

    df = df.sort_values("Year")
    # 线性插值到目标年份（边界用端点值外推）
    x = df["Year"].to_numpy()
    y = df["Production"].to_numpy()
    prod_interp = np.interp(target_years, x, y, left=y[0], right=y[-1])
    return dict(zip(target_years, prod_interp))

# ===================== 读取 South Africa 的 Energy 行 =====================
sheet_name = resolve_sheet_name(ENERGY_XLSX, SHEET_HINT)
df = pd.read_excel(ENERGY_XLSX, sheet_name=sheet_name)

# 统一列名，兼容 tCO2eq / CO2 / CO2eq
df.columns = [str(c).strip() for c in df.columns]
lower_map = {c.lower(): c for c in df.columns}
val_col = lower_map.get("tco2eq") or lower_map.get("co2") or lower_map.get("co2eq")
if not val_col:
    raise ValueError("未找到数值列：需要 tCO2eq / CO2 / CO2eq 之一。")

# 规范关键列并保序
need = {"process": "Process", "year": "Year"}
for k, v in need.items():
    if k not in lower_map:
        raise ValueError(f"缺少列：{v}（或其大小写变体）")
    df = df.rename(columns={lower_map[k]: v})
df = df.rename(columns={val_col: "Value"})

# 仅保留 Energy 行；记录输入顺序
df = df.reset_index().rename(columns={"index": "Order"})
df["Process"] = df["Process"].astype(str).str.strip()
df = df[df["Process"].str.lower() == "energy"].copy()
if df.empty:
    raise ValueError(f"sheet '{sheet_name}' 中没有 'Energy' 行。")
df["Year"] = df["Year"].astype(int)

# 目标年份（通常就是 2020/2025…/2050）
years_energy = sorted(df["Year"].unique().tolist())
# 读取并插值到这些五年点
prod_map = read_production_5y(PROD_XLSX, SHEET_HINT, years_energy)  # 年→产量(ton)

# ===================== 预计算 Feed 的“单位产量”90%区间宽度（kg / 吨产量） =====================
# 注意：feed_cf_* 是 kg/吨产量，所以单位产量排放应为：
# per_unit (kg/吨产量) = (feed_samples / feed_mean) * feed_cf_samples
feed_samples    = np.random.normal(feed_mean, feed_std, N_SIMS)
feed_samples    = np.clip(feed_samples, feed_lower, feed_upper)      # 无量纲比例
feed_cf_samples = np.random.normal(feed_cf_mean, feed_cf_std, N_SIMS)
feed_cf_samples = np.clip(feed_cf_samples, feed_cf_lower, feed_cf_upper)  # kg / 吨产量

per_unit_samples_kg_per_ton = (feed_samples / feed_mean) * feed_cf_samples
feed_width_unit_kg_per_ton  = ci95_width(per_unit_samples_kg_per_ton)     # 这里现为90%宽度

# ===================== 逐年计算 Energy 宽度 + Feed 宽度(×产量→tCO2e) =====================
rows = []
for r in df.itertuples(index=False):
    year = int(r.Year)
    mean_energy = float(r.Value)  # 已是 tCO2e（五年点）

    # Energy：±25% 截断正态 → 宽度（tCO2e）
    lower = mean_energy * (1 - ENERGY_PCT)
    upper = mean_energy * (1 + ENERGY_PCT)
    energy_samples = clipped_normal_by_bounds(mean_energy, lower, upper, N_SIMS)
    energy_width = ci95_width(energy_samples)  # 这里现为90%宽度

    # Feed：单位产量宽度（kg/吨产量）× 当年产量（吨）÷1000 = tCO2e
    prod_ton = float(prod_map[year])            # 该年产量（ton）
    feed_width_t = feed_width_unit_kg_per_ton * prod_ton / 1000.0

    rows.append({
        "Year": year,
        "CI95_Width_Energy_tCO2eq": energy_width,
        "CI95_Width_Feed_tCO2eq":   feed_width_t,
        "CI95_Width_Sum_tCO2eq":    energy_width + feed_width_t,
        "Production_Used_ton":      prod_ton,
        "N_Sims": N_SIMS
    })

# 第一张表：按输入顺序输出
sum_df = pd.DataFrame(rows)
order_map = df[["Year", "Order"]].drop_duplicates()
sum_df = (
    sum_df
    .merge(order_map, on="Year", how="left")
    .sort_values("Order")
    .drop(columns=["Order"])
    .reset_index(drop=True)
)

# ===================== 第二张表：2020–2050 年度线性插值（平均递增） =====================
interp_years = np.arange(INTERP_START, INTERP_END + 1)

cols_to_interp = [
    "CI95_Width_Energy_tCO2eq",
    "CI95_Width_Feed_tCO2eq",
    "CI95_Width_Sum_tCO2eq",
]
base = sum_df.set_index("Year")[cols_to_interp]

# 线性插值；边界用前后值填充，确保 2020–2050 全覆盖
treat = (
    base
    .reindex(interp_years)
    .interpolate(method="linear")
    .ffill()
    .bfill()
).copy()

# Year 回到列并排序；追加“误差棒一半长度”
treat["Year"] = treat.index.astype(int)
treat = treat.reset_index(drop=True)
treat = treat[["Year"] + cols_to_interp]
treat["Half_Errorbar_tCO2eq"] = treat["CI95_Width_Sum_tCO2eq"] / 2.0
treat["N_Sims"] = N_SIMS  # 方便追溯

# ===================== 写入结果（两个 sheet） =====================
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="w") as writer:
    sum_df.to_excel(writer, sheet_name=SHEET_SUM, index=False)
    treat.to_excel(writer, sheet_name=SHEET_TREAT, index=False)

print(f"✅ 完成（sheet='{sheet_name}'）：Feed 宽度按年×产量换算为 tCO2e，Energy 维持 ±25%")
print(f"📄 已输出：{OUTPUT_XLSX} -> '{SHEET_SUM}', '{SHEET_TREAT}'（含 Half_Errorbar_tCO2eq）")
