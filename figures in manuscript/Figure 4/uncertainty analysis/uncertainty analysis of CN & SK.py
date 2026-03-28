import numpy as np
import pandas as pd

# ===================== 全局配置 =====================
np.random.seed(42)

# Monte Carlo 次数
N_SIMS = 10_000

# 不确定性比例（±p%） —— 仍然用 ±25% 来设定正态分布的“裁剪上下界”和由此反推 std
FEED_REST_PCT = 0.25
ENERGY_PCT    = 0.25

# 输出区间：90% → P5 / P95
LOWER_Q = 5.0
UPPER_Q = 95.0

# 说明：上下界→std 的换算仍按 95%（±1.96σ）来推导，
# 因为“±25% 是上下界”的设定没有改变；若将来希望把 ±25% 解释为 90% 区间，
# 只需把 Z_FOR_BOUNDS 改为 1.645 即可。
Z_FOR_BOUNDS = 1.96

# Natural Feed（龙须菜）参数（单位：
#   input_* = 吨投入 / 单位产量；
#   ef_*    = tCO2eq / 吨投入   ← 允许为负值（汇碳））
NATURAL_FEED_PARAMS = {
    "CN": {  # China
        "input_mean": 20.0,
        "input_lower": 18.0,
        "input_upper": 22.0,
        "ef_mean": -0.05689,
        "ef_lower": -0.03463,
        "ef_upper": -0.07886,
    },
    "SK": {  # South Korea
        "input_mean": 20.0,
        "input_lower": 18.0,
        "input_upper": 22.0,
        "ef_mean": -0.05689,
        "ef_lower": -0.03463,
        "ef_upper": -0.07886,
    },
}

# Excel 文件
FEED_REST_XLSX = "feed_rest_data.xlsx"
ENERGY_XLSX    = "energy_data.xlsx"
PRODUCTION_XLSX= "production.xlsx"

# 国家与 sheet 识别关键词
COUNTRY_SHEETS = {
    "CN": ["China", "CN", "China Mainland", "PRC"],
    "SK": ["South Korea", "Korea", "Republic of Korea", "KR", "KOR"]
}

# 输出文件
OUTPUT_FILES = {
    "CN": "result of CN.xlsx",
    "SK": "result of SK.xlsx"
}

# 五年节点 + 插值范围
FIVE_YEAR_GRID = list(range(2020, 2051, 5))  # [2020, 2025, ..., 2050]
INTERP_START = 2020
INTERP_END   = 2050

SHEET_SUM    = "CI90_Interval_Sum_5y"  # 五年一次明细（各部分区间 & 总区间，90%）
SHEET_TREAT  = "treat"                 # 仅对“总区间”逐年插值 + Half error bar（90%）

# ===================== 工具函数 =====================
def resolve_sheet_name(xlsx_path, hints):
    xf = pd.ExcelFile(xlsx_path)
    sheets = xf.sheet_names
    # exact
    for h in hints:
        if h in sheets:
            return h
    # case-insensitive exact
    lower_map = {s.lower(): s for s in sheets}
    for h in hints:
        if h.lower() in lower_map:
            return lower_map[h.lower()]
    # contains
    for h in hints:
        for s in sheets:
            if h.lower() in s.lower():
                return s
    raise ValueError(f"找不到匹配的 sheet（{xlsx_path}）：hints={hints}；可用={sheets}")

def clipped_normal_by_bounds(mean, lower, upper, size):
    # 仍按“上下界≈±1.96σ”反推 std，并裁剪到 [lower, upper]
    lo, hi = (lower, upper) if lower <= upper else (upper, lower)
    std = (hi - lo) / (2 * Z_FOR_BOUNDS)
    samples = np.random.normal(mean, std, size)
    return np.clip(samples, lo, hi)

def p_lo_hi(samples, q_lo=LOWER_Q, q_hi=UPPER_Q):
    """返回 (P{q_lo}, P{q_hi})，此处为 (P5, P95)"""
    l = float(np.percentile(samples, q_lo))
    u = float(np.percentile(samples, q_hi))
    return l, u

def read_value_series_5y(xlsx_path, sheet_hints):
    """
    读取 FEED_rest 或 Energy：Year, Value（tCO2eq）
    仅保留 2020–2050 且在五年网格上的年份
    """
    sheet = resolve_sheet_name(xlsx_path, sheet_hints)
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    val_col = lower_map.get("tco2eq") or lower_map.get("co2") or lower_map.get("co2eq")
    if not val_col:
        raise ValueError(f"{xlsx_path} / '{sheet}' 未找到数值列（tCO2eq/CO2/CO2eq）")
    year_col = lower_map.get("year")
    if not year_col:
        raise ValueError(f"{xlsx_path} / '{sheet}' 未找到年份列（Year）")
    out = df[[year_col, val_col]].dropna()
    out.columns = ["Year", "Value"]
    out["Year"] = out["Year"].astype(int)
    out = out[(out["Year"] >= INTERP_START) & (out["Year"] <= INTERP_END)]
    out = out[out["Year"].isin(FIVE_YEAR_GRID)]
    out = out.groupby("Year", as_index=False).first().sort_values("Year").reset_index(drop=True)
    return out  # Year, Value (tCO2eq)

def compute_component_ci_per_year(means_df, pct):
    """
    给定 Year,Value（tCO2eq），按 ±pct 截断正态，
    返回两个 dict: {Year->P5}, {Year->P95}
    """
    lowers, uppers = {}, {}
    for r in means_df.itertuples(index=False):
        year = int(r.Year); mean = float(r.Value)
        lo = mean * (1 - pct); hi = mean * (1 + pct)
        sims = clipped_normal_by_bounds(mean, lo, hi, N_SIMS)
        l, u = p_lo_hi(sims)  # 90% 区间
        lowers[year] = l
        uppers[year] = u
    return lowers, uppers

def compute_nf_ci_per_unit_t(params):
    """
    Natural Feed（龙须菜）——允许 EF 为负（汇碳）：
    - input（t/单位产量） & EF（tCO2eq/t投入）用上下界反推 std（仍按1.96）；
    - input 仍下截到 0（物理量不为负）；EF 不截断；
    - 相乘得到“每单位产量”的排放样本（tCO2eq/单位产量）；
    - 返回 (P5, P95)。
    """
    in_mean = float(params["input_mean"])
    in_lo, in_hi = sorted([float(params["input_lower"]), float(params["input_upper"])])
    ef_mean = float(params["ef_mean"])
    ef_lo,  ef_hi  = sorted([float(params["ef_lower"]),  float(params["ef_upper"])])

    in_std = (in_hi - in_lo) / (2 * Z_FOR_BOUNDS)
    ef_std = (ef_hi - ef_lo) / (2 * Z_FOR_BOUNDS)

    input_samples = np.random.normal(in_mean, in_std, N_SIMS)
    ef_samples    = np.random.normal(ef_mean, ef_std, N_SIMS)

    input_samples = np.clip(input_samples, 0, None)  # 仅对投入量做下截 0

    emission_per_unit_t = input_samples * ef_samples  # tCO2eq / 单位产量（可为负）
    return p_lo_hi(emission_per_unit_t)  # (per-unit P5, P95)

def read_production_series_5y(xlsx_path, sheet_hints):
    """
    读取产量序列；若不是严格 5 年点也没关系，
    先筛 2020–2050，再线性插值到五年网格（2020、2025…、2050）
    """
    sheet = resolve_sheet_name(xlsx_path, sheet_hints)
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    year_col = lower_map.get("year")
    if not year_col:
        raise ValueError(f"{xlsx_path} / '{sheet}' 未找到年份列（Year）")
    prod_col = (lower_map.get("production") or lower_map.get("prod") or
                lower_map.get("output") or lower_map.get("quantity") or lower_map.get("qty"))
    if not prod_col:
        raise ValueError(f"{xlsx_path} / '{sheet}' 未找到产量列（Production/Prod/Output/Quantity/Qty）")

    df = df[[year_col, prod_col]].dropna()
    df.columns = ["Year", "Production"]
    df["Year"] = df["Year"].astype(int)
    df = df[(df["Year"] >= INTERP_START) & (df["Year"] <= INTERP_END)]

    # 线性插值到五年网格
    base = df.set_index("Year")[["Production"]]
    grid = pd.DataFrame({"Year": FIVE_YEAR_GRID}).set_index("Year")
    prod_5y = base.reindex(grid.index).interpolate("linear").ffill().bfill().reset_index()
    return prod_5y  # Year, Production（仅五年点）

def make_treat_table_from_total_interval(sum_df):
    """
    只对“总区间”逐年插值（2020–2050）：
    分别对 Lower 与 Upper 做线性插值，再计算 Width 与 Half_Errorbar（90%）
    """
    years = np.arange(INTERP_START, INTERP_END + 1)
    base = sum_df.set_index("Year")[["Lower_90_Total_tCO2eq", "Upper_90_Total_tCO2eq"]]
    treat = (base.reindex(years).interpolate("linear").ffill().bfill()).reset_index()
    treat.columns = ["Year", "Lower_90_Total_tCO2eq", "Upper_90_Total_tCO2eq"]
    treat["CI90_Width_Total_tCO2eq"] = (treat["Upper_90_Total_tCO2eq"] - treat["Lower_90_Total_tCO2eq"]).clip(lower=0.0)
    treat["Half_Errorbar_tCO2eq"] = treat["CI90_Width_Total_tCO2eq"] / 2.0
    treat["N_Sims"] = N_SIMS
    return treat

def process_country(tag, sheet_hints, out_xlsx, nf_params):
    """
    - 读取 FEED_rest（五年点）、Energy（五年点）、Production（五年点）
    - 分别计算三部分的 90% 区间（下界/上界），Natural Feed 需乘五年产量（EF 允许为负）
    - “总区间”= 各部分下界相加 / 各部分上界相加；宽度与 Half 为加和后的结果（90%）
    - 导出：五年点明细（各部分与总区间）；以及 treat（仅对总区间逐年插值）
    """
    # 1) 读 FEED_rest、Energy 的五年点（单位：tCO2eq）
    fr_df_5y = read_value_series_5y(FEED_REST_XLSX, sheet_hints)
    en_df_5y = read_value_series_5y(ENERGY_XLSX,    sheet_hints)

    # 2) 各自的 90% 区间（五年点）
    fr_low, fr_up = compute_component_ci_per_year(fr_df_5y, FEED_REST_PCT)
    en_low, en_up = compute_component_ci_per_year(en_df_5y, ENERGY_PCT)

    # 3) 产量（五年点）
    prod_5y = read_production_series_5y(PRODUCTION_XLSX, sheet_hints)
    prod_map_5y = dict(zip(prod_5y["Year"], prod_5y["Production"]))

    # 4) Natural Feed：每单位产量 (P5, P95) × 五年产量
    nf_per_unit_l, nf_per_unit_u = compute_nf_ci_per_unit_t(nf_params)

    # 5) 合并
    rows = []
    for y in FIVE_YEAR_GRID:
        if y < INTERP_START or y > INTERP_END:
            continue
        fr_l = fr_low.get(y, 0.0); fr_u = fr_up.get(y, 0.0)
        en_l = en_low.get(y, 0.0); en_u = en_up.get(y, 0.0)

        prod_y = float(prod_map_5y[y])
        nf_l = nf_per_unit_l * prod_y
        nf_u = nf_per_unit_u * prod_y

        tot_l = fr_l + en_l + nf_l
        tot_u = fr_u + en_u + nf_u
        width_total = max(tot_u - tot_l, 0.0)

        rows.append({
            "Year": y,
            "Lower_90_FEED_rest_tCO2eq": fr_l, "Upper_90_FEED_rest_tCO2eq": fr_u,
            "Lower_90_Energy_tCO2eq":    en_l, "Upper_90_Energy_tCO2eq":    en_u,
            "Lower_90_NaturalFeed_tCO2eq": nf_l, "Upper_90_NaturalFeed_tCO2eq": nf_u,
            "Lower_90_Total_tCO2eq": tot_l, "Upper_90_Total_tCO2eq": tot_u,
            "CI90_Width_Total_tCO2eq": width_total,
            "Half_Errorbar_tCO2eq": width_total / 2.0,
            "Production_Used": prod_y,
            "N_Sims": N_SIMS
        })

    sum_df = pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)

    # 6) treat：仅对“总区间”逐年插值（Lower/Upper 分别插）
    treat_df = make_treat_table_from_total_interval(sum_df)

    # 7) 写出
    with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="w") as writer:
        sum_df.to_excel(writer, sheet_name=SHEET_SUM, index=False)
        treat_df.to_excel(writer, sheet_name=SHEET_TREAT, index=False)

    print(f"✅ {tag}: 完成 → {out_xlsx}（'{SHEET_SUM}' 五年一次；'{SHEET_TREAT}' 年度插值=对总区间插值）")

# ===================== 主流程：China & South Korea =====================
process_country(
    tag="China",
    sheet_hints=COUNTRY_SHEETS["CN"],
    out_xlsx=OUTPUT_FILES["CN"],
    nf_params=NATURAL_FEED_PARAMS["CN"]
)

process_country(
    tag="South Korea",
    sheet_hints=COUNTRY_SHEETS["SK"],
    out_xlsx=OUTPUT_FILES["SK"],
    nf_params=NATURAL_FEED_PARAMS["SK"]
)
