import numpy as np
import pandas as pd


np.random.seed(42)


feed_mean  = 1.5
feed_lower = 1.25
feed_upper = 1.75
feed_std   = (feed_upper - feed_lower) / (2 * 1.645)  # 90%: 1.645


feed_cf_mean  = 1736.476
feed_cf_lower = feed_cf_mean * (1 - 0.25)
feed_cf_upper = feed_cf_mean * (1 + 0.25)
feed_cf_std   = (feed_cf_upper - feed_cf_lower) / (2 * 1.645)  # 90%: 1.645


ENERGY_PCT = 0.25


N_SIMS = 10_000


ENERGY_XLSX   = "energy_data.xlsx"
PROD_XLSX     = "production.xlsx"
SHEET_HINT    = "South Africa"

OUTPUT_XLSX   = "result of ZA.xlsx"
SHEET_SUM     = "CI95_Width_Sum"
SHEET_TREAT   = "treat"
INTERP_START  = 2020
INTERP_END    = 2050
FIVE_YEARS    = list(range(2020, 2051, 5))


def clipped_normal_by_bounds(mean, lower, upper, size):

    if lower > upper:
        lower, upper = upper, lower
    std = (upper - lower) / (2 * 1.645)  # 90%: 1.645
    samples = np.random.normal(mean, std, size)
    return np.clip(samples, lower, upper)

def ci95_width(x):

    l = float(np.percentile(x, 5.0))
    u = float(np.percentile(x, 95.0))
    return abs(u - l)

def resolve_sheet_name(xlsx_path, hint):

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
    raise ValueError(f"no matched sheet: {hint}。available: {sheets}")

def read_production_5y(xlsx_path, hint, target_years):

    sheet = resolve_sheet_name(xlsx_path, hint)
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    ycol = lower_map.get("year")
    pcol = (lower_map.get("production") or lower_map.get("prod") or
            lower_map.get("output") or lower_map.get("quantity") or lower_map.get("qty"))
    if not ycol or not pcol:
        raise ValueError(f"{xlsx_path}/{sheet} need Year and Production")

    df = df[[ycol, pcol]].copy()
    df.columns = ["Year", "Production"]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Production"] = pd.to_numeric(df["Production"], errors="coerce")
    df = df.dropna().astype({"Year": int})
    df = df[(df["Year"] >= INTERP_START) & (df["Year"] <= INTERP_END)]
    if df.empty:
        raise ValueError("产量数据在 2020–2050 区间为空")

    df = df.sort_values("Year")

    x = df["Year"].to_numpy()
    y = df["Production"].to_numpy()
    prod_interp = np.interp(target_years, x, y, left=y[0], right=y[-1])
    return dict(zip(target_years, prod_interp))


sheet_name = resolve_sheet_name(ENERGY_XLSX, SHEET_HINT)
df = pd.read_excel(ENERGY_XLSX, sheet_name=sheet_name)

df.columns = [str(c).strip() for c in df.columns]
lower_map = {c.lower(): c for c in df.columns}
val_col = lower_map.get("tco2eq") or lower_map.get("co2") or lower_map.get("co2eq")
if not val_col:
    raise ValueError("no value column：need tCO2eq / CO2 / CO2eq。")


need = {"process": "Process", "year": "Year"}
for k, v in need.items():
    if k not in lower_map:
        raise ValueError(f"lose column：{v}")
    df = df.rename(columns={lower_map[k]: v})
df = df.rename(columns={val_col: "Value"})


df = df.reset_index().rename(columns={"index": "Order"})
df["Process"] = df["Process"].astype(str).str.strip()
df = df[df["Process"].str.lower() == "energy"].copy()
if df.empty:
    raise ValueError(f"sheet '{sheet_name}' loses 'Energy' row。")
df["Year"] = df["Year"].astype(int)


years_energy = sorted(df["Year"].unique().tolist())

prod_map = read_production_5y(PROD_XLSX, SHEET_HINT, years_energy)


feed_samples    = np.random.normal(feed_mean, feed_std, N_SIMS)
feed_samples    = np.clip(feed_samples, feed_lower, feed_upper)
feed_cf_samples = np.random.normal(feed_cf_mean, feed_cf_std, N_SIMS)
feed_cf_samples = np.clip(feed_cf_samples, feed_cf_lower, feed_cf_upper)

per_unit_samples_kg_per_ton = (feed_samples / feed_mean) * feed_cf_samples
feed_width_unit_kg_per_ton  = ci95_width(per_unit_samples_kg_per_ton)


rows = []
for r in df.itertuples(index=False):
    year = int(r.Year)
    mean_energy = float(r.Value)


    lower = mean_energy * (1 - ENERGY_PCT)
    upper = mean_energy * (1 + ENERGY_PCT)
    energy_samples = clipped_normal_by_bounds(mean_energy, lower, upper, N_SIMS)
    energy_width = ci95_width(energy_samples)


    prod_ton = float(prod_map[year])
    feed_width_t = feed_width_unit_kg_per_ton * prod_ton / 1000.0

    rows.append({
        "Year": year,
        "CI95_Width_Energy_tCO2eq": energy_width,
        "CI95_Width_Feed_tCO2eq":   feed_width_t,
        "CI95_Width_Sum_tCO2eq":    energy_width + feed_width_t,
        "Production_Used_ton":      prod_ton,
        "N_Sims": N_SIMS
    })


sum_df = pd.DataFrame(rows)
order_map = df[["Year", "Order"]].drop_duplicates()
sum_df = (
    sum_df
    .merge(order_map, on="Year", how="left")
    .sort_values("Order")
    .drop(columns=["Order"])
    .reset_index(drop=True)
)


interp_years = np.arange(INTERP_START, INTERP_END + 1)

cols_to_interp = [
    "CI95_Width_Energy_tCO2eq",
    "CI95_Width_Feed_tCO2eq",
    "CI95_Width_Sum_tCO2eq",
]
base = sum_df.set_index("Year")[cols_to_interp]


treat = (
    base
    .reindex(interp_years)
    .interpolate(method="linear")
    .ffill()
    .bfill()
).copy()


treat["Year"] = treat.index.astype(int)
treat = treat.reset_index(drop=True)
treat = treat[["Year"] + cols_to_interp]
treat["Half_Errorbar_tCO2eq"] = treat["CI95_Width_Sum_tCO2eq"] / 2.0
treat["N_Sims"] = N_SIMS


with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="w") as writer:
    sum_df.to_excel(writer, sheet_name=SHEET_SUM, index=False)
    treat.to_excel(writer, sheet_name=SHEET_TREAT, index=False)

print(f"finish（sheet='{sheet_name}'）")
print(f"Output：{OUTPUT_XLSX}")
