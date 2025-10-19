import numpy as np
import pandas as pd


np.random.seed(42)


N_SIMS = 10_000


FEED_REST_PCT = 0.25
ENERGY_PCT    = 0.25


LOWER_Q = 5.0
UPPER_Q = 95.0


Z_FOR_BOUNDS = 1.96


NATURAL_FEED_PARAMS = {
    "CN": {  # China
        "input_mean": 20.0,
        "input_lower": 18.0,
        "input_upper": 22.0,
        "ef_mean": -0.05689,
        "ef_lower": -0.03463,
        "ef_upper": -0.07886,
    },
    "KR": {  # South Korea
        "input_mean": 20.0,
        "input_lower": 18.0,
        "input_upper": 22.0,
        "ef_mean": -0.05689,
        "ef_lower": -0.03463,
        "ef_upper": -0.07886,
    },
}


FEED_REST_XLSX = "feed_rest_data.xlsx"
ENERGY_XLSX    = "energy_data.xlsx"
PRODUCTION_XLSX= "production.xlsx"


COUNTRY_SHEETS = {
    "CN": ["China", "CN", "China Mainland", "PRC"],
    "KR": ["South Korea", "Korea", "Republic of Korea", "KR", "KOR"]
}


OUTPUT_FILES = {
    "CN": "result of CN.xlsx",
    "KR": "result of KR.xlsx"
}


FIVE_YEAR_GRID = list(range(2020, 2051, 5))  # [2020, 2025, ..., 2050]
INTERP_START = 2020
INTERP_END   = 2050

SHEET_SUM    = "CI90_Interval_Sum_5y"
SHEET_TREAT  = "treat"


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
    raise ValueError(f"no matched sheet（{xlsx_path}）：hints={hints}；available={sheets}")

def clipped_normal_by_bounds(mean, lower, upper, size):

    lo, hi = (lower, upper) if lower <= upper else (upper, lower)
    std = (hi - lo) / (2 * Z_FOR_BOUNDS)
    samples = np.random.normal(mean, std, size)
    return np.clip(samples, lo, hi)

def p_lo_hi(samples, q_lo=LOWER_Q, q_hi=UPPER_Q):

    l = float(np.percentile(samples, q_lo))
    u = float(np.percentile(samples, q_hi))
    return l, u

def read_value_series_5y(xlsx_path, sheet_hints):

    sheet = resolve_sheet_name(xlsx_path, sheet_hints)
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    val_col = lower_map.get("tco2eq") or lower_map.get("co2") or lower_map.get("co2eq")
    if not val_col:
        raise ValueError(f"{xlsx_path} / '{sheet}' no value column（tCO2eq/CO2/CO2eq）")
    year_col = lower_map.get("year")
    if not year_col:
        raise ValueError(f"{xlsx_path} / '{sheet}' no year column（Year）")
    out = df[[year_col, val_col]].dropna()
    out.columns = ["Year", "Value"]
    out["Year"] = out["Year"].astype(int)
    out = out[(out["Year"] >= INTERP_START) & (out["Year"] <= INTERP_END)]
    out = out[out["Year"].isin(FIVE_YEAR_GRID)]
    out = out.groupby("Year", as_index=False).first().sort_values("Year").reset_index(drop=True)
    return out  # Year, Value (tCO2eq)

def compute_component_ci_per_year(means_df, pct):

    lowers, uppers = {}, {}
    for r in means_df.itertuples(index=False):
        year = int(r.Year); mean = float(r.Value)
        lo = mean * (1 - pct); hi = mean * (1 + pct)
        sims = clipped_normal_by_bounds(mean, lo, hi, N_SIMS)
        l, u = p_lo_hi(sims)
        lowers[year] = l
        uppers[year] = u
    return lowers, uppers

def compute_nf_ci_per_unit_t(params):

    in_mean = float(params["input_mean"])
    in_lo, in_hi = sorted([float(params["input_lower"]), float(params["input_upper"])])
    ef_mean = float(params["ef_mean"])
    ef_lo,  ef_hi  = sorted([float(params["ef_lower"]),  float(params["ef_upper"])])

    in_std = (in_hi - in_lo) / (2 * Z_FOR_BOUNDS)
    ef_std = (ef_hi - ef_lo) / (2 * Z_FOR_BOUNDS)

    input_samples = np.random.normal(in_mean, in_std, N_SIMS)
    ef_samples    = np.random.normal(ef_mean, ef_std, N_SIMS)

    input_samples = np.clip(input_samples, 0, None)

    emission_per_unit_t = input_samples * ef_samples
    return p_lo_hi(emission_per_unit_t)

def read_production_series_5y(xlsx_path, sheet_hints):

    sheet = resolve_sheet_name(xlsx_path, sheet_hints)
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    year_col = lower_map.get("year")
    if not year_col:
        raise ValueError(f"{xlsx_path} / '{sheet}' no year column（Year）")
    prod_col = (lower_map.get("production") or lower_map.get("prod") or
                lower_map.get("output") or lower_map.get("quantity") or lower_map.get("qty"))
    if not prod_col:
        raise ValueError(f"{xlsx_path} / '{sheet}' no production column（Production/Prod/Output/Quantity/Qty）")

    df = df[[year_col, prod_col]].dropna()
    df.columns = ["Year", "Production"]
    df["Year"] = df["Year"].astype(int)
    df = df[(df["Year"] >= INTERP_START) & (df["Year"] <= INTERP_END)]


    base = df.set_index("Year")[["Production"]]
    grid = pd.DataFrame({"Year": FIVE_YEAR_GRID}).set_index("Year")
    prod_5y = base.reindex(grid.index).interpolate("linear").ffill().bfill().reset_index()
    return prod_5y

def make_treat_table_from_total_interval(sum_df):

    years = np.arange(INTERP_START, INTERP_END + 1)
    base = sum_df.set_index("Year")[["Lower_90_Total_tCO2eq", "Upper_90_Total_tCO2eq"]]
    treat = (base.reindex(years).interpolate("linear").ffill().bfill()).reset_index()
    treat.columns = ["Year", "Lower_90_Total_tCO2eq", "Upper_90_Total_tCO2eq"]
    treat["CI90_Width_Total_tCO2eq"] = (treat["Upper_90_Total_tCO2eq"] - treat["Lower_90_Total_tCO2eq"]).clip(lower=0.0)
    treat["Half_Errorbar_tCO2eq"] = treat["CI90_Width_Total_tCO2eq"] / 2.0
    treat["N_Sims"] = N_SIMS
    return treat

def process_country(tag, sheet_hints, out_xlsx, nf_params):


    fr_df_5y = read_value_series_5y(FEED_REST_XLSX, sheet_hints)
    en_df_5y = read_value_series_5y(ENERGY_XLSX,    sheet_hints)


    fr_low, fr_up = compute_component_ci_per_year(fr_df_5y, FEED_REST_PCT)
    en_low, en_up = compute_component_ci_per_year(en_df_5y, ENERGY_PCT)


    prod_5y = read_production_series_5y(PRODUCTION_XLSX, sheet_hints)
    prod_map_5y = dict(zip(prod_5y["Year"], prod_5y["Production"]))


    nf_per_unit_l, nf_per_unit_u = compute_nf_ci_per_unit_t(nf_params)


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


    treat_df = make_treat_table_from_total_interval(sum_df)


    with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="w") as writer:
        sum_df.to_excel(writer, sheet_name=SHEET_SUM, index=False)
        treat_df.to_excel(writer, sheet_name=SHEET_TREAT, index=False)

    print(f" {tag}: finish → {out_xlsx}（'{SHEET_SUM}' five year/step；'{SHEET_TREAT}' Annual interpolation = Interpolation of the total range）")


process_country(
    tag="China",
    sheet_hints=COUNTRY_SHEETS["CN"],
    out_xlsx=OUTPUT_FILES["CN"],
    nf_params=NATURAL_FEED_PARAMS["CN"]
)

process_country(
    tag="South Korea",
    sheet_hints=COUNTRY_SHEETS["KR"],
    out_xlsx=OUTPUT_FILES["KR"],
    nf_params=NATURAL_FEED_PARAMS["KR"]
)
