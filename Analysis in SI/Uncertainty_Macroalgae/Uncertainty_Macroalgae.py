
import os
import numpy as np
import pandas as pd

# --- headless plotting backend ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, gaussian_kde

# ===================== CONFIG =====================
EXCEL_PATH     = "uncertainty_macroalgae.xlsx"
SHEET_NAME     = 0
SEED           = 42
N_SIMS         = 10_000
CI_LEVEL       = 0.95
USE_TRUNCNORM  = True
SYMMETRIC_SD_AROUND_MEAN = True
ALLOW_SD_COLUMN = True
EPS_A          = 1e-6
DPI            = 600
FONT_FAMILY    = "Arial"
DEBUG_PRINT    = True


RESULT_NAME = "Net carbon sequestration of macroalgae"
UNIT_TEX    = r"kg CO$_2$ eq. ton$^{-1}$"
RESULT_LABEL = f"{RESULT_NAME} ({UNIT_TEX})"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)
matplotlib.rcParams["font.family"] = FONT_FAMILY


# ===================== Utilities =====================
def sd_from_minmax(minv: float, meanv: float, maxv: float, symmetric: bool = True) -> float:
    if symmetric:
        half_span = max(abs(meanv - minv), abs(maxv - meanv))
        return half_span / 1.96 if half_span > 0 else 0.0
    else:
        return (maxv - minv) / (2.0 * 1.96)

def _norm_sample(mean: float, sd: float, size: int) -> np.ndarray:
    if sd <= 0:
        return np.full(size, mean, dtype=float)
    return np.random.normal(loc=mean, scale=sd, size=size)

def _truncnorm_sample(mean: float, sd: float, low: float, high: float, size: int) -> np.ndarray:
    if high <= low:
        return np.full(size, mean, dtype=float)
    if sd <= 0:
        return np.clip(np.full(size, mean, dtype=float), low, high)
    a, b = (low - mean) / sd, (high - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)

def _smart_sample(mean: float, sd: float, low: float, high: float, size: int) -> np.ndarray:
    return (_truncnorm_sample(mean, sd, low, high, size)
            if USE_TRUNCNORM else
            _norm_sample(mean, sd, size))

def _normalize_headers(df: pd.DataFrame) -> dict:
    return {str(c).lower().strip(): c for c in df.columns}

def _get_columns(df: pd.DataFrame):
    colmap = _normalize_headers(df)
    wanted = ["varname", "abbrev", "min", "mean", "max"]
    found = {w: colmap[w] for w in wanted if w in colmap}

    aliases = {"varname": ["var", "variable", "name"], "abbrev": ["abbr", "short", "code"]}
    for key, alts in aliases.items():
        if key not in found:
            for alt in alts:
                if alt in colmap:
                    found[key] = colmap[alt]; break

    if any(k not in found for k in wanted):
        if df.shape[1] < 5:
            raise ValueError("Excel requires at least 5 columns：[VarName, Abbrev, min, mean, max]")
        cols = list(df.columns[:5])
        found = {"varname": cols[0], "abbrev": cols[1], "min": cols[2], "mean": cols[3], "max": cols[4]}

    found["sd"] = colmap.get("sd") if ALLOW_SD_COLUMN else None
    return found

def load_params(df: pd.DataFrame):
    cols = _get_columns(df)
    df["_abbrev_lc"] = df[cols["abbrev"]].astype(str).str.strip().str.lower()
    df["_var_lc"]    = df[cols["varname"]].astype(str).str.strip().str.lower()

    def row_to_stats(row):
        vmin  = pd.to_numeric(row[cols["min"]],  errors="coerce")
        vmean = pd.to_numeric(row[cols["mean"]], errors="coerce")
        vmax  = pd.to_numeric(row[cols["max"]],  errors="coerce")
        if cols["sd"] is not None and cols["sd"] in row:
            vsd_raw = pd.to_numeric(row[cols["sd"]], errors="coerce")
            vsd = float(vsd_raw) if pd.notna(vsd_raw) else float(sd_from_minmax(vmin, vmean, vmax, SYMMETRIC_SD_AROUND_MEAN))
        else:
            vsd = float(sd_from_minmax(vmin, vmean, vmax, SYMMETRIC_SD_AROUND_MEAN))
        return {"min": float(vmin), "mean": float(vmean), "max": float(vmax), "sd": float(vsd)}


    rdoc_by_abbrev = df[df["_abbrev_lc"] == "rdoc"]
    if not rdoc_by_abbrev.empty:
        rdoc = row_to_stats(rdoc_by_abbrev.iloc[0])
    else:
        rdoc_by_name = df[df["_var_lc"].str.contains("rdoc")]
        if rdoc_by_name.empty:
            raise ValueError("RDOC (Abbreviation = 'RDOC' or VarName contains 'RDOC') was not found in the table")
        rdoc = row_to_stats(rdoc_by_name.iloc[0])

    # a/b/c/d
    abcd = {}
    for key in ["a", "b", "c", "d"]:
        sel = df[df["_abbrev_lc"] == key]
        if sel.empty:
            raise ValueError(f"The parameter was not found in the table: {key}")
        abcd[key] = row_to_stats(sel.iloc[0])

    return rdoc, abcd


# ===================== Main =====================
def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)


    rdoc, abcd = load_params(df)


    RDOC = _smart_sample(rdoc["mean"], rdoc["sd"], rdoc["min"], rdoc["max"], N_SIMS)
    a    = _smart_sample(abcd["a"]["mean"], abcd["a"]["sd"], abcd["a"]["min"], abcd["a"]["max"], N_SIMS)
    b    = _smart_sample(abcd["b"]["mean"], abcd["b"]["sd"], abcd["b"]["min"], abcd["b"]["max"], N_SIMS)
    c    = _smart_sample(abcd["c"]["mean"], abcd["c"]["sd"], abcd["c"]["min"], abcd["c"]["max"], N_SIMS)
    d    = _smart_sample(abcd["d"]["mean"], abcd["d"]["sd"], abcd["d"]["min"], abcd["d"]["max"], N_SIMS)


    a_safe = np.where(a < EPS_A, EPS_A, a)


    POCs = RDOC / a_safe * b
    POCb = RDOC / a_safe * c
    POCt = RDOC / a_safe * d
    C    = RDOC + POCs + POCb + POCt



    def stats(x):
        return pd.Series({
            "mean": float(np.mean(x)),
            "std":  float(np.std(x, ddof=1)),
            "p2.5": float(np.percentile(x, 2.5)),
            "p50":  float(np.percentile(x, 50)),
            "p97.5":float(np.percentile(x, 97.5)),
        })
    sanity = pd.DataFrame({
        "RDOC": stats(RDOC), "a": stats(a), "b": stats(b), "c": stats(c), "d": stats(d),
        "POCs": stats(POCs), "POCb": stats(POCb), "POCt": stats(POCt), "C": stats(C),
    }).T
    sanity.to_csv(os.path.join(OUTPUT_DIR, "sanity_stats.csv"))


    pd.DataFrame({
        "a": a, "b": b, "c": c, "d": d,
        "RDOC": RDOC, "POCs": POCs, "POCb": POCb, "POCt": POCt,
        RESULT_NAME: C
    }).to_csv(os.path.join(OUTPUT_DIR, "mc_samples.csv"), index=False)


    alpha = 1 - CI_LEVEL
    low_q, high_q = 100 * (alpha/2), 100 * (1 - alpha/2)
    summary = {
        "mean": float(np.mean(C)),
        "std":  float(np.std(C, ddof=1)),
        f"p{low_q:.1f}":  float(np.percentile(C, low_q)),
        "p50.0":          float(np.percentile(C, 50)),
        f"p{high_q:.1f}": float(np.percentile(C, high_q)),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, "summary_stats.csv"), index=False)


    plt.figure()
    plt.hist(C, bins=50, density=True, alpha=0.5,
             edgecolor="black", linewidth=0.5)
    try:
        xs = np.linspace(C.min(), C.max(), 400)
        plt.plot(xs, gaussian_kde(C)(xs), linewidth=2)
    except Exception:
        pass
    plt.xlabel(RESULT_LABEL)
    plt.ylabel("Probability density (per " + UNIT_TEX + ")")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distribution_hist_kde.png"), dpi=DPI, transparent=True)
    plt.close()

    print("\nDone. Files saved in:", os.path.abspath(OUTPUT_DIR))

if __name__ == "__main__":
    main()
