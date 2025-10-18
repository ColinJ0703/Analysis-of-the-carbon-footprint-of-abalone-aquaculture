import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# === 4PL Logistic function ===
def logistic_4pl(x, A1, A2, x0, p):
    return A2 + (A1 - A2) / (1 + (x / x0) ** p)

# === Input Excel file ===
file = 'Production.xlsx'

# 读取所有 sheet 名
sheet_names = pd.ExcelFile(file).sheet_names

# 全局字体设置
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16  # 全局字体大小

# 遍历每个 sheet
for sheet in sheet_names:
    # === 读取数据 ===
    df = pd.read_excel(file, sheet_name=sheet).iloc[:, :2].copy()
    df.columns = [str(c) for c in df.columns]
    x_label = df.columns[0]
    y_label = df.columns[1]

    years = df.iloc[:, 0].values.astype(float)
    production = df.iloc[:, 1].values.astype(float)

    # === Shift years for numerical stability ===
    year0 = years.min()
    x_data = years - year0 + 1e-6  # ensure positive

    # === Initial guess for 4PL ===
    A1_0 = min(production)
    A2_0 = max(production)
    x0_0 = np.median(x_data)
    p0 = 1.0
    init_params = [A1_0, A2_0, x0_0, p0]

    # === Bounds ===
    bounds_lower = [-np.inf, -np.inf, x_data.min(), 0]
    bounds_upper = [np.inf, np.inf, x_data.max(), np.inf]

    # === Fit 4PL ===
    popt, pcov = curve_fit(
        logistic_4pl, x_data, production,
        p0=init_params, bounds=(bounds_lower, bounds_upper), maxfev=100000
    )
    A1, A2, x0_rel, p = popt
    x0_year = year0 + x0_rel  # Convert to actual year

    # === Compute R² ===
    y_pred = logistic_4pl(x_data, *popt)
    ss_res = np.sum((production - y_pred) ** 2)
    ss_tot = np.sum((production - np.mean(production)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # === Generate fitted curve ===
    x_fit_years = np.linspace(years.min(), years.max(), 300)
    x_fit_rel = x_fit_years - year0 + 1e-6
    y_fit = logistic_4pl(x_fit_rel, *popt)

    # === Plot ===
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar chart (Observed data, no legend)
    if len(years) > 1:
        year_gaps = np.diff(np.sort(years))
        bar_width = 0.8 * (year_gaps[year_gaps > 0].min() if np.any(year_gaps > 0) else 1.0)
    else:
        bar_width = 0.6
    ax.bar(
        years, production,
        width=bar_width,
        color="#6BB392",
        edgecolor="none",
        alpha=0.85,
        zorder=2
    )

    # Logistic curve with function + parameters in legend
    function_str = r"$y = A_{2} + \frac{A_{1} - A_{2}}{1 + \left(\frac{x}{x_{0}}\right)^{p}}$"
    param_str1 = rf"$A_1$={A1:.2f}, $A_2$={A2:.2f}, $x_0$={x0_year:.2f}"
    param_str2 = rf"$p$={p:.2f}, $R^2$={r2:.3f}"
    ax.plot(
        x_fit_years, y_fit,
        '-', linewidth=2.0, color='red',
        label=f"Logistic model:\n{function_str}\n{param_str1}\n{param_str2}",
        zorder=3
    )

    # 保留 2022 竖线
    ax.axvline(2022, linestyle='--', linewidth=1.5, color='gray', alpha=0.9, zorder=1)

    # 自动刻度，但去掉 2022 标签
    xticks = ax.get_xticks()
    xtick_labels = ["" if int(round(t)) == 2022 else str(int(t)) for t in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    # Set x-axis limits ±2 years padding
    ax.set_xlim(years.min() - 2, years.max() + 2)

    # Axes, grid, legend
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=16)

    plt.tight_layout()
    out_name = f"{sheet}.png"
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close()

    # Console output for each sheet
    print(f"[{sheet}] Fitted parameters (Logistic model): "
          f"A1={A1:.6f}, A2={A2:.6f}, x0(year)={x0_year:.6f}, p={p:.6f}, R²={r2:.4f}")
    print(f"Figure saved as {out_name}")
