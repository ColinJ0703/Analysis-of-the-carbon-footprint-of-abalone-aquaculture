import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

file = 'Temperature.xlsx'


xls = pd.ExcelFile(file)
sheet_names = xls.sheet_names
if len(sheet_names) < 2:
    raise ValueError("Excel 至少需要包含两个 sheet。")

def load_sheet(sheet_name):
    df = pd.read_excel(file, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    required = {'Temperature', 'mean', 'SD'}
    if not required.issubset(df.columns):
        raise ValueError(f"Sheet '{sheet_name}' 缺少必需列：{required - set(df.columns)}")
    df = df.dropna(subset=['Temperature', 'mean', 'SD']).sort_values('Temperature')
    return df

df1 = load_sheet(sheet_names[0])
df2 = load_sheet(sheet_names[1])


fig, ax = plt.subplots(figsize=(8, 5))

def plot_with_band(ax, df, label, color):
    x = df['Temperature'].values
    y = df['mean'].values
    sd = df['SD'].values
    ax.plot(x, y, marker='o', linewidth=2, label=f"{label} (±SD)", color=color)
    ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.2)

plot_with_band(ax, df1, sheet_names[0], color='tab:blue')
plot_with_band(ax, df2, sheet_names[1], color='tab:orange')


ax.set_xlabel('Temperature (°C)') 


ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig('temperature.png', dpi=300)
# plt.show()
