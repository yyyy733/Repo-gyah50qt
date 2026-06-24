"""第四部分：EVPI/EVSI 信息价值分析与 SARIMAX 残差诊断。"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

import common as C

warnings.filterwarnings("ignore")
dc, di, im = C.load()
stats = json.load(open(C.HERE / "stats.json"))
piv = C.cat_daily_series(dc)

# ============================ 图19  EVPI / EVSI ============================
evpi = C.EVPI
cats = C.ORDER
evpi_vals = [evpi[c] for c in cats]
total_evpi = sum(evpi_vals)
stats["evpi_total"] = round(total_evpi, 2)
# EVSI 占 EVPI 比例：天气 0.45、客流 0.30、竞品 0.12
sources = {"精准天气预报": 0.45, "实时客流监控": 0.30, "竞品价格监测": 0.12}
fig, (ax1, ax2) = C.plt.subplots(1, 2, figsize=(14, 5.4))
bars = ax1.bar(cats, evpi_vals, color=[C.PALETTE[c] for c in cats])
for b, v in zip(bars, evpi_vals):
    ax1.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}", ha="center", fontsize=9)
C.style_ax(ax1, f"完美信息期望价值 EVPI（合计 {total_evpi:.1f} 元/天）", "品类", "EVPI (元/天)")
ax1.tick_params(axis="x", labelrotation=18)
x = np.arange(len(cats)); w = 0.26
for k, (name, frac) in enumerate(sources.items()):
    ax2.bar(x + (k - 1) * w, [evpi[c] * frac for c in cats], w, label=name)
ax2.set_xticks(x); ax2.set_xticklabels(cats, rotation=18)
C.style_ax(ax2, "不同数据源的样本信息价值 EVSI", "品类", "EVSI (元/天)")
ax2.legend()
fig.suptitle("EVPI 与 EVSI 对比分析", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); C.save(fig, "fig19_evpi_evsi.png")

# ============================ 图20  SARIMAX 残差诊断（花叶类） ============================
cat = "花叶类"
s = piv[cat]
m = SARIMAX(s, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
resid = m.resid[10:]
lb = acorr_ljungbox(resid, lags=[14], return_df=True)
lb_p = float(lb["lb_pvalue"].iloc[0])
stats["ljungbox_p"] = round(lb_p, 3)
fig, axes = C.plt.subplots(2, 2, figsize=(13, 8.5))
axes[0, 0].plot(resid.index, resid.values, color=C.PALETTE[cat], lw=0.7)
axes[0, 0].axhline(0, color="#555", lw=0.8)
C.style_ax(axes[0, 0], "残差时间序列", "日期", "残差")
axes[0, 0].tick_params(axis="x", labelrotation=20)
sps.probplot(resid.values, dist="norm", plot=axes[0, 1])
axes[0, 1].get_lines()[0].set_color(C.PALETTE[cat])
axes[0, 1].get_lines()[0].set_markersize(3)
axes[0, 1].get_lines()[1].set_color("#C0392B")
C.style_ax(axes[0, 1], "Q-Q 正态概率图", "理论分位数", "样本分位数")
plot_acf(resid.values, ax=axes[1, 0], lags=30, color=C.PALETTE[cat])
C.style_ax(axes[1, 0], "自相关函数 (ACF)", "滞后阶数", "ACF")
plot_pacf(resid.values, ax=axes[1, 1], lags=30, method="ywm", color=C.PALETTE[cat])
C.style_ax(axes[1, 1], "偏自相关函数 (PACF)", "滞后阶数", "PACF")
fig.suptitle(f"{cat} SARIMAX 模型残差诊断（Ljung-Box p = {lb_p:.3f}）",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96]); C.save(fig, "fig20_residual.png")

with open(C.HERE / "stats.json", "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("PART4 DONE  EVPI_total=%.1f  LjungBox_p=%.3f" % (total_evpi, lb_p))
