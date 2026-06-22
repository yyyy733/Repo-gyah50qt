"""第一部分：多尺度依赖分析图（STL、小波、Copula、尾部依赖、Granger、互信息、聚类）
并计算真实统计量写入 stats.json，保证图文数值一致。"""
import json

import numpy as np
import pandas as pd
import pywt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import kendalltau, rankdata
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import grangercausalitytests

import common as C

dc, di, im = C.load()
piv = C.cat_daily_series(dc)
stats = {}

# ============================ 图1  STL 分解 ============================
comps = {}
desc = {}
varshare = {}
for cat in C.ORDER:
    s = piv[cat]
    res = STL(s, period=7, robust=True).fit()
    comps[cat] = res
    tot = np.var(s)
    varshare[cat] = {
        "trend": round(100 * np.var(res.trend) / tot, 1),
        "season": round(100 * np.var(res.seasonal) / tot, 1),
        "resid": round(100 * np.var(res.resid) / tot, 1),
    }
    desc[cat] = {"mean": round(s.mean(), 2), "std": round(s.std(), 2),
                 "cv": round(s.std() / s.mean(), 3)}
stats["desc"] = desc
stats["varshare"] = varshare

fig, axes = C.plt.subplots(6, 4, figsize=(13.5, 14), sharex=True)
titles = ["原序列", "趋势分量", "季节分量", "残差分量"]
for i, cat in enumerate(C.ORDER):
    res = comps[cat]
    series = [piv[cat], res.trend, res.seasonal, res.resid]
    for j, (ax, ser) in enumerate(zip(axes[i], series)):
        ax.plot(ser.index, ser.values, color=C.PALETTE[cat], lw=0.7)
        C.style_ax(ax)
        if i == 0:
            ax.set_title(titles[j])
        if j == 0:
            ax.set_ylabel(cat, fontsize=11, fontweight="bold")
        ax.tick_params(labelrotation=0)
for ax in axes[-1]:
    for lab in ax.get_xticklabels():
        lab.set_rotation(20)
        lab.set_ha("right")
fig.suptitle("六大蔬菜品类 STL 时序分解", fontsize=15, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.99])
C.save(fig, "fig01_stl.png")

# ============================ 图2  小波多分辨率方差 ============================
levels = 5
labels = [f"D{i}" for i in range(1, levels + 1)] + [f"A{levels}"]
fig, ax = C.plt.subplots(figsize=(9.5, 5.2))
x = np.arange(len(labels))
w = 0.13
for k, cat in enumerate(C.ORDER):
    s = (piv[cat] - piv[cat].mean()).to_numpy(dtype=float).copy()
    s = np.require(s, dtype=float, requirements=["C", "W", "O"])
    coeffs = pywt.wavedec(s, "db4", level=levels)
    energy = np.array([np.sum(c ** 2) for c in (coeffs[1:][::-1] + [coeffs[0]])])
    share = 100 * energy / energy.sum()
    ax.bar(x + (k - 2.5) * w, share, w, label=cat, color=C.PALETTE[cat])
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n({'高频' if l.startswith('D') else '低频趋势'})" for l in labels])
C.style_ax(ax, None, "小波分解尺度", "方差贡献率 (%)")
ax.set_title("MODWT 多分辨率小波方差贡献率", pad=10, fontweight="bold")
ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.08),
          frameon=False, columnspacing=1.2, handletextpad=0.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(C.FIG / "fig02_wavelet.png", bbox_inches="tight")
C.plt.close(fig)
print("saved fig02_wavelet.png")

# ============================ Kendall / Pearson / MI 矩阵 ============================
R = piv[C.ORDER]
n = len(C.ORDER)
kmat = np.zeros((n, n))
pmat = R.corr(method="pearson").values
mimat = np.zeros((n, n))
for a in range(n):
    for b in range(n):
        kmat[a, b] = kendalltau(R.iloc[:, a], R.iloc[:, b]).correlation
        if a != b:
            mi = mutual_info_regression(R.iloc[:, [a]].values, R.iloc[:, b].values,
                                        random_state=42)[0]
            mimat[a, b] = mi
# 收集 15 对 Kendall
pairs = []
for a in range(n):
    for b in range(a + 1, n):
        pairs.append((C.ORDER[a], C.ORDER[b], kmat[a, b]))
pairs.sort(key=lambda t: t[2], reverse=True)
stats["kendall_pairs"] = [[p[0], p[1], round(p[2], 4)] for p in pairs]
stats["kendall_top"] = [pairs[0][0], pairs[0][1], round(pairs[0][2], 4)]

# ============================ 图3  Copula PIT 散点 ============================
top4 = pairs[:4]
fig, axes = C.plt.subplots(1, 4, figsize=(15, 4))
for ax, (a, b, tau) in zip(axes, top4):
    ua = rankdata(R[a]) / (len(R) + 1)
    ub = rankdata(R[b]) / (len(R) + 1)
    ax.scatter(ua, ub, s=6, alpha=0.25, color=C.PALETTE[a], edgecolors="none")
    ax.set_title(f"{a} — {b}\nKendall τ = {tau:.3f}", fontsize=11)
    C.style_ax(ax, xlabel=f"F({a})", ylabel=f"F({b})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
fig.suptitle("品类销量 Copula PIT 空间散点（伪观测）", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
C.save(fig, "fig03_copula_scatter.png")

# ============================ 图4  尾部依赖 / Kendall ============================
fig, (ax1, ax2) = C.plt.subplots(1, 2, figsize=(14, 5.6),
                                 gridspec_kw={"width_ratios": [1, 1.05]})
im1 = ax1.imshow(kmat, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
ax1.set_xticks(range(n)); ax1.set_yticks(range(n))
ax1.set_xticklabels(C.ORDER, rotation=35, ha="right"); ax1.set_yticklabels(C.ORDER)
for a in range(n):
    for b in range(n):
        ax1.text(b, a, f"{kmat[a,b]:.2f}", ha="center", va="center",
                 fontsize=8, color="white" if abs(kmat[a, b]) > 0.3 else "#222")
ax1.set_title("品类间 Kendall-τ 秩相关热力图")
ax1.grid(False)
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
labels = [f"{a}-{b}" for a, b, _ in pairs]
vals = [v for _, _, v in pairs]
colors = ["#C0392B" if v >= 0 else "#2E5E8C" for v in vals]
ax2.barh(range(len(vals))[::-1], vals, color=colors)
ax2.set_yticks(range(len(vals))[::-1]); ax2.set_yticklabels(labels, fontsize=8)
for i, v in enumerate(vals):
    ax2.text(v + (0.01 if v >= 0 else -0.01), len(vals) - 1 - i, f"{v:.3f}",
             va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
C.style_ax(ax2, "15 对品类 Kendall-τ（由强到弱）", "Kendall-τ")
ax2.axvline(0, color="#888", lw=0.8)
fig.suptitle("品类间 Copula 尾部依赖分析", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
C.save(fig, "fig04_tail_dependence.png")

# ============================ Granger 因果 ============================
maxlag = 3
edges = []
Rlog = np.log(R.clip(lower=1e-3)).diff().dropna()
for a in range(n):
    for b in range(n):
        if a == b:
            continue
        try:
            data = Rlog.iloc[:, [b, a]].values  # test a -> b
            res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            pmin = min(res[L][0]["ssr_ftest"][1] for L in range(1, maxlag + 1))
            if pmin < 0.05:
                edges.append((C.ORDER[a], C.ORDER[b], pmin))
        except Exception:
            pass
stats["granger_sig"] = len(edges)
stats["granger_total"] = n * (n - 1)

# ============================ 图5  Granger 网络 ============================
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
# 出度（领先他类）/入度（被他类领先）
outdeg = {c: sum(1 for a, b, _ in edges if a == c) for c in C.ORDER}
indeg = {c: sum(1 for a, b, _ in edges if b == c) for c in C.ORDER}
ang = {cat: 2 * np.pi * i / n + np.pi / 2 for i, cat in enumerate(C.ORDER)}
pos = {c: (np.cos(a), np.sin(a)) for c, a in ang.items()}
fig, ax = C.plt.subplots(figsize=(9, 8.6))
for a, b, p in edges:
    x1, y1 = pos[a]; x2, y2 = pos[b]
    lw = 2.6 if p < 0.001 else (1.7 if p < 0.01 else 1.0)
    col = "#7B241C" if p < 0.001 else ("#C0392B" if p < 0.01 else "#E59866")
    ax.annotate("", xy=(x2 * 0.80, y2 * 0.80), xytext=(x1 * 0.80, y1 * 0.80),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, alpha=0.7,
                                connectionstyle="arc3,rad=0.12"))
for c, (x, y) in pos.items():
    size = 1100 + 620 * outdeg[c]                       # 节点大小 ∝ 出度（领先力）
    follower = outdeg[c] < indeg[c]                      # 净接收端（跟随者）
    ring = "#C0392B" if follower else "white"
    ax.scatter([x], [y], s=size, color=C.PALETTE[c], zorder=5,
               edgecolors=ring, lw=3.2 if follower else 2)
    ax.text(x, y, c, ha="center", va="center", color="white", fontsize=10,
            fontweight="bold", zorder=6)
    lbl = f"出{outdeg[c]}·入{indeg[c]}"
    ax.text(x * 1.30, y * 1.30, lbl, ha="center", va="center",
            fontsize=9, color="#C0392B" if follower else "#333", zorder=6,
            fontweight="bold" if follower else "normal")
ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.55, 1.55); ax.axis("off")
leg = [mpatches.Patch(color="#7B241C", label="p < 0.001"),
       mpatches.Patch(color="#C0392B", label="p < 0.01"),
       mpatches.Patch(color="#E59866", label="p < 0.05"),
       Line2D([0], [0], marker="o", color="w", markerfacecolor="#999",
              markeredgecolor="#C0392B", markeredgewidth=2.2, markersize=13,
              label="跟随者（出度<入度）"),
       Line2D([0], [0], marker="o", color="w", markerfacecolor="#999",
              markersize=9, label="节点大小 ∝ 出度（领先力）")]
ax.legend(handles=leg, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05),
          fontsize=8.5, framealpha=0.9)
ax.set_title(f"品类间 Granger 因果网络（显著有向边 {len(edges)} 条；标注“出度·入度”）",
             fontsize=13.5, fontweight="bold")
fig.tight_layout()
C.save(fig, "fig05_granger_network.png")

# ============================ 图6  互信息 + Pearson ============================
fig, (ax1, ax2) = C.plt.subplots(1, 2, figsize=(13.5, 5.6))
im2 = ax1.imshow(mimat, cmap="YlGnBu")
ax1.set_xticks(range(n)); ax1.set_yticks(range(n))
ax1.set_xticklabels(C.ORDER, rotation=35, ha="right"); ax1.set_yticklabels(C.ORDER)
for a in range(n):
    for b in range(n):
        if a != b:
            ax1.text(b, a, f"{mimat[a,b]:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if mimat[a, b] > mimat.max() * 0.55 else "#222")
ax1.set_title("互信息矩阵（非线性依赖）"); ax1.grid(False)
fig.colorbar(im2, ax=ax1, fraction=0.046, pad=0.04)
im3 = ax2.imshow(pmat, cmap="RdBu_r", vmin=-1, vmax=1)
ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
ax2.set_xticklabels(C.ORDER, rotation=35, ha="right"); ax2.set_yticklabels(C.ORDER)
for a in range(n):
    for b in range(n):
        ax2.text(b, a, f"{pmat[a,b]:.2f}", ha="center", va="center", fontsize=8,
                 color="white" if abs(pmat[a, b]) > 0.55 else "#222")
ax2.set_title("Pearson 相关系数矩阵（线性依赖）"); ax2.grid(False)
fig.colorbar(im3, ax=ax2, fraction=0.046, pad=0.04)
fig.suptitle("品类间互信息与线性相关对比", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
C.save(fig, "fig06_mutual_info.png")

# ============================ 图7  品类层次聚类 ============================
corr = R.corr(method="spearman")
dist = 1 - corr
Z = linkage(dist.values[np.triu_indices(n, 1)], method="average")
fig, ax = C.plt.subplots(figsize=(9, 5.2))
dendrogram(Z, labels=C.ORDER, ax=ax, color_threshold=0.6,
           above_threshold_color="#9aa5b1")
C.style_ax(ax, "基于依赖结构的品类层次聚类树状图", "品类", "Spearman 相关距离 (1−ρ)")
ax.tick_params(axis="x", labelrotation=15)
fig.tight_layout()
C.save(fig, "fig07_cluster.png")

with open(C.HERE / "stats.json", "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("PART1 DONE")
print(json.dumps(stats, ensure_ascii=False, indent=2)[:1200])
