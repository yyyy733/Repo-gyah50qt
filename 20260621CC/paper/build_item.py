"""问题一补充：单品销量分布规律与相互关系。"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

import common as C

warnings.filterwarnings("ignore")
dc, di, im = C.load()
stats = json.load(open(C.HERE / "stats.json"))

# ---------------- 单品销量汇总 ----------------
g = di.groupby(["item_code", "item_name", "cat"]).agg(
    total_qty=("qty_net", "sum"),
    days=("date", "nunique"),
    mean_qty=("qty_net", "mean"),
    std_qty=("qty_net", "std"),
).reset_index()
g = g[g.total_qty > 0].copy()
g["cv"] = g["std_qty"] / g["mean_qty"]
g = g.sort_values("total_qty", ascending=False).reset_index(drop=True)
n_item = len(g)


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    nn = len(x)
    cum = np.cumsum(x)
    return (nn + 1 - 2 * np.sum(cum) / cum[-1]) / nn


G = gini(g.total_qty.values)
share = g.total_qty.values / g.total_qty.sum()
cum_share = np.cumsum(share)
top10 = cum_share[9] * 100
top20pct_idx = int(np.ceil(0.2 * n_item)) - 1
top20pct_share = cum_share[top20pct_idx] * 100
stats["item"] = {"n_item": int(n_item), "gini": round(float(G), 3),
                 "top10_share": round(float(top10), 1),
                 "top20pct_share": round(float(top20pct_share), 1)}
print("item stats:", stats["item"])

# ============================ 图22  洛伦兹曲线 / 帕累托集中度 ============================
fig, (ax1, ax2) = C.plt.subplots(1, 2, figsize=(13.5, 5.4))
# 洛伦兹曲线
xx = np.insert(np.arange(1, n_item + 1) / n_item, 0, 0)
yy = np.insert(cum_share, 0, 0)
ax1.plot([0, 1], [0, 1], "--", color="#95A5A6", label="绝对均等线")
ax1.plot(xx, yy, color="#2E5E8C", lw=2.4, label=f"洛伦兹曲线 (Gini={G:.3f})")
ax1.fill_between(xx, yy, xx, color="#2E5E8C", alpha=0.12)
ax1.axhline(top20pct_share / 100, color="#C0392B", ls=":", lw=1.2)
ax1.axvline(0.2, color="#C0392B", ls=":", lw=1.2)
ax1.scatter([0.2], [top20pct_share / 100], color="#C0392B", zorder=5)
ax1.annotate(f"前20%单品\n占销量 {top20pct_share:.1f}%", (0.2, top20pct_share / 100),
             xytext=(0.27, top20pct_share / 100 - 0.18), fontsize=9, color="#C0392B",
             arrowprops=dict(arrowstyle="->", color="#C0392B"))
C.style_ax(ax1, "单品销量集中度：洛伦兹曲线", "累计单品比例", "累计销量比例")
ax1.legend(loc="upper left"); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
# 帕累托：销量前40单品的占比与累计占比
m = 40
ax2b = ax2.twinx()
ax2.bar(range(m), share[:m] * 100, color="#5499C7", label="单品销量占比")
ax2b.plot(range(m), cum_share[:m] * 100, color="#C0392B", lw=2, marker="o", ms=3,
          label="累计占比")
ax2b.axhline(80, color="#7D3C98", ls="--", lw=1)
ax2b.text(m * 0.55, 82, "80% 线", color="#7D3C98", fontsize=9)
C.style_ax(ax2, "销量前 40 单品的帕累托图", "单品排名", "单品销量占比 (%)")
ax2b.set_ylabel("累计销量占比 (%)"); ax2b.set_ylim(0, 100)
for s in ("top",):
    ax2b.spines[s].set_visible(False)
fig.suptitle("单品销量分布的长尾与集中度特征", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); C.save(fig, "fig22_item_pareto.png")

# ============================ 图23  单品销量分布 + 热销单品 ============================
fig, (ax1, ax2) = C.plt.subplots(1, 2, figsize=(14, 5.6),
                                 gridspec_kw={"width_ratios": [1, 1.05]})
logq = np.log10(g.total_qty.values)
ax1.hist(logq, bins=28, color="#27926B", edgecolor="white", alpha=0.85)
C.style_ax(ax1, "单品累计销量分布（对数尺度，近似对数正态）",
           "log₁₀(累计销量 / kg)", "单品数量")
ax1.axvline(np.median(logq), color="#C0392B", ls="--",
            label=f"中位数 {10**np.median(logq):.0f} kg")
ax1.legend()
top = g.head(20).iloc[::-1]
ax2.barh(range(20), top.total_qty.values,
         color=[C.PALETTE[c] for c in top.cat.values])
ax2.set_yticks(range(20))
ax2.set_yticklabels([f"{r.item_name[:10]}" for r in top.itertuples()], fontsize=8)
C.style_ax(ax2, "累计销量前 20 单品", "三年累计销量 (kg)")
import matplotlib.patches as mpatches
ax2.legend(handles=[mpatches.Patch(color=C.PALETTE[c], label=c) for c in C.ORDER],
           fontsize=7.5, ncol=2, loc="lower right")
fig.suptitle("单品销量分布规律与热销单品", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); C.save(fig, "fig23_item_dist.png")

# ============================ 图24  单品间相关与层次聚类 ============================
TOPN = 25
top_items = g.head(TOPN)["item_code"].tolist()
name_map = dict(zip(g.item_code, g.item_name))
cat_map = dict(zip(g.item_code, g.cat))
wide = di[di.item_code.isin(top_items)].pivot_table(
    index="date", columns="item_code", values="qty_net", aggfunc="sum")
wide = wide.reindex(columns=top_items)
full = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
wide = wide.reindex(full).fillna(0)
corr = wide.corr(method="spearman")
labels = [name_map[c][:8] for c in top_items]
# 层次聚类排序
dist = 1 - corr.values
np.fill_diagonal(dist, 0)
dist = (dist + dist.T) / 2
Z = linkage(squareform(dist, checks=False), method="average")
fig, (ax1, ax2) = C.plt.subplots(1, 2, figsize=(15, 6.6),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
dend = dendrogram(Z, labels=labels, ax=ax2, color_threshold=0.7,
                  above_threshold_color="#9aa5b1", orientation="right")
C.style_ax(ax2, "热销单品层次聚类（Spearman 相关距离）", "相关距离 (1−ρ)")
ax2.tick_params(axis="y", labelsize=7.5)
order = dend["leaves"]
cm = corr.values[np.ix_(order, order)]
im2 = ax1.imshow(cm, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
ax1.set_xticks(range(TOPN)); ax1.set_yticks(range(TOPN))
ax1.set_xticklabels([labels[i] for i in order], rotation=90, fontsize=7)
ax1.set_yticklabels([labels[i] for i in order], fontsize=7)
ax1.set_title("销量前 25 单品 Spearman 相关热力图", fontweight="bold")
ax1.grid(False)
fig.colorbar(im2, ax=ax1, fraction=0.046, pad=0.04)
fig.suptitle("单品间销量相互关系：相关结构与聚类", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); C.save(fig, "fig24_item_relation.png")

# 类内单品异质性（CV 分布）统计
within = g.groupby("cat")["cv"].median().reindex(C.ORDER)
stats["item"]["within_cv_median"] = {k: round(float(v), 2) for k, v in within.items()}
# 最强单品正相关对
cc = np.array(corr.values, dtype=float, copy=True)
np.fill_diagonal(cc, np.nan)
iu = np.unravel_index(np.nanargmax(cc), cc.shape)
stats["item"]["top_pair"] = [name_map[top_items[iu[0]]], name_map[top_items[iu[1]]],
                             round(float(cc[iu]), 3)]
print("within_cv_median:", stats["item"]["within_cv_median"])
print("top item pair:", stats["item"]["top_pair"])

with open(C.HERE / "stats.json", "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("ITEM ANALYSIS DONE")
