"""Q1: distribution patterns and interrelationships of vegetable sales."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import common as c

plt = c.setup_plot()
FIG = c.FIG
res = {}

di = c.build_daily()           # daily item-level
dc = c.build_daily_cat()       # daily category-level
items = c.build()["items"]

CATS = ["花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"]

# ---------- 1. category totals & share ----------
cat_tot = di.groupby("cat_name")["qty"].sum().reindex(CATS)
cat_amt = di.groupby("cat_name")["amount"].sum().reindex(CATS)
res["cat_total_qty"] = cat_tot.round(1).to_dict()
res["cat_total_amount"] = cat_amt.round(1).to_dict()
res["cat_item_count"] = items.groupby("cat_name")["item_id"].nunique().reindex(CATS).to_dict()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
cat_tot.plot(kind="bar", ax=ax[0], color="#4C72B0")
ax[0].set_title("各品类总销量（千克）")
ax[0].set_ylabel("总销量 / kg"); ax[0].set_xlabel("")
ax[0].tick_params(axis="x", rotation=20)
ax[1].pie(cat_tot.values, labels=cat_tot.index, autopct="%1.1f%%", startangle=90,
          colors=plt.cm.Set2.colors)
ax[1].set_title("各品类销量占比")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q1_cat_total.png")); plt.close(fig)

# ---------- 2. seasonality: monthly category time series ----------
di["ym"] = di["date"].dt.to_period("M").dt.to_timestamp()
monthly = di.groupby(["ym", "cat_name"])["qty"].sum().reset_index()
piv_m = monthly.pivot(index="ym", columns="cat_name", values="qty").reindex(columns=CATS)
fig, ax = plt.subplots(figsize=(12, 4.6))
for col in piv_m.columns:
    ax.plot(piv_m.index, piv_m[col], marker="", label=col, linewidth=1.6)
ax.set_title("各品类月销量时间序列（2020-07 至 2023-06）")
ax.set_ylabel("月销量 / kg"); ax.legend(ncol=3, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q1_monthly_ts.png")); plt.close(fig)

# month-of-year seasonal profile (avg across years)
di["month"] = di["date"].dt.month
seas = di.groupby(["month", "cat_name"])["qty"].sum().reset_index()
# normalize per category to show shape
piv_s = seas.pivot(index="month", columns="cat_name", values="qty").reindex(columns=CATS)
piv_s_norm = piv_s / piv_s.mean()
fig, ax = plt.subplots(figsize=(10, 4.4))
for col in piv_s_norm.columns:
    ax.plot(piv_s_norm.index, piv_s_norm[col], marker="o", label=col)
ax.set_xticks(range(1, 13))
ax.set_title("各品类销量的月度季节性（按月汇总并归一化）")
ax.set_xlabel("月份"); ax.set_ylabel("相对销量（=1 为均值）")
ax.legend(ncol=3, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q1_seasonality.png")); plt.close(fig)

# ---------- 3. day-of-week effect ----------
di["dow"] = di["date"].dt.dayofweek
dow = di.groupby(["dow"])["qty"].sum()
daily_total = di.groupby("date")["qty"].sum()
dow_mean = di.groupby(["date"]).agg(qty=("qty", "sum")).reset_index()
dow_mean["dow"] = dow_mean["date"].dt.dayofweek
dow_avg = dow_mean.groupby("dow")["qty"].mean()
labels = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(labels, dow_avg.values, color="#55A868")
ax.set_title("一周内各天的平均日销量（全品类合计）")
ax.set_ylabel("平均日销量 / kg")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q1_dow.png")); plt.close(fig)
res["dow_avg_qty"] = dow_avg.round(1).to_dict()
res["weekend_vs_weekday"] = {
    "weekday": round(dow_avg.iloc[:5].mean(), 1),
    "weekend": round(dow_avg.iloc[5:].mean(), 1)}

# ---------- 4. item-level distribution (long tail / lognormal) ----------
item_tot = di.groupby("item_id")["qty"].sum().sort_values(ascending=False)
res["n_items_sold"] = int(item_tot.shape[0])
res["top10_share"] = round(item_tot.head(10).sum() / item_tot.sum() * 100, 1)
res["top20pct_share"] = round(
    item_tot.head(int(len(item_tot) * 0.2)).sum() / item_tot.sum() * 100, 1)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].hist(np.log10(item_tot.values), bins=30, color="#C44E52", alpha=0.85)
ax[0].set_title("单品总销量分布（对数坐标）")
ax[0].set_xlabel("log10(单品总销量/kg)"); ax[0].set_ylabel("单品数")
# Lorenz / cumulative concentration
cum = np.cumsum(item_tot.values) / item_tot.sum()
xx = np.arange(1, len(cum) + 1) / len(cum)
ax[1].plot(xx * 100, cum * 100, color="#8172B3")
ax[1].plot([0, 100], [0, 100], "k--", linewidth=0.8)
ax[1].set_title("单品销量集中度（累积曲线）")
ax[1].set_xlabel("单品累积占比 %"); ax[1].set_ylabel("销量累积占比 %")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q1_item_dist.png")); plt.close(fig)

# lognormal fit test on daily item qty
ln = np.log(di["qty"][di["qty"] > 0])
res["daily_qty_lognormal_skew_of_log"] = round(float(stats.skew(ln)), 3)

# top items table
top10 = item_tot.head(10).reset_index().merge(
    items[["item_id", "item_name", "cat_name"]], on="item_id")
res["top10_items"] = top10[["item_name", "cat_name", "qty"]].assign(
    qty=lambda d: d["qty"].round(1)).to_dict("records")

# ---------- 5. category correlation matrix ----------
piv_cat = dc.pivot(index="date", columns="cat_name", values="qty").reindex(columns=CATS)
piv_cat = piv_cat.fillna(0)
corr_p = piv_cat.corr(method="pearson")
corr_s = piv_cat.corr(method="spearman")
res["cat_corr_pearson"] = corr_p.round(3).to_dict()


def heat(mat, title, fname):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(mat.values, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            ax.text(j, i, f"{mat.values[i,j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(title); fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)


heat(corr_p, "各品类日销量相关系数矩阵（Pearson）", "q1_cat_corr.png")

# ---------- 6. item-level correlation among top items ----------
top_items = item_tot.head(30).index
piv_it = di[di["item_id"].isin(top_items)].pivot_table(
    index="date", columns="item_id", values="qty", aggfunc="sum").fillna(0)
name_map = items.set_index("item_id")["item_name"].to_dict()
corr_it = piv_it.corr()
# strongest positive pairs
pairs = []
cols = corr_it.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        pairs.append((name_map.get(cols[i], cols[i]), name_map.get(cols[j], cols[j]),
                      round(corr_it.values[i, j], 3)))
pairs.sort(key=lambda x: -x[2])
res["top_item_pairs_pos"] = pairs[:10]
res["top_item_pairs_neg"] = sorted(pairs, key=lambda x: x[2])[:5]

fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(corr_it.values, cmap="RdYlBu_r", vmin=-1, vmax=1)
labels_it = [name_map.get(x, x) for x in corr_it.columns]
ax.set_xticks(range(len(labels_it))); ax.set_xticklabels(labels_it, rotation=90, fontsize=6)
ax.set_yticks(range(len(labels_it))); ax.set_yticklabels(labels_it, fontsize=6)
ax.set_title("销量前30单品的日销量相关系数热力图")
fig.colorbar(im, fraction=0.046)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q1_item_corr.png")); plt.close(fig)

# ---------- 7. clustering of items by weekly/seasonal profile ----------
# build per-item monthly normalized profile for items with enough data
it_month = di.groupby(["item_id", "month"])["qty"].sum().reset_index()
prof = it_month.pivot(index="item_id", columns="month", values="qty").fillna(0)
prof = prof[prof.sum(axis=1) > 50]            # active items
profn = prof.div(prof.sum(axis=1), axis=0)    # share across months
k = 4
km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(profn.values)
prof_lbl = pd.Series(km.labels_, index=profn.index)
res["cluster_sizes"] = pd.Series(km.labels_).value_counts().sort_index().to_dict()
fig, ax = plt.subplots(figsize=(10, 4.6))
for cl in range(k):
    center = profn[prof_lbl == cl].mean()
    ax.plot(center.index, center.values, marker="o", label=f"聚类{cl+1} (n={int((prof_lbl==cl).sum())})")
ax.set_xticks(range(1, 13))
ax.set_title("单品月度销售形态聚类（K-means, 归一化月度占比）")
ax.set_xlabel("月份"); ax.set_ylabel("月度销量占比")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q1_clusters.png")); plt.close(fig)

with open(os.path.join(c.ANALYSIS, "q1_results.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("Q1 done. keys:", list(res.keys()))
print(json.dumps({k: res[k] for k in ["cat_total_qty", "top10_share", "top20pct_share",
      "weekend_vs_weekday", "cat_corr_pearson"]}, ensure_ascii=False, indent=2))
