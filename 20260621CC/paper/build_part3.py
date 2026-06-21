"""第三部分：MILP 单品选择-补货-定价优化、影子价格、灵敏度分析。"""
import json
import warnings

import numpy as np
import pandas as pd
import pulp
from scipy.stats import norm

import common as C

warnings.filterwarnings("ignore")
dc, di, im = C.load()
stats = json.load(open(C.HERE / "stats.json"))

# 候选集：2023-06-24~30 有销售的可售单品（取销量规模前 49 为可选品种）
win = di[(di.date >= "2023-06-24") & (di.date <= "2023-06-30")].copy()
cv_item = (di.groupby("item_code")["qty_net"].std() /
           di.groupby("item_code")["qty_net"].mean()).to_dict()
agg = win.groupby(["item_code", "item_name", "cat"]).agg(
    mean_qty=("qty_net", "mean"),
    price=("avg_retail_price", "mean"),
    cost=("wholesale_price", "mean"),
).reset_index()
loss = im.set_index("item_code")["loss_rate"].to_dict()
agg["loss"] = agg["item_code"].map(loss).fillna(0.1)
agg = agg[(agg.mean_qty > 0) & (agg.price > agg.cost)].reset_index(drop=True)
agg = agg.sort_values("mean_qty", ascending=False).head(49).reset_index(drop=True)
n = len(agg)
MIN_DISP = 2.5


def item_profit(mu_d, price, cost, lossr, cv):
    """单品在最小陈列量约束下的单日期望利润（对数正态报童，含过量损耗）。"""
    cv = max(min(cv, 1.5), 0.2)
    sigma = np.sqrt(np.log(1 + cv ** 2))
    mu = np.log(max(mu_d, 1e-3)) - sigma ** 2 / 2
    cr = (price - cost) / (price + lossr * cost)
    Q_star = float(np.exp(mu + sigma * norm.ppf(np.clip(cr, 1e-3, 0.999))))
    Q = max(MIN_DISP, Q_star)               # 受最小陈列量约束
    z = (np.log(Q) - mu) / sigma
    Emin = mu_d * norm.cdf((np.log(Q) - mu - sigma ** 2) / sigma) + Q * (1 - norm.cdf(z))
    profit = price * Emin - cost * Q - lossr * cost * (Q - Emin)
    return profit, Q


agg["unit_profit"] = agg["price"] - agg["cost"] / (1 - agg["loss"])
res = [item_profit(r.mean_qty, r.price, r.cost, r.loss, cv_item.get(r.item_code, 0.6))
       for r in agg.itertuples()]
agg["exp_profit"] = [r[0] for r in res]
agg["Q"] = [r[1] for r in res]

# MILP：纯品种选择（Q 已由各单品报童最优给定），最大化期望利润
# 27<=Σy<=33；每品类>=1。低需求单品在 2.5kg 约束下期望利润为负，构成内部最优。
prob = pulp.LpProblem("sku_selection", pulp.LpMaximize)
y = {i: pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)}
prob += pulp.lpSum(agg.loc[i, "exp_profit"] * y[i] for i in range(n))
prob += pulp.lpSum(y[i] for i in range(n)) >= 27
prob += pulp.lpSum(y[i] for i in range(n)) <= 33
for cat in C.ORDER:
    idx = agg.index[agg.cat == cat].tolist()
    if idx:
        prob += pulp.lpSum(y[i] for i in idx) >= 1
prob.solve(pulp.PULP_CBC_CMD(msg=0))
sel = [i for i in range(n) if y[i].value() and y[i].value() > 0.5]
total_q = sum(agg.loc[i, "Q"] for i in sel)
total_profit = pulp.value(prob.objective)
agg["sel"] = [1 if i in sel else 0 for i in range(n)]
stats["milp"] = {"n_candidates": n, "n_selected": len(sel),
                 "total_q": round(total_q, 1), "total_profit": round(total_profit, 1),
                 "cats_covered": int(agg[agg.sel == 1]["cat"].nunique())}
print("MILP:", stats["milp"])

# ============================ 图15  MILP 品种选择 ============================
seldf = agg[agg.sel == 1].sort_values(["cat", "Q"], ascending=[True, False])
fig, (ax1, ax2) = C.plt.subplots(1, 2, figsize=(14.5, 7),
                                 gridspec_kw={"width_ratios": [1, 1]})
ypos = 0; yticks = []; ylabels = []
for cat in C.ORDER:
    cd = seldf[seldf.cat == cat]
    for _, r in cd.iterrows():
        ax1.barh(ypos, r["Q"], color=C.PALETTE[cat])
        yticks.append(ypos); ylabels.append(f"{r['item_name'][:8]}")
        ypos += 1
    ypos += 0.6
ax1.set_yticks(yticks); ax1.set_yticklabels(ylabels, fontsize=7.5)
C.style_ax(ax1, f"MILP 最优品种选择（共 {len(sel)} 个）", "补货量 (kg)")
ax1.invert_yaxis()
import matplotlib.patches as mpatches
ax1.legend(handles=[mpatches.Patch(color=C.PALETTE[c], label=c) for c in C.ORDER],
           fontsize=8, ncol=2, loc="lower right")
top15 = seldf.nlargest(15, "Q")
ax2.barh(range(15), top15["Q"].values[::-1],
         color=[C.PALETTE[c] for c in top15["cat"].values[::-1]])
ax2.set_yticks(range(15))
ax2.set_yticklabels([f"{r['item_name'][:9]}" for _, r in top15.iloc[::-1].iterrows()],
                    fontsize=8)
C.style_ax(ax2, "补货量前 15 单品", "补货量 (kg)")
fig.suptitle("MILP 单品选择-补货-定价优化结果", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96]); C.save(fig, "fig15_milp_selection.png")

# ============================ 图16  品种数量影子价格 ============================
ks = list(range(15, 46, 2))
profits = []
for k in ks:
    p2 = pulp.LpProblem("k", pulp.LpMaximize)
    yy = {i: pulp.LpVariable(f"y{i}", cat="Binary") for i in range(n)}
    p2 += pulp.lpSum(agg.loc[i, "exp_profit"] * yy[i] for i in range(n))
    p2 += pulp.lpSum(yy[i] for i in range(n)) == k
    for cat in C.ORDER:
        idx = agg.index[agg.cat == cat].tolist()
        if idx:
            p2 += pulp.lpSum(yy[i] for i in idx) >= 1
    p2.solve(pulp.PULP_CBC_CMD(msg=0))
    profits.append(pulp.value(p2.objective))
profits = np.array(profits)
marg = np.gradient(profits, ks)
fig, (ax1, ax2) = C.plt.subplots(1, 2, figsize=(13.5, 5.2))
ax1.axvspan(27, 33, color="#27926B", alpha=0.12, label="可行区间 [27,33]")
ax1.plot(ks, profits, "o-", color="#2E5E8C", lw=2)
kbest = ks[int(np.argmax(profits))]
ax1.axvline(kbest, color="#C0392B", ls="--", label=f"最优拐点 ≈ {kbest}")
C.style_ax(ax1, "品种数约束与总利润关系", "品种数 k", "总期望利润 (元)"); ax1.legend()
ax2.bar(ks, marg, color=["#27926B" if m >= 0 else "#C0392B" for m in marg], width=1.4)
ax2.axhline(0, color="#555", lw=0.8)
C.style_ax(ax2, "影子价格：增加一个品种的边际价值", "品种数 k", "边际利润 (元/品种)")
fig.suptitle("品种数量约束的影子价格分析", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); C.save(fig, "fig16_shadow_price.png")

# ============================ 图17  灵敏度（损耗率×最小陈列量） ============================
base_loss = agg["loss"].mean()
lf = np.linspace(0.7, 1.3, 40)         # 损耗率倍数
md = np.linspace(2.5, 4.0, 40)          # 最小陈列量
LF, MD = np.meshgrid(lf, md)
Zp = np.zeros_like(LF)
up = agg["unit_profit"].values; mq = agg["mean_qty"].values; ls = agg["loss"].values
pr = agg["price"].values; co = agg["cost"].values
for i in range(LF.shape[0]):
    for j in range(LF.shape[1]):
        loss_adj = np.clip(ls * LF[i, j], 0, 0.6)
        upi = pr - co / (1 - loss_adj)
        qcap = np.maximum(MD[i, j], mq * 1.3)
        order = np.argsort(-upi * qcap)[:30]
        Zp[i, j] = np.sum((upi * np.maximum(qcap, MD[i, j]))[order])
fig, ax = C.plt.subplots(figsize=(9, 6.2))
cf = ax.contourf(LF, MD, Zp, levels=18, cmap="YlOrRd")
cs = ax.contour(LF, MD, Zp, levels=8, colors="white", linewidths=0.6)
ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
ax.axvline(1.0, color="#2E5E8C", ls="--", lw=1.2, label="基准损耗率")
ax.axhline(2.5, color="#6A4C93", ls="--", lw=1.2, label="基准最小陈列量 2.5kg")
C.style_ax(ax, "总利润等高线（损耗率倍数 × 最小陈列量）", "损耗率倍数", "最小陈列量 (kg)")
ax.legend(loc="upper right"); fig.colorbar(cf, ax=ax, label="总期望利润 (元)")
ax.grid(False); fig.tight_layout(); C.save(fig, "fig17_sensitivity.png")

with open(C.HERE / "stats.json", "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("PART3 DONE")
