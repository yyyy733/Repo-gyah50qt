"""Q3: single-item replenishment & pricing for 2023-07-01.

Candidate items = items actually sold during 2023-06-24..30.
Choose 27-33 items, each order >= 2.5 kg (min display), cover all 6 categories,
maximise total profit. Pricing is tied to Q2 category markup; item demand is
scaled from its recent level by category price-elasticity.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import pulp
import common as c

plt = c.setup_plot()
FIG = c.FIG
res = {}

di = c.build_daily()
items = c.build()["items"]
wholesale = c.build()["wholesale"]
loss = c.build()["loss"]

CATS = ["花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"]

# Q2 outputs (category elasticity & recommended markup)
with open(os.path.join(c.ANALYSIS, "q2_results.json"), encoding="utf-8") as f:
    q2 = json.load(f)
elas_cat = q2["elasticity"]
markup_cat = {r["cat"]: r["avg_markup"] / 100.0 for r in q2["weekly_summary"]}

MIN_DISPLAY = 2.5
target = pd.Timestamp("2023-07-01")          # Saturday
cand_start, cand_end = pd.Timestamp("2023-06-24"), pd.Timestamp("2023-06-30")

# ----- candidate items: sold in 2023-06-24..30 -----
cand = di[(di["date"] >= cand_start) & (di["date"] <= cand_end)].copy()
cand_items = sorted(cand["item_id"].unique())
res["n_candidates"] = len(cand_items)

# Saturday demand factor from full history (Sat vs overall daily mean)
di["dow"] = di["date"].dt.dayofweek
tot = di.groupby("date")["qty"].sum()
dtot = tot.to_frame("q"); dtot["dow"] = dtot.index.dayofweek
sat_factor = dtot[dtot["dow"] == 5]["q"].mean() / dtot["q"].mean()
res["sat_factor"] = round(float(sat_factor), 3)

# recent cost: last available wholesale price on/before target per item
wh = wholesale[wholesale["date"] <= target]
recent_cost = wh.sort_values("date").groupby("item_id")["cost"].last()
loss_map = loss.set_index("item_id")["loss_rate"] / 100.0
name_map = items.set_index("item_id")["item_name"]
cat_map = items.set_index("item_id")["cat_name"]

rows = []
for it in cand_items:
    sub = cand[cand["item_id"] == it]
    cat = cat_map.get(it)
    if cat not in elas_cat:
        continue
    # recent avg daily demand during candidate week (active days only -> mean per day present)
    days_present = sub["date"].nunique()
    q_recent = sub["qty"].sum() / 7.0          # avg over the 7-day window
    p_recent = (sub["amount"].sum() / sub["qty"].sum())
    w = recent_cost.get(it, np.nan)
    if not np.isfinite(w) or w <= 0:
        w = sub.eval("amount/qty").mean() / (1 + markup_cat[cat])  # fallback
    L = float(loss_map.get(it, 0.10))
    e = elas_cat[cat]
    mk = markup_cat[cat]
    p_new = w * (1 + mk)
    # demand scaled by elasticity from recent price to new price, plus Saturday uplift
    q_new = q_recent * sat_factor * (p_new / p_recent) ** e
    rows.append({
        "item_id": it, "item_name": name_map.get(it), "cat": cat,
        "cost": w, "loss": L, "p_recent": p_recent, "price": p_new,
        "markup_%": mk * 100, "q_recent": q_recent, "demand": max(q_new, 0.0),
        "days_present": days_present})

cdf = pd.DataFrame(rows)

# profit per item if selected (order = max(demand/(1-L), MIN_DISPLAY))
def item_profit(r):
    need_purchase = r["demand"] / (1 - r["loss"]) if r["demand"] > 0 else 0.0
    order = max(need_purchase, MIN_DISPLAY)
    sellable = order * (1 - r["loss"])
    sold = min(sellable, r["demand"]) if r["demand"] > 0 else 0.0
    profit = sold * r["price"] - order * r["cost"]
    return pd.Series({"order": order, "sold": sold, "profit": profit})

cdf = pd.concat([cdf, cdf.apply(item_profit, axis=1)], axis=1)

# ----- MILP: choose 27-33 items, >=1 per category, maximise profit -----
prob = pulp.LpProblem("q3_selection", pulp.LpMaximize)
y = {r.item_id: pulp.LpVariable(f"y_{r.item_id}", cat="Binary") for r in cdf.itertuples()}
prob += pulp.lpSum(r.profit * y[r.item_id] for r in cdf.itertuples())
prob += pulp.lpSum(y.values()) >= 27
prob += pulp.lpSum(y.values()) <= 33
for cat in CATS:
    ids = cdf[cdf["cat"] == cat]["item_id"].tolist()
    if ids:
        prob += pulp.lpSum(y[i] for i in ids) >= 1     # cover every category
prob.solve(pulp.PULP_CBC_CMD(msg=0))

cdf["selected"] = cdf["item_id"].map(lambda i: int(pulp.value(y[i])))
sel = cdf[cdf["selected"] == 1].copy().sort_values(["cat", "profit"], ascending=[True, False])

res["n_selected"] = int(sel.shape[0])
res["selected_by_cat"] = sel.groupby("cat")["item_id"].count().reindex(CATS).fillna(0).astype(int).to_dict()
res["total_profit_q3"] = round(float(sel["profit"].sum()), 1)
res["total_order_q3"] = round(float(sel["order"].sum()), 1)

out = sel[["item_name", "cat", "cost", "loss", "price", "markup_%",
           "demand", "order", "profit"]].copy()
out["loss"] = (out["loss"] * 100).round(2)
for col in ["cost", "price", "demand", "order", "profit"]:
    out[col] = out[col].round(2)
out["markup_%"] = out["markup_%"].round(1)
out.rename(columns={"loss": "loss_%"}, inplace=True)
out.to_csv(os.path.join(c.ANALYSIS, "q3_plan.csv"), index=False, encoding="utf-8-sig")

with open(os.path.join(c.ANALYSIS, "q3_results.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

# plot: order qty by selected item, colored by category
fig, ax = plt.subplots(figsize=(13, 5.2))
colors = {cat: col for cat, col in zip(CATS, plt.cm.Set2.colors)}
sel2 = sel.sort_values("order", ascending=False)
ax.bar(range(len(sel2)), sel2["order"], color=[colors[c_] for c_ in sel2["cat"]])
ax.set_xticks(range(len(sel2)))
ax.set_xticklabels(sel2["item_name"], rotation=90, fontsize=7)
ax.set_ylabel("建议补货量 / kg")
ax.set_title("2023-07-01 单品补货量（按品类着色）")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=colors[c_], label=c_) for c_ in CATS], ncol=6, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q3_item_orders.png")); plt.close(fig)

print("Q3 done")
print("candidates:", res["n_candidates"], "selected:", res["n_selected"])
print("by cat:", res["selected_by_cat"])
print("total profit:", res["total_profit_q3"], "total order kg:", res["total_order_q3"])
print(out.head(35).to_string())
