"""Q2: relationship between category sales volume and cost-plus markup;
optimal daily replenishment & pricing for 2023-07-01..07-07 to maximize profit.

Demand model (per category):  ln Q = a + e*ln(P) + month dummies + dow dummies + trend
  e  = price elasticity of demand
Profit per day given retail price P and wholesale cost w, loss rate L:
  to sell Q kg we must purchase Q/(1-L) kg
  profit = Q*P - (Q/(1-L))*w = Q*(P - w/(1-L))
Optimal P maximizes profit; demand Q(P) from the fitted model.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar
import common as c

plt = c.setup_plot()
FIG = c.FIG
res = {}

dc = c.build_daily_cat().copy()
dc = dc.sort_values("date")
CATS = ["花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"]

# clean: drop nonpositive, clip extreme markups per category
dc = dc[(dc["qty"] > 0) & (dc["price"] > 0) & (dc["cost"] > 0)].copy()
dc["month"] = dc["date"].dt.month
dc["dow"] = dc["date"].dt.dayofweek
dc["t"] = (dc["date"] - dc["date"].min()).dt.days
dc["lnq"] = np.log(dc["qty"])
dc["lnp"] = np.log(dc["price"])

# ----- recent cost & loss (last 30 days of data) per category -----
last_day = dc["date"].max()
recent = dc[dc["date"] >= last_day - pd.Timedelta(days=30)]
recent_cost = recent.groupby("cat_name")["cost"].mean()
recent_loss = recent.groupby("cat_name")["loss_rate"].mean() / 100.0

# target week
target_dates = pd.date_range("2023-07-01", "2023-07-07")

elas = {}
models = {}
plan_rows = []

fig_s, axs = plt.subplots(2, 3, figsize=(15, 8))
axs = axs.ravel()

for idx, cat in enumerate(CATS):
    d = dc[dc["cat_name"] == cat].copy()
    # trim extreme markup days (data noise) at 1st/99th pct of price
    lo, hi = d["price"].quantile([0.01, 0.99])
    d = d[(d["price"] >= lo) & (d["price"] <= hi)]
    # regression with seasonal & weekday controls
    model = smf.ols("lnq ~ lnp + C(month) + C(dow) + t", data=d).fit()
    e = model.params["lnp"]
    elas[cat] = round(float(e), 3)
    models[cat] = model

    # markup bounds from history: realistic operating band (25th-75th pct).
    # NOTE: estimated demand is price-inelastic (|e|<1) for every category, so the
    # unconstrained profit optimum is a corner solution; we cap the markup at the
    # historically realised P75 to keep the recommendation credible.
    mk = (d["price"] / d["cost"] - 1.0)
    r_lo, r_hi = float(mk.quantile(0.25)), float(mk.quantile(0.75))
    r_lo = max(r_lo, 0.05)

    w = float(recent_cost[cat])
    L = float(recent_loss[cat])
    w_eff = w / (1 - L)

    # baseline for each target day: predict lnq holding price at recent mean,
    # then K = Q_pred / P^e  (so Q(P)=K*P^e)
    for dt in target_dates:
        Xrow = pd.DataFrame({
            "lnp": [np.log(d["price"].iloc[-30:].mean())],
            "month": [dt.month], "dow": [dt.dayofweek],
            "t": [(dt - dc["date"].min()).days]})
        lnq_hat = model.predict(Xrow).iloc[0]
        P_ref = np.exp(Xrow["lnp"].iloc[0])
        K = np.exp(lnq_hat) / (P_ref ** e)

        def neg_profit(P):
            Q = K * (P ** e)
            return -(Q * (P - w_eff))

        P_min = w * (1 + r_lo)
        P_max = w * (1 + r_hi)
        opt = minimize_scalar(neg_profit, bounds=(P_min, P_max), method="bounded")
        P_star = float(opt.x)
        Q_star = float(K * (P_star ** e))
        repl = Q_star / (1 - L)
        profit = Q_star * (P_star - w_eff)
        plan_rows.append({
            "date": dt.strftime("%Y-%m-%d"), "cat": cat,
            "cost": round(w, 3), "loss_%": round(L * 100, 2),
            "price": round(P_star, 2), "markup_%": round((P_star / w - 1) * 100, 1),
            "demand_kg": round(Q_star, 1), "replenish_kg": round(repl, 1),
            "profit": round(profit, 1)})

    # scatter markup vs qty for the relationship plot
    ax = axs[idx]
    ax.scatter(mk, d["qty"], s=6, alpha=0.25, color="#4C72B0")
    # fitted curve at avg cost/seasonal
    rr = np.linspace(r_lo, r_hi, 50)
    base_lnq = model.predict(pd.DataFrame({
        "lnp": [np.log(w)], "month": [7], "dow": [5],
        "t": [(target_dates[0] - dc["date"].min()).days]})).iloc[0]
    Kc = np.exp(base_lnq) / (w ** e)
    Pcurve = w * (1 + rr)
    Qcurve = Kc * (Pcurve ** e)
    ax.plot(rr, Qcurve, color="#C44E52", lw=2, label=f"拟合 e={e:.2f}")
    ax.set_title(f"{cat}：加成率 vs 销量")
    ax.set_xlabel("成本加成率"); ax.set_ylabel("日销量/kg")
    ax.set_xlim(0, min(2.0, r_hi * 1.1)); ax.legend(fontsize=8)

fig_s.suptitle("各品类：成本加成率与日销量关系（散点+拟合需求曲线）", y=1.02)
fig_s.tight_layout(); fig_s.savefig(os.path.join(FIG, "q2_markup_demand.png")); plt.close(fig_s)

plan = pd.DataFrame(plan_rows)
plan.to_csv(os.path.join(c.ANALYSIS, "q2_plan.csv"), index=False, encoding="utf-8-sig")

# weekly summary per category
wk = plan.groupby("cat").agg(
    week_replenish=("replenish_kg", "sum"),
    avg_price=("price", "mean"),
    avg_markup=("markup_%", "mean"),
    week_profit=("profit", "sum")).round(2)
res["elasticity"] = elas
res["weekly_summary"] = wk.reset_index().to_dict("records")
res["recent_cost"] = recent_cost.round(3).to_dict()
res["recent_loss_%"] = (recent_loss * 100).round(2).to_dict()
res["total_week_profit"] = round(plan["profit"].sum(), 1)
res["total_week_replenish"] = round(plan["replenish_kg"].sum(), 1)

# elasticity & R2 table
res["model_R2"] = {cat: round(models[cat].rsquared, 3) for cat in CATS}

with open(os.path.join(c.ANALYSIS, "q2_results.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

# daily replenishment plot
fig, ax = plt.subplots(figsize=(11, 4.6))
pv = plan.pivot(index="date", columns="cat", values="replenish_kg").reindex(columns=CATS)
pv.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
ax.set_title("2023-07-01~07-07 各品类每日建议补货量")
ax.set_ylabel("补货量 / kg"); ax.set_xlabel(""); ax.legend(ncol=3, fontsize=8)
ax.tick_params(axis="x", rotation=30)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q2_daily_replenish.png")); plt.close(fig)

print("Q2 done")
print("elasticity:", json.dumps(elas, ensure_ascii=False))
print("R2:", json.dumps(res["model_R2"], ensure_ascii=False))
print(wk.to_string())
print("total week profit:", res["total_week_profit"], "total replenish:", res["total_week_replenish"])
