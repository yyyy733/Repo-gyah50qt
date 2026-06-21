"""Shared data loading and cleaning utilities for the 2023 CUMCM Problem C analysis."""
import os
import glob
import pandas as pd
import numpy as np


def setup_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc"]:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"
    return plt


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(REPO, "analysis")
CACHE = os.path.join(ANALYSIS, "cache")
FIG = os.path.join(REPO, "paper", "figures")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def _find(prefix_digit):
    # files are named 附件1.xlsx etc. Match by ordering of xlsx files.
    xs = sorted(glob.glob(os.path.join(REPO, "*.xlsx")))
    return xs


def load_raw():
    xs = sorted(glob.glob(os.path.join(REPO, "*.xlsx")))
    # Identify by content rather than name (names are non-ascii / order-dependent)
    items = wholesale = sales = loss = None
    loss_cat = None
    for x in xs:
        xl = pd.ExcelFile(x)
        for sh in xl.sheet_names:
            df = xl.parse(sh)
            cols = list(df.columns)
            if any("销售单价" in str(c) for c in cols):
                sales = df
            elif any("批发价格" in str(c) for c in cols):
                wholesale = df
            elif any("损耗率" in str(c) for c in cols) and any("单品编码" in str(c) for c in cols):
                loss = df
            elif any("损耗率" in str(c) for c in cols) and any("小分类" in str(c) for c in cols):
                loss_cat = df
            elif any("分类名称" in str(c) for c in cols):
                items = df
    return items, sales, wholesale, loss, loss_cat


def rename_std(items, sales, wholesale, loss):
    items = items.rename(columns={
        "单品编码": "item_id", "单品名称": "item_name",
        "分类编码": "cat_id", "分类名称": "cat_name"})
    sales = sales.rename(columns={
        "销售日期": "date", "扫码销售时间": "time", "单品编码": "item_id",
        "销量(千克)": "qty", "销售单价(元/千克)": "price",
        "销售类型": "sale_type", "是否打折销售": "discounted"})
    wholesale = wholesale.rename(columns={
        "日期": "date", "单品编码": "item_id", "批发价格(元/千克)": "cost"})
    loss = loss.rename(columns={
        "单品编码": "item_id", "单品名称": "item_name", "损耗率(%)": "loss_rate"})
    return items, sales, wholesale, loss


def build():
    """Load, clean, merge; cache to parquet. Returns dict of frames."""
    f_sales = os.path.join(CACHE, "sales.parquet")
    f_items = os.path.join(CACHE, "items.parquet")
    f_whole = os.path.join(CACHE, "wholesale.parquet")
    f_loss = os.path.join(CACHE, "loss.parquet")
    if all(os.path.exists(p) for p in [f_sales, f_items, f_whole, f_loss]):
        return {
            "sales": pd.read_parquet(f_sales),
            "items": pd.read_parquet(f_items),
            "wholesale": pd.read_parquet(f_whole),
            "loss": pd.read_parquet(f_loss),
        }
    items, sales, wholesale, loss, loss_cat = load_raw()
    items, sales, wholesale, loss = rename_std(items, sales, wholesale, loss)

    sales["date"] = pd.to_datetime(sales["date"])
    wholesale["date"] = pd.to_datetime(wholesale["date"])
    # returns: 销售类型 == 退货 -> negative qty handling
    sales["is_return"] = sales["sale_type"].astype(str).str.contains("退")
    sales["discounted"] = sales["discounted"].astype(str).str.contains("是")

    items.to_parquet(f_items)
    sales.to_parquet(f_sales)
    wholesale.to_parquet(f_whole)
    loss.to_parquet(f_loss)
    return {"sales": sales, "items": items, "wholesale": wholesale, "loss": loss}


def build_daily():
    """Daily item-level aggregated table merged with cost, category, loss."""
    f = os.path.join(CACHE, "daily_item.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    d = build()
    sales, items, wholesale, loss = d["sales"], d["items"], d["wholesale"], d["loss"]
    # net quantity: returns count negative
    s = sales.copy()
    s["signed_qty"] = np.where(s["is_return"], -s["qty"].abs(), s["qty"])
    s["amount"] = s["signed_qty"] * s["price"]
    g = s.groupby(["date", "item_id"]).agg(
        qty=("signed_qty", "sum"),
        amount=("amount", "sum"),
        n_tx=("signed_qty", "size"),
        disc_qty=("signed_qty", lambda x: x[s.loc[x.index, "discounted"]].sum()),
    ).reset_index()
    g = g[g["qty"] > 0].copy()
    g["price"] = g["amount"] / g["qty"]
    g = g.merge(items[["item_id", "item_name", "cat_id", "cat_name"]], on="item_id", how="left")
    g = g.merge(wholesale[["date", "item_id", "cost"]], on=["date", "item_id"], how="left")
    g = g.merge(loss[["item_id", "loss_rate"]], on="item_id", how="left")
    g["markup"] = g["price"] / g["cost"] - 1.0
    g["profit"] = (g["price"] - g["cost"]) * g["qty"]
    g.to_parquet(f)
    return g


def build_daily_cat():
    """Daily category-level table: qty-weighted cost & price, markup."""
    di = build_daily()
    di = di.dropna(subset=["cost"])
    g = di.groupby(["date", "cat_name"]).apply(
        lambda x: pd.Series({
            "qty": x["qty"].sum(),
            "amount": x["amount"].sum(),
            "cost_amt": (x["cost"] * x["qty"]).sum(),
            "loss_rate": np.average(x["loss_rate"], weights=x["qty"]),
            "n_items": x["item_id"].nunique(),
        }), include_groups=False).reset_index()
    g["price"] = g["amount"] / g["qty"]
    g["cost"] = g["cost_amt"] / g["qty"]
    g["markup"] = g["price"] / g["cost"] - 1.0
    g["profit"] = g["amount"] - g["cost_amt"]
    return g


if __name__ == "__main__":
    d = build()
    s = d["sales"]
    print("sales rows:", len(s))
    print("date range:", s["date"].min(), s["date"].max())
    print("sale_type counts:\n", s["sale_type"].value_counts())
    print("returns:", s["is_return"].sum(), "discounted:", s["discounted"].sum())
    print("items:", len(d["items"]))
    print("categories:\n", d["items"]["cat_name"].value_counts())
    print("wholesale rows:", len(d["wholesale"]))
    print("loss rows:", len(d["loss"]), "mean loss%:", d["loss"]["loss_rate"].mean())
