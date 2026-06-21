"""
2023 高教社杯全国大学生数学建模竞赛 C 题
蔬菜类商品的自动定价与补货决策 —— 数据预处理脚本

输入（位于同目录）:
  1.xlsx  附件1 商品信息 (单品编码, 单品名称, 分类编码, 分类名称)
  2.xlsx  附件2 销售流水明细 (销售日期, 扫码销售时间, 单品编码, 销量(千克),
                              销售单价(元/千克), 销售类型, 是否打折销售)
  3.xlsx  附件3 商品批发价格 (日期, 单品编码, 批发价格(元/千克))
  4.xlsx  附件4 商品近期损耗率 (单品编码, 单品名称, 损耗率(%))

输出（写入 processed/ 子目录）:
  item_master.csv         单品主表(附件1+附件4)
  daily_item.csv          单品×日 聚合明细(净销量/营收/加权零售价/批发价/损耗率/成本加成/单位毛利)
  daily_category.csv      品类×日 聚合明细
  cleaned_transactions.parquet  清洗后的逐笔流水(含品类/单品名/销售额/退货标记)
  processed_data.xlsx     汇总工作簿(便于直接查看)
"""

import pandas as pd
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "processed"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# 1. 读取并规范列名（源文件中文列名存在字体导致的乱码，统一改为清晰英文/标准名）
# ----------------------------------------------------------------------------
a1 = pd.read_excel(HERE / "1.xlsx")
a1.columns = ["item_code", "item_name", "cat_code", "cat_name"]

a2 = pd.read_excel(HERE / "2.xlsx")
a2.columns = ["date", "time", "item_code", "qty", "price", "sale_type", "discount"]

a3 = pd.read_excel(HERE / "3.xlsx")
a3.columns = ["date", "item_code", "wholesale_price"]

a4 = pd.read_excel(HERE / "4.xlsx", sheet_name="Sheet1")
a4.columns = ["item_code", "item_name", "loss_rate_pct"]

# 类型规范
a1["item_code"] = a1["item_code"].astype("int64")
a2["item_code"] = a2["item_code"].astype("int64")
a3["item_code"] = a3["item_code"].astype("int64")
a4["item_code"] = a4["item_code"].astype("int64")
a2["date"] = pd.to_datetime(a2["date"])
a3["date"] = pd.to_datetime(a3["date"])

# ----------------------------------------------------------------------------
# 2. 单品主表：附件1 + 附件4 损耗率
# ----------------------------------------------------------------------------
item_master = a1.merge(a4[["item_code", "loss_rate_pct"]], on="item_code", how="left")
item_master["loss_rate"] = item_master["loss_rate_pct"] / 100.0

# ----------------------------------------------------------------------------
# 3. 流水清洗
#    - 退货：销售类型=='退货'，其销量为负；标准化为 is_return 标记
#    - 销售额 = 销量 * 销售单价（退货为负，表示冲减）
#    - 合并品类/单品名
#    - 异常值标记：单笔销量<=0 但非退货、或销量极端偏大（>3*IQR 上界）
# ----------------------------------------------------------------------------
tx = a2.copy()
tx["is_return"] = tx["sale_type"].astype(str).str.contains("退").astype(int)
tx["is_discount"] = tx["discount"].astype(str).str.contains("是").astype(int)
tx["amount"] = tx["qty"] * tx["price"]
tx = tx.merge(a1[["item_code", "item_name", "cat_code", "cat_name"]],
              on="item_code", how="left")

# 异常标记（仅标记不删除，保留可追溯）
q1, q3 = tx.loc[tx.qty > 0, "qty"].quantile([0.25, 0.75])
iqr = q3 - q1
upper = q3 + 3 * iqr
tx["qty_outlier"] = ((tx.qty > upper) & (tx.is_return == 0)).astype(int)
tx["nonpos_nonreturn"] = ((tx.qty <= 0) & (tx.is_return == 0)).astype(int)

tx = tx[["date", "time", "item_code", "item_name", "cat_code", "cat_name",
         "qty", "price", "amount", "sale_type", "discount",
         "is_return", "is_discount", "qty_outlier", "nonpos_nonreturn"]]

# ----------------------------------------------------------------------------
# 4. 单品×日 聚合
# ----------------------------------------------------------------------------
sale = tx[tx.is_return == 0]
ret = tx[tx.is_return == 1]

g_sale = sale.groupby(["date", "item_code"]).agg(
    qty_sold=("qty", "sum"),
    revenue=("amount", "sum"),
    n_txn=("qty", "size"),
    discount_qty=("qty", lambda s: s[sale.loc[s.index, "is_discount"] == 1].sum()),
).reset_index()
# 销量加权平均零售价
g_sale["avg_retail_price"] = (g_sale["revenue"] / g_sale["qty_sold"]).round(4)

g_ret = ret.groupby(["date", "item_code"]).agg(
    qty_returned=("qty", lambda s: -s.sum()),  # 转正
    return_amount=("amount", lambda s: -s.sum()),
).reset_index()

daily_item = g_sale.merge(g_ret, on=["date", "item_code"], how="left")
daily_item[["qty_returned", "return_amount"]] = daily_item[
    ["qty_returned", "return_amount"]].fillna(0.0)
daily_item["qty_net"] = daily_item["qty_sold"] - daily_item["qty_returned"]
daily_item["revenue_net"] = daily_item["revenue"] - daily_item["return_amount"]

# 合并品类、单品名、损耗率、批发价
daily_item = daily_item.merge(
    item_master[["item_code", "item_name", "cat_code", "cat_name", "loss_rate"]],
    on="item_code", how="left")
daily_item = daily_item.merge(a3, on=["date", "item_code"], how="left")

# 成本加成定价率 = (零售价 - 批发价) / 批发价
daily_item["markup_rate"] = (
    (daily_item["avg_retail_price"] - daily_item["wholesale_price"])
    / daily_item["wholesale_price"]).round(4)
# 单位毛利（考虑损耗）: 售价 - 批发价/(1-损耗率)
eff_cost = daily_item["wholesale_price"] / (1 - daily_item["loss_rate"])
daily_item["unit_gross_profit"] = (daily_item["avg_retail_price"] - eff_cost).round(4)

daily_item = daily_item[[
    "date", "item_code", "item_name", "cat_code", "cat_name",
    "qty_sold", "qty_returned", "qty_net", "n_txn", "discount_qty",
    "revenue", "return_amount", "revenue_net",
    "avg_retail_price", "wholesale_price", "loss_rate",
    "markup_rate", "unit_gross_profit",
]].sort_values(["date", "item_code"]).reset_index(drop=True)

# ----------------------------------------------------------------------------
# 5. 品类×日 聚合
# ----------------------------------------------------------------------------
daily_category = daily_item.groupby(["date", "cat_code", "cat_name"]).agg(
    qty_net=("qty_net", "sum"),
    qty_sold=("qty_sold", "sum"),
    qty_returned=("qty_returned", "sum"),
    revenue_net=("revenue_net", "sum"),
    n_items=("item_code", "nunique"),
).reset_index()
# 品类销量加权平均零售价、加权平均批发价
wp = daily_item.dropna(subset=["wholesale_price"]).copy()
wp["w"] = wp["qty_sold"]
cat_w = wp.groupby(["date", "cat_code"]).apply(
    lambda d: pd.Series({
        "avg_retail_price": np.average(d["avg_retail_price"], weights=d["w"])
        if d["w"].sum() > 0 else np.nan,
        "avg_wholesale_price": np.average(d["wholesale_price"], weights=d["w"])
        if d["w"].sum() > 0 else np.nan,
    }), include_groups=False).reset_index()
daily_category = daily_category.merge(cat_w, on=["date", "cat_code"], how="left")
daily_category["avg_retail_price"] = daily_category["avg_retail_price"].round(4)
daily_category["avg_wholesale_price"] = daily_category["avg_wholesale_price"].round(4)
daily_category = daily_category.sort_values(["date", "cat_code"]).reset_index(drop=True)

# ----------------------------------------------------------------------------
# 6. 输出
# ----------------------------------------------------------------------------
item_master.to_csv(OUT / "item_master.csv", index=False, encoding="utf-8-sig")
daily_item.to_csv(OUT / "daily_item.csv", index=False, encoding="utf-8-sig")
daily_category.to_csv(OUT / "daily_category.csv", index=False, encoding="utf-8-sig")
tx.to_parquet(OUT / "cleaned_transactions.parquet", index=False)

with pd.ExcelWriter(OUT / "processed_data.xlsx", engine="openpyxl") as xw:
    item_master.to_excel(xw, sheet_name="item_master", index=False)
    daily_category.to_excel(xw, sheet_name="daily_category", index=False)
    daily_item.head(2000).to_excel(xw, sheet_name="daily_item_sample", index=False)

# ----------------------------------------------------------------------------
# 7. 预处理质量报告
# ----------------------------------------------------------------------------
report = []
report.append("=== 数据预处理质量报告 ===")
report.append(f"附件1 单品数: {len(a1)}  品类数: {a1.cat_name.nunique()}")
report.append(f"  品类分布: {a1.cat_name.value_counts().to_dict()}")
report.append(f"附件2 原始流水: {len(a2)} 条  时间跨度: {a2.date.min().date()} ~ {a2.date.max().date()}")
report.append(f"  销售记录: {(tx.is_return==0).sum()}  退货记录: {tx.is_return.sum()}")
report.append(f"  打折记录: {tx.is_discount.sum()}")
report.append(f"  销量异常(>Q3+3IQR={upper:.2f}kg)标记: {tx.qty_outlier.sum()}")
report.append(f"  非退货非正销量标记: {tx.nonpos_nonreturn.sum()}")
report.append(f"  参与销售的单品数: {sale.item_code.nunique()} / {len(a1)} (未售出 {len(a1)-sale.item_code.nunique()})")
report.append(f"附件3 批发价记录: {len(a3)}")
report.append(f"  daily_item 中缺失批发价的行: {daily_item.wholesale_price.isna().sum()} / {len(daily_item)}")
report.append(f"附件4 损耗率: {len(a4)} 个单品, 范围 {a4.loss_rate_pct.min():.2f}% ~ {a4.loss_rate_pct.max():.2f}%")
report.append(f"输出: daily_item {daily_item.shape}, daily_category {daily_category.shape}, cleaned_transactions {tx.shape}")
report_txt = "\n".join(report)
print(report_txt)
(OUT / "preprocess_report.txt").write_text(report_txt, encoding="utf-8")
