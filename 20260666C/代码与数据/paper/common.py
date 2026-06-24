"""共享绘图样式与数据工具（学术期刊风格）。"""
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROC = ROOT / "processed"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# ----------------------------- 字体 -----------------------------
_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_FONT)
CJK = fm.FontProperties(fname=_FONT).get_name()
mpl.rcParams["font.sans-serif"] = [CJK, "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

# ----------------------------- 主题 -----------------------------
mpl.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#E3E6EA",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "font.size": 11,
})

# 六大品类规范名称与配色（与正文一致）
CAT = {
    1011010101: "花叶类",
    1011010504: "辣椒类",
    1011010801: "食用菌",
    1011010201: "花菜类",
    1011010402: "水生根茎类",
    1011010501: "茄类",
}
ORDER = ["花叶类", "辣椒类", "食用菌", "花菜类", "水生根茎类", "茄类"]
PALETTE = {
    "花叶类": "#2E5E8C",
    "辣椒类": "#C0392B",
    "食用菌": "#27926B",
    "花菜类": "#B07A2B",
    "水生根茎类": "#6A4C93",
    "茄类": "#8C6D4F",
}
SEQ = ["#2E5E8C", "#C0392B", "#27926B", "#B07A2B", "#6A4C93", "#8C6D4F"]


def load():
    """读取处理后的数据并附规范品类名。"""
    dc = pd.read_csv(PROC / "daily_category.csv", parse_dates=["date"])
    di = pd.read_csv(PROC / "daily_item.csv", parse_dates=["date"])
    im = pd.read_csv(PROC / "item_master.csv")
    for d in (dc, di):
        d["cat"] = d["cat_code"].map(CAT)
    im["cat"] = im["cat_code"].map(CAT)
    return dc, di, im


def cat_daily_series(dc):
    """返回品类×日的净销量宽表（按完整日期重索引、线性插值补缺）。"""
    piv = dc.pivot_table(index="date", columns="cat", values="qty_net", aggfunc="sum")
    piv = piv.reindex(columns=ORDER)
    full = pd.date_range(piv.index.min(), piv.index.max(), freq="D")
    piv = piv.reindex(full).interpolate(limit_direction="both")
    return piv


def style_ax(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def save(fig, name):
    out = FIG / name
    fig.savefig(out)
    plt.close(fig)
    print("saved", out.name)
    return out


# 与正文表格一致的模型结果（保证图文数值一致）
ELASTICITY = {  # 表7  2SLS 价格弹性（工具变量=批发价）
    "花叶类": -0.359, "辣椒类": -0.373,
    "食用菌": -1.164, "花菜类": -0.568,
    "茄类": -0.535, "水生根茎类": -1.882,
}
NEWSVENDOR = {  # 表9  (p*, Q*, profit)
    "花叶类": (19.36, 133.67, 1190.51), "辣椒类": (31.04, 72.47, 930.08),
    "花菜类": (43.50, 15.03, 222.24), "茄类": (24.93, 22.63, 196.56),
    "水生根茎类": (18.11, 10.54, 27.76), "食用菌": (14.00, 25.16, 49.96),
}
SARIMAX_ACC = {  # 表8  (MAPE %, RMSE kg)  —— 已与图10统一
    "辣椒类": (25.0, 23.4), "花叶类": (31.2, 61.8), "食用菌": (38.5, 22.7),
    "花菜类": (42.1, 11.5), "水生根茎类": (56.3, 23.1), "茄类": (74.7, 13.9),
}
EVPI = {  # 表12
    "花叶类": 88.95, "辣椒类": 66.01, "食用菌": 40.01,
    "水生根茎类": 39.48, "花菜类": 36.92, "茄类": 32.38,
}
KENDALL = {  # 表4 关键品类对
    ("花叶类", "辣椒类"): 0.4659, ("花叶类", "食用菌"): 0.4345,
    ("辣椒类", "食用菌"): 0.3897, ("花菜类", "辣椒类"): 0.3521,
    ("水生根茎类", "花叶类"): 0.3012, ("花菜类", "水生根茎类"): 0.2814,
    ("茄类", "食用菌"): -0.0556,
}
