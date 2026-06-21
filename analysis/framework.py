"""Generate the overall research-framework (整体思路框架图) for the paper."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt = c.setup_plot()
fig, ax = plt.subplots(figsize=(13.5, 10))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

C_DATA = "#DCE6F1"; C_PREP = "#FCE4D6"; C_Q = "#E2EFDA"; C_M = "#FFF2CC"; C_OUT = "#F4CCCC"


def box(x, y, w, h, text, fc, fs=10, bold=False, ec="#7F7F7F"):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.4,rounding_size=1.2",
                       fc=fc, ec=ec, lw=1.3)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(x1, y1, x2, y2, color="#404040"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle="-|>", mutation_scale=16, lw=1.5, color=color,
                 connectionstyle="arc3,rad=0"))


ax.text(50, 97, "蔬菜类商品自动定价与补货决策 —— 整体研究思路框架", ha="center",
        va="center", fontsize=16, fontweight="bold")

# Layer 1: data sources
box(50, 89, 86, 6.5,
    "数据源：附件1 商品信息(6品类/251单品)  |  附件2 销售流水(2020.7–2023.6, 87.8万条)  |  "
    "附件3 批发价格  |  附件4 损耗率", C_DATA, fs=10, bold=True)
arrow(50, 85.6, 50, 82.5)

# Layer 2: preprocessing
box(50, 78.5, 86, 7,
    "数据预处理与特征构建：剔除退货/异常价  ·  单品×日聚合(销量、加权售价)  ·  "
    "关联批发成本/损耗率/品类  ·  构造加成率、利润、星期/月份特征", C_PREP, fs=10, bold=True)

# fork to four questions
for qx in [18, 39.5, 61, 82]:
    arrow(50, 75, qx, 70.5)

# Layer 3: four questions
box(18, 64, 30, 12,
    "问题1\n分布规律与关联关系\n\n· 品类/单品销量分布(长尾、对数正态)\n"
    "· 季节性与星期效应\n· Pearson/Spearman 相关\n· K-means 销售形态聚类", C_Q, fs=8.5, bold=False)
box(39.5, 64, 14, 12,
    "问题2\n品类补货\n与定价\n(以周为单位)", C_Q, fs=9, bold=True)
box(61, 64, 14, 12,
    "问题3\n单品补货\n与定价\n(7.1当日)", C_Q, fs=9, bold=True)
box(82, 64, 26, 12,
    "问题4\n数据采集建议\n\n· 期初/期末库存与报废\n· 单品进价与到货时间\n"
    "· 天气/节假日/客流\n· 竞品价与产地", C_Q, fs=8.5)

# methods layer for Q2 & Q3
arrow(39.5, 58, 39.5, 52.5)
arrow(61, 58, 61, 52.5)
box(39.5, 46, 16, 11,
    "对数需求弹性模型\nlnQ=a+e·lnP\n+月份+星期+趋势\n→ 估计价格弹性 e", C_M, fs=8.5)
box(61, 46, 16, 11,
    "0-1 整数规划(MILP)\n选27–33单品·全品类覆盖\n单品≥2.5kg 最小陈列\n目标:总利润最大", C_M, fs=8.5)

arrow(39.5, 40.5, 39.5, 35.5)
arrow(61, 40.5, 61, 35.5)
box(39.5, 30, 18, 9,
    "利润最优化\nπ=Q·(P−w/(1−L))\n在历史加成带内求最优加成率\n→ 每日补货量+定价", C_M, fs=8.5)
box(61, 30, 18, 9,
    "单品需求按弹性缩放\nQ=Q0·(P/P0)^e·周六系数\n→ 单品补货量+定价", C_M, fs=8.5)

# Q1 result arrow down
arrow(18, 58, 18, 18.5)
arrow(39.5, 25.5, 39.5, 18.5)
arrow(61, 25.5, 61, 18.5)
arrow(82, 58, 82, 18.5)

# results layer
box(18, 14, 30, 8,
    "结果1：花叶类占42%；销量右偏长尾(前20%单品占≈84%)；\n"
    "周末显著高于工作日；同类/跨类正相关，茄类相对独立", C_OUT, fs=8)
box(39.5, 14, 16, 8, "结果2：\n6品类未来一周\n日补货量与定价\n(周利润≈7013元)", C_OUT, fs=8)
box(61, 14, 16, 8, "结果3：\n7.1 选33单品\n补货量与定价\n(日利润≈806元)", C_OUT, fs=8)
box(82, 14, 26, 8,
    "结果4：明确需补采集的数据\n清单及其对预测/优化\n各环节的作用与理由", C_OUT, fs=8)

for qx in [18, 39.5, 61, 82]:
    arrow(qx, 10, 50, 5.8)
box(50, 3.2, 70, 4.5,
    "结论与决策建议：成本加成 + 需求驱动的动态补货定价体系，支撑商超日常补货与定价决策", C_DATA,
    fs=10.5, bold=True)

fig.savefig(os.path.join(c.FIG, "framework.png"), dpi=160)
fig.savefig(os.path.join(c.REPO, "paper", "framework.png"), dpi=160)
print("framework saved")
