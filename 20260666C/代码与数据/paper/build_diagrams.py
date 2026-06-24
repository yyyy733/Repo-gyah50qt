"""框架/流程类示意图：总体路线图(新增)、三阶段框架、MILP流程、供应链信息流、数据生态。"""
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import common as C
plt = C.plt


def box(ax, x, y, w, h, text, fc, ec, fs=10, tc="#1b1b1b", bold=False):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.05",
                       linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=4, wrap=True)


def arrow(ax, p1, p2, color="#566573", lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                                 lw=lw, color=color, zorder=2,
                                 shrinkA=2, shrinkB=2))


# ============================ 图0  论文总体研究路线图（新增） ============================
fig, ax = plt.subplots(figsize=(13.5, 8))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 8); ax.axis("off")
ax.text(6.75, 7.7, "蔬菜类商品自动定价与补货决策 — 总体研究路线图",
        ha="center", fontsize=15, fontweight="bold")
# 列标题
heads = [("数据基础", 1.6, "#D6EAF8"), ("四个问题", 4.4, "#FCF3CF"),
         ("方法与模型", 8.0, "#D5F5E3"), ("决策输出", 11.7, "#FADBD8")]
for t, x, c in heads:
    box(ax, x, 6.9, 2.4, 0.55, t, c, "#34495E", fs=11.5, bold=True)
# 数据
data = ["附件1 商品信息\n(251 单品/6 品类)", "附件2 销售流水\n(87.9 万条)",
        "附件3 批发价格\n(5.6 万条)", "附件4 损耗率\n(251 单品)"]
dy = [5.9, 4.6, 3.3, 2.0]
for t, y in zip(data, dy):
    box(ax, 1.6, y, 2.4, 0.95, t, "#EBF5FB", "#5499C7", fs=8.5)
# 问题
probs = ["问题一\n分布规律与\n品类关联", "问题二\n品类级补货量\n与定价(7日)",
         "问题三\n单品选择\n补货与定价", "问题四\n额外数据\n价值评估"]
py = [5.9, 4.6, 3.3, 2.0]
for t, y in zip(probs, py):
    box(ax, 4.4, y, 2.2, 0.98, t, "#FEF9E7", "#D4AC0D", fs=8.8, bold=True)
# 方法
meths = ["STL+小波+Copula\n+Granger+互信息", "2SLS 弹性 + SARIMAX\n+ 报童(KKT)+DRO",
         "MILP 品种选择\n(分段线性化+分支定界)", "EVPI / EVSI\n信息价值框架"]
my = [5.9, 4.6, 3.3, 2.0]
for t, y in zip(meths, my):
    box(ax, 8.0, y, 2.9, 0.98, t, "#E9F7EF", "#27AE60", fs=8.8)
# 输出
outs = ["品类关联网络\n与聚类结构", "6 品类 7 日\n定价-补货计划", "30 单品组合\n+补货+定价方案",
        "数据采购\n优先级建议"]
oy = [5.9, 4.6, 3.3, 2.0]
for t, y in zip(outs, oy):
    box(ax, 11.7, y, 2.4, 0.98, t, "#FDEDEC", "#CB4335", fs=8.8)
# 连接：问题→方法→输出
for y in py:
    arrow(ax, (5.5, y), (6.55, y), color="#B7950B")
    arrow(ax, (9.45, y), (10.5, y), color="#239B56")
# 数据汇入问题（汇聚箭头）
for y in dy:
    arrow(ax, (2.8, y), (3.3, 4.0 if y in (5.9, 2.0) else y), color="#7FB3D5", lw=1.2)
box(ax, 6.75, 0.7, 9.5, 0.6,
    "贯穿主线：数据驱动 → 统计建模 → 随机优化 → 决策落地与信息价值量化",
    "#F4F6F7", "#566573", fs=10, bold=True)
fig.tight_layout(); C.save(fig, "fig00_roadmap.png")

# ============================ 图8  三阶段集成框架 ============================
fig, ax = plt.subplots(figsize=(13.5, 7.2))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.2); ax.axis("off")
ax.text(6.75, 6.9, "图8  联合定价-库存优化三阶段集成框架",
        ha="center", fontsize=14, fontweight="bold")
stages = [("第一阶段\n多尺度依赖分析", 2.3, "#D6EAF8", "#2E86C1"),
          ("第二阶段\n联合定价-库存优化", 6.75, "#D5F5E3", "#229954"),
          ("第三阶段\nMILP 品种选择", 11.2, "#FDEBD0", "#CA6F1E")]
contents = [
    ["数据预处理 (附件1-4)", "STL 时序分解\n(趋势/季节/残差)", "MODWT 小波\n多分辨率分析",
     "Copula 依赖结构\n尾部依赖分析", "Granger 因果网络\n与互信息分析"],
    ["2SLS 价格弹性\n(批发价=IV)", "SARIMAX 需求预测\n(MAPE 评估)",
     "报童模型\nQ*=F^{-1}(CR)", "联合(p*,Q*)优化\nKKT 条件求解", "鲁棒优化\n(DRO 不确定集)"],
    ["可售单品筛选\n(49→30 种)", "MILP 建模\ny_j∈{0,1}", "品类覆盖约束\n种数约束[27,33]",
     "分段线性化\n+分支定界", "影子价格分析\n灵敏度分析"],
]
xs = [2.3, 6.75, 11.2]
for (title, x, fc, ec), col in zip(stages, contents):
    box(ax, x, 6.1, 3.6, 0.7, title, fc, ec, fs=11, bold=True)
    y = 5.2
    for t in col:
        box(ax, x, y, 3.3, 0.74, t, "white", ec, fs=8.6)
        y -= 0.92
arrow(ax, (4.2, 4.0), (4.95, 4.0), color="#34495E", lw=2.4)
arrow(ax, (8.6, 4.0), (9.4, 4.0), color="#34495E", lw=2.4)
box(ax, 6.75, 0.55, 12.6, 0.6,
    "最终输出：7 日品类补货计划 + 单品组合方案 + EVPI/EVSI 数据价值量化",
    "#FEF9E7", "#B7950B", fs=10.5, bold=True)
fig.tight_layout(); C.save(fig, "fig08_framework.png")

# ============================ 图14  MILP 流程图 ============================
fig, ax = plt.subplots(figsize=(7.5, 9.5))
ax.set_xlim(0, 7.5); ax.set_ylim(0, 9.5); ax.axis("off")
ax.text(3.75, 9.2, "图14  MILP 品种选择-补货-定价优化流程",
        ha="center", fontsize=12.5, fontweight="bold")
flow = [
    ("输入：49 种可售单品\n(单品需求/价格/成本/损耗率)", "#D6EAF8", "#2E86C1", 8.4),
    ("各单品报童最优:\nQ_j*=max(2.5, F^{-1}(CR_j))", "#EBF5FB", "#5499C7", 7.2),
    ("建立 MILP 目标函数\nmax Σ y_j·E[π_j(p_j,Q_j)]", "#D5F5E3", "#229954", 6.0),
    ("约束: 27≤Σy_j≤33;\n每品类≥1; 最小陈列量 2.5kg", "#E9F7EF", "#27AE60", 4.8),
    ("分段线性化 (SOS2)\n+ 分支定界求解", "#FDEBD0", "#CA6F1E", 3.6),
    ("最优品种组合\n+ 补货量 + 定价方案", "#FADBD8", "#CB4335", 2.4),
]
for t, fc, ec, y in flow:
    box(ax, 3.75, y, 5.0, 0.95, t, fc, ec, fs=9.5,
        bold=(y in (8.4, 2.4)))
for i in range(len(flow) - 1):
    arrow(ax, (3.75, flow[i][3] - 0.5), (3.75, flow[i+1][3] + 0.5),
          color="#34495E", lw=2)
fig.tight_layout(); C.save(fig, "fig14_milp_flow.png")

# ============================ 图18  供应链信息流 ============================
fig, ax = plt.subplots(figsize=(13.5, 5.2))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 5.2); ax.axis("off")
ax.text(6.75, 4.9, "图18  蔬菜供应链结构与信息流拓扑",
        ha="center", fontsize=13.5, fontweight="bold")
nodes = [("上游产区", 1.2, "#D5F5E3", "#229954"),
         ("批发市场\n(c_j 工具变量)", 3.6, "#D6EAF8", "#2E86C1"),
         ("超市补货\n与定价", 6.4, "#FCF3CF", "#B7950B"),
         ("消费者购买", 9.2, "#FADBD8", "#CB4335"),
         ("销售流水\nPOS", 11.9, "#E8DAEF", "#7D3C98")]
for t, x, fc, ec in nodes:
    box(ax, x, 3.0, 2.0, 1.0, t, fc, ec, fs=9.5, bold=True)
for i in range(len(nodes) - 1):
    arrow(ax, (nodes[i][1] + 1.0, 3.0), (nodes[i+1][1] - 1.0, 3.0),
          color="#34495E", lw=2)
# 反馈回路 + 数据增强节点
arrow(ax, (11.9, 2.5), (6.4, 1.4), color="#CB4335", lw=1.4, style="-|>")
ax.text(9.0, 1.55, "需求反馈 (SARIMAX/弹性估计)", ha="center", fontsize=9,
        color="#CB4335")
aug = [("天气预报", 3.6), ("竞品价格", 6.4), ("客流监控", 9.2)]
for t, x in aug:
    box(ax, x, 4.15, 1.7, 0.5, t, "#F9EBEA", "#E59866", fs=8.5)
    arrow(ax, (x, 3.9), (x, 3.5), color="#E59866", lw=1.2)
fig.tight_layout(); C.save(fig, "fig18_supply_chain.png")

# ============================ 图21  数据生态系统 ============================
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
ax.text(6, 6.7, "图21  智能蔬菜供应链数据生态系统 (EVSI 框架)",
        ha="center", fontsize=13.5, fontweight="bold")
sources = [("批发价格\n(已有)", 1.6, 5.3, "#D6EAF8", "#2E86C1"),
           ("门店 POS\n(已有)", 1.6, 3.8, "#D6EAF8", "#2E86C1"),
           ("天气预报\n误差↓20%", 1.6, 2.3, "#FDEBD0", "#CA6F1E"),
           ("客流监控\n误差↓15%", 1.6, 0.9, "#FDEBD0", "#CA6F1E")]
for t, x, y, fc, ec in sources:
    box(ax, x, y, 2.0, 1.0, t, fc, ec, fs=8.8)
box(ax, 5.6, 3.1, 2.8, 1.6, "统一决策支持平台\n机器学习自适应训练\n需求预测精度提升",
    "#D5F5E3", "#229954", fs=9.5, bold=True)
box(ax, 9.8, 4.2, 2.6, 1.2, "优化定价与补货决策\nEVPI=Σ损失上界", "#FADBD8", "#CB4335", fs=9)
box(ax, 9.8, 2.0, 2.6, 1.2, "数据采购优先级\n天气>客流>竞品", "#FEF9E7", "#B7950B", fs=9)
for _, x, y, _, _ in sources:
    arrow(ax, (x + 1.0, y), (4.2, 3.1), color="#7FB3D5", lw=1.3)
arrow(ax, (7.0, 3.4), (8.5, 4.2), color="#239B56", lw=1.8)
arrow(ax, (7.0, 2.8), (8.5, 2.0), color="#239B56", lw=1.8)
fig.tight_layout(); C.save(fig, "fig21_data_ecosystem.png")

print("DIAGRAMS DONE")
