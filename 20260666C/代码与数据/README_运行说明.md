# 代码与数据 — 运行说明

本文件夹包含论文《蔬菜类商品的自动定价与补货决策》的全部核心源代码、数据文件及运行说明。

## 一、目录结构

```
代码与数据/
├── 1.xlsx 2.xlsx 3.xlsx 4.xlsx     原始附件（附件1商品信息 / 附件2销售流水 / 附件3批发价 / 附件4损耗率）
├── preprocess.py                    数据预处理（读取附件 1–4，输出 processed/）
├── processed/                       预处理后数据（已随附，可直接用于建模脚本）
│   ├── item_master.csv              单品主表（251 单品）
│   ├── daily_item.csv               单品 × 日明细
│   ├── daily_category.csv           品类 × 日明细
│   ├── cleaned_transactions.parquet 清洗后逐笔流水（878,503 条，零删除）
│   ├── processed_data.xlsx          汇总工作簿（便于直接查看）
│   └── preprocess_report.txt        数据质量报告
└── paper/                           各问题求解脚本
    ├── common.py                    统一数据加载、品类日销量矩阵、绘图风格与关键参数（含价格弹性）
    ├── build_part1.py               问题一：STL 分解 / MODWT(db4) 小波 / Student-t Copula / Granger 因果与互信息
    ├── build_item.py                问题一：单品级销量分布（长尾/帕累托）与单品间相互关系
    ├── build_part2.py               问题二：2SLS 价格弹性 / SARIMAX 预测 / 报童联合定价-库存优化 / DRO
    ├── build_part3.py               问题三：MILP 品种选择-补货-定价优化（PuLP+CBC）与灵敏度分析
    ├── build_part4.py               问题四：EVPI/EVSI 信息价值分析与 SARIMAX 残差诊断
    ├── build_diagrams.py            全篇思路框架图等示意图
    └── stats.json                   各脚本共享的统计结果缓存（由 build_part1.py 等生成/更新）
```

## 二、运行环境

Python 3.11，主要依赖：

```
pip install pandas numpy scipy matplotlib scikit-learn statsmodels openpyxl pulp pywavelets
```

其中 `statsmodels` 用于 SARIMAX 预测与 Granger 因果检验，`pulp` 用于 MILP 求解（调用 CBC），
`pywavelets` 用于 MODWT 小波多分辨率分析。

## 三、运行顺序

| 顺序 | 脚本 | 生成内容 |
|---|---|---|
| 0 | `python preprocess.py`（在本文件夹下执行） | 读取附件 1–4，生成 `processed/` 数据 |
| 1 | `python paper/build_part1.py` | 问题一多尺度依赖分析（STL/小波/Copula/Granger） |
| 2 | `python paper/build_item.py` | 问题一单品级分布与相互关系分析 |
| 3 | `python paper/build_part2.py` | 问题二定价-补货联合优化 |
| 4 | `python paper/build_part3.py` | 问题三 MILP 品种选择 |
| 5 | `python paper/build_part4.py` | 问题四 EVPI/EVSI 与残差诊断 |
| 6 | `python paper/build_diagrams.py` | 全篇思路框架图 |

说明：
1. `processed/` 数据已随附，若仅需复现图表可跳过第 0 步，直接从 `paper/` 下运行建模脚本。
2. 所有图表输出至 `paper/figures/` 目录，数值结果写入 `paper/stats.json`。
3. 各脚本均固定随机种子（`np.random.seed(42)`），可完整复现论文中的数值结果与图形。
4. 路径基于相对结构自动解析（`preprocess.py` 与附件、`processed/` 同级；`paper/` 脚本读取上级 `processed/`），请保持上述目录结构不变。
