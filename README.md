# 上证综指波动率建模：ARIMA-GARCH 分析框架

> 基于上证综合指数（sh.000001）日度价格序列，构建完整的时间序列分析与波动率建模管线，涵盖数据采集、预处理、探索性分析、ARIMA 均值模型（对数收盘价）和 GARCH 波动率模型（对数收益率）全流程。

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/dependency-Poetry-cyan.svg)](https://python-poetry.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14-orange.svg)](https://www.statsmodels.org/)
[![arch](https://img.shields.io/badge/arch-8.0-green.svg)](https://arch.readthedocs.io/)

---

## 目录

- [上证综指波动率建模：ARIMA-GARCH 分析框架](#上证综指波动率建模arima-garch-分析框架)
  - [目录](#目录)
  - [项目背景](#项目背景)
  - [技术栈](#技术栈)
  - [项目结构](#项目结构)
  - [数据说明](#数据说明)
  - [建模流程](#建模流程)
    - [ARIMA 均值模型](#arima-均值模型)
    - [GARCH 波动率模型](#garch-波动率模型)
  - [关键结论](#关键结论)
  - [可视化成果](#可视化成果)
  - [快速开始](#快速开始)
    - [1. 克隆项目并安装依赖](#1-克隆项目并安装依赖)
    - [2. 采集数据（可选，已提供原始 CSV）](#2-采集数据可选已提供原始-csv)
    - [3. 运行完整分析（推荐按顺序执行 Notebook）](#3-运行完整分析推荐按顺序执行-notebook)
    - [4. 快速调用核心模块示例](#4-快速调用核心模块示例)
  - [模块说明](#模块说明)
  - [测试](#测试)
  - [局限性与未来方向](#局限性与未来方向)

---

## 项目背景

A 股市场波动率建模对风险管理、期权定价和投资组合构建具有重要意义。本项目以**上证综合指数（000001.SH）**为研究对象，以 2000—2024 年完整历史（5,932 个交易日）做 EDA，以 2016—2020 年（1,217 天）为训练集、2021—2025 年（1,211 天）为测试集，系统验证并实现以下假设：

1. **价格水平非平稳**：对数收盘价含单位根（ADF p = 0.15），一阶差分（对数收益率）后平稳（ADF p < 1e-29）。
2. **收益率存在波动聚集**：收益率平方序列 Ljung-Box 检验所有滞后 p < 0.05，ARCH 效应显著。
3. **ARIMA 建模对数收盘价**：d=1 差分保证平稳，模型捕捉条件均值线性结构。
4. **GARCH 建模对数收益率**：对条件异方差进行参数化，量化波动率持久性。

---

## 技术栈

| 类别 | 工具 |
|------|------|
| 语言 | Python 3.13 |
| 依赖管理 | Poetry |
| 数据采集 | [Baostock](http://baostock.com/) |
| 数据处理 | pandas, numpy |
| 统计检验 | statsmodels, scipy |
| ARIMA | statsmodels, pmdarima (auto_arima) |
| GARCH | arch |
| 可视化 | matplotlib |
| 数据持久化 | MongoDB（可选），CSV |
| 测试 | pytest |
| 分析环境 | Jupyter Notebook |

---

## 项目结构

```
sse_price_volatility/
├── data/
│   ├── raw/                          # Baostock 下载的原始 CSV
│   └── processed/                    # 预处理后的数据（含 log_return）
├── notebooks/
│   ├── 01_eda.ipynb                  # 探索性数据分析
│   ├── 02_arima.ipynb                # ARIMA 均值模型
│   └── 03_garch.ipynb                # GARCH 波动率模型
├── outputs/
│   ├── figures/                      # 所有可视化图表
│   └── reports/                      # 分析报告 Markdown
├── src/
│   ├── data_loader.py                # 原始 CSV 加载
│   ├── preprocess.py                 # 清洗、特征工程全管线
│   ├── eda.py                        # 统计检验与绘图工具
│   ├── arima_model.py                # ARIMA 建模（搜索/拟合/预测/诊断）
│   ├── garch_model.py                # GARCH 建模（搜索/拟合/预测/诊断）
│   ├── baostock_helper.py            # Baostock 数据拉取封装
│   ├── sh_index_ingest.py            # 数据摄取脚本
│   ├── report.py                     # Markdown 报告生成器
│   └── plot_config.py                # 全局 matplotlib 中文字体配置
├── tests/
│   ├── test_preprocess.py
│   └── test_arima_model.py
├── pyproject.toml
└── README.md
```

---

## 数据说明

| 字段 | 值 |
|------|----|
| 标的 | 上证综合指数 `sh.000001` |
| 数据源 | Baostock（免费 A 股历史数据 API） |
| EDA 数据集 | 2000-01-05 — 2024-06-28（5,932 个交易日，完整历史） |
| 训练集 | 2016-01-05 — 2020-12-31（1,217 个交易日） |
| 测试集 | 2021-01-05 — 2025-12-31（1,211 个交易日） |
| 频率 | 日度 OHLCV |
| 衍生特征 | `log_close = ln(close)`，`log_return = ln(Pₜ / Pₜ₋₁)` |

> EDA（`01_eda.ipynb`）使用完整 2000—2024 历史刻画长期统计特性；ARIMA 与 GARCH 建模使用 2016—2020 训练集，以反映近期市场机制并控制计算量。

原始数据保存于 `data/raw/`，经预处理管线输出至 `data/processed/`，全程**不覆盖原始文件**。

---

## 建模流程

```
原始 CSV
    │
    ▼
preprocess.py
  ├─ validate_schema    → 断言必要列存在
  ├─ clean_dataframe    → 日期解析 / 去重 / 填充 / 排序
  ├─ add_features       → log_close, log_return
    │
    ▼
eda.py
  ├─ summary_stats      → 描述统计（偏度、峰度）
  ├─ check_stationarity → ADF 单位根检验
  ├─ ljung_box_test     → 自相关 / ARCH 效应检验
  └─ 可视化             → 价格图、收益率分布、ACF/PACF、Q-Q 图
    │
    ▼
arima_model.py                     garch_model.py
  ├─ prepare_series                  ├─ prepare_returns
  ├─ search_order (AIC/BIC)          ├─ search_garch_order (AIC/BIC)
  ├─ fit_arima  → ARIMA(3,1,4)       ├─ fit_garch  → GARCH(1,2)
  ├─ forecast_arima                  ├─ forecast_garch
  ├─ rolling_forecast_arima          ├─ rolling_forecast_garch
  └─ residual_diagnostics            └─ garch_diagnostics
    │                                     │
    └──────────────┬──────────────────────┘
                   ▼
             report.py  →  outputs/reports/report.md
```

### ARIMA 均值模型

- 对 **`log_close`** 序列进行 ADF 检验（p = 0.15，非平稳），`prepare_series` 自动确定 d=1。
- 以 AIC 为准则网格搜索 p ∈ {0…4}，q ∈ {0…4}，共 25 个候选模型。
- **最终阶数：ARIMA(3, 1, 4)**（AIC = −7374.61，BIC = −7333.78；`choose_best_order` 排除未收敛的 ARIMA(4,1,3)，选取 AIC 最优且已收敛的阶次）。
- 残差 Ljung-Box 检验：所有 20 个滞后 p > 0.05，残差近似白噪声 ✓。
- 滚动预测：`rolling_forecast_arima` 以 `refit_every=5` 在测试集逐步重新拟合，1-step-ahead 预测。
- **样本外评估（对数收盘价，2021—2025）**：MAE = 0.0070，RMSE = 0.0100，MAPE = 0.087%。

### GARCH 波动率模型

- 对 **`log_return`** 序列进行 ARCH-LM 检验（收益率平方 LB 所有 10 滞后 p < 0.05）✓，确认波动聚集。
- 网格搜索 GARCH(p, q)，p, q ∈ {1, 2}（AIC 准则）。
- **最终阶数：GARCH(1, 2)**（AIC 最优）。
- **波动率持久性 α + β = 0.9898**，近积分 GARCH，波动率冲击消散极缓。
- 诊断：标准化残差平方 Ljung-Box 在滞后 1–3 仍显著（p < 0.05），提示模型未完全消除短期 ARCH 效应，建议后续考虑 GJR-GARCH 或 Student-t 分布。
- 滚动预测：`rolling_forecast_garch` 在测试集上进行逐步滚动波动率预测。

---

## 关键结论

1. **价格水平含单位根**：对数收盘价 ADF p = 0.15（非平稳），对数收益率 ADF p < 1e-29（平稳），一阶差分足以消除单位根。
2. **显著的波动聚集**：收益率平方序列 Ljung-Box 检验 10 个滞后全部 p < 0.05，ARCH 效应高度显著，GARCH 建模有充分依据。
3. **ARIMA(3,1,4) 均值模型**：对数收盘价建模，AIC = −7374.61；残差 Ljung-Box 所有 20 滞后 p > 0.05（白噪声 ✓）；样本外 MAE = 0.0070，RMSE = 0.0100，MAPE = 0.087%。
4. **GARCH(1,2) 基准波动率模型**：持久性 α+β = 0.9898，近积分 GARCH；标准化残差平方在滞后 2+ 仍显著（p < 0.05），对称正态 GARCH 未能完全消除 ARCH 效应 ⚠️
5. **GJR-GARCH(1,2) 显著改善拟合**：ΔAIC = −205（−5697 vs −5491）。杠杆系数 γ = 0.018 > 0，确认负向冲击对波动率的放大效果更强（**杠杆效应**）；持久性降至 0.947。残差 ARCH 效应在短滞后仍部分残留，后续可尝试 EGARCH 或更高阶规格。
6. **样本外定量评估**：在 1,211 个测试交易日上，GARCH vs GJR-GARCH 的 RMSE（vs |r|）分别为 0.0075 / 0.0076，MAE 为 0.0058 / 0.0058，QLIKE 损失为 **−8.348** / −8.337（越小越好）。GARCH(1,2) 在 RMSE 和 QLIKE 指标上略优，两者差异较小。
7. **VaR 回测：Kupiec + Christoffersen 检验全部通过**：GARCH 1% VaR 穿越率 = 0.99%（12 次 / 1,211 天，Kupiec p = 0.97），GJR-GARCH 1% VaR 穿越率 = 0.91%（11 次，p = 0.74）；GARCH 5% 穿越率 = 5.45%（66 次，p = 0.48），GJR-GARCH 5% 穿越率 = 5.53%（67 次，p = 0.40）。进一步通过 Christoffersen（1998）条件覆盖检验：GARCH 1% LR_ind = 2.63（p_ind = 0.10），LR_cc = 2.63（p_cc = 0.27）；GJR-GARCH 1% LR_cc = 0.31（p_cc = 0.86）——独立性与条件覆盖原假设均未拒绝，两模型 VaR 违约不存在时序聚集，校准质量良好。
8. **短期预测局限**：模型预测反映历史统计规律，**不构成任何投资建议或交易信号**。

---

## 可视化成果

| 图表 | 说明 |
|------|------|
| `eda_close_price.png` | 上证综指收盘价走势（2000—2024 完整历史） |
| `eda_return_distribution.png` | 日度对数收益率分布（直方图 + KDE + 正态拟合） |
| `eda_return_acf_pacf.png` | 收益率 ACF / PACF（确认短程自相关结构） |
| `eda_close_acf_pacf.png` | 对数收盘价 ACF / PACF（非平稳参照） |
| `eda_return_qq.png` | 收益率 Q-Q 图（验证厚尾特征） |
| `arima_insample_fit.png` | ARIMA 样本内拟合值 vs 实际对数收盘价 |
| `arima_residual_diagnostics.png` | ARIMA 残差四联诊断图 |
| `arima_forecast_vs_actual.png` | 滚动 1-step-ahead ARIMA 预测 vs 实际值 |
| `garch_volatility_clustering.png` | 收益率及其平方序列（ARCH 效应可视化） |
| `garch_conditional_volatility.png` | GARCH 条件波动率时间序列 |
| `garch_residual_diagnostics.png` | GARCH(1,2) 标准化残差四联诊断图 |
| `garch_forecast_vs_realized.png` | GARCH 波动率预测 vs 已实现波动率 |
| `garch_rolling_forecast_vs_realized.png` | 滚动 GARCH 波动率预测 vs 已实现波动率 |
| `gjr_garch_residual_diagnostics.png` | GJR-GARCH(1,2) 标准化残差四联诊断图 |
| `garch_vs_gjr_conditional_volatility.png` | GARCH vs GJR-GARCH 条件波动率对比 |
| `gjr_garch_rolling_forecast_vs_realized.png` | GJR-GARCH 滚动预测 vs 已实现波动率（3 面板对比）|
| `vol_forecast_scatter.png` | 预测波动率 vs \|收益率\|代理散点图（GARCH / GJR-GARCH）|
| `var_backtest.png` | 1% / 5% VaR 穿越时间线图（GARCH 与 GJR-GARCH 对比）|

所有图表保存于 `outputs/figures/`。

---

## 快速开始

### 1. 克隆项目并安装依赖

```bash
git clone <repo-url>
cd sse_price_volatility
poetry install
```

### 2. 采集数据（可选，已提供原始 CSV）

```bash
poetry run python src/sh_index_ingest.py
```

### 3. 运行完整分析（推荐按顺序执行 Notebook）

```bash
poetry run jupyter notebook
# 依次运行 notebooks/01_eda.ipynb → 02_arima.ipynb → 03_garch.ipynb
```

### 4. 快速调用核心模块示例

```python
from src.data_loader import load_train
from src.preprocess ikmport preprocess
from src.arima_model import search_order, fit_arima, forecast_arima
from src.garch_model import prepare_returns, fit_garch, forecast_garch

# 加载并预处理
raw = load_train()
df = preprocess(raw, save_path="data/processed/train_processed.csv")

# ARIMA 均值模型（实验最优阶次 (4,1,3)，AIC=-7380.13）
order, aic = search_order(df["log_close"], d=1, p_range=(0, 4), q_range=(0, 4))
result = fit_arima(df["log_close"], order=order)
fc = forecast_arima(result, steps=10)

# GARCH 波动率模型（实验最优阶次 (1,2)）
returns = prepare_returns(df["log_return"])
garch_res = fit_garch(returns, p=1, q=2)
vol_fc = forecast_garch(garch_res, steps=10)
```

---

## 模块说明

| 模块 | 职责 |
|------|------|
| `data_loader.py` | 按股票代码和日期范围加载原始 CSV，返回 `pd.DataFrame` |
| `preprocess.py` | 数据校验、清洗、特征工程（`log_close`、`log_return`）、持久化 |
| `eda.py` | 描述统计、ADF 检验、Ljung-Box 检验、时序图/分布图/ACF/Q-Q 图 |
| `arima_model.py` | 平稳性检查、阶数搜索（AIC/BIC）、模型拟合、预测、残差诊断 |
| `garch_model.py` | ARCH 效应检验、阶数搜索、模型拟合、方差预测、标准化残差诊断 |
| `evaluate.py` | 点预测指标（MAE/RMSE/MAPE）、波动率指标（RMSE/QLIKE）、VaR 回测（Kupiec POF + Christoffersen CC）|
| `baostock_helper.py` | Baostock 登录/登出及历史行情下载封装 |
| `sh_index_ingest.py` | 端到端数据摄取脚本（调用 helper → 落盘 CSV） |
| `report.py` | 汇总模型结果，生成 Markdown 分析报告 |
| `plot_config.py` | 统一配置 matplotlib 中文字体与风格 |

---

## 测试

```bash
poetry run pytest tests/ -v
```

测试覆盖：

- `test_preprocess.py`：schema 校验、清洗逻辑、特征工程正确性、边界情况处理。
- `test_arima_model.py`：`prepare_series` 平稳化、`fit_arima` 返回值结构、预测输出维度。

---

## 局限性与未来方向

**当前局限**

- **残差 ARCH 效应未完全消除**：GARCH(1,2) 和 GJR-GARCH(1,2) 的标准化残差平方在滞后 2+ 仍显著（p < 0.05），短期波动聚集未被完全捕捉。
- **GJR-GARCH 杠杆项 t 统计量偏低**：γ 的 t-stat = 0.145，统计显著性不足，杠杆效应的经济意义需结合 AIC 改善量（ΔAIC = 205）综合判断。
- 默认使用 Student-t 误差分布，GARCH 标准化残差峰度 > 5，**厚尾**特征仍明显。
- 训练窗口 2016—2020 仅覆盖一个主要机制，测试期结构性变化可能降低样本外泛化能力。
- 单变量模型，未引入宏观因子或跨市场信号。

**后续改进方向**

- [ ] **EGARCH**：对数方差方程天然捕捉非对称性，且无需非负参数约束
- [ ] **GJR-GARCH(2,2)**：提高阶次以消除短滞后残余 ARCH 效应
- [ ] **Skewed-t 分布**：进一步拟合偏度和峰度（kurtosis ≈ 5.9）
- [x] **VaR 回测**：Kupiec POF + Christoffersen CC 条件覆盖检验，全部通过 ✓（已完成）
- [x] **收敛稳健性验证**：`choose_best_order` 已排除未收敛的 ARIMA(4,1,3)，实际采用 ARIMA(3,1,4)（已完成）
- [ ] **Markov Switching GARCH**：捕捉波动率机制转换（如危机 vs 平稳期）
- [ ] **DCC-GARCH**：与港股（恒生）、美股（S&P 500）联动分析

---

> **免责声明**：本项目纯属学术研究，所有模型输出均不构成投资建议。
