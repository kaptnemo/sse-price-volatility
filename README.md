# 上证综指波动率建模：ARIMA-GARCH 分析框架

> 基于上证综合指数（sh.000001）日度收益率序列，构建完整的时间序列分析与波动率建模管线，涵盖数据采集、预处理、探索性分析、均值模型（ARIMA）和波动率模型（GARCH）全流程。

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/dependency-Poetry-cyan.svg)](https://python-poetry.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14-orange.svg)](https://www.statsmodels.org/)
[![arch](https://img.shields.io/badge/arch-8.0-green.svg)](https://arch.readthedocs.io/)

---

## 目录

- [项目背景](#项目背景)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [数据说明](#数据说明)
- [建模流程](#建模流程)
- [关键结论](#关键结论)
- [可视化成果](#可视化成果)
- [快速开始](#快速开始)
- [模块说明](#模块说明)
- [测试](#测试)
- [局限性与未来方向](#局限性与未来方向)

---

## 项目背景

A 股市场波动率建模对风险管理、期权定价和投资组合构建具有重要意义。本项目以**上证综合指数（000001.SH）**为研究对象，利用 2016—2020 年共 1,217 个交易日的日度 OHLCV 数据作为训练集，以 2021—2025 年 1,211 个交易日作为测试集，系统验证并实现以下假设：

1. **价格水平非平稳**：对数收盘价含单位根，一阶差分（对数收益率）后平稳。
2. **收益率存在波动聚集**：ARCH-LM 检验显著，适合 GARCH 族模型。
3. **ARIMA-GARCH 联合建模**：ARIMA 捕捉条件均值线性结构，GARCH 建模条件方差异方差性。

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
| 训练集 | 2016-01-05 — 2020-12-31（1,217 个交易日） |
| 测试集 | 2021-01-05 — 2025-12-31（1,211 个交易日） |
| 频率 | 日度 OHLCV |
| 衍生特征 | `log_close = ln(close)`，`log_return = ln(Pₜ / Pₜ₋₁)` |

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
  ├─ fit_arima  → ARIMA(4,1,3)       ├─ fit_garch  → GARCH(1,2)
  ├─ forecast_arima                  ├─ forecast_garch
  ├─ rolling_forecast_arima          ├─ rolling_forecast_garch
  └─ residual_diagnostics            └─ garch_diagnostics
    │                                     │
    └──────────────┬──────────────────────┘
                   ▼
             report.py  →  outputs/reports/report.md
```

### ARIMA 均值模型

- 对 `log_return` 序列进行 ADF 检验，确认平稳后搜索最优阶数。
- 以 AIC 为准则通过网格搜索或 `auto_arima` 确定 **(p, d, q)**。
- **最终阶数：ARIMA(4, 1, 3)**（AIC = −7380.13，BIC = −7339.30）。
- 残差 Ljung-Box 检验验证无显著自相关（所有滞后 p > 0.05）。
- 支持 `rolling_forecast_arima` 在测试集上进行逐步 1-step-ahead 滚动预测。

### GARCH 波动率模型

- 对 ARIMA 残差序列进行 ARCH-LM 检验，确认存在波动聚集效应。
- 网格搜索 GARCH(p, q)，以 AIC 选择最优阶数。
- **最终阶数：GARCH(1, 2)**（AIC 准则，网格搜索范围 p, q ∈ {1, 2}）。
- 标准化残差 Ljung-Box 检验验证条件方差结构已被充分捕捉。
- 支持 `rolling_forecast_garch` 在测试集上进行逐步滚动波动率预测。

---

## 关键结论

1. **价格水平含单位根**：对数收盘价 ADF 检验无法拒绝原假设，一阶差分后收益率平稳，符合随机游走假设。
2. **显著的波动聚集**：对数收益率的平方序列 Ljung-Box 检验 p < 0.05，ARCH 效应显著，GARCH 模型适用。
3. **ARIMA 残差无显著自相关**：ARIMA(4,1,3) 残差 Ljung-Box 检验所有滞后 p > 0.05，均值结构捕捉充分。
4. **GARCH(1,2) 消除条件异方差**：标准化残差平方项无显著自相关，联合模型充分刻画了条件均值与条件方差动态。
5. **短期预测局限**：模型预测反映历史统计规律，**不构成任何投资建议或交易信号**。

---

## 可视化成果

| 图表 | 说明 |
|------|------|
| `eda_close_price.png` | 上证综指收盘价走势（2016—2020 训练期） |
| `eda_return_distribution.png` | 日度对数收益率分布（直方图 + KDE + 正态拟合） |
| `eda_return_acf_pacf.png` | 收益率 ACF / PACF（确认短程自相关结构） |
| `eda_close_acf_pacf.png` | 对数收盘价 ACF / PACF（非平稳参照） |
| `eda_return_qq.png` | 收益率 Q-Q 图（验证厚尾特征） |
| `arima_insample_fit.png` | ARIMA 样本内拟合值 vs 实际对数收盘价 |
| `arima_residual_diagnostics.png` | ARIMA 残差四联诊断图 |
| `arima_forecast_vs_actual.png` | 滚动 1-step-ahead ARIMA 预测 vs 实际值 |
| `garch_volatility_clustering.png` | 收益率及其平方序列（ARCH 效应可视化） |
| `garch_conditional_volatility.png` | GARCH 条件波动率时间序列 |
| `garch_residual_diagnostics.png` | GARCH 标准化残差四联诊断图 |
| `garch_forecast_vs_realized.png` | GARCH 波动率预测 vs 已实现波动率 |
| `garch_rolling_forecast_vs_realized.png` | 滚动 GARCH 波动率预测 vs 已实现波动率 |

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
from src.preprocess import preprocess
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

- 模型假设对称波动响应，未捕捉**杠杆效应**（负收益放大波动）。
- 默认使用正态误差分布，金融收益率通常具有**厚尾**特征。
- 单变量模型，未引入宏观因子或跨市场信号。

**后续改进方向**

- [ ] **GJR-GARCH / EGARCH**：建模非对称波动率（杠杆效应）
- [ ] **Student-t / Skewed-t 分布**：更好拟合厚尾特征，AIC/BIC 对比
- [ ] **Walk-forward 回测**：滚动窗口样本外评估，更稳健的预测误差度量
- [ ] **Markov Switching GARCH**：捕捉波动率机制转换（如危机 vs 平稳期）
- [ ] **DCC-GARCH**：与港股（恒生）、美股（S&P 500）联动分析
- [ ] **自动化管线**：Prefect / Airflow 编排全流程（采集 → 预处理 → 建模 → 报告）

---

> **免责声明**：本项目纯属学术研究，所有模型输出均不构成投资建议。
