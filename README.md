# Paper Trading Agent — 双引擎量化模拟交易系统

## 目录

- [一、系统概述](#一系统概述)
- [二、LLM 多智能体交易引擎](#二llm-多智能体交易引擎)
- [三、24 因子规则化回测引擎](#三24-因子规则化回测引擎)
- [四、止盈止损](#四止盈止损)
- [五、多策略并行交易](#五多策略并行交易)
- [六、自动选取最优策略](#六自动选取最优策略)
- [七、自然语言意图识别](#七自然语言意图识别)
- [八、双引擎对比](#八双引擎对比)
- [九、技术栈](#九技术栈)
- [十、默认投资组合](#十默认投资组合)
- [十一、使用方法](#十一使用方法)

---

## 一、系统概述

Paper Trading Agent 是一个 **LLM + 规则双引擎量化模拟交易系统**，支持美股多策略交易与历史回测。系统由两个核心 Python 模块构成：

| 模块 | 职责 |
|------|------|
| **Paper_Trading_Agent.py** | LLM 驱动的多智能体交易引擎，集成 Azure OpenAI GPT-4o 进行市场分析决策，通过 Alpaca Paper API 执行交易，支持止盈止损、多策略并行、自动优选策略和自然语言意图识别 |
| **BacktestEngine.py** | 纯规则化回测引擎，基于 24 个技术因子做确定性策略回测，无需 LLM，输出 12 项完整绩效指标 |

### 系统架构

```
┌──────────────────────────────────────────────────────┐
│               Paper_Trading_Agent.py                 │
│                                                      │
│  DataAgent -> MacroAgent -> StockAnalystAgent        │
│                                |                     │
│              RiskManagerAgent <- Backtest Results    │
│                       |                              │
│              FrequencyDecider                        │
│                       |                              │
│              AlpacaExecutor <- Alpaca Paper API      │
│                                                      │
├──────────────────────────────────────────────────────┤
│                BacktestEngine.py                     │
│                                                      │
│  FactorLibrary -> BacktestRunner -> EvaluationMetrics│
│  (24 factors)   (deterministic)    (12 metrics)      │
└──────────────────────────────────────────────────────┘
```

---

## 二、LLM 多智能体交易引擎

### 分层智能体架构

| 层级 | 智能体 | 职责 |
|------|--------|------|
| 数据层 | **DataAgent** | 通过 yfinance 拉取美股行情及 SPY/VIX，计算 15 项技术指标（SMA/EMA/RSI/MACD/布林带/ATR/随机指标/波动率/动量/回撤/量价趋势） |
| 宏观层 | **MacroAgent** | 分析 SPY 收益率和 VIX 指数，通过 GPT-4o 判断市场阶段（Bull/Bear/Ranging/Panic），输出 0-10 风险偏好评分 |
| 个股层 | **StockAnalystAgent** | 结合技术面数据、宏观环境和回测上下文，对每只股票生成 BUY/SELL/HOLD 信号及置信度 |
| 风控层 | **RiskManagerAgent** | 组合层面风险评估，计算 VaR、最大回撤、波动率，结合回测绩效输出 Low/Medium/High 风险等级及调仓建议 |
| 频率层 | **FrequencyDecider** | LLM 根据 VIX、市场阶段、实现波动率和趋势强度，动态推荐最优调仓频率 |
| 执行层 | **AlpacaExecutor** | 对接 Alpaca Paper Trading API，执行下单、Bracket Order 止盈止损、账户查询和持仓管理 |

### 调仓执行顺序

为确保买入时有充足的购买力（buying power），`rebalance()` 方法采用三阶段执行：

1. **阶段一**：将调仓计划拆分为卖出列表和买入列表
2. **阶段二**：先提交所有卖单（取消对应标的的挂单后以市价卖出）
3. **阶段三**：等待 3 秒让卖单成交，重新查询账户购买力，买入按金额从大到小排序，逐个检查预算，超出则缩量或跳过

### LLM 自动回测触发

`decide_backtest()` 通过 LLM 动态判断是否需要触发回测：

- VIX > 30 或市场处于 Panic/Bear 阶段 → **强烈触发**
- 风险等级 High 且上次回测超过 30 天 → **触发**
- Bull 市场 + 低波动 + 近期已回测 → **跳过**

### 调仓频率动态决策

`decide_rebalance_frequency()` 根据实时市场状态推荐调仓频率：

| 市场状态 | 推荐频率 |
|---------|---------|
| VIX > 35 或 Panic | daily（每日） |
| VIX > 25 或 Bear | weekly（每周） |
| VIX 15–25 或 Ranging | weekly / biweekly |
| VIX < 15 且 Bull 且强趋势 | monthly（每月） |

---

## 三、24 因子规则化回测引擎

### 因子库（5 大类，24 因子）

| 类别 | 因子 | 数量 |
|------|------|------|
| **趋势** | SMA 20/50 交叉、SMA 50/200 交叉（金叉/死叉）、EMA 12/26 交叉、ADX 14 | 4 |
| **动量** | RSI 14、MACD 信号线、MACD 柱、动量 20d/60d/120d、随机指标 14、CCI 20、Williams %R 14 | 7 |
| **波动率** | 布林带位置、布林带挤压、波动率 20d、波动率区间、ATR 14 | 5 |
| **成交量** | 量比、量价趋势、OBV 趋势 | 3 |
| **风险/其他** | Beta（对标 SPY）、5 日反转、60 日回撤 | 3 |

每个因子输出标准化到 [-1, 1] 的信号值，正值看多，负值看空。

### 策略配置

每套策略通过 `strategy_config.json` 管理，每个策略包含：

| 配置项 | 说明 | 可选值 |
|--------|------|--------|
| `name` | 策略名称 | 自定义字符串 |
| `start_date` / `end_date` | 回测区间 | YYYY-MM-DD |
| `tickers` | 交易标的列表 | 32 只股票池任意子集 |
| `initial_capital` | 初始资金 | 任意正数 |
| `factors` | 因子选择及参数 | 权重、方向、多空阈值 |
| `signal_logic` | 信号合成逻辑 | weighted_sum / majority_vote / top_n |
| `rebalance_frequency` | 调仓频率 | daily / weekly / biweekly / monthly |
| `position_sizing` | 仓位分配方式 | equal_weight / factor_score / risk_parity |
| `max_positions` | 最大持仓数 | 整数 |

### 回测流程

```
加载策略配置 → 拉取历史数据 → 一次性计算全部因子全时段信号矩阵
→ 按调仓频率循环 → 调仓日计算目标权重并记录交易 → 每日计算组合收益
→ 评估 12 项绩效指标 → 单策略详细报告 + 多策略横向对比表
```

### 绩效评估（12 项指标）

| 指标 | 说明 |
|------|------|
| Total Return | 总收益率 |
| Annualized Return | 年化收益率 |
| Sharpe Ratio | 夏普比率（风险调整收益） |
| Calmar Ratio | 卡尔玛比率（收益 / 最大回撤） |
| Sortino Ratio | 索提诺比率（仅惩罚下行波动） |
| Max Drawdown | 最大回撤 |
| Volatility (ann.) | 年化波动率 |
| Beta | 相对 SPY 的 Beta 系数 |
| Win Rate | 日胜率（正收益天数占比） |
| Profit/Loss Ratio | 盈亏比（平均盈利 / 平均亏损） |
| Profit Factor | 盈利因子（总盈利 / 总亏损） |
| VaR / CVaR 95% | 风险价值 / 条件风险价值 |

---

## 四、止盈止损

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TAKE_PROFIT_PCT` | 15% | 盈利达成本价 15% 触发止盈 |
| `STOP_LOSS_PCT` | 8% | 亏损达成本价 8% 触发止损 |

两种使用方式：

- **Bracket Order（括号订单）**：买入时自动附带 TP 限价单 + SL 止损单，OCO 关系（一个触发则另一个自动取消）
- **事后补挂**：对已有持仓批量附加 TP/SL 订单，支持自定义百分比

`check_tpsl_status()` 实时检查每个持仓的浮动盈亏百分比，标注是否接近或触及 TP/SL 阈值。

---

## 五、多策略并行交易

`run_multi_strategies()` 支持同时运行多套策略，资金按比例分配后合并权重：

```
策略 #0 (动量型,  8因子) → 权重 [AAPL 50%, NVDA 50%]
策略 #2 (价值型,  6因子) → 权重 [JNJ  60%, PG   40%]
策略 #3 (反转型,  5因子) → 权重 [XOM  40%, BAC  60%]

资金 1:1:1 → 最终合并权重:
  AAPL 16.7%, NVDA 16.7%, JNJ 20%, PG 13.3%, XOM 13.3%, BAC 20%
```

流程：拉取所有策略涉及的全部 ticker → 每个策略独立计算因子和权重 → 按资金比例加权合并 → LLM 风控审查 → 执行调仓（可选 Bracket TP/SL）。

---

## 六、自动选取最优策略

`auto_select_and_trade()` 实现全自动"回测→排名→交易"流水线：

1. **回测全部**：对 `strategy_config.json` 中所有策略逐一回测
2. **多指标排名**：在 6 个维度上分别排名（Sharpe、Calmar、Sortino、Profit Factor、Max Drawdown、Win Rate），计算平均排名
3. **选取最优**：平均排名最低的策略胜出
4. **模拟交易**：使用最优策略的因子配置和标的池执行规则化交易

支持通过交互菜单 `[13]` 或命令行 `--auto-select` 触发。

---

## 七、自然语言意图识别

交互菜单 `[14]` 支持用自然语言（中文或英文）描述需求，系统通过 GPT-4o 自动匹配到已有功能：

| 匹配结果 | 行为 |
|---------|------|
| 匹配到 1 个功能，置信度 ≥ 0.7 | 确认后直接执行 |
| 匹配到 1 个功能，置信度 < 0.7 | 提示低置信度，需用户明确确认 |
| 匹配到 2+ 个功能 | 列出所有候选，让用户进一步选择 |
| 无匹配 | 提示无法识别，展示全部已有功能列表 |

例如输入"帮我回测一下所有策略"→ 自动匹配到 `[5] Backtest All Strategies`。

---

## 八、双引擎对比

| 维度 | LLM 多智能体引擎 | 24 因子规则引擎 |
|------|-----------------|----------------|
| 决策方式 | GPT-4o 推理 | 确定性算法 |
| 因子来源 | 15 项技术指标 | 24 因子库 |
| 宏观分析 | 有（MacroAgent） | 无 |
| 风控 | 有（RiskManagerAgent） | 仅因子阈值过滤 |
| 频率决策 | LLM 动态推荐 | 策略配置固定 |
| TP/SL | Bracket Order 自动挂载 | 可选 |
| 多策略并行 | 不支持 | 支持 |
| 成本 | Azure OpenAI API 调用 | 零 API 成本 |
| 速度 | 较慢（LLM + Rate Limit） | 快速（纯计算） |

---

## 九、技术栈

| 组件 | 技术 |
|------|------|
| LLM | Azure OpenAI GPT-4o |
| 行情数据 | yfinance (Yahoo Finance) |
| 交易接口 | Alpaca Markets Paper Trading API |
| 数值计算 | NumPy / Pandas |
| 语言 | Python 3 |

---

## 十、默认投资组合

**LLM 交易引擎默认标的（10 只）**：

> AAPL, MSFT, JPM, JNJ, XOM, AMZN, NVDA, UNH, BAC, PG

**回测引擎策略池（32 只）**：

> AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, JNJ, XOM, UNH, BAC, PG, V, MA, HD, DIS, NFLX, ADBE, CRM, AMD, INTC, PFE, WMT, KO, PEP, CSCO, QCOM, TXN, AVGO, COST, ABBV

---

## 十一、使用方法

### 1. 启动交互式菜单

```bash
python Paper_Trading_Agent.py
```

无任何参数启动，进入交互式菜单。启动后显示以下界面：

```
╔══════════════════════════════════════════════════╗
║              PAPER TRADING AGENT                 ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  [1]  LLM Multi-Agent Trading (Dry Run)          ║
║  [2]  LLM Multi-Agent Trading (Real Orders)      ║
║  [3]  Strategy-Based Trading (Dry Run)           ║
║  [4]  Strategy-Based Trading (Real Orders)       ║
║  [5]  Backtest All Strategies                    ║
║  [6]  Backtest Single Strategy                   ║
║  [7]  Live Trading + Auto Backtest               ║
║  [8]  Regenerate Strategy Config                 ║
║  [9]  View Alpaca Account Summary                ║
║  [10] LLM Rebalance Frequency Analysis           ║
║  [11] Multi-Strategy Parallel Trading            ║
║  [12] Manage TP/SL (Attach + Status Check)       ║
║  [13] Auto-Select Best Strategy & Trade          ║
║  [14] Natural Language Input (AI Intent Recog.)  ║
║  [0]  Exit                                       ║
║                                                  ║
╚══════════════════════════════════════════════════╝
```

### 2. 菜单各项功能说明

| 选项 | 功能 | 说明 |
|------|------|------|
| **[1]** | LLM 多智能体交易（模拟） | 运行完整 LLM 流水线：DataAgent 拉数据 → MacroAgent 判市场 → StockAnalystAgent 出信号 → RiskManagerAgent 风控 → 计算目标权重，**不下单** |
| **[2]** | LLM 多智能体交易（实盘） | 同上，但会向 Alpaca 纸交易账户提交真实订单。需要确认后执行 |
| **[3]** | 策略规则化交易（模拟） | 从 `strategy_config.json` 中选取一个策略，用 24 因子规则引擎计算信号和权重，**不下单**。速度快、零 API 成本 |
| **[4]** | 策略规则化交易（实盘） | 同上，提交真实订单。需要确认后执行 |
| **[5]** | 回测全部策略 | 对 `strategy_config.json` 中所有策略逐一回测，打印每个策略的详细报告和横向对比表 |
| **[6]** | 回测单个策略 | 选择其中一个策略单独回测，查看详细绩效指标和交易记录 |
| **[7]** | 实时交易 + 自动回测 | 先执行 LLM 模拟交易，再由 LLM 根据市场状态自动判断是否需要触发回测 |
| **[8]** | 重新生成策略配置 | 随机生成新的策略组合并覆盖 `strategy_config.json`，可指定生成数量 |
| **[9]** | 查看账户摘要 | 显示 Alpaca 纸交易账户的现金、持仓、市值、浮动盈亏等信息 |
| **[10]** | 调仓频率分析 | 让 LLM 根据当前 VIX、市场阶段、波动率和趋势强度推荐最优调仓频率 |
| **[11]** | 多策略并行交易 | 同时运行 2+ 个策略，资金按比例分配，权重合并后统一调仓。支持自定义资金分配比例和 Bracket TP/SL |
| **[12]** | 止盈止损管理 | 查看所有持仓的浮盈状态（是否接近 TP/SL），为持仓批量附加止盈止损单 |
| **[13]** | 自动选取最优策略交易 | 回测全部策略 → 多指标排名 → 自动选最优 → 用该策略执行模拟/实盘交易 |
| **[14]** | 自然语言输入 | 用中文或英文描述需求，AI 自动识别意图并匹配到对应功能 |
| **[0]** | 退出 | 退出系统 |

### 3. 命令行参数一览

#### 模式选择（互斥）

| 参数 | 说明 |
|------|------|
| （无参数） | 启动交互式菜单 |
| `--live` | 提交真实订单到 Alpaca 纸交易账户 |
| `--dry-run` | 完整流水线分析但不提交订单（默认行为） |
| `--backtest` | 仅运行回测，不执行交易 |
| `--auto-backtest` | 先执行交易，再由 LLM 判断是否需要回测 |
| `--generate-config` | 重新生成 strategy_config.json |
| `--summary` | 查看 Alpaca 账户摘要并退出 |

#### 可选参数

| 参数 | 说明 |
|------|------|
| `--use-strategy N` | 使用策略配置中第 N 个策略（0-based），绕过 LLM，纯规则信号 |
| `--auto-select` | 回测全部策略，自动选择最优，执行模拟/实盘交易 |
| `--strategy N` | 配合 `--backtest`，只回测第 N 个策略 |
| `--strategies N` | 配合 `--generate-config`，生成 N 套策略（默认 5） |
| `--tickers A,B,C` | 自定义交易标的（逗号分隔，覆盖默认组合） |
| `--date YYYY-MM-DD` | 指定交易日（默认当天） |
| `--quiet` | 减少控制台输出 |

### 4. 命令行使用示例

#### LLM 多智能体交易

```bash
# 模拟交易（分析 + 显示计划调仓，不实际下单）
python Paper_Trading_Agent.py

# 实盘交易（分析 + 提交真实 Alpaca 纸交易订单）
python Paper_Trading_Agent.py --live

# 指定日期执行
python Paper_Trading_Agent.py --live --date 2026-05-20

# 自定义标的
python Paper_Trading_Agent.py --live --tickers AAPL,TSLA,NVDA,META,GOOGL
```

#### 规则化策略交易

```bash
# 使用策略 #2 模拟交易（不实际下单）
python Paper_Trading_Agent.py --use-strategy 2

# 使用策略 #0 实盘交易
python Paper_Trading_Agent.py --live --use-strategy 0

# 结合自定义标的和日期
python Paper_Trading_Agent.py --live --use-strategy 1 --tickers AAPL,MSFT,GOOGL --date 2026-05-20
```

#### 回测

```bash
# 回测所有策略
python Paper_Trading_Agent.py --backtest

# 回测单个策略（按索引，0-based）
python Paper_Trading_Agent.py --backtest --strategy 0

# 静默模式（减少输出）
python Paper_Trading_Agent.py --backtest --strategy 2 --quiet
```

#### 自动选取最优策略

```bash
# 回测全部 → 排名 → 选最优 → 模拟交易
python Paper_Trading_Agent.py --auto-select

# 回测全部 → 排名 → 选最优 → 实盘交易
python Paper_Trading_Agent.py --auto-select --live
```

#### 自动回测模式

```bash
# 先运行 LLM 交易，再让 LLM 自动判断是否需要回测
python Paper_Trading_Agent.py --auto-backtest

# 结合策略交易 + 自动回测
python Paper_Trading_Agent.py --auto-backtest --use-strategy 1
```

#### 策略配置管理

```bash
# 生成默认 5 套策略
python Paper_Trading_Agent.py --generate-config

# 生成 10 套策略
python Paper_Trading_Agent.py --generate-config --strategies 10
```

#### 账户查看

```bash
# 查看 Alpaca 纸交易账户摘要
python Paper_Trading_Agent.py --summary
```

### 5. 典型工作流

#### 场景一：验证策略思路

```bash
# 1. 生成策略配置
python Paper_Trading_Agent.py --generate-config --strategies 10

# 2. 回测所有策略，对比绩效
python Paper_Trading_Agent.py --backtest

# 3. 选最优策略做模拟交易验证
python Paper_Trading_Agent.py --use-strategy 3
```

#### 场景二：日常交易流程

```
1. 启动交互菜单 → 选择 [9] 查看账户状态
2. 选择 [10] 让 LLM 分析当前最优调仓频率
3. 选择 [1] 或 [3] 做模拟交易，检查信号是否合理
4. 选择 [2] 或 [4] 执行实盘交易
5. 选择 [12] 为持仓挂上止盈止损单
```

#### 场景三：一键全自动

```bash
# 回测所有策略 → 自动选最优 → 模拟交易
python Paper_Trading_Agent.py --auto-select
```

#### 场景四：多策略实盘部署

```
1. 交互菜单选择 [11]
2. 输入策略编号，如 0,1,3
3. 选择 Real Orders 实盘
4. 选择启用 Bracket TP/SL
5. 三个策略资金等比例分配，权重合并后统一调仓
```

#### 场景五：自然语言快速操作

```
1. 交互菜单选择 [14]
2. 输入 "回测一下所有策略" → 自动执行 [5]
3. 输入 "查看我的账户" → 自动执行 [9]
4. 输入 "帮我选最好的策略交易" → 自动执行 [13]
```
