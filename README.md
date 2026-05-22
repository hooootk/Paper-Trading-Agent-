# Paper Trading Agent

## 目录

- [一、LLM 多智能体交易引擎](#一llm-多智能体交易引擎)
- [二、24 因子规则化回测引擎](#二24-因子规则化回测引擎)
- [三、止盈止损](#三止盈止损)
- [四、多策略并行](#四多策略并行)
- [五、双引擎对比](#五双引擎对比)
- [六、交互方式](#六交互方式)
- [七、技术栈](#七技术栈)
- [八、默认投资组合](#八默认投资组合)
- [九、使用方法](#九使用方法)

---

## 概述

Paper Trading Agent 是一个 **LLM + 规则双引擎量化模拟交易系统**，支持美股多策略交易与回测。系统由两个核心模块构成：

- **Paper_Trading_Agent.py**：LLM 驱动的多智能体交易引擎，集成 Azure OpenAI GPT-4o 进行市场分析决策，通过 Alpaca Paper API 执行交易，支持止盈止损和多策略并行
- **BacktestEngine.py**：纯规则化回测引擎，基于 24 个技术因子做确定性策略回测，无需 LLM，输出 12 项完整绩效指标

---

## 一、LLM 多智能体交易引擎

### 分层智能体架构

```
DataAgent → MacroAgent → StockAnalystAgent → RiskManagerAgent → AlpacaExecutor
                                                      ↓
                                          FrequencyDecider (调仓频率建议)
```

| 层级 | 组件 | 职责 |
|------|------|------|
| 数据层 | **DataAgent** | yfinance 拉取美股行情及 SPY/VIX，计算 15 项技术指标（SMA/EMA/RSI/MACD/布林带/ATR/随机指标/波动率/动量/回撤/量价趋势） |
| 宏观层 | **MacroAgent** | 分析 SPY 和 VIX，判断市场阶段（Bull/Bear/Ranging/Panic），输出 0-10 风险偏好评分 |
| 个股层 | **StockAnalystAgent** | 结合技术面 + 宏观环境 + 回测上下文，对每只股票生成 BUY/SELL/HOLD 信号及置信度 |
| 风控层 | **RiskManagerAgent** | 组合层面风险评估，计算 VaR、最大回撤、波动率，输出 Low/Medium/High 风险等级及调仓建议 |
| 频率层 | **FrequencyDecider** | LLM 根据 VIX、市场阶段、波动率、趋势强度，动态推荐调仓频率（daily/weekly/biweekly/monthly） |
| 执行层 | **AlpacaExecutor** | 对接 Alpaca Paper Trading API，执行下单、Bracket Order 止盈止损、账户管理 |

### LLM 自动回测触发

`decide_backtest()` 通过 LLM 动态判断是否需要回测：VIX > 30 或市场 Panic/Bear 时强烈触发，Bull 低波且近期已回测则跳过。

### 调仓频率动态决策

`decide_rebalance_frequency()` 根据实时市场状态推荐调仓频率：

| 市场状态 | 推荐频率 |
|---------|---------|
| VIX > 35 或 Panic | daily（每日） |
| VIX > 25 或 Bear | weekly（每周） |
| VIX 15-25 或 Ranging | weekly 或 biweekly |
| VIX < 15 且 Bull 且强趋势 | monthly（每月） |

---

## 二、24 因子规则化回测引擎

### 因子库（5 大类，24 因子）

| 类别 | 因子 | 数量 |
|------|------|------|
| **趋势** | SMA 20/50 交叉、SMA 50/200 交叉（金叉/死叉）、EMA 12/26 交叉、ADX 14 | 4 |
| **动量** | RSI 14、MACD 信号线、MACD 柱、动量 20d/60d/120d、随机指标 14、CCI 20、Williams %R 14 | 7 |
| **波动率** | 布林带位置、布林带挤压、波动率 20d、波动率区间、ATR 14 | 5 |
| **成交量** | 量比、量价趋势、OBV 趋势 | 3 |
| **风险/其他** | Beta（对标 SPY）、5 日反转、60 日回撤 | 3 |

每个因子输出标准化到 [-1, 1] 的信号值，正向表示看多，负向表示看空。

### 策略配置

每套策略通过 JSON 配置管理：

- **因子选择**：从 24 因子池中任选子集
- **因子参数**：权重、方向（做多/反转）、多空阈值
- **信号逻辑**：加权求和 / 多数投票 / Top-N
- **仓位管理**：等权 / 因子得分加权 / 风险平价
- **调仓频率**：每日 / 每周 / 双周 / 每月
- **最大持仓数**：限制同时持有的标的数量

策略由 `StrategyConfig.generate_full_config()` 随机生成多套配置，自动写入 `strategy_config.json`。支持手动编辑或通过菜单重新生成。

### 回测流程

```
加载策略配置 → 拉取历史数据 → 一次性计算全部因子全时段信号矩阵
→ 按调仓频率循环 → 调仓日计算目标权重并记录交易 → 每日计算组合收益
→ 评估 12 项绩效指标 → 单策略详细报告 + 多策略横向对比
```

### 绩效评估（12 项指标）

| 指标 | 说明 |
|------|------|
| Total Return | 总收益率 |
| Annualized Return | 年化收益率 |
| Sharpe Ratio | 夏普比率 |
| Calmar Ratio | 卡尔玛比率（收益/最大回撤） |
| Sortino Ratio | 索提诺比率（仅下行波动） |
| Max Drawdown | 最大回撤 |
| Volatility (ann.) | 年化波动率 |
| Beta | 相对 SPY 的 Beta 系数 |
| Win Rate | 日胜率 |
| Profit/Loss Ratio | 盈亏比 |
| Profit Factor | 盈利因子（总盈利/总亏损） |
| VaR / CVaR 95% | 风险价值 / 条件风险价值 |

---

## 三、止盈止损

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TAKE_PROFIT_PCT` | 15% | 盈利达成本价 15% 触发止盈 |
| `STOP_LOSS_PCT` | 8% | 亏损达成本价 8% 触发止损 |

两种使用方式：

- **Bracket Order**：买入时自动附带 TP 限价单 + SL 止损单，OCO 关系（一个触发，另一个自动取消）
- **事后补挂**：对已有持仓批量附加 TP/SL 订单，支持自定义百分比

同时提供 `check_tpsl_status()` 实时检查每个持仓的盈亏百分比，标注是否接近或触及 TP/SL 阈值。

---

## 四、多策略并行

`run_multi_strategies()` 支持同时运行多套策略，资金按比例分配：

```
策略 #0 (动量型, 8因子) → 算权重 [AAPL 50%, NVDA 50%]
策略 #2 (价值型, 6因子) → 算权重 [JNJ 60%, PG 40%]
策略 #3 (反转型, 5因子) → 算权重 [XOM 40%, BAC 60%]

资金 1:1:1 → 最终合并权重:
  AAPL 16.7%, NVDA 16.7%, JNJ 20%, PG 13.3%, XOM 13.3%, BAC 20%
```

流程：拉取所有策略涉及的全部 ticker → 每个策略独立计算因子和权重 → 按资金比例加权合并 → LLM 风控 → 执行调仓（可选 Bracket TP/SL）。

---

## 五、双引擎对比

| 维度 | LLM 多智能体引擎 | 24 因子规则引擎 |
|------|-----------------|----------------|
| 决策方式 | GPT-4o 推理 | 确定性算法 |
| 因子来源 | 15 项技术指标 | 24 因子库 |
| 宏观分析 | 有（MacroAgent） | 无 |
| 风控 | 有（RiskManagerAgent） | 仅因子阈值过滤 |
| 频率决策 | LLM 动态推荐 | 策略配置固定 |
| TP/SL | Bracket Order 自动挂载 | 可选 |
| 多策略 | 不支持 | 支持并行 |
| 成本 | Azure OpenAI API 调用 | 零 API 成本 |
| 速度 | 较慢（LLM + Rate Limit） | 快速（纯计算） |

---

## 六、交互方式

### 命令行参数（适合脚本化）

```bash
python Paper_Trading_Agent.py --live              # LLM 实盘
python Paper_Trading_Agent.py --use-strategy 2    # 策略 2 模拟交易
python Paper_Trading_Agent.py --backtest          # 回测全部策略
python Paper_Trading_Agent.py --summary           # 查看账户
```

### 交互式菜单（无参数启动）

```
╔══════════════════════════════════════════════════╗
║              PAPER TRADING AGENT                 ║
╠══════════════════════════════════════════════════╣
║  [1]  LLM Multi-Agent Trading (Dry Run)         ║
║  [2]  LLM Multi-Agent Trading (Real Orders)     ║
║  [3]  Strategy-Based Trading (Dry Run)          ║
║  [4]  Strategy-Based Trading (Real Orders)      ║
║  [5]  Backtest All Strategies                   ║
║  [6]  Backtest Single Strategy                  ║
║  [7]  Live Trading + Auto Backtest              ║
║  [8]  Regenerate Strategy Config                ║
║  [9]  View Alpaca Account Summary               ║
║  [10] LLM Rebalance Frequency Analysis          ║
║  [11] Multi-Strategy Parallel Trading           ║
║  [12] Manage TP/SL (Attach + Status Check)      ║
║  [0]  Exit                                      ║
╚══════════════════════════════════════════════════╝
```

---

## 七、技术栈

| 组件 | 技术 |
|------|------|
| LLM | Azure OpenAI GPT-4o |
| 行情数据 | yfinance (Yahoo Finance) |
| 交易接口 | Alpaca Markets Paper Trading API |
| 数值计算 | NumPy / Pandas |
| 语言 | Python 3 |

---

## 八、默认投资组合

**Paper_Trading_Agent**：AAPL、MSFT、JPM、JNJ、XOM、AMZN、NVDA、UNH、BAC、PG

**BacktestEngine 策略池**：上述 10 只 + GOOGL、META、TSLA、V、MA、HD、DIS、NFLX、ADBE、CRM、AMD、INTC、PFE、WMT、KO、PEP、CSCO、QCOM、TXN、AVGO、COST、ABBV（共 32 只）

---

## 九、使用方法

### 1. 直接启动（交互式菜单）

```bash
python Paper_Trading_Agent.py
```

无任何参数启动，进入交互式菜单，显示 13 个功能选项。输入数字选择要执行的操作，每次执行完毕后菜单重新显示，输入 `0` 退出。

### 2. LLM 多智能体交易

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

### 3. 规则化策略交易

```bash
# 使用策略 #2 模拟交易（不实际下单）
python Paper_Trading_Agent.py --use-strategy 2

# 使用策略 #0 实盘交易
python Paper_Trading_Agent.py --live --use-strategy 0

# 结合自定义标的和日期
python Paper_Trading_Agent.py --live --use-strategy 1 --tickers AAPL,MSFT,GOOGL --date 2026-05-20
```

### 4. 回测

```bash
# 回测所有策略
python Paper_Trading_Agent.py --backtest

# 回测单个策略（按索引，0-based）
python Paper_Trading_Agent.py --backtest --strategy 0

# 静默模式（减少输出）
python Paper_Trading_Agent.py --backtest --strategy 2 --quiet
```

### 5. 自动回测模式

```bash
# 先运行 LLM 交易，再让 LLM 自动判断是否需要回测
python Paper_Trading_Agent.py --auto-backtest

# 结合策略交易 + 自动回测
python Paper_Trading_Agent.py --auto-backtest --use-strategy 1
```

### 6. 策略配置管理

```bash
# 生成默认 5 套策略
python Paper_Trading_Agent.py --generate-config

# 生成 10 套策略
python Paper_Trading_Agent.py --generate-config --strategies 10
```

### 7. 账户查看

```bash
# 查看 Alpaca 纸交易账户摘要
python Paper_Trading_Agent.py --summary
```

### 8. 全部命令行参数一览

| 参数 | 说明 |
|------|------|
| `--live` | 提交真实 Alpaca 纸交易订单（不加则为模拟运行） |
| `--dry-run` | 显式指定模拟运行（默认行为） |
| `--backtest` | 仅运行回测，不执行交易 |
| `--auto-backtest` | 先执行交易，再由 LLM 判断是否需要回测 |
| `--generate-config` | 重新生成 strategy_config.json |
| `--summary` | 查看 Alpaca 账户摘要并退出 |
| `--use-strategy N` | 使用策略配置中第 N 个策略（0-based），绕过 LLM，纯规则信号 |
| `--strategy N` | 配合 `--backtest`，只回测第 N 个策略 |
| `--strategies N` | 配合 `--generate-config`，生成 N 套策略（默认 5） |
| `--tickers A,B,C` | 自定义交易标的（逗号分隔，覆盖默认组合） |
| `--date YYYY-MM-DD` | 指定交易日（默认当天） |
| `--quiet` | 减少控制台输出 |

### 9. 典型工作流

**场景一：验证策略思路**

```bash
# 1. 生成策略配置
python Paper_Trading_Agent.py --generate-config --strategies 10

# 2. 回测所有策略，对比绩效
python Paper_Trading_Agent.py --backtest

# 3. 选最优策略做模拟交易验证
python Paper_Trading_Agent.py --use-strategy 3
```

**场景二：日常交易**

```bash
# 1. 启动交互菜单，选 [9] 查看账户
# 2. 选 [10] 让 LLM 分析当前应该用多高的调仓频率
# 3. 选 [1] 或 [3] 做模拟交易，检查信号是否合理
# 4. 选 [2] 或 [4] 执行实盘交易
# 5. 选 [12] 为持仓挂上止盈止损单
```

**场景三：多策略实盘部署**

```bash
# 交互菜单中选 [11]，输入策略编号如 0,1,3
# 选择 Real Orders
# 选择启用 Bracket TP/SL
# 三个策略的资金等比例分配，权重合并后统一调仓
```
