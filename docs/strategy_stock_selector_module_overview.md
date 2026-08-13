# StockSelector 三策略模块功能总览

## 1. 文档目的
本文档统一说明当前基于 StockSelector 的三套策略实现，覆盖以下信息：
1. 策略定位与流程。
2. 选股条件与触发频率。
3. 运行方式与输入参数。
4. 输入输出文件与路径。
5. 依赖模块与耦合关系。

---

## 2. 三套策略清单
1. 周度基本面 + 布林/ADX 共振策略。
2. 月度 Excel 选股调仓策略。
3. 科创超跌价值回归策略（含信号买入版与入池即买版）。

---

## 3. 策略一：周度基本面 + 布林/ADX 共振

### 3.1 对应文件
1. 策略主文件： [akq_stock_strategy_weekly_mixed_bollinger.py](../akq_stock_strategy_weekly_mixed_bollinger.py)
2. 选股模块： [akq_module_stock_selector.py](../akq_module_stock_selector.py)
3. 数据模块： [akq_module_tusharedatamanager.py](../akq_module_tusharedatamanager.py)
4. 轻量验证： [scripts/validate_mixed_bollinger_light.py](../scripts/validate_mixed_bollinger_light.py)
5. 详细说明： [docs/strategy_weekly_mixed_bollinger_system_description.md](./strategy_weekly_mixed_bollinger_system_description.md)

### 3.2 选股条件与触发
1. 选股来源：StockSelector.select。
2. 选股频率：每周构建一次候选池（W-MON）。
3. 条件口径：行业过滤、ST/退市、流动性、毛利率与估值等 10 条 mixed 条件。
4. 策略内触发：日线 on_bar 信号驱动，周池仅用于新开仓门控。

### 3.3 运行方式
1. 直接运行：python akq_stock_strategy_weekly_mixed_bollinger.py
2. 环境变量：需要设置 TUSHARE_TOKEN。
3. 可调参数入口：
   - 选股参数：MIXED_BOLLINGER_SELECTOR_FILTER_PARAMS
   - 回测参数：main() 内 start_date, end_date, 仓位与风控参数。

### 3.4 输入输出
1. 输入：
   - 市场与财务数据缓存目录：selector_data, tsdata
   - 周度候选池缓存：selector_data/weekly_universe_cache/
2. 输出：
   - 回测报告：reports/weekly_mixed_bollinger_adx_YYYYMMDD_HHMMSS.html
   - 结构化产物：reports/runs/weekly_mixed_bollinger_adx_YYYYMMDD_HHMMSS/

---

## 4. 策略二：月度 Excel 选股调仓

### 4.1 对应文件
1. 策略主文件： [akq_stock_strategy_monthly_excel_rebalance.py](../akq_stock_strategy_monthly_excel_rebalance.py)
2. 选股模块： [akq_module_stock_selector.py](../akq_module_stock_selector.py)
3. 数据模块： [akq_module_tusharedatamanager.py](../akq_module_tusharedatamanager.py)

### 4.2 选股条件与触发
1. 当前支持两种模式：
   - 模式A：先跑 StockSelector 月度选股，输出 Excel，再回测。
   - 模式B：直接读取既有 Excel 回测。
2. 选股频率：按月（MS）调用 StockSelector.run_monthly。
3. 调仓频率：每月首个交易日执行一次 on_daily_rebalance。
4. 调仓规则：每月按 Excel 当月顺序取前 top_n，缺失则清仓。

### 4.3 运行方式
1. 直接运行：python akq_stock_strategy_monthly_excel_rebalance.py
2. 环境变量：需要设置 TUSHARE_TOKEN。
3. 可调参数入口：
   - 选股参数：MONTHLY_EXCEL_REBALANCE_SELECTOR_FILTER_PARAMS
   - 选股区间：selector_start_date, selector_end_date
   - 是否先选股：run_selector_first
   - 调仓参数：top_n, target_exposure

### 4.4 输入输出
1. 输入：
   - 选股输出 Excel：reports/stock_selection_results_*.xlsx
   - 行情缓存目录：tsdata
2. 输出：
   - 选股 Excel：reports/stock_selection_results_YYYYMMDD_HHMMSS.xlsx
   - 回测报告：reports/monthly_excel_rebalance_strategy_YYYYMMDD_HHMMSS.html

---

## 5. 策略三：科创超跌价值回归

### 5.1 对应文件
1. 信号买入版： [akq_stock_strategy_star_value_reversion.py](../akq_stock_strategy_star_value_reversion.py)
2. 入池即买版： [akq_stock_strategy_star_value_reversion_direct_buy.py](../akq_stock_strategy_star_value_reversion_direct_buy.py)
3. 选股模块： [akq_module_stock_selector.py](../akq_module_stock_selector.py)
4. 轻量验证： [scripts/validate_star_value_reversion_light.py](../scripts/validate_star_value_reversion_light.py)
5. 设计文档： [docs/strategy_star_value_reversion_strategy_design.md](./strategy_star_value_reversion_strategy_design.md)

### 5.2 选股条件与触发
1. 选股入口：StockSelector.select_star_value_reversion。
2. 选股频率：按月构建股票池（MS）。
3. 典型条件：
   - 仅 688 科创标的。
   - 上市时长窗口。
   - 超跌回撤阈值。
   - 不亏损、增长、质押/商誉/负债率等财务过滤。
   - 可选技术过滤（enable_tech_filter）。
4. 两个交易版本：
   - 信号买入版：BOLL 或 MACD 触发买入。
   - 入池即买版：当月入池即买入。

### 5.3 运行方式
1. 信号买入版：python akq_stock_strategy_star_value_reversion.py
2. 入池即买版：python akq_stock_strategy_star_value_reversion_direct_buy.py
3. 环境变量：需要设置 TUSHARE_TOKEN。
4. 可调参数入口：
   - STAR_VALUE_REVERSION_FILTER_PARAMS
   - build_monthly_star_universe 的 star_filter_params

### 5.4 输入输出
1. 输入：
   - 月度池缓存：selector_data/monthly_star_universe_cache/
   - 行情与财务缓存：selector_data, tsdata
2. 输出：
   - 信号买入报告：reports/star_value_reversion_YYYYMMDD_HHMMSS.html
   - 入池即买报告：reports/star_value_reversion_direct_buy_YYYYMMDD_HHMMSS.html

---

## 6. 模块依赖与关系
1. 共同核心：三套策略都依赖 StockSelector 作为选股引擎。
2. 数据依赖：三套策略都依赖 TushareStockDataManager 拉取和缓存行情。
3. 配置路径：参数应由策略文件显式传入，StockSelector 负责执行。
4. 耦合边界：
   - 月度 Excel 调仓策略只消费选股结果，不内置选股逻辑细节。
   - 科创 direct_buy 复用了 star 策略中的 build_monthly_star_universe 与参数定义。

---

## 7. 当前建议
1. 策略命名已体现时间粒度：weekly_mixed_bollinger。
2. 推荐继续保持“策略文件显式传参 + selector 执行”的职责边界。
3. 如需进一步解耦，可把科创共用构建函数与参数常量提取到独立模块。