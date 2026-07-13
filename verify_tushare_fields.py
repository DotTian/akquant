"""
验证 Tushare 基本面数据字段可用性
测试项:
  1. 毛利率 (grossprofit_margin)
  2. 研发投入占营收比 (rd_expense / revenue)
  3. 净利润 (是否亏损)
  4. ST/退市状态 (namechange + delist_date)
  5. PEG 计算依赖 (PE_TTM + 净利润增长率)
  6. 已有字段确认 (PE, 市值, 换手率, 日交易额)
"""

import os
import pandas as pd
import tushare as ts

# 初始化
token = os.getenv('TUSHARE_TOKEN')
if not token:
    raise EnvironmentError("请设置 TUSHARE_TOKEN 环境变量")

ts.set_token(token)
pro = ts.pro_api()

# 测试股票 (选择一只样本，例如 600519 贵州茅台)
symbol = '600519'
ts_code = f'{symbol}.SH'
test_date = '20251231'  # 确保有财报数据

print("=" * 60)
print("Tushare 基本面数据字段验证")
print(f"测试股票: {symbol} ({ts_code})")
print("=" * 60)

# ─────────────────────────────────────────
# 1. 毛利率 (fina_indicator)
# ─────────────────────────────────────────
print("\n📊 [1/6] 毛利率 - fina_indicator")
try:
    df = pro.fina_indicator(ts_code=ts_code, start_date='20240101', end_date='20251231',
                            fields='end_date,grossprofit_margin')
    print(f"  获取到 {len(df)} 条记录")
    if not df.empty:
        print(df.head().to_string())
    else:
        print("  ⚠️ 返回空（可能日期无数据）")
except Exception as e:
    print(f"  ❌ 错误: {e}")

# ─────────────────────────────────────────
# 2. 研发费用占营收比 (rd_expense / revenue)
# ─────────────────────────────────────────
print("\n📊 [2/6] 研发费用占比 - fina_indicator + income")
try:
    # 研发费用 (从 income 接口获取，字段名 rd_exp)
    df_rd = pro.income(ts_code=ts_code, start_date='20240101', end_date='20251231',
                              fields='end_date,rd_exp')
    # 营业收入 (取自 income 接口)
    df_income = pro.income(ts_code=ts_code, start_date='20240101', end_date='20251231',
                          fields='end_date,total_revenue')
    if not df_rd.empty and not df_income.empty:
        merged = pd.merge(df_rd, df_income, on='end_date')
        merged['rd_ratio'] = merged['rd_expense'] / merged['total_revenue'] * 100
        print(f"  合并后 {len(merged)} 条记录")
        print(merged[['end_date', 'rd_expense', 'total_revenue', 'rd_ratio']].head().to_string())
    else:
        print("  ⚠️ 数据为空")
except Exception as e:
    print(f"  ❌ 错误: {e}")

# ─────────────────────────────────────────
# 3. 是否亏损 (净利润 n_income)
# ─────────────────────────────────────────
print("\n📊 [3/6] 是否亏损 - income (净利润)")
try:
    df = pro.income(ts_code=ts_code, start_date='20240101', end_date='20251231',
                    fields='end_date,n_income_attr_p')
    if not df.empty:
        df['is_loss'] = df['n_income_attr_p'] < 0
        print(f"  获取到 {len(df)} 条记录")
        print(df[['end_date', 'n_income_attr_p', 'is_loss']].head().to_string())
    else:
        print("  ⚠️ 返回空")
except Exception as e:
    print(f"  ❌ 错误: {e}")

# ─────────────────────────────────────────
# 4. ST / 退市状态 (namechange + stock_basic delist_date)
# ─────────────────────────────────────────
print("\n📊 [4/6] ST/退市状态 - namechange & stock_basic")
try:
    # 查更名记录
    df_name = pro.namechange(ts_code=ts_code)
    if not df_name.empty:
        st_records = df_name[df_name['name'].str.contains(r'\*?ST', regex=True)]
        if not st_records.empty:
            print(f"  ⚠️ 历史 ST 记录: {len(st_records)} 条")
            print(st_records[['ann_date', 'name']].to_string())
        else:
            print("  ✅ 无 ST 记录")
    else:
        print("  ✅ 无更名记录")

    # 查退市状态
    df_basic = pro.stock_basic(ts_code=ts_code, fields='ts_code,name,delist_date')
    if not df_basic.empty:
        delist = df_basic.iloc[0]['delist_date']
        if pd.isna(delist):
            print("  ✅ 未退市")
        else:
            print(f"  ❌ 已退市: {delist}")
    else:
        print("  ⚠️ stock_basic 无数据")
except Exception as e:
    print(f"  ❌ 错误: {e}")

# ─────────────────────────────────────────
# 5. PEG 计算依赖 (PE_TTM + 净利润增长率)
# ─────────────────────────────────────────
print("\n📊 [5/6] PEG 依赖 - daily_basic (PE_TTM) + fina_indicator (q_profit_yoy)")
try:
    # PE_TTM
    df_daily = pro.daily_basic(ts_code=ts_code, trade_date=test_date,
                               fields='ts_code,trade_date,pe_ttm')
    pe_ttm = None
    if not df_daily.empty:
        pe_ttm = df_daily.iloc[0]['pe_ttm']
        print(f"  PE_TTM ({test_date}): {pe_ttm}")

    # 净利润同比增长率 (季度)
    df_q = pro.fina_indicator(ts_code=ts_code, start_date='20240101', end_date='20251231',
                              fields='end_date,q_profit_yoy')
    if not df_q.empty:
        latest_growth = df_q.iloc[0]['q_profit_yoy']
        print(f"  最新季度净利润同比增长率: {latest_growth}%")
        if pe_ttm and latest_growth:
            # PEG = PE_TTM / (增长率 * 100)
            peg = pe_ttm / (latest_growth * 100)
            print(f"  🧮 计算 PEG ≈ {peg:.2f}")
    else:
        print("  ⚠️ 净利润增长率数据为空")
except Exception as e:
    print(f"  ❌ 错误: {e}")

# ─────────────────────────────────────────
# 6. 已有字段确认 (PE, 市值, 换手率, 日交易额)
# ─────────────────────────────────────────
print("\n📊 [6/6] 已有字段确认 - daily_basic")
try:
    df = pro.daily_basic(ts_code=ts_code, trade_date=test_date,
                         fields='ts_code,trade_date,pe,total_mv,circ_mv,turnover_rate,turnover_rate_f,volume_ratio')
    if not df.empty:
        row = df.iloc[0]
        print(f"  PE: {row['pe']}")
        print(f"  总市值 (亿): {row['total_mv']/10000:.2f}")
        print(f"  流通市值 (亿): {row['circ_mv']/10000:.2f}")
        print(f"  换手率 (%): {row['turnover_rate']}")
        print(f"  换手率(自由流通股): {row['turnover_rate_f']}")
        # 日交易额从已有的 daily 数据获取 (amount)
        df_daily = pro.daily(ts_code=ts_code, trade_date=test_date,
                             fields='ts_code,trade_date,amount')
        if not df_daily.empty:
            print(f"  日交易额 (元): {df_daily.iloc[0]['amount']}")
    else:
        print("  ⚠️ 无数据")
except Exception as e:
    print(f"  ❌ 错误: {e}")

print("\n" + "=" * 60)
print("验证完成")