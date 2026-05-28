# 策略逻辑详解
# 利弗莫尔的核心是“顺势而为”和“让利润奔跑”。上述代码通过三个关键步骤实现这一理念：
# 关键点确认（入场）
# 定义：利弗莫尔只在他称之为“关键点”的价格水平上交易。这通常是一个明确的阻力位突破。
# 实现：代码计算过去20个交易日的最高点作为阻力位。只有当当前价格突破这一“关键点”时，系统才会发出第一次买入信号。这避免了在横盘震荡中频繁交易。
# 金字塔加仓（资金管理）
# 定义：绝不向下摊平亏损头寸，只在上涨过程中加仓。最初的仓位通常是试探性的，当市场证明你是正确的（价格上涨），才逐步追加。
# 实现：
# 第一笔买入占用约20%的资金。
# 设定一个涨幅步长（如5%），当价格从上次买入价上涨5%时，触发第二次买入。
# 后续加仓金额通常是递减的（如20%， 15%， 10%），呈“金字塔”形状，确保底仓最重，顶部最轻。
# 移动止损（退出）
# 定义：利弗莫尔认为绝不能让利润变成亏损，一旦趋势回调超过一定幅度（通常为10%），说明判断可能错误，应立即清仓离场。
# 实现：系统记录每一次加仓的价格。只要价格相比最后一次加仓价回撤超过10%，不管当前是否盈利，都会无条件清仓，锁定剩余利润或截断亏损。

# livermore_strategy.py
import akshare as ak
import pandas as pd
import numpy as np
import akquant as aq
from akquant import Strategy
from akq_module_tusharedatamanager import TushareStockDataManager  
import os
from datetime import datetime

class LivermoreBreakoutStrategy(Strategy):
    """
    利弗莫尔关键点突破 + 金字塔式建仓策略
    核心逻辑：
    1. 关键点确认：突破近期（20日）最高点为买入信号。
    2. 试探性建仓：首次买入总仓位的20%（约20%资金）。
    3. 上涨加仓（金字塔）：价格每上涨一定百分比（如5%），加仓一次。
    4. 止损（硬性回撤）：相比上次加仓价格下跌10%，止损卖出全部仓位。
    """
    
    def __init__(self,para_end_date=None, lookback_length=20, pyramid_step_pct=0.05, stop_loss_pct=0.10):
        # ========== 策略配置参数 ==========
        # 资金管理：基础仓位系数（基于总资金的比例，如0.2表示20%）
        self.base_risk_ratio = 0.20 
        # 金字塔加仓触发涨幅（相对于上次买入价，上涨X%加仓）
        self.pyramid_step_pct = 0.05 
        # 移动止损线（利弗莫尔原则：绝不让盈利变亏损，或回撤超过10%）
        self.stop_loss_pct = 0.10 
        # 趋势关键点：突破N日最高点视为关键点
        self.lookback_length = 20 
        
        self.para_end_date = para_end_date  # 回测结束日期参数
        
        # ========== 状态跟踪 ==========
        self.last_buy_price = 0.0    # 上一次买入的价格
        self.highest_price_since_buy = 0.0  # 买入后的最高价（用于移动止损）
        self.current_trench = 1      # 当前是第几批仓位（用于金字塔管理）
        self.is_active = False        # 是否持有头寸

        self.set_history_depth(self.lookback_length + 1)  # 设置历史数据深度，确保能获取足够的历史数据计算关键点

    def on_bar(self, bar):
        """
        核心逻辑：每根K线调用一次
        """
        symbol = bar.symbol
        current_price = bar.close
        current_date_str = bar.timestamp_iso  # ✅ 获取当前日期

        
        # 1. 获取历史数据用于计算突破 (使用 get_history 避免未来函数)
        # 获取过去 lookback_length+1 天的数据
        hist = self.get_history(self.lookback_length + 1,symbol,  field="high")
        # 如果不包含当前Bar，或者数据不足，返回
        if hist is None or len(hist) < self.lookback_length + 1:
            return
            
        # 计算关键点阈值：过去 N 天的最高点（不包括今天，避免未来函数/信号闪烁）
        # 注意：get_history 通常包含当前 bar，如果包含，我们需要用 [:-1] 排除
        # 假设返回的列表中索引0为最旧，-1为最新（当前bar）
        recent_highs = hist[:-1]  # 去掉当前bar
        threshold = np.max(recent_highs)
        
        # 获取当前持仓
        position = self.get_position(symbol)

        # 如果是回测的最后一天，不买只卖；倒数第二天清仓以确保 T+1 生效
        #current_date = datetime.strptime(current_date_str, "%Y-%m-%d %H:%M:%S") # 转换为 datetime 对象

        # 修改为：直接去掉时区信息
        current_date_str_clean = current_date_str.replace('T', ' ').replace('Z', '')
        current_date = datetime.strptime(current_date_str_clean, "%Y-%m-%d %H:%M:%S")

        days_to_end = (self.para_end_date - current_date).days if self.para_end_date is not None else None

        if position > 0 and (days_to_end is not None and days_to_end <= 1):
            if position > 0:
                print(f"{current_date_str}【回测结束清仓】{symbol} 清空所有持仓 ({position}股)")
                self.close_position(symbol)
            # 停止策略后续操作
            self.is_active = False
            self.current_trench = 1
            self.last_buy_price = 0
            return
        
        # ========== 信号生成与动作执行 ==========
        # 情况 A：空仓状态 -> 寻找关键点突破买入信号
        if position == 0 and not self.is_active:
            # 如果当前价格突破了过去 N 日的最高点（关键点买入）
            if current_price > threshold:
                # 计算首次买入股数（基于总资金的 base_risk_ratio）
                total_cash = self.get_cash()  # 获取当前可用资金
                target_value = total_cash * self.base_risk_ratio
                # 使用限价单或市价单，这里简化使用市价买入
                quantity = int(target_value / current_price)
                # A股1手=100股
                quantity = (quantity // 100) * 100
                
                if quantity > 0:
                    self.buy(symbol, quantity)
                    print(f"{current_date_str}【关键点突破】价格 {current_price:.2f} > 阻力位 {threshold:.2f}")
                    print(f"--> 执行第1批买入: {quantity} 股，价格 {current_price:.2f}")
                    
                    # 更新状态
                    self.is_active = True
                    self.last_buy_price = current_price
                    self.highest_price_since_buy = current_price
                    self.current_trench = 1
                    
        # 情况 B：持有头寸状态 -> 管理止损和金字塔加仓
        elif position > 0 and self.is_active:
            # 更新最高价（用于移动止损）
            if current_price > self.highest_price_since_buy:
                self.highest_price_since_buy = current_price
            
            # 1. 止损逻辑：跌破上次加仓价格的 10%
            stop_price = self.last_buy_price * (1 - self.stop_loss_pct)
            if current_price < stop_price:
                print(f"{current_date_str}【触发止损】价格 {current_price:.2f} < 止损线 {stop_price:.2f}")
                self.close_position(symbol)
                # 重置状态
                self.is_active = False
                self.current_trench = 1
                self.last_buy_price = 0
                return
            
            # 2. 加仓逻辑（金字塔式）
            # 计算是否达到了下一阶段的加仓点
            next_buy_price = self.last_buy_price * (1 + self.pyramid_step_pct)
            
            # 定义金字塔系数：第1批20%，第2批20%，第3批20%，第4批40% (模拟利弗莫尔规则)
            # 这里简化：一共只加2次仓，每次加仓金额递减或固定
            # 实际交易中可配置，这里演示如果达到条件且现金充足则加仓
            if current_price >= next_buy_price:
                # 限制最大加仓次数（例如最多加到3批）
                max_trenches = 4
                if self.current_trench >= max_trenches:
                    return
                    
                total_cash = self.get_cash()  # 获取当前可用资金
                # 加仓资金比例：后续批次每次加剩余资金的 20% 或固定比例
                # 为了保证金字塔，这里越加越少（例如第2批加15%，第3批加10%）
                add_ratio = 0.0
                if self.current_trench == 1:
                    add_ratio = 0.20
                elif self.current_trench == 2:
                    add_ratio = 0.20
                elif self.current_trench == 3:
                    add_ratio = 0.20
                elif self.current_trench == 4:
                    add_ratio = 0.40

                if add_ratio <= 0.0:
                    return

                target_add_value = total_cash * add_ratio
                add_quantity = int(target_add_value / current_price)
                add_quantity = (add_quantity // 100) * 100
                
                if add_quantity > 0:
                    self.buy(symbol, add_quantity)
                    print(f"{current_date_str}【金字塔加仓】第{self.current_trench+1}批买入: {add_quantity} 股，价格 {current_price:.2f}")
                    self.last_buy_price = current_price  # 更新持仓成本基准
                    self.current_trench += 1

    def on_stop(self) -> None:
        """
        策略停止时调用 - 回测结束时强制清仓
        """
        # 在on stop中清空所有持仓，但是因为是T+1，所以只能在下一个交易日开盘时清仓，这里我们直接调用close_position，实际回测框架会处理T+1逻辑
        # 改到on bar中，在回测结束的最后一个bar触发清仓，确保不违反T+1规则


        # # 将 int 时间戳转换为 datetime
        # current_date = self.format_time(self.ctx.current_time)
        # # 获取所有持仓标的
        # for symbol in list(self.ctx.positions.keys()):
        #     position = self.get_position(symbol)
        #     if position > 0:
        #         print(f"{current_date}【回测结束清仓】{symbol} 清空所有持仓 ({position}股)")
        #         self.close_position(symbol)
        
        # # 重置状态
        # self.is_active = False
        # self.current_trench = 1
        # self.last_buy_price = 0
        
        # print("========== 回测结束，所有仓位已清空 ==========")


# ========== 运行回测 ==========
if __name__ == "__main__":
    # 1. 获取数据
    symbol = "688690"  # 替换为你想测试的股票代码
    start_date = "20210101" 
    end_date = "20260527"
    DATA_DIR = "tsdata"  # 数据存储目录
    
    print(f"正在获取 {symbol} 数据...")
    
    mytoken = os.getenv('TUSHARE_TOKEN')
    print('TUSHARE_TOKEN set:', bool(mytoken))

    manager = TushareStockDataManager(
        token= mytoken,  # 替换为你的实际 Token # type: ignore
        data_dir=DATA_DIR,
        request_interval=1.5  # 请求间隔 1.5 秒
    )
    df = manager.get_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
    
    
    print(f"数据获取成功，共 {len(df)} 条记录")
    
    para_end_date = datetime.strptime(end_date, "%Y%m%d")  # 转换为 datetime 对象

    print(f"回测结束日期设置为: {para_end_date}")

    # 2. 创建策略实例
    strategy = LivermoreBreakoutStrategy(
        para_end_date=para_end_date,
        # period=20,
        # std_dev=2,
        # buy_threshold=-0.0001,
        # sell_threshold=0.0001
    )
    
    # 3. 运行回测
    result = aq.run_backtest(
        strategy=strategy,
        data=df,

        initial_cash=1_000_000,      # 初始资金100万
        commission_rate=0.0003,      # 万三佣金
        stamp_tax_rate=0.001,        # 千一印花税（卖出收取）
        min_commission=5.0,          # 最低佣金5元
        t_plus_one=True,             # A股T+1制度
        lot_size=100,                # 最小交易单位100股
        slippage={"type": "percent", "value": 0.001},              # 滑点

        fill_policy={
            "price_basis": "open",   # 使用开盘价
            "bar_offset": 1          # 下根bar成交
        },
        show_progress=True
    )
    
    # 4. 输出结果
    print("\n=== 回测结果 ===")
    print(result.metrics_df)
    
    # 5. 生成报告
    report_dir = "./reports"
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{report_dir}/livermore_strategy_{symbol}_{timestamp}.html"
    
    result.report(
        filename=report_path,
        title=f"利弗莫尔突破策略报告 ({symbol})",
        market_data=df,
        include_trade_kline=True
    )
    
    print(f"\n报告已保存至: {report_path}")