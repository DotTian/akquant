"""
股票趋势识别模块 - 综合多种方法判断上升/震荡/下降趋势

使用方法:
    from trend_classifier import TrendClassifier
    classifier = TrendClassifier()
    trend, confidence = classifier.classify(df['close'], df['high'], df['low'])
    print(f"当前趋势: {trend}, 置信度: {confidence:.0%}")
"""

import os
import platform

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import matplotlib.font_manager as fm
import mplfinance as mpf
import numpy as np
import pandas as pd
from scipy import stats

# ── 修复中文字体乱码（自动适配 Windows / Linux）──
def _get_cjk_fonts():
    if platform.system() == 'Windows':
        return ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    else:
        return ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']

plt.rcParams['font.sans-serif'] = _get_cjk_fonts()
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示

from typing import Tuple, Optional, List, Dict

from akq_module_tusharedatamanager import TushareStockDataManager
from akq_module_stockinfo import StockInfoManager


class TrendClassifier:
    """
    趋势分类器 - 综合四种方法识别股票处于上升/震荡/下降状态
    
    四种识别方法:
        1. 斜率法 - 线性回归判断价格斜率
        2. 均线排列法 - 短期/中期/长期均线相对位置
        3. ADX法 - 趋势强度指标
        4. 波动率法 - 布林带宽度 + 价格位置
    """
    
    def __init__(self,
                 # 斜率法参数
                 slope_window: int = 20,
                 slope_threshold: float = 0.01,
                 slope_significance: float = 0.05,
                 
                 # 均线法参数
                 ma_fast: int = 5,
                 ma_medium: int = 20,
                 ma_slow: int = 60,
                 
                 # ADX法参数
                 adx_period: int = 14,
                 adx_trend_threshold: float = 25,
                 adx_range_threshold: float = 20,
                 
                 # 波动率法参数
                 bb_window: int = 20,
                 bb_width_threshold: float = 0.05,
                 
                 # 综合投票参数
                 min_votes: int = 3,      # 最少需要多少方法一致
                 require_confidence: float = 0.6  # 最低置信度要求
                 ):
        
        # 斜率法参数
        self.slope_window = slope_window
        self.slope_threshold = slope_threshold
        self.slope_significance = slope_significance
        
        # 均线法参数
        self.ma_fast = ma_fast
        self.ma_medium = ma_medium
        self.ma_slow = ma_slow
        
        # ADX法参数
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        
        # 波动率法参数
        self.bb_window = bb_window
        self.bb_width_threshold = bb_width_threshold
        
        # 投票参数
        self.min_votes = min_votes
        self.require_confidence = require_confidence
        
        # 存储最近一次分类的详细信息
        self.last_details: Dict = {}
    
    # ==================== 方法1: 斜率法 ====================
    def _classify_by_slope(self, prices: pd.Series) -> Tuple[str, float]:
        """
        基于价格斜率判断趋势
        返回: (趋势, 置信度)
        """
        if len(prices) < self.slope_window:
            return "unknown", 0.0
        
        recent = prices.iloc[-self.slope_window:]
        x = np.arange(len(recent))
        
        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent.values)
        
        # 归一化斜率（相对于价格水平）
        normalized_slope = slope / recent.mean()
        
        # 判断趋势
        if normalized_slope > self.slope_threshold and p_value < self.slope_significance:
            # 斜率显著为正 -> 上升趋势
            confidence = min(0.9, 0.5 + abs(normalized_slope) * 10)
            return "uptrend", confidence
        elif normalized_slope < -self.slope_threshold and p_value < self.slope_significance:
            # 斜率显著为负 -> 下降趋势
            confidence = min(0.9, 0.5 + abs(normalized_slope) * 10)
            return "downtrend", confidence
        else:
            # 斜率不显著 -> 震荡
            confidence = 0.5 + (self.slope_threshold - abs(normalized_slope)) * 10
            confidence = min(0.8, max(0.3, confidence))
            return "range", confidence
    
    # ==================== 方法2: 均线排列法 ====================
    def _classify_by_ma(self, prices: pd.Series) -> Tuple[str, float]:
        """
        基于均线排列判断趋势
        """
        if len(prices) < self.ma_slow:
            return "unknown", 0.0
        
        ma_fast = prices.rolling(self.ma_fast).mean().iloc[-1]
        ma_medium = prices.rolling(self.ma_medium).mean().iloc[-1]
        ma_slow = prices.rolling(self.ma_slow).mean().iloc[-1]
        
        # 计算均线间的距离比例（用于置信度）
        fast_medium_ratio = (ma_fast - ma_medium) / ma_medium if ma_medium != 0 else 0
        medium_slow_ratio = (ma_medium - ma_slow) / ma_slow if ma_slow != 0 else 0
        
        # 多头排列: fast > medium > slow
        if ma_fast > ma_medium > ma_slow:
            # 距离越大，置信度越高
            confidence = min(0.95, 0.5 + abs(fast_medium_ratio) * 20 + abs(medium_slow_ratio) * 10)
            return "uptrend", confidence
        
        # 空头排列: fast < medium < slow
        elif ma_fast < ma_medium < ma_slow:
            confidence = min(0.95, 0.5 + abs(fast_medium_ratio) * 20 + abs(medium_slow_ratio) * 10)
            return "downtrend", confidence
        
        # 均线缠绕 -> 震荡
        else:
            # 计算均线离散程度，越低越震荡
            ma_values = [ma_fast, ma_medium, ma_slow]
            ma_cv = np.std(ma_values) / np.mean(ma_values) if np.mean(ma_values) != 0 else 0
            confidence = min(0.8, 0.3 + (1 - ma_cv * 20))
            return "range", max(0.4, confidence)
    
    # ==================== 方法3: ADX法 ====================
    def _compute_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[float, float, float]:
        """
        计算ADX指标
        返回: (adx, plus_di, minus_di)
        """
        period = self.adx_period
        
        # 计算+DM和-DM
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        # 计算TR (True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 平滑计算
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        # 计算DX和ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]
    
    def _classify_by_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[str, float]:
        """
        基于ADX指标判断趋势
        """
        if len(close) < self.adx_period * 2:
            return "unknown", 0.0
        
        adx, plus_di, minus_di = self._compute_adx(high, low, close)
        
        # ADX < 20 -> 无趋势（震荡）
        if adx < self.adx_range_threshold:
            confidence = 0.5 + (self.adx_range_threshold - adx) / self.adx_range_threshold * 0.3
            confidence = min(0.8, max(0.4, confidence))
            return "range", confidence
        
        # ADX > 25 且有明确方向
        elif adx > self.adx_trend_threshold:
            di_diff = abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
            confidence = min(0.9, 0.5 + (adx - self.adx_trend_threshold) / 50 + di_diff)
            
            if plus_di > minus_di:
                return "uptrend", confidence
            else:
                return "downtrend", confidence
        
        # 20 <= ADX <= 25 -> 弱趋势，倾向于震荡
        else:
            return "range", 0.5
    
    # ==================== 方法4: 波动率法 ====================
    def _classify_by_volatility(self, prices: pd.Series) -> Tuple[str, float]:
        """
        基于波动率和价格位置判断趋势
        """
        if len(prices) < self.bb_window:
            return "unknown", 0.0
        
        recent = prices.iloc[-self.bb_window:]
        current_price = prices.iloc[-1]
        
        # 计算布林带宽度
        ma = recent.mean()
        std = recent.std()
        bb_width = (2 * std) / ma if ma != 0 else 0
        
        # 价格位置（相对于window内高低点）
        price_min = recent.min()
        price_max = recent.max()
        if price_max == price_min:
            price_position = 0.5
        else:
            price_position = (current_price - price_min) / (price_max - price_min)
        
        # 震荡判断：波动率低
        if bb_width < self.bb_width_threshold:
            confidence = min(0.85, 0.5 + (self.bb_width_threshold - bb_width) * 10)
            return "range", confidence
        
        # 上升判断：波动率较高 且 价格在上方25%区域
        elif price_position > 0.75:
            confidence = min(0.85, 0.5 + (price_position - 0.75) * 2 + (bb_width - self.bb_width_threshold) * 5)
            return "uptrend", confidence
        
        # 下降判断：波动率较高 且 价格在下方25%区域
        elif price_position < 0.25:
            confidence = min(0.85, 0.5 + (0.25 - price_position) * 2 + (bb_width - self.bb_width_threshold) * 5)
            return "downtrend", confidence
        
        # 中间区域 -> 震荡或弱趋势
        else:
            confidence = 0.5 - abs(price_position - 0.5) * 0.5
            return "range", max(0.3, confidence)
    
    # ==================== 综合投票 ====================
    def classify(self, 
                 close: pd.Series, 
                 high: Optional[pd.Series] = None, 
                 low: Optional[pd.Series] = None,
                 debug: bool = False) -> Tuple[str, float]:
        """
        综合四种方法判断趋势状态
        
        参数:
            close: 收盘价序列
            high: 最高价序列（可选，用于ADX法）
            low: 最低价序列（可选，用于ADX法）
            debug: 是否打印详细信息
            
        返回:
            (趋势, 置信度)
            趋势: uptrend / range / downtrend / unknown
            置信度: 0-1之间
        """
        if len(close) < max(self.ma_slow, self.slope_window):
            return "unknown", 0.0
        
        # 存储各方法的结果
        results: List[Tuple[str, float]] = []
        method_names = []
        
        # 方法1: 斜率法
        trend1, conf1 = self._classify_by_slope(close)
        results.append((trend1, conf1))
        method_names.append("斜率法")
        
        # 方法2: 均线法
        trend2, conf2 = self._classify_by_ma(close)
        results.append((trend2, conf2))
        method_names.append("均线法")
        
        # 方法3: ADX法（需要high/low数据）
        if high is not None and low is not None and len(high) == len(close):
            trend3, conf3 = self._classify_by_adx(high, low, close)
            results.append((trend3, conf3))
            method_names.append("ADX法")
        else:
            results.append(("unknown", 0.0))
            method_names.append("ADX法(跳过)")
        
        # 方法4: 波动率法
        trend4, conf4 = self._classify_by_volatility(close)
        results.append((trend4, conf4))
        method_names.append("波动率法")
        
        # 计算各趋势的投票得分（加权）
        scores = {"uptrend": 0.0, "range": 0.0, "downtrend": 0.0}
        method_results = []
        
        for i, (trend, conf) in enumerate(results):
            if trend != "unknown":
                scores[trend] += conf
                method_results.append((method_names[i], trend, conf))
        
        # 找出最高分趋势
        if max(scores.values()) == 0:
            return "unknown", 0.0
        
        best_trend = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_trend] / total_score if total_score > 0 else 0
        
        # 存储详细信息供调试
        self.last_details = {
            "scores": scores,
            "methods": method_results,
            "best_trend": best_trend,
            "confidence": confidence
        }
        
        if debug:
            self.print_debug()
        
        return best_trend, confidence
    
    def print_debug(self):
        """打印调试信息"""
        print("\n" + "=" * 60)
        print("趋势识别详情")
        print("=" * 60)
        
        for method, trend, conf in self.last_details["methods"]:
            trend_cn = {"uptrend": "↑上升", "range": "→震荡", "downtrend": "↓下降", "unknown": "?"}.get(trend, trend)
            print(f"  {method:8s}: {trend_cn} (置信度: {conf:.0%})")
        
        print("-" * 40)
        print(f"  综合得分: 上升={self.last_details['scores']['uptrend']:.2f}, "
              f"震荡={self.last_details['scores']['range']:.2f}, "
              f"下降={self.last_details['scores']['downtrend']:.2f}")
        print(f"  最终判断: {self.last_details['best_trend']} (置信度: {self.last_details['confidence']:.0%})")
        print("=" * 60)


# ==================== 绘制函数 ====================
def plot_trend_chart(df: pd.DataFrame,
                    results_df: pd.DataFrame,
                    symbol: str,
                    save_path: Optional[str] = None,
                    stock_name: Optional[str] = None):
    """
    绘制带趋势标注的日K线图

    Parameters
    ----------
    df : 原始OHLC数据 (index 为日期)
    results_df : classify 输出的结果 (columns: date, close, trend, confidence)
    symbol : 股票代码
    save_path : 保存路径，如果为 None 则显示
    stock_name : 股票中文名称（可选），如果提供则显示在标题中
    """
    # 显示用的标的名
    display_name = f"{symbol} {stock_name}" if stock_name else symbol
    # 确保 df 的 index 是 DatetimeIndex
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 构建 trend map：date -> (trend, confidence)
    trend_map = {}
    for _, row in results_df.iterrows():
        d = row['date']
        if isinstance(d, pd.Timestamp):
            d = d.strftime('%Y-%m-%d')
        trend_map[d] = (row['trend'], row['confidence'])

    # ── 准备 mplfinance addplot ──
    addplots = []

    # MA5
    ma5 = df['close'].rolling(5).mean()
    addplots.append(mpf.make_addplot(ma5, color='orange', width=0.8, label='MA5'))

    # MA20
    ma20 = df['close'].rolling(20).mean()
    addplots.append(mpf.make_addplot(ma20, color='purple', width=0.8, label='MA20'))

    # MA60
    ma60 = df['close'].rolling(60).mean()
    addplots.append(mpf.make_addplot(ma60, color='teal', width=0.8, label='MA60'))

    # ── 构建趋势背景色标记 ──
    # mplfinance 中通过 fill_between / vspan 等方式做背景不太方便，
    # 这里采用 mplfinance + 外部 axvspan 的方式

        # 先用 mplfinance 画 K 线（字体配置传入 style 防止被 mplfinance 覆盖）
    mc = mpf.make_marketcolors(
        up='red', down='green', edge='inherit', wick='inherit', volume='inherit'
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=':',
        y_on_right=False,
        # windows 字体
        # rc={
        #     'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
        #     'axes.unicode_minus': False,
        # }
        # 修改后 Linux 字体
        rc={
            'font.sans-serif': _get_cjk_fonts(),
            'axes.unicode_minus': False,
        }
    )

    fig, axes = mpf.plot(
        df,
        type='candle',
        style=style,
        addplot=addplots,
        volume=True,
        title=f"{display_name} 日K线图 + 趋势识别",
        ylabel='价格',
        ylabel_lower='成交量',
        figratio=(18, 9),
        figsize=(16, 8),
        returnfig=True,
        datetime_format='%Y-%m',
        xrotation=30,
    )

    # ax 是第一个子图（K线图）
    ax = axes[0]

    # 颜色映射
    trend_colors = {
        'uptrend': '#4CAF50',    # 绿色
        'downtrend': '#F44336',  # 红色
        'range': '#FFC107',      # 黄色
        'unknown': '#E0E0E0',    # 灰色
    }

    # 获取数据索引对应的 x 坐标
    # mplfinance 的 x 轴是数值索引
    x_index = np.arange(len(df))
    x_to_date = dict(zip(x_index, df.index))
    date_to_x = {}
    for i, d in enumerate(df.index):
        date_to_x[d.strftime('%Y-%m-%d')] = i

            # 遍历每个交易日，用 axvspan 画出趋势区间
    # 策略：连续相同趋势合并成一个区间
    if len(results_df) > 0:
        segments = []
        current_trend = None
        seg_start = None

        for _, row in results_df.iterrows():
            d = row['date']
            trend = row['trend']
            if isinstance(d, pd.Timestamp):
                d_str = d.strftime('%Y-%m-%d')
            else:
                d_str = str(d)

            if d_str not in date_to_x:
                continue
            x = date_to_x[d_str]

            if trend != current_trend:
                if current_trend is not None and seg_start is not None:
                    segments.append((seg_start, x - 0.5, current_trend))
                current_trend = trend
                seg_start = max(0, x - 0.5)

        # 最后一段
        if current_trend is not None and seg_start is not None:
            segments.append((seg_start, len(df) - 0.5, current_trend))

        # 绘制半透明色块（zorder=3 确保显示在K线之上）
        for seg_start, seg_end, trend in segments:
            color = trend_colors.get(trend, '#E0E0E0')
            ax.axvspan(seg_start, seg_end, alpha=0.35, color=color, zorder=3, edgecolor=color, linewidth=0.5)

        # ── 添加趋势变化标记（显示在成交量子图上方，不干扰K线） ──
        prev_trend = None
        # 获取成交量子图（mplfinance 返回 axes: [0]=K线, [2]=成交量）
        ax_vol = axes[2]
        for _, row in results_df.iterrows():
            d = row['date']
            trend = row['trend']
            confidence = row['confidence']
            if isinstance(d, pd.Timestamp):
                d_str = d.strftime('%Y-%m-%d')
            else:
                d_str = str(d)

            if d_str not in date_to_x:
                continue
            x = date_to_x[d_str]

            # 只在趋势变化时标注，显示在成交量图上
            if trend != prev_trend:
                marker_color = trend_colors.get(trend, 'gray')
                trend_cn = {'uptrend': '↑', 'downtrend': '↓', 'range': '→'}.get(trend, '')
                label = f"{trend_cn} {confidence:.0%}" if confidence > 0 else trend_cn
                y_vol_top = ax_vol.get_ylim()[1]
                ax_vol.annotate(
                    label,
                    xy=(x, y_vol_top),
                    fontsize=9,
                    fontweight='bold',
                    color=marker_color,
                    ha='center',
                    va='bottom',
                    xytext=(0, 6),
                    textcoords='offset points',
                    bbox=dict(
                        boxstyle='round,pad=0.2',
                        facecolor=marker_color,
                        alpha=0.15,
                        edgecolor='none'
                    )
                )
            prev_trend = trend

    # ── 图例 ──
    handles = [
        mpatches.Patch(color='#4CAF50', alpha=0.25, label='↑ 上升趋势'),
        mpatches.Patch(color='#FFC107', alpha=0.25, label='→ 震荡'),
        mpatches.Patch(color='#F44336', alpha=0.25, label='↓ 下降趋势'),
    ]
    ax.legend(
        handles=handles,
        loc='upper left',
        fontsize=8,
        framealpha=0.9,
        title='趋势状态 (背景色)',
        title_fontsize=9,
    )

    # ── 标题上加趋势统计 ──
    uptrend_pct = (results_df['trend'] == 'uptrend').mean()
    range_pct = (results_df['trend'] == 'range').mean()
    downtrend_pct = (results_df['trend'] == 'downtrend').mean()
    ax.set_title(
        f"{display_name} 日K线 + 趋势识别\n"
        f"[上升 {uptrend_pct:.0%} | 震荡 {range_pct:.0%} | 下降 {downtrend_pct:.0%}]\n"
        f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}",
        fontsize=12,
    )

    # ── 保存或显示 ──
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[图表已保存] {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 配置多个 symbol
    symbols = ["300724", "688270"]   # 可在此添加更多股票代码
    start_date = "20260105"
    end_date = "20260701"
    DATA_DIR = "tsdata"
    report_dir = "./reports"

    mytoken = os.getenv('TUSHARE_TOKEN')
    print('TUSHARE_TOKEN set:', bool(mytoken))

    manager = TushareStockDataManager(
        token=mytoken,
        data_dir=DATA_DIR,
        request_interval=1.5
    )

    # 初始化股票信息管理器（获取中文名称）
    stock_info = StockInfoManager(
        token=mytoken,
        data_dir="stock_info",
        request_interval=1.2
    )

    classifier = TrendClassifier()

    for symbol in symbols:
        # 获取中文名称
        cn_name = stock_info.get_stock_name(symbol)
        display_label = f"{symbol} ({cn_name})" if cn_name else symbol

        print(f"\n{'='*60}")
        print(f"正在获取 {display_label} 数据...")
        df = manager.get_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)

        if df.empty:
            print(f"警告: {symbol} 未获取到数据，跳过")
            continue

        print(f"数据获取成功，共 {len(df)} 条记录")
        print(f"数据范围: {df.index[0]} 至 {df.index[-1]}, 共{len(df)}个交易日")

        # 分别判断每天的趋势（滚动窗口）
        results = []
        for i in range(60, len(df)):
            end_idx = i + 1
            close = df['close'].iloc[:end_idx]
            high = df['high'].iloc[:end_idx]
            low = df['low'].iloc[:end_idx]

            trend, confidence = classifier.classify(close, high, low)
            results.append({
                'date': df.index[i],
                'close': close.iloc[-1],
                'trend': trend,
                'confidence': confidence
            })

        results_df = pd.DataFrame(results)

        # 输出最近10天的结果（带中文名）
        print(f"\n{display_label} 最近10天趋势判断:")
        print(results_df.tail(10).to_string())

        # 统计各趋势占比
        print(f"\n{display_label} 趋势分布统计:")
        print(results_df['trend'].value_counts())

        # 测试单次详细输出（最后一天）
        print("\n" + "=" * 60)
        print(f"{display_label} 单次详细识别示例（最后一天）:")
        last_close = df['close']
        last_high = df['high']
        last_low = df['low']
        trend, conf = classifier.classify(last_close, last_high, last_low, debug=True)

        # ── 绘制带趋势标注的 K 线图（标题带中文名）──
        print(f"\n正在生成 {symbol} 趋势K线图...")
        os.makedirs(report_dir, exist_ok=True)
        chart_path = f"{report_dir}/trend_chart_{symbol}_{start_date}_{end_date}.png"
        plot_trend_chart(df, results_df, symbol, save_path=chart_path, stock_name=cn_name)

    print("\n" + "="*60)
    print("全部股票处理完成。")
    print("="*60)