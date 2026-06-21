"""
股票日线背离信号检测模块
- MACD 与价格背离（顶/底）
- 量与价背离（顶/底）

使用方法：
    from akq_module_divergence import DivergenceDetector
    detector = DivergenceDetector()
    divergences = detector.detect(close, high, low, volume)
    for d in divergences:
        print(d['type'], d['subtype'], d['date'], d['price'])
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import mplfinance as mpf
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

# ── 修复中文字体乱码 (Windows) ──
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False

from typing import Tuple, Optional, List, Dict

from akq_module_tusharedatamanager import TushareStockDataManager


class DivergenceDetector:
    """
    背离检测器
    """

    def __init__(self,
                 # MACD 参数
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 # 极值点查找窗口（左右各多少个K线作为局部区域）
                 extremum_order: int = 5,
                 # 量价背离要求：若价格创新高但成交量低于前高成交量的比率阈值
                 volume_confirm_ratio: float = 0.95,
                 ):
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.extremum_order = extremum_order
        self.volume_confirm_ratio = volume_confirm_ratio

        # 存储最近检测结果，用于绘图
        self.last_divergences: List[Dict] = []

    # ==================== MACD 计算 ====================
    def _compute_macd(self, close: pd.Series) -> pd.DataFrame:
        """返回包含 DIF, DEA, MACD 柱的 DataFrame"""
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.macd_signal, adjust=False).mean()
        macd_bar = 2 * (dif - dea)   # 常用 MACD 柱 = (DIF - DEA)*2
        return pd.DataFrame({'DIF': dif, 'DEA': dea, 'MACD_BAR': macd_bar}, index=close.index)

    # ==================== 极值点查找 ====================
    @staticmethod
    def _find_peaks(series: pd.Series, order: int = 5) -> np.ndarray:
        """
        返回局部极大值的索引数组 (使用 scipy.signal.argrelextrema)
        """
        indices = argrelextrema(series.values, np.greater, order=order)[0]
        return indices

    @staticmethod
    def _find_valleys(series: pd.Series, order: int = 5) -> np.ndarray:
        """
        返回局部极小值的索引数组
        """
        indices = argrelextrema(series.values, np.less, order=order)[0]
        return indices

    # ==================== 背离检测主逻辑 ====================
    def detect(self,
               close: pd.Series,
               high: pd.Series,
               low: pd.Series,
               volume: pd.Series,
               debug: bool = False) -> List[Dict]:
        """
        检测所有背离信号，返回列表，每个元素为 dict：
        {
            'date':      pd.Timestamp,
            'price':     float,
            'type':      'top' 或 'bottom',   # 顶背离 / 底背离
            'subtype':   'macd' 或 'volume',   # MACD背离 / 量价背离
            'confidence': float,               # 置信度（可基于偏离程度）
        }
        """
        divergences = []

        # 计算 MACD 指标
        macd_df = self._compute_macd(close)
        dif = macd_df['DIF']
        macd_bar = macd_df['MACD_BAR']

        # --- 1. 寻找价格局部极值点 ---
        peaks_idx = self._find_peaks(close, order=self.extremum_order)
        valleys_idx = self._find_valleys(close, order=self.extremum_order)

        # 确保有足够数据进行对比（至少两个极值）
        if len(peaks_idx) < 2 and len(valleys_idx) < 2:
            if debug:
                print("数据不足，无法检测背离")
            return divergences

        # --- 2. MACD 背离检测（使用 DIF） ---
        # 顶背离：价格新高，DIF 未创新高
        for i in range(1, len(peaks_idx)):
            idx_cur = peaks_idx[i]
            idx_prev = peaks_idx[i - 1]
            price_cur = close.iloc[idx_cur]
            price_prev = close.iloc[idx_prev]
            dif_cur = dif.iloc[idx_cur]
            dif_prev = dif.iloc[idx_prev]

            if price_cur > price_prev and dif_cur < dif_prev:
                # 价格新高，DIF 未新高 -> 顶背离
                confidence = min(0.9, (price_cur / price_prev - 1) * 10 + 0.5)
                divergences.append({
                    'date': close.index[idx_cur],
                    'price': price_cur,
                    'type': 'top',
                    'subtype': 'macd',
                    'confidence': confidence
                })

        # 底背离：价格新低，DIF 未创新低
        for i in range(1, len(valleys_idx)):
            idx_cur = valleys_idx[i]
            idx_prev = valleys_idx[i - 1]
            price_cur = low.iloc[idx_cur]  # 使用最低价
            price_prev = low.iloc[idx_prev]
            dif_cur = dif.iloc[idx_cur]
            dif_prev = dif.iloc[idx_prev]

            if price_cur < price_prev and dif_cur > dif_prev:
                confidence = min(0.9, (price_prev / price_cur - 1) * 10 + 0.5)
                divergences.append({
                    'date': close.index[idx_cur],
                    'price': low.iloc[idx_cur],
                    'type': 'bottom',
                    'subtype': 'macd',
                    'confidence': confidence
                })

        # --- 3. 量价背离检测 ---
        # 顶背离：价格新高，成交量未创新高（使用成交量确认）
        for i in range(1, len(peaks_idx)):
            idx_cur = peaks_idx[i]
            idx_prev = peaks_idx[i - 1]
            price_cur = close.iloc[idx_cur]
            price_prev = close.iloc[idx_prev]
            vol_cur = volume.iloc[idx_cur]
            vol_prev = volume.iloc[idx_prev]

            if price_cur > price_prev and vol_cur < vol_prev * self.volume_confirm_ratio:
                confidence = min(0.85, (price_cur / price_prev - 1) * 8 + 0.5)
                divergences.append({
                    'date': close.index[idx_cur],
                    'price': price_cur,
                    'type': 'top',
                    'subtype': 'volume',
                    'confidence': confidence
                })

        # 底背离：价格新低，成交量未创新低（成交量缩小）
        for i in range(1, len(valleys_idx)):
            idx_cur = valleys_idx[i]
            idx_prev = valleys_idx[i - 1]
            price_cur = low.iloc[idx_cur]
            price_prev = low.iloc[idx_prev]
            vol_cur = volume.iloc[idx_cur]
            vol_prev = volume.iloc[idx_prev]

            if price_cur < price_prev and vol_cur > vol_prev * (1 / self.volume_confirm_ratio):
                # 成交量放大而非缩小，其实才是底背离的正常情况（放量下跌后缩量或放量确认？）
                # 标准量价底背离：价格新低，成交量不创新低（拒绝下跌）→ 成交量应相对放大，此处简化：成交量高于前低
                confidence = min(0.85, (price_prev / price_cur - 1) * 8 + 0.5)
                divergences.append({
                    'date': close.index[idx_cur],
                    'price': low.iloc[idx_cur],
                    'type': 'bottom',
                    'subtype': 'volume',
                    'confidence': confidence
                })

        # 排序按日期
        divergences.sort(key=lambda x: x['date'])
        self.last_divergences = divergences

        if debug:
            self._print_debug(close)

        return divergences

    def _print_debug(self, close: pd.Series):
        print(f"\n{'=' * 60}")
        print(f"背离信号检测结果（共 {len(self.last_divergences)} 个）")
        print(f"{'=' * 60}")
        for d in self.last_divergences:
            type_cn = "顶背离" if d['type'] == 'top' else "底背离"
            subtype_cn = "MACD" if d['subtype'] == 'macd' else "量价"
            print(f"  {d['date'].strftime('%Y-%m-%d')} | {type_cn}({subtype_cn}) "
                  f"价格:{d['price']:.2f} 置信度:{d['confidence']:.0%}")
        print(f"{'=' * 60}\n")


# ==================== 绘图函数 ====================
def plot_divergence_chart(df: pd.DataFrame,
                          divergences: List[Dict],
                          symbol: str,
                          save_path: Optional[str] = None):
    """
    绘制带背离标注的日K线图（包含MACD和成交量子图）

    Parameters
    ----------
    df : 原始OHLCV数据 (index 为日期)
    divergences : detect 返回的背离列表
    symbol : 股票代码
    save_path : 保存路径，如果为 None 则显示
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 计算 MACD 用于子图
    macd_fast, macd_slow, macd_signal = 12, 26, 9
    ema_fast = df['close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=macd_slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=macd_signal, adjust=False).mean()
    macd_bar = 2 * (dif - dea)

    # 构建 mplfinance addplot
    addplots = [
        mpf.make_addplot(ma5, color='orange', width=0.8, label='MA5')
        for ma5 in [df['close'].rolling(5).mean()]
    ]
    addplots.extend([
        mpf.make_addplot(df['close'].rolling(20).mean(), color='purple', width=0.8, label='MA20'),
        mpf.make_addplot(df['close'].rolling(60).mean(), color='teal', width=0.8, label='MA60'),
    ])

    # 成交量子图
    volume_plot = mpf.make_addplot(df['volume'], panel=1, color='gray', width=0.8, ylabel='成交量')

    # MACD 子图
    macd_panel = mpf.make_addplot(dif, panel=2, color='blue', width=0.7, ylabel='DIF')
    dea_panel = mpf.make_addplot(dea, panel=2, color='orange', width=0.7)
    macd_bar_panel = mpf.make_addplot(macd_bar, panel=2, type='bar', color='red', width=0.7, alpha=0.5)

    all_plots = addplots + [volume_plot, macd_panel, dea_panel, macd_bar_panel]

    # 风格设置
    mc = mpf.make_marketcolors(
        up='red', down='green', edge='inherit', wick='inherit', volume='inherit'
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=':',
        y_on_right=False,
        rc={
            'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
            'axes.unicode_minus': False,
        }
    )

    fig, axes = mpf.plot(
        df,
        type='candle',
        style=style,
        addplot=all_plots,
        volume=False,  # 我们手动加了 volume 子图
        title=f"{symbol} 日K线图 + 背离信号",
        ylabel='价格',
        ylabel_lower='',
        figratio=(16, 10),
        figsize=(16, 9),
        returnfig=True,
        datetime_format='%Y-%m',
        xrotation=30,
        panel_ratios=(3, 1, 1.5),
    )

    ax_candle = axes[0]
    ax_vol = axes[1]  # 成交量子图
    ax_macd = axes[2]  # MACD 子图

    # 在K线图上标注背离
    # 颜色定义
    color_top_macd = '#FF5722'   # 橙红
    color_bottom_macd = '#4CAF50'
    color_top_volume = '#E91E63'  # 粉红
    color_bottom_volume = '#2196F3'

    # 创建日期到x坐标的映射
    date_to_x = {d.strftime('%Y-%m-%d'): i for i, d in enumerate(df.index)}

    for d in divergences:
        d_str = d['date'].strftime('%Y-%m-%d') if isinstance(d['date'], pd.Timestamp) else str(d['date'])
        if d_str not in date_to_x:
            continue
        x = date_to_x[d_str]
        price = d['price']

        # 根据类型选择颜色和标记
        if d['type'] == 'top':
            if d['subtype'] == 'macd':
                color = color_top_macd
                marker = 'v'
                label_text = 'MACD顶背离'
            else:
                color = color_top_volume
                marker = 'v'
                label_text = '量价顶背离'
            y_offset = -20  # 向下偏移
        else:  # bottom
            if d['subtype'] == 'macd':
                color = color_bottom_macd
                marker = '^'
                label_text = 'MACD底背离'
            else:
                color = color_bottom_volume
                marker = '^'
                label_text = '量价底背离'
            y_offset = 20   # 向上偏移

        # 标注箭头
        ax_candle.annotate(
            label_text,
            xy=(x, price),
            xytext=(0, y_offset),
            textcoords='offset points',
            fontsize=7,
            color=color,
            fontweight='bold',
            ha='center',
            va='bottom' if d['type'] == 'top' else 'top',
            arrowprops=dict(
                arrowstyle='->',
                color=color,
                lw=1.5,
                connectionstyle='arc3,rad=0'
            ),
            bbox=dict(
                boxstyle='round,pad=0.15',
                facecolor=color,
                alpha=0.12,
                edgecolor=color,
            )
        )
        # 在顶部或底部画一个小标记点
        ax_candle.scatter(x, price, color=color, s=60, marker=marker, zorder=5)

    # 添加图例
    handles = [
        mpatches.Patch(color=color_top_macd, alpha=0.4, label='MACD顶背离'),
        mpatches.Patch(color=color_bottom_macd, alpha=0.4, label='MACD底背离'),
        mpatches.Patch(color=color_top_volume, alpha=0.4, label='量价顶背离'),
        mpatches.Patch(color=color_bottom_volume, alpha=0.4, label='量价底背离'),
    ]
    ax_candle.legend(handles=handles, loc='upper left', fontsize=7, framealpha=0.8)

    # 标题
    uptrend_pct = None
    # 可加统计信息
    ax_candle.set_title(
        f"{symbol} 日K线 + 背离信号\n"
        f"共检测到 {len([d for d in divergences if d['type']=='top'])} 个顶背离, "
        f"{len([d for d in divergences if d['type']=='bottom'])} 个底背离\n"
        f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}",
        fontsize=12,
    )

    # MACD 子图标题
    ax_macd.set_title('MACD (12,26,9)')

    # 保存或显示
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[图表已保存] {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ==================== 使用示例 (main) ====================
if __name__ == "__main__":
    # 1. 获取数据
    symbol = "688131"
    start_date = "20250101"
    end_date = "20260621"
    DATA_DIR = "tsdata"

    print(f"正在获取 {symbol} 数据...")

    mytoken = os.getenv('TUSHARE_TOKEN')
    print('TUSHARE_TOKEN set:', bool(mytoken))

    manager = TushareStockDataManager(
        token=mytoken,
        data_dir=DATA_DIR,
        request_interval=1.5
    )
    df = manager.get_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)

    print(f"数据获取成功，共 {len(df)} 条记录")
    print(f"数据范围: {df.index[0]} 至 {df.index[-1]}")

    # 2. 创建背离检测器
    detector = DivergenceDetector(
        extremum_order=5,        # 左右各5根K线确定局部极值
        volume_confirm_ratio=0.95
    )

    # 3. 运行检测
    # 需要足够的预热数据（至少包含MACD计算所需的26+9个周期）
    # 但检测器内部会从极值点开始，所以我们全部传入
    divergences = detector.detect(
        close=df['close'],
        high=df['high'],
        low=df['low'],
        volume=df['volume'],
        debug=True
    )

    # 4. 输出结果
    print("\n背离信号明细:")
    for d in divergences:
        print(f"{d['date'].strftime('%Y-%m-%d')} | {d['type']} | {d['subtype']} | 价格:{d['price']:.2f} | 置信度:{d['confidence']:.0%}")

    # 5. 绘制图表
    print("\n正在生成背离信号K线图...")
    report_dir = "./reports"
    os.makedirs(report_dir, exist_ok=True)
    chart_path = f"{report_dir}/divergence_chart_{symbol}_{start_date}_{end_date}.png"
    plot_divergence_chart(df, divergences, symbol, save_path=chart_path)