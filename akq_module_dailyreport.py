"""
每日报告生成模块 - 综合分析多只股票的背离与趋势，生成 HTML 报告

功能：
    - 对指定的股票列表，获取指定日期范围的数据
    - 使用 DivergenceDetector 检测 MACD/量价背离
    - 使用 TrendClassifier 识别每日趋势
    - 获取股票基本信息（名称、行业、市盈率、总市值等）
    - 生成包含 K 线图、背离标注、趋势标注的图表
    - 将以上内容汇总为一个独立的 HTML 文件

使用方法：
    from akq_module_dailyreport import DailyReportGenerator
    generator = DailyReportGenerator(token='your_tushare_token')
    generator.generate_report(
        symbols=["300724", "600519"],
        start_date="20250101",
        output_path="./report/daily_report.html"
    )
"""

import base64
import io
import os
import platform
import sys
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 GUI 问题
import matplotlib.pyplot as plt
import mplfinance as mpf

# 确保可以导入同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from akq_module_tusharedatamanager import TushareStockDataManager
from akq_module_stockinfo import StockInfoManager
from akq_module_divergence import DivergenceDetector, plot_divergence_chart
from akq_module_trendclassifier import TrendClassifier, plot_trend_chart

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# HTML 报告模板（Bootstrap 5 内联）
# ──────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>股票每日分析报告</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
            body {{ background-color: #f8f9fa; padding-top: 20px; }}
            .report-header {{ text-align: center; margin-bottom: 30px; }}
            .report-header h1 {{ color: #0d6efd; }}
            .report-meta {{ color: #6c757d; font-size: 0.9rem; }}
            .stock-section {{ margin-bottom: 40px; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 20px; }}
            .stock-title {{ font-size: 1.5rem; font-weight: bold; margin-bottom: 5px; }}
            .stock-code {{ color: #6c757d; }}
            .info-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin: 15px 0; }}
            .info-item {{ background: #f0f4ff; padding: 8px 12px; border-radius: 6px; }}
            .info-label {{ font-size: 0.75rem; color: #6c757d; }}
            .info-value {{ font-size: 1rem; font-weight: 600; color: #0d6efd; }}
            .chart-container {{ margin-top: 15px; }}
            .chart-container img {{ width: 100%; max-width: 1000px; display: block; margin: 0 auto; border-radius: 8px; border: 1px solid #dee2e6; }}
            .divergence-list {{ margin: 10px 0; }}
            .divergence-item {{ display: inline-block; padding: 3px 10px; margin: 3px; border-radius: 12px; font-size: 0.8rem; font-weight: 500; }}
            .top-macd {{ background-color: #ff5722; color: white; }}
            .top-volume {{ background-color: #e91e63; color: white; }}
            .bottom-macd {{ background-color: #4caf50; color: white; }}
            .bottom-volume {{ background-color: #2196f3; color: white; }}
            .trend-bar {{ height: 24px; border-radius: 12px; overflow: hidden; margin: 10px 0; display: flex; }}
            .trend-segment {{ display: flex; align-items: center; justify-content: center; font-size: 0.7rem; color: white; font-weight: bold; }}
            .footer {{ text-align: center; margin-top: 30px; padding: 20px; color: #6c757d; }}
        </style>
</head>
<body>
    <div class="container">
        <div class="report-header">
            <h1>📊 股票每日分析报告</h1>
            <p class="report-meta">
                生成时间: {generate_time} &nbsp;|&nbsp; 数据范围: {start_date} ~ {end_date} &nbsp;|&nbsp; 共 {stock_count} 只股票
            </p>
        </div>

        {stock_sections}

        <div class="footer">
            <p>报告由 akq_module_dailyreport 自动生成</p>
        </div>
    </div>
</body>
</html>
"""

STOCK_SECTION_TEMPLATE = """
<div class="stock-section">
    <div class="d-flex justify-content-between align-items-start">
        <div>
            <span class="stock-title">{stock_name}</span>
            <span class="stock-code">{stock_symbol}</span>
        </div>
        <span class="badge bg-primary fs-6">{industry}</span>
    </div>
    <div class="info-grid">
        <div class="info-item">
            <div class="info-label">交易所</div>
            <div class="info-value">{exchange}</div>
        </div>
        <div class="info-item">
            <div class="info-label">上市日期</div>
            <div class="info-value">{list_date}</div>
        </div>
        <div class="info-item">
            <div class="info-label">市盈率 (PE)</div>
            <div class="info-value">{pe}</div>
        </div>
        <div class="info-item">
            <div class="info-label">总市值 (亿)</div>
            <div class="info-value">{total_mv}</div>
        </div>
        <div class="info-item">
            <div class="info-label">流通市值 (亿)</div>
            <div class="info-value">{circ_mv}</div>
        </div>
        <div class="info-item">
            <div class="info-label">换手率</div>
            <div class="info-value">{turnover_rate}</div>
        </div>
        <div class="info-item">
            <div class="info-label">数据条数</div>
            <div class="info-value">{data_count}</div>
        </div>
        <div class="info-item">
            <div class="info-label">最近收盘</div>
            <div class="info-value">{last_close:.2f}</div>
        </div>
    </div>

    <!-- 背离信号摘要 -->
    <div class="divergence-list">
        <strong>背离信号:</strong>
        {divergence_badges}
        <span class="text-muted ms-2">(共 {divergence_count} 个)</span>
    </div>

    <!-- 趋势分布条 -->
    <div class="trend-bar">
        <div class="trend-segment" style="width: {uptrend_pct}%; background-color: #4caf50;">↑ {uptrend_pct:.1f}%</div>
        <div class="trend-segment" style="width: {range_pct}%; background-color: #ffc107; color: #333;">→ {range_pct:.1f}%</div>
        <div class="trend-segment" style="width: {downtrend_pct}%; background-color: #f44336;">↓ {downtrend_pct:.1f}%</div>
    </div>

    <!-- 背离图 -->
    <div class="chart-container">
        <h5>📈 背离检测图</h5>
        <img src="data:image/png;base64,{divergence_chart}" alt="背离图">
    </div>

    <!-- 趋势图 -->
    <div class="chart-container">
        <h5>📈 趋势识别图</h5>
        <img src="data:image/png;base64,{trend_chart}" alt="趋势图">
    </div>
</div>
"""

# ──────────────────────────────────────────────
# 工具函数：图 → base64
# ──────────────────────────────────────────────
def _fig_to_base64(fig) -> str:
    """将 matplotlib figure 保存为 base64 字符串（PNG 格式）"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    buf.close()
    return img_base64


# ──────────────────────────────────────────────
# 主类
# ──────────────────────────────────────────────
class DailyReportGenerator:
    """
    每日报告生成器
    """

    def __init__(self,
                 token: str,
                 data_dir: str = "tsdata",
                 stock_info_dir: str = "stock_info",
                 request_interval: float = 1.5):
        self.token = token
        self.data_dir = data_dir
        self.request_interval = request_interval

        # 初始化各个管理器
        self.data_manager = TushareStockDataManager(
            token=token,
            data_dir=data_dir,
            request_interval=request_interval
        )
        self.stock_info = StockInfoManager(
            token=token,
            data_dir=stock_info_dir,
            request_interval=request_interval * 0.8
        )
        self.divergence_detector = DivergenceDetector(
            extremum_order=5,
            volume_confirm_ratio=0.95
        )
        self.trend_classifier = TrendClassifier()

    # ── 获取财务数据（市盈率、市值等） ──
    def _get_financial_data(self, symbol: str, trade_date: str) -> Dict:
        """
        从 tushare daily_basic 获取某只股票在某个交易日的财务数据
        返回 dict: {'pe': xx, 'total_mv': xx, 'circ_mv': xx, 'turnover_rate': xx}
        """
        import tushare as ts
        pro = ts.pro_api()
        try:
            ts_code = self._to_ts_code(symbol)
            df = pro.daily_basic(ts_code=ts_code, trade_date=trade_date, fields='pe,total_mv,circ_mv,turnover_rate')
            if df is not None and not df.empty:
                row = df.iloc[0]
                return {
                    'pe': f"{row['pe']:.2f}" if pd.notna(row['pe']) else 'N/A',
                    'total_mv': f"{row['total_mv'] / 10000:.2f}" if pd.notna(row['total_mv']) else 'N/A',
                                        'circ_mv': f"{row['circ_mv'] / 10000:.2f}" if pd.notna(row['circ_mv']) else 'N/A',
                    'turnover_rate': f"{row['turnover_rate']:.2f}%" if pd.notna(row['turnover_rate']) else 'N/A',
                }
        except Exception as e:
            logger.warning(f"获取 {symbol} 财务数据失败: {e}")
        return {'pe': 'N/A', 'total_mv': 'N/A', 'circ_mv': 'N/A', 'turnover_rate': 'N/A'}

    @staticmethod
    def _to_ts_code(symbol: str) -> str:
        """将股票代码转为 tushare 格式，兼容已带后缀的输入。"""
        if symbol is None:
            return ''
        code = str(symbol).strip().upper()
        if '.' in code:
            code = code.split('.', 1)[0]
        code = code.zfill(6)
        if code.startswith(('43', '83', '87', '88', '92')):
            return f"{code}.BJ"
        if code.startswith(('688', '600', '601', '603', '605')):
            return f"{code}.SH"
        return f"{code}.SZ"

    # ── 生成背离图片（base64） ──
    def _generate_divergence_chart(self, df: pd.DataFrame, symbol: str, stock_name: str) -> str:
        """生成背离检测图，返回 base64 字符串"""
        divergences = self.divergence_detector.detect(
            close=df['close'],
            high=df['high'],
            low=df['low'],
            volume=df['volume']
        )

        # 复用原有绘图函数，通过临时文件获取图片
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            # 设置中文字体（自动适配 Windows / Linux）
            if platform.system() == 'Windows':
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
            else:
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            plot_divergence_chart(
                df=df,
                divergences=divergences,
                symbol=symbol,
                save_path=tmp_path,
                stock_name=stock_name
            )
            with open(tmp_path, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        return img_base64

    # ── 生成趋势图片（base64） ──
    def _generate_trend_chart(self, df: pd.DataFrame, symbol: str, stock_name: str) -> str:
        """生成趋势识别图，返回 base64 字符串"""
        # 计算每日趋势结果（滚动窗口，需要至少60个数据点）
        results = []
        for i in range(60, len(df)):
            close = df['close'].iloc[:i+1]
            high = df['high'].iloc[:i+1]
            low = df['low'].iloc[:i+1]
            trend, confidence = self.trend_classifier.classify(close, high, low)
            results.append({
                'date': df.index[i],
                'close': close.iloc[-1],
                'trend': trend,
                'confidence': confidence
            })
        results_df = pd.DataFrame(results)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            # 设置中文字体（自动适配 Windows / Linux）
            if platform.system() == 'Windows':
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
            else:
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            plot_trend_chart(
                df=df,
                results_df=results_df,
                symbol=symbol,
                save_path=tmp_path,
                stock_name=stock_name
            )
            with open(tmp_path, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        return img_base64

    # ── 生成单个股票的报告片段（HTML） ──
    def _process_stock(self, symbol: str, start_date: str, end_date: str) -> Optional[str]:
        """
        处理单只股票，返回 HTML 片段（若失败返回 None）
        """
        logger.info(f"正在处理股票: {symbol}")

        # 1. 获取行情数据
        try:
            df = self.data_manager.get_stock_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            return None

        if df.empty:
            logger.warning(f"{symbol} 数据为空，跳过")
            return None

        # 2. 获取股票基本信息
        info = self.stock_info.get_stock_info(symbol)
        stock_name = info.get('name', '未知') if info else '未知'
        industry = info.get('industry', '未知') if info else '未知'
        exchange = info.get('exchange', '未知') if info else '未知'
        list_date = info.get('list_date', '未知') if info else '未知'
        if isinstance(list_date, pd.Timestamp):
            list_date = list_date.strftime('%Y-%m-%d')

        # 3. 获取财务数据（使用最新交易日）
        last_trade_date = df.index[-1].strftime('%Y%m%d')
        fin = self._get_financial_data(symbol, last_trade_date)

        # 4. 检测背离
        divergences = self.divergence_detector.detect(
            close=df['close'],
            high=df['high'],
            low=df['low'],
            volume=df['volume']
        )

        # 5. 生成背离信号标签
        divergence_badges = []
        for d in divergences:
            badge_class = f"{d['type']}-{d['subtype']}"
            label = {
                'top-macd': 'MACD顶背离',
                'top-volume': '量价顶背离',
                'bottom-macd': 'MACD底背离',
                'bottom-volume': '量价底背离'
            }.get(badge_class, badge_class)
            divergence_badges.append(
                f'<span class="divergence-item {badge_class}">{label}</span>'
            )
        divergence_html = ' '.join(divergence_badges) if divergence_badges else '<span class="text-muted">无背离信号</span>'

        # 6. 计算趋势分布
        trend_results = []
        for i in range(60, len(df)):
            close = df['close'].iloc[:i+1]
            high = df['high'].iloc[:i+1]
            low = df['low'].iloc[:i+1]
            trend, _ = self.trend_classifier.classify(close, high, low)
            trend_results.append(trend)
        total_trend = len(trend_results)
        if total_trend > 0:
            uptrend_pct = trend_results.count('uptrend') / total_trend * 100
            range_pct = trend_results.count('range') / total_trend * 100
            downtrend_pct = trend_results.count('downtrend') / total_trend * 100
        else:
            uptrend_pct = range_pct = downtrend_pct = 0

        # 7. 生成图表 base64
        logger.info(f"生成 {symbol} 背离图...")
        div_chart_b64 = self._generate_divergence_chart(df, symbol, stock_name)
        logger.info(f"生成 {symbol} 趋势图...")
        trend_chart_b64 = self._generate_trend_chart(df, symbol, stock_name)

        # 8. 填充模板
        last_close = df['close'].iloc[-1]
        section = STOCK_SECTION_TEMPLATE.format(
            stock_name=stock_name,
            stock_symbol=symbol,
            industry=industry,
            exchange=exchange,
            list_date=list_date,
            pe=fin['pe'],
            total_mv=fin['total_mv'],
            circ_mv=fin['circ_mv'],
            turnover_rate=fin['turnover_rate'],
            data_count=len(df),
            last_close=last_close,
            divergence_badges=divergence_html,
            divergence_count=len(divergences),
            uptrend_pct=uptrend_pct,
            range_pct=range_pct,
            downtrend_pct=downtrend_pct,
            divergence_chart=div_chart_b64,
            trend_chart=trend_chart_b64
        )
        return section

    # ── 生成完整报告 ──
    def generate_report(self,
                        symbols: List[str],
                        start_date: str,
                        end_date: Optional[str] = None,
                        output_path: str = "./daily_report.html") -> str:
        """
        生成每日报告 HTML 文件

        Parameters:
        -----------
        symbols : List[str]
            股票代码列表（纯数字，如 ['300724', '600519']）
        start_date : str
            起始日期，格式 YYYYMMDD 或 YYYY-MM-DD
        end_date : str, optional
            结束日期，默认当天
        output_path : str
            输出 HTML 文件路径

        Returns:
        --------
        str : 生成的 HTML 文件路径
        """
        start_date_clean = start_date.replace('-', '')
        if end_date:
            end_date_clean = end_date.replace('-', '')
        else:
            end_date_clean = datetime.now().strftime('%Y%m%d')

        logger.info(f"开始生成报告，共 {len(symbols)} 只股票，日期范围 {start_date_clean} ~ {end_date_clean}")

        sections = []
        success_count = 0
        for symbol in symbols:
            section = self._process_stock(symbol, start_date_clean, end_date_clean)
            if section:
                sections.append(section)
                success_count += 1

        if not sections:
            logger.error("没有成功处理任何股票，无法生成报告")
            raise RuntimeError("没有可用的股票数据")

        # 生成时间
        generate_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        html = HTML_TEMPLATE.format(
            generate_time=generate_time,
            start_date=start_date_clean,
            end_date=end_date_clean,
            stock_count=success_count,
            stock_sections='\n'.join(sections)
        )

        # 写入文件
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding='utf-8')
        logger.info(f"✅ 报告已生成: {output.absolute()}")
        return str(output.absolute())


# ──────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import os

    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        print("请设置 TUSHARE_TOKEN 环境变量")
        exit(1)

    # 示例 symbols
    symbols = ["002901", "002755", "000933", "000807", "688629", "688690", "301393", "301095", "002738", "301358", "688131",]
    start_date = "20260101"

    generator = DailyReportGenerator(
        token=token,
        data_dir="tsdata",
        stock_info_dir="stock_info",
        request_interval=0.5
    )

    # 生成带日期时间的报告文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./reports/daily_report_{timestamp}.html"

    output = generator.generate_report(
        symbols=symbols,
        start_date=start_date,
        output_path=output_path
    )
    print(f"报告已保存至: {output}")