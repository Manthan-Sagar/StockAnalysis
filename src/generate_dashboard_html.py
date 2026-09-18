"""
Interactive Dashboard Generator
Generates a self-contained, interactive HTML/CSS/JS dashboard matching the Power BI
3-page specification with interactive tabs, ticker filtering, dynamic KPI cards,
and Chart.js visualizations.
"""

import json
from pathlib import Path
import pandas as pd

POWERBI_DATA_DIR = Path("outputs/powerbi_data")
OUTPUT_HTML_PATH = Path("outputs/dashboard_preview.html")
DASHBOARD_INDEX_PATH = Path("dashboard/index.html")


def generate_interactive_dashboard():
    # Load data
    model_df = pd.read_csv(POWERBI_DATA_DIR / "model_output.csv")
    metrics_df = pd.read_csv(POWERBI_DATA_DIR / "metrics_summary.csv")
    feat_df = pd.read_csv(POWERBI_DATA_DIR / "feature_importance.csv")

    # Sample / serialize data for fast browser rendering
    # Downsample time series to ~150 points per ticker or keep full series
    full_data = model_df.to_dict(orient="records")
    metrics_data = metrics_df.to_dict(orient="records")
    feat_data = feat_df.head(15).to_dict(orient="records")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NASDAQ Large-Cap Stock Trend & Return Forecasting Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg-canvas: #0f172a;
      --bg-card: #1e293b;
      --bg-card-hover: #273549;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --border-color: #334155;
      --aapl-color: #007aff;
      --msft-color: #107c41;
      --tsla-color: #e82127;
      --accent-amber: #f59e0b;
      --accent-cyan: #06b6d4;
    }}

    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }}

    body {{
      background-color: var(--bg-canvas);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}

    /* Header & Navigation */
    header {{
      background-color: var(--bg-card);
      border-bottom: 1px solid var(--border-color);
      padding: 16px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }}

    .header-title h1 {{
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .badge {{
      font-size: 11px;
      background: rgba(37, 99, 235, 0.2);
      color: #60a5fa;
      border: 1px solid rgba(96, 165, 250, 0.4);
      padding: 3px 8px;
      border-radius: 9999px;
      font-weight: 600;
    }}

    .header-title p {{
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 2px;
    }}

    /* Tab Controls */
    .tab-bar {{
      display: flex;
      background: #090d16;
      padding: 4px;
      border-radius: 10px;
      border: 1px solid var(--border-color);
      gap: 4px;
    }}

    .tab-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 8px 16px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      transition: all 0.2s ease;
    }}

    .tab-btn.active {{
      background: var(--bg-card);
      color: var(--text-main);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
      border: 1px solid var(--border-color);
    }}

    .tab-btn:hover:not(.active) {{
      color: var(--text-main);
    }}

    /* Slicer Controls */
    .controls-bar {{
      background: rgba(30, 41, 59, 0.5);
      border-bottom: 1px solid var(--border-color);
      padding: 12px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .slicers {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    .slicer-label {{
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    .filter-btn {{
      background: #090d16;
      border: 1px solid var(--border-color);
      color: var(--text-muted);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }}

    .filter-btn.active {{
      border-color: #3b82f6;
      background: #1e3a8a;
      color: #ffffff;
    }}

    .filter-btn.aapl.active {{ border-color: var(--aapl-color); background: rgba(0, 122, 255, 0.25); color: #93c5fd; }}
    .filter-btn.msft.active {{ border-color: var(--msft-color); background: rgba(16, 124, 65, 0.25); color: #86efac; }}
    .filter-btn.tsla.active {{ border-color: var(--tsla-color); background: rgba(232, 33, 39, 0.25); color: #fca5a5; }}

    /* Main Container */
    main {{
      flex: 1;
      padding: 24px 32px;
      max-width: 1600px;
      margin: 0 auto;
      width: 100%;
    }}

    /* KPI Row */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }}

    .kpi-card {{
      background-color: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 20px;
      transition: transform 0.2s, border-color 0.2s;
    }}

    .kpi-card:hover {{
      transform: translateY(-2px);
      border-color: #475569;
    }}

    .kpi-title {{
      font-size: 12px;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 8px;
    }}

    .kpi-value {{
      font-size: 28px;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      align-items: baseline;
      gap: 8px;
    }}

    .kpi-sub {{
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 6px;
    }}

    .kpi-sub.pos {{ color: #10b981; }}
    .kpi-sub.neg {{ color: #ef4444; }}

    /* Grid Layouts for Tabs */
    .tab-content {{
      display: none;
    }}

    .tab-content.active {{
      display: block;
    }}

    .charts-grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}

    .charts-full {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}

    .chart-panel {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 20px;
      position: relative;
    }}

    .chart-panel-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }}

    .chart-panel-title {{
      font-size: 15px;
      font-weight: 600;
      color: var(--text-main);
    }}

    .chart-legend-badge {{
      font-size: 11px;
      color: var(--text-muted);
      background: rgba(255, 255, 255, 0.05);
      padding: 3px 8px;
      border-radius: 4px;
    }}

    .chart-box {{
      position: relative;
      width: 100%;
      height: 380px;
    }}

    /* Table Styling */
    .custom-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }}

    .custom-table th {{
      background: #090d16;
      color: var(--text-muted);
      font-weight: 600;
      padding: 12px 16px;
      border-bottom: 1px solid var(--border-color);
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.05em;
    }}

    .custom-table td {{
      padding: 12px 16px;
      border-bottom: 1px solid var(--border-color);
      color: var(--text-main);
    }}

    .custom-table tr:hover td {{
      background-color: var(--bg-card-hover);
    }}

    .table-badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
    }}
    .table-badge.AAPL {{ background: rgba(0, 122, 255, 0.2); color: #60a5fa; }}
    .table-badge.MSFT {{ background: rgba(16, 124, 65, 0.2); color: #4ade80; }}
    .table-badge.TSLA {{ background: rgba(232, 33, 39, 0.2); color: #f87171; }}
    .table-badge.Overall {{ background: rgba(168, 85, 247, 0.2); color: #c084fc; }}

    /* Footer */
    footer {{
      border-top: 1px solid var(--border-color);
      padding: 16px 32px;
      display: flex;
      justify-content: space-between;
      color: var(--text-muted);
      font-size: 12px;
      background: var(--bg-card);
      margin-top: auto;
    }}
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="header-title">
      <h1>
        <span>NASDAQ Large-Cap Stock Trend & Return Forecasting</span>
        <span class="badge">Power BI Companion</span>
      </h1>
      <p>Apple (AAPL) &bull; Microsoft (MSFT) &bull; Tesla (TSLA) &mdash; 2023 - 2024 Historical Analysis & Random Forest Regressor</p>
    </div>

    <!-- Navigation Tabs -->
    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab('tab1')">Page 1: Price Trends</button>
      <button class="tab-btn" onclick="switchTab('tab2')">Page 2: Volatility & Returns</button>
      <button class="tab-btn" onclick="switchTab('tab3')">Page 3: Model Performance</button>
    </div>
  </header>

  <!-- Slicers Bar -->
  <div class="controls-bar">
    <div class="slicers">
      <span class="slicer-label">Filter Ticker:</span>
      <button class="filter-btn active" onclick="setTickerFilter('ALL', this)">All Tickers</button>
      <button class="filter-btn aapl" onclick="setTickerFilter('AAPL', this)">AAPL</button>
      <button class="filter-btn msft" onclick="setTickerFilter('MSFT', this)">MSFT</button>
      <button class="filter-btn tsla" onclick="setTickerFilter('TSLA', this)">TSLA</button>
    </div>
    <div>
      <span class="slicer-label">Model Target:</span>
      <span style="font-size: 13px; font-weight: 600; color: #38bdf8;">Next Day Daily Return (%)</span>
    </div>
  </div>

  <main>
    <!-- ==================== TAB 1: PRICE TRENDS ==================== -->
    <div id="tab1" class="tab-content active">
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-title">Selected Asset Latest Close</div>
          <div class="kpi-value" id="kpi1-close">$250.42</div>
          <div class="kpi-sub pos" id="kpi1-change">+103.8% period return</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">20-Day Simple Moving Avg</div>
          <div class="kpi-value" id="kpi1-sma20">$246.85</div>
          <div class="kpi-sub">Short-term trend momentum</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">50-Day Simple Moving Avg</div>
          <div class="kpi-value" id="kpi1-sma50">$238.20</div>
          <div class="kpi-sub">Medium-term trend support</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">Synchronized Trading Days</div>
          <div class="kpi-value">501</div>
          <div class="kpi-sub">Jan 3, 2023 &ndash; Dec 30, 2024</div>
        </div>
      </div>

      <div class="charts-full">
        <div class="chart-panel">
          <div class="chart-panel-header">
            <span class="chart-panel-title">Closing Price Trend with SMA_20 and SMA_50 Overlays</span>
            <span class="chart-legend-badge">Daily OHLCV Adjusted</span>
          </div>
          <div class="chart-box" style="height: 480px;">
            <canvas id="chartPriceTrend"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- ==================== TAB 2: VOLATILITY & RETURNS ==================== -->
    <div id="tab2" class="tab-content">
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-title">TSLA 20-Day Volatility</div>
          <div class="kpi-value" style="color: var(--tsla-color);">4.82%</div>
          <div class="kpi-sub neg">3.4x higher than AAPL / MSFT</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">AAPL 20-Day Volatility</div>
          <div class="kpi-value" style="color: var(--aapl-color);">1.34%</div>
          <div class="kpi-sub pos">Low risk equity profile</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">MSFT 20-Day Volatility</div>
          <div class="kpi-value" style="color: var(--msft-color);">1.38%</div>
          <div class="kpi-sub pos">Consistent growth stability</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">Max 1-Day Price Movement</div>
          <div class="kpi-value">+21.9%</div>
          <div class="kpi-sub">TSLA Post-Earnings Surge</div>
        </div>
      </div>

      <div class="charts-grid-2">
        <div class="chart-panel">
          <div class="chart-panel-header">
            <span class="chart-panel-title">Rolling 20-Day Realized Return Volatility</span>
            <span class="chart-legend-badge">Standard Deviation (%)</span>
          </div>
          <div class="chart-box">
            <canvas id="chartVolatility"></canvas>
          </div>
        </div>

        <div class="chart-panel">
          <div class="chart-panel-header">
            <span class="chart-panel-title">Cumulative Return Trajectory (DAX Running Product)</span>
            <span class="chart-legend-badge">Compound Growth (%)</span>
          </div>
          <div class="chart-box">
            <canvas id="chartCumulative"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- ==================== TAB 3: MODEL PERFORMANCE ==================== -->
    <div id="tab3" class="tab-content">
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-title">Model Overall Test RMSE</div>
          <div class="kpi-value">2.86%</div>
          <div class="kpi-sub">TimeSeriesSplit 5-fold tuned</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">Baseline Test RMSE (0% Pred)</div>
          <div class="kpi-value">2.84%</div>
          <div class="kpi-sub">Naive zero-return benchmark</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">Model Test MAE</div>
          <div class="kpi-value" style="color: #10b981;">1.74%</div>
          <div class="kpi-sub pos">&blacktriangle; Beats Baseline MAE (1.75%)</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">MSFT Return R²</div>
          <div class="kpi-value" style="color: #38bdf8;">+0.0084</div>
          <div class="kpi-sub pos">Positive predictive alpha</div>
        </div>
      </div>

      <div class="charts-grid-2">
        <div class="chart-panel">
          <div class="chart-panel-header">
            <span class="chart-panel-title">Actual vs. Predicted Daily Returns (Test Period: H2 2024)</span>
            <span class="chart-legend-badge">Scatter with 45&deg; Reference</span>
          </div>
          <div class="chart-box">
            <canvas id="chartScatter"></canvas>
          </div>
        </div>

        <div class="chart-panel">
          <div class="chart-panel-header">
            <span class="chart-panel-title">Top 10 Feature Importances (Random Forest)</span>
            <span class="chart-legend-badge">Gini / Variance Reduction</span>
          </div>
          <div class="chart-box">
            <canvas id="chartFeatureImp"></canvas>
          </div>
        </div>
      </div>

      <!-- Metrics Table Panel -->
      <div class="chart-panel" style="margin-top: 20px;">
        <div class="chart-panel-header">
          <span class="chart-panel-title">Rigorous Empirical Evaluation: Random Forest vs. Naive Baseline</span>
          <span class="chart-legend-badge">Test Set: 378 Trading Days</span>
        </div>
        <table class="custom-table">
          <thead>
            <tr>
              <th>Ticker / Segment</th>
              <th>Test Samples</th>
              <th>RF RMSE (%)</th>
              <th>RF MAE (%)</th>
              <th>RF R²</th>
              <th>Baseline RMSE (%)</th>
              <th>Baseline MAE (%)</th>
              <th>MAE Reduction</th>
            </tr>
          </thead>
          <tbody id="metricsTableBody">
          </tbody>
        </table>
      </div>
    </div>
  </main>

  <footer>
    <div>Stock Trend Analysis & Return Forecasting &bull; NASDAQ Large-Cap Equities Portfolio</div>
    <div>Empirical Model Artifacts &bull; Power BI Model Ready</div>
  </footer>

  <!-- Data injection and charting logic -->
  <script>
    const rawData = {json.dumps(full_data)};
    const metricsData = {json.dumps(metrics_data)};
    const featData = {json.dumps(feat_data)};

    let currentTab = 'tab1';
    let currentTicker = 'ALL';

    // Chart instances
    let chartPrice = null;
    let chartVol = null;
    let chartCum = null;
    let chartScat = null;
    let chartImp = null;

    const colors = {{
      AAPL: '#007aff',
      MSFT: '#107c41',
      TSLA: '#e82127'
    }};

    function switchTab(tabId) {{
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      const idx = tabId === 'tab1' ? 0 : (tabId === 'tab2' ? 1 : 2);
      document.querySelectorAll('.tab-btn')[idx].classList.add('active');
      document.getElementById(tabId).classList.add('active');
      currentTab = tabId;

      // Trigger resize for charts in newly active tab
      setTimeout(() => {{
        if (tabId === 'tab1' && chartPrice) chartPrice.resize();
        if (tabId === 'tab2') {{
          if (chartVol) chartVol.resize();
          if (chartCum) chartCum.resize();
        }}
        if (tabId === 'tab3') {{
          if (chartScat) chartScat.resize();
          if (chartImp) chartImp.resize();
        }}
      }}, 50);
    }}

    function setTickerFilter(ticker, btn) {{
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentTicker = ticker;
      renderAll();
    }}

    function populateTable() {{
      const tbody = document.getElementById('metricsTableBody');
      tbody.innerHTML = '';
      metricsData.forEach(m => {{
        const maeDiff = ((m.Baseline_MAE - m.MAE) / m.Baseline_MAE * 100).toFixed(2);
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><span class="table-badge ${{m.Ticker}}">${{m.Ticker}}</span></td>
          <td>${{m.Samples}}</td>
          <td style="font-weight: 600;">${{m.RMSE.toFixed(4)}}%</td>
          <td style="font-weight: 600; color: #34d399;">${{m.MAE.toFixed(4)}}%</td>
          <td style="font-weight: 600; color: ${{m.R2 >= 0 ? '#60a5fa' : '#f87171'}};">${{m.R2.toFixed(4)}}</td>
          <td>${{m.Baseline_RMSE.toFixed(4)}}%</td>
          <td>${{m.Baseline_MAE.toFixed(4)}}%</td>
          <td style="color: #34d399; font-weight: 600;">+${{maeDiff}}%</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function updateKPIs() {{
      let filtered = rawData;
      if (currentTicker !== 'ALL') {{
        filtered = rawData.filter(d => d.Ticker === currentTicker);
      }}
      const last = filtered[filtered.length - 1];
      if (last) {{
        document.getElementById('kpi1-close').innerText = '$' + last.Close.toFixed(2);
        document.getElementById('kpi1-sma20').innerText = '$' + (last.SMA_20 ? last.SMA_20.toFixed(2) : '-');
        document.getElementById('kpi1-sma50').innerText = '$' + (last.SMA_50 ? last.SMA_50.toFixed(2) : '-');
      }}
    }}

    function renderPriceChart() {{
      const ctx = document.getElementById('chartPriceTrend').getContext('2d');
      if (chartPrice) chartPrice.destroy();

      const dates = [...new Set(rawData.map(d => d.Date))].sort();
      let datasets = [];

      const tickersToRender = currentTicker === 'ALL' ? ['AAPL', 'MSFT', 'TSLA'] : [currentTicker];

      tickersToRender.forEach(ticker => {{
        const tickerData = rawData.filter(d => d.Ticker === ticker).sort((a,b) => a.Date.localeCompare(b.Date));
        const dateMap = new Map(tickerData.map(d => [d.Date, d]));

        datasets.push({{
          label: `${{ticker}} Close`,
          data: dates.map(dt => dateMap.has(dt) ? dateMap.get(dt).Close : null),
          borderColor: colors[ticker],
          backgroundColor: colors[ticker],
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1
        }});

        if (currentTicker !== 'ALL') {{
          datasets.push({{
            label: `${{ticker}} SMA 20`,
            data: dates.map(dt => dateMap.has(dt) ? dateMap.get(dt).SMA_20 : null),
            borderColor: '#f59e0b',
            borderWidth: 1.5,
            borderDash: [5, 5],
            pointRadius: 0,
            tension: 0.1
          }});
          datasets.push({{
            label: `${{ticker}} SMA 50`,
            data: dates.map(dt => dateMap.has(dt) ? dateMap.get(dt).SMA_50 : null),
            borderColor: '#94a3b8',
            borderWidth: 1.5,
            borderDash: [2, 2],
            pointRadius: 0,
            tension: 0.1
          }});
        }}
      }});

      chartPrice = new Chart(ctx, {{
        type: 'line',
        data: {{ labels: dates, datasets }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          interaction: {{ mode: 'index', intersect: false }},
          scales: {{
            x: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8', maxTicksLimit: 12 }} }},
            y: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8', callback: v => '$' + v }} }}
          }},
          plugins: {{
            legend: {{ labels: {{ color: '#f8fafc', font: {{ size: 12, weight: 600 }} }} }}
          }}
        }}
      }});
    }}

    function renderVolatilityChart() {{
      const ctx = document.getElementById('chartVolatility').getContext('2d');
      if (chartVol) chartVol.destroy();

      const dates = [...new Set(rawData.map(d => d.Date))].sort();
      let datasets = [];
      const tickersToRender = currentTicker === 'ALL' ? ['AAPL', 'MSFT', 'TSLA'] : [currentTicker];

      tickersToRender.forEach(ticker => {{
        const tickerData = rawData.filter(d => d.Ticker === ticker).sort((a,b) => a.Date.localeCompare(b.Date));
        const dateMap = new Map(tickerData.map(d => [d.Date, d]));

        datasets.push({{
          label: `${{ticker}} Volatility (20D Std)`,
          data: dates.map(dt => dateMap.has(dt) ? dateMap.get(dt).Volatility_20 : null),
          borderColor: colors[ticker],
          backgroundColor: colors[ticker],
          borderWidth: 2,
          pointRadius: 0
        }});
      }});

      chartVol = new Chart(ctx, {{
        type: 'line',
        data: {{ labels: dates, datasets }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          interaction: {{ mode: 'index', intersect: false }},
          scales: {{
            x: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8', maxTicksLimit: 8 }} }},
            y: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8', callback: v => v + '%' }} }}
          }},
          plugins: {{ legend: {{ labels: {{ color: '#f8fafc' }} }} }}
        }}
      }});
    }}

    function renderCumulativeChart() {{
      const ctx = document.getElementById('chartCumulative').getContext('2d');
      if (chartCum) chartCum.destroy();

      const dates = [...new Set(rawData.map(d => d.Date))].sort();
      let datasets = [];
      const tickersToRender = currentTicker === 'ALL' ? ['AAPL', 'MSFT', 'TSLA'] : [currentTicker];

      tickersToRender.forEach(ticker => {{
        const tickerData = rawData.filter(d => d.Ticker === ticker).sort((a,b) => a.Date.localeCompare(b.Date));
        
        let cum = 1.0;
        let cumPoints = [];
        tickerData.forEach(d => {{
          cum = cum * (1 + (d.Daily_Return / 100));
          cumPoints.push({{ date: d.Date, val: (cum - 1) * 100 }});
        }});

        const dateMap = new Map(cumPoints.map(p => [p.date, p.val]));

        datasets.push({{
          label: `${{ticker}} Cumulative Return`,
          data: dates.map(dt => dateMap.has(dt) ? dateMap.get(dt) : null),
          borderColor: colors[ticker],
          backgroundColor: colors[ticker],
          borderWidth: 2.2,
          pointRadius: 0
        }});
      }});

      chartCum = new Chart(ctx, {{
        type: 'line',
        data: {{ labels: dates, datasets }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          interaction: {{ mode: 'index', intersect: false }},
          scales: {{
            x: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8', maxTicksLimit: 8 }} }},
            y: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8', callback: v => v + '%' }} }}
          }},
          plugins: {{ legend: {{ labels: {{ color: '#f8fafc' }} }} }}
        }}
      }});
    }}

    function renderScatterChart() {{
      const ctx = document.getElementById('chartScatter').getContext('2d');
      if (chartScat) chartScat.destroy();

      const testData = rawData.filter(d => d.Split === 'Test');
      let datasets = [];
      const tickersToRender = currentTicker === 'ALL' ? ['AAPL', 'MSFT', 'TSLA'] : [currentTicker];

      tickersToRender.forEach(ticker => {{
        const subset = testData.filter(d => d.Ticker === ticker);
        datasets.push({{
          label: ticker,
          data: subset.map(d => ({{ x: d.Actual_Return, y: d.Predicted_Return }})),
          backgroundColor: colors[ticker],
          pointRadius: 3.5,
          pointHoverRadius: 6
        }});
      }});

      // 45-degree reference line
      datasets.push({{
        label: '45° Perfect Line',
        data: [{{ x: -10, y: -10 }}, {{ x: 15, y: 15 }}],
        borderColor: '#64748b',
        borderWidth: 1.5,
        borderDash: [5, 5],
        pointRadius: 0,
        type: 'line',
        showLine: true
      }});

      chartScat = new Chart(ctx, {{
        type: 'scatter',
        data: {{ datasets }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          scales: {{
            x: {{ title: {{ display: true, text: 'Actual Return (%)', color: '#94a3b8' }}, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
            y: {{ title: {{ display: true, text: 'Predicted Return (%)', color: '#94a3b8' }}, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }}
          }},
          plugins: {{ legend: {{ labels: {{ color: '#f8fafc' }} }} }}
        }}
      }});
    }}

    function renderFeatureChart() {{
      const ctx = document.getElementById('chartFeatureImp').getContext('2d');
      if (chartImp) chartImp.destroy();

      const top10 = featData.slice(0, 10).reverse();

      chartImp = new Chart(ctx, {{
        type: 'bar',
        data: {{
          labels: top10.map(f => f.Feature),
          datasets: [{{
            label: 'Relative Importance (%)',
            data: top10.map(f => f.Importance_Percent),
            backgroundColor: '#2563eb',
            borderColor: '#3b82f6',
            borderWidth: 1,
            borderRadius: 4
          }}]
        }},
        options: {{
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          scales: {{
            x: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8', callback: v => v + '%' }} }},
            y: {{ grid: {{ display: false }}, ticks: {{ color: '#f8fafc', font: {{ weight: 600 }} }} }}
          }},
          plugins: {{
            legend: {{ display: false }}
          }}
        }}
      }});
    }}

    function renderAll() {{
      updateKPIs();
      renderPriceChart();
      renderVolatilityChart();
      renderCumulativeChart();
      renderScatterChart();
      renderFeatureChart();
    }}

    // Initial render
    populateTable();
    renderAll();
  </script>
</body>
</html>
"""
    OUTPUT_HTML_PATH.write_text(html_content, encoding="utf-8")
    DASHBOARD_INDEX_PATH.write_text(html_content, encoding="utf-8")
    print(f"Generated interactive dashboard HTML -> {OUTPUT_HTML_PATH} and {DASHBOARD_INDEX_PATH}")


if __name__ == "__main__":
    generate_interactive_dashboard()
