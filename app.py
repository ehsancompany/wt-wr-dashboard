"""
Standalone WaveTrend + Williams %R Dashboard
Willy (WR21+EMA13) only — STD removed
UTC+2 timestamps on x-axis
Run: python app.py
"""
import os
import ccxt
import numpy as np
import threading
import time
from flask import Flask, jsonify, render_template_string
from datetime import datetime, timezone, timedelta

app = Flask(__name__)

COINS      = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT',
    'XRP/USDT', 'TRX/USDT', 'DOGE/USDT', 'ADA/USDT',
    'AVAX/USDT', 'LINK/USDT',
]
TIMEFRAMES = ['1m', '3m', '5m', '15m', '1h']
LIMIT      = 500   # increased from 100 — needed for WT warmup

exchange = ccxt.binance({
    'enableRateLimit': True,
    'timeout': 10000,   
    'options': {'defaultType': 'spot'},
})
UTC2 = timezone(timedelta(hours=2))

state = {
    'coin':         'BTC/USDT',
    'data':         {},
    'last_updated': '—',
    'error':        '',
    'interval':     15,
    'loading':      False,
}

# ─── WaveTrend [LazyBear] ─────────────────────────────────────
def wavetrend_lazybear(highs, lows, closes, n1=10, n2=21):
    n = len(closes)
    if n < n2 + 4:
        return [], []
    h  = np.array(highs,  dtype=float)
    l  = np.array(lows,   dtype=float)
    c  = np.array(closes, dtype=float)
    ap = (h + l + c) / 3.0

    def ema(src, period):
        k   = 2.0 / (period + 1)
        out = np.empty(len(src))
        out[0] = src[0]
        for i in range(1, len(src)):
            out[i] = src[i] * k + out[i-1] * (1 - k)
        return out

    def sma(src, period):
        out = np.full(len(src), np.nan)
        for i in range(period - 1, len(src)):
            out[i] = np.mean(src[i - period + 1:i + 1])
        return out

    esa = ema(ap, n1)
    d   = ema(np.abs(ap - esa), n1)
    d   = np.where(d < 1e-10, 1e-10, d)
    ci  = (ap - esa) / (0.015 * d)
    wt1 = ema(ci, n2)
    wt2 = sma(wt1, 4)   # proper NaN-aware SMA, no rounding mid-calc

    # Round only at boundary
    wt1_out = [None if np.isnan(x) else round(float(x), 4) for x in wt1]
    wt2_out = [None if np.isnan(x) else round(float(x), 4) for x in wt2]
    return wt1_out, wt2_out


# ─── The Willy — WR(21) + EMA(13) ────────────────────────────
def williams_r_willy(highs, lows, closes, period=21, ema_period=13):
    h, l, c = np.array(highs, dtype=float), np.array(lows, dtype=float), np.array(closes, dtype=float)
    n   = len(c)
    raw = np.full(n, np.nan)
    for i in range(period - 1, n):
        hh    = np.max(h[i - period + 1 : i + 1])
        ll    = np.min(l[i - period + 1 : i + 1])
        denom = hh - ll
        raw[i] = -50.0 if denom < 1e-10 else 100.0 * (c[i] - hh) / denom

    k      = 2.0 / (ema_period + 1)
    smooth = np.full(n, np.nan)
    first  = next((i for i, v in enumerate(raw) if not np.isnan(v)), None)
    if first is not None:
        smooth[first] = raw[first]
        for i in range(first + 1, n):
            if not np.isnan(raw[i]):
                prev      = smooth[i-1] if not np.isnan(smooth[i-1]) else raw[i]
                smooth[i] = raw[i] * k + prev * (1 - k)

    raw_l    = [None if np.isnan(x) else round(float(x), 4) for x in raw]
    smooth_l = [None if np.isnan(x) else round(float(x), 4) for x in smooth]
    return raw_l, smooth_l


# ─── Fetch one timeframe ──────────────────────────────────────
def fetch_tf(coin, tf):
    try:
        ohlcv = exchange.fetch_ohlcv(coin, timeframe=tf, limit=LIMIT)
    except Exception as e:
        print("❌ FETCH ERROR:", coin, tf, str(e))
        return None

    if not ohlcv:
        return None

    try:
        ts     = [o[0] for o in ohlcv]
        highs  = [o[2] for o in ohlcv]
        lows   = [o[3] for o in ohlcv]
        closes = [o[4] for o in ohlcv]

        wt1, wt2                = wavetrend_lazybear(highs, lows, closes)
        willy_raw, willy_smooth = williams_r_willy(highs, lows, closes)

        pad = len(closes) - len(wt1)
        wt1 = [None] * pad + wt1
        wt2 = [None] * pad + wt2

        cur_wt1 = wt1[-1] if wt1 and wt1[-1] is not None else 0
        cur_wt2 = wt2[-1] if wt2 and wt2[-1] is not None else 0

        def wr_sig(w):
            if w is None: return 'neutral'
            if w < -80:   return 'oversold'
            if w > -20:   return 'overbought'
            return 'neutral'

        def wt_sig(w1, w2):
            if w1 is None or w2 is None: return 'neutral'
            if w1 > w2 and w1 < -53:    return 'buy'
            if w1 < w2 and w1 > 53:     return 'sell'
            return 'neutral'

        def fmt_label(ts_ms, tf):
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC2)
            if tf == '1h':
                return dt.strftime('%m/%d %H:%M')
            return dt.strftime('%H:%M')

        labels = [fmt_label(t, tf) for t in ts]

        return {
            'labels':        labels,
            'closes':        closes,
            'wt1':           wt1,
            'wt2':           wt2,
            'cur_wt1':       round(cur_wt1, 2),
            'cur_wt2':       round(cur_wt2, 2),
            'wt_signal':     wt_sig(cur_wt1, cur_wt2),
            'willy_raw':     willy_raw,
            'willy_smooth':  willy_smooth,
            'cur_willy':     round(willy_raw[-1], 2)    if willy_raw    and willy_raw[-1]    is not None else -50,
            'cur_willy_ema': round(willy_smooth[-1], 2) if willy_smooth and willy_smooth[-1] is not None else -50,
            'willy_signal':  wr_sig(willy_raw[-1] if willy_raw else None),
            'price':         closes[-1] if closes else 0,
        }

    except Exception as e:
        print("❌ PROCESS ERROR:", coin, tf, str(e))
        return None


# ─── Background loop ──────────────────────────────────────────
def fetch_all_parallel(coin):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_tf, coin, tf): tf for tf in TIMEFRAMES}
        for future in as_completed(futures):
            tf = futures[future]
            try:
                r = future.result()
                if r:
                    results[tf] = r
            except Exception:
                pass
    return results

def update_loop():
    while True:
        try:
            state['loading']      = True
            state['data']         = fetch_all_parallel(state['coin'])
            state['last_updated'] = datetime.now(UTC2).strftime('%H:%M:%S')
            state['error']        = ''
        except Exception as e:
            state['error'] = str(e)
        finally:
            state['loading'] = False
        time.sleep(max(1, state['interval']))


# ─── Routes ───────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/data')
def data():
    return jsonify(state)

@app.route('/set_interval/<int:sec>')
def set_interval(sec):
    state['interval'] = max(5, min(60, sec))
    return jsonify({'ok': True, 'interval': state['interval']})

@app.route('/switch_coin/<coin>')
def switch_coin(coin):
    decoded = coin.replace('_', '/')
    if decoded in COINS:
        state['coin']    = decoded
        state['data']    = {}
        state['loading'] = True
        threading.Thread(target=_immediate_fetch, daemon=True).start()
    return jsonify({'ok': True, 'coin': state['coin']})

def _immediate_fetch():
    try:
        state['data']         = fetch_all_parallel(state['coin'])
        state['last_updated'] = datetime.now(UTC2).strftime('%H:%M:%S')
        state['error']        = ''
    except Exception as e:
        state['error'] = str(e)
    finally:
        state['loading'] = False


# ─── HTML ─────────────────────────────────────────────────────
HTML = '''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>WT + Willy Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&display=swap');
:root {
  --bg:      #080c10; --surface: #0e1318; --border: #1c2530;
  --text:    #c8d8e8; --muted:   #4a6070;
  --blue:    #38bdf8; --green:   #22d3a0; --red:    #f43f5e;
  --cyan:    #22d0ee; --grey:    #94a3b8;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:'JetBrains Mono',monospace; padding:20px; min-height:100vh; }

.header { display:flex; align-items:center; justify-content:space-between; margin-bottom:20px; padding-bottom:16px; border-bottom:1px solid var(--border); flex-wrap:wrap; gap:12px; }
.header h1 { font-size:16px; font-weight:600; color:var(--blue); letter-spacing:2px; text-transform:uppercase; }
.header-right { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.updated { font-size:11px; color:var(--muted); }
.coins { display:flex; gap:6px; flex-wrap:wrap; }
.coin-btn { padding:5px 14px; border-radius:4px; border:1px solid var(--border); background:transparent; color:var(--muted); font-family:'JetBrains Mono',monospace; font-size:12px; cursor:pointer; transition:all .15s; }
.coin-btn:hover, .coin-btn.active { border-color:var(--blue); color:var(--blue); background:rgba(56,189,248,.08); }

/* ── Loading overlay ── */
#loading-overlay {
  display:none; position:fixed; inset:0;
  background:rgba(8,12,16,0.78); backdrop-filter:blur(3px);
  z-index:999; align-items:center; justify-content:center; flex-direction:column; gap:18px;
}
#loading-overlay.visible { display:flex; }
.spinner { width:40px; height:40px; border:3px solid var(--border); border-top-color:var(--blue); border-radius:50%; animation:spin .7s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
.loading-coin { font-size:14px; font-weight:700; color:var(--blue); letter-spacing:3px; }
.loading-sub  { font-size:10px; color:var(--muted); letter-spacing:2px; }

.summary { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-bottom:20px; }
.summary-card { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:12px 14px; position:relative; overflow:hidden; }
.summary-card::before { content:''; position:absolute; top:0;left:0;right:0; height:2px; background:var(--border); transition:background .3s; }
.summary-card.buy::before, .summary-card.oversold::before   { background:var(--green); }
.summary-card.sell::before,.summary-card.overbought::before { background:var(--red); }
.tf-label  { font-size:10px; color:var(--muted); letter-spacing:1px; margin-bottom:8px; }
.price-val { font-size:13px; color:var(--text); margin-bottom:6px; }
.indicator-row { display:flex; justify-content:space-between; margin-top:4px; font-size:11px; }
.sig-badge { padding:2px 8px; border-radius:3px; font-size:10px; font-weight:600; letter-spacing:.5px; }
.sig-buy,.sig-oversold    { background:rgba(34,211,160,.15); color:var(--green); border:1px solid rgba(34,211,160,.3); }
.sig-sell,.sig-overbought { background:rgba(244,63,94,.15);  color:var(--red);   border:1px solid rgba(244,63,94,.3); }
.sig-neutral              { background:rgba(74,96,112,.2);   color:var(--muted); border:1px solid var(--border); }

.panels { display:flex; flex-direction:column; gap:16px; }
.panel  { background:var(--surface); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
.panel-header { display:flex; align-items:center; justify-content:space-between; padding:12px 16px; border-bottom:1px solid var(--border); background:rgba(14,19,24,.8); flex-wrap:wrap; gap:8px; }
.panel-tf    { font-size:13px; font-weight:700; color:var(--blue); }
.panel-stats { display:flex; gap:20px; font-size:11px; align-items:center; }
.stat        { display:flex; flex-direction:column; align-items:flex-end; gap:2px; }
.stat-label  { color:var(--muted); font-size:10px; }
.stat-val    { font-weight:600; }

.panel-charts { display:grid; grid-template-columns:1fr 1fr; }
.chart-section { padding:12px 16px 8px; }
.chart-section:first-child { border-right:1px solid var(--border); }
.chart-title  { font-size:10px; color:var(--muted); letter-spacing:1px; text-transform:uppercase; }
.chart-legend { display:flex; gap:12px; margin:4px 0 4px; }
.legend-item  { display:flex; align-items:center; gap:4px; font-size:9px; color:var(--muted); }
.legend-dot   { width:18px; height:2px; border-radius:1px; }
/* taller wrap to fit x-axis labels */
.chart-wrap   { position:relative; height:110px; }
.zone-ob { position:absolute; top:0;    right:4px; font-size:9px; color:rgba(244,63,94,.5); pointer-events:none; }
.zone-os { position:absolute; bottom:18px; right:4px; font-size:9px; color:rgba(34,211,160,.5); pointer-events:none; }

.loading { text-align:center; padding:60px; color:var(--muted); font-size:13px; letter-spacing:2px; }
.dot-anim::after { content:''; animation:dots 1.5s steps(4,end) infinite; }
@keyframes dots { 0%,20%{content:''} 40%{content:'.'} 60%{content:'..'} 80%,100%{content:'...'} }
::-webkit-scrollbar{width:4px} ::-webkit-scrollbar-track{background:var(--bg)} ::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
</style>
</head>
<body>

<div id="loading-overlay">
  <div class="spinner"></div>
  <div class="loading-coin" id="loading-coin-name">—</div>
  <div class="loading-sub">FETCHING DATA</div>
</div>

<div class="header">
  <h1>⚡ WT · Willy · Multi-Timeframe</h1>
  <div class="header-right">
    <div class="coins">
      <button class="coin-btn active" id="cb-BTC"  onclick="switchCoin('BTC/USDT')">BTC</button>
      <button class="coin-btn"        id="cb-ETH"  onclick="switchCoin('ETH/USDT')">ETH</button>
      <button class="coin-btn"        id="cb-BNB"  onclick="switchCoin('BNB/USDT')">BNB</button>
      <button class="coin-btn"        id="cb-SOL"  onclick="switchCoin('SOL/USDT')">SOL</button>
      <button class="coin-btn"        id="cb-XRP"  onclick="switchCoin('XRP/USDT')">XRP</button>
      <button class="coin-btn"        id="cb-TRX"  onclick="switchCoin('TRX/USDT')">TRX</button>
      <button class="coin-btn"        id="cb-DOGE" onclick="switchCoin('DOGE/USDT')">DOGE</button>
      <button class="coin-btn"        id="cb-ADA"  onclick="switchCoin('ADA/USDT')">ADA</button>
      <button class="coin-btn"        id="cb-AVAX" onclick="switchCoin('AVAX/USDT')">AVAX</button>
      <button class="coin-btn"        id="cb-LINK" onclick="switchCoin('LINK/USDT')">LINK</button>
    </div>
    <div style="display:flex;align-items:center;gap:8px">
      <span style="font-size:10px;color:var(--muted)">UPDATE</span>
      <input type="range" min="5" max="60" value="5" id="interval-slider"
        style="width:80px;accent-color:var(--blue);cursor:pointer"
        oninput="setUpdateInterval(parseInt(this.value))">
      <span id="interval-val" style="font-size:11px;color:var(--blue);width:28px">5s</span>
    </div>
    <div class="updated" id="updated">—</div>
  </div>
</div>

<div class="summary" id="summary"><div class="loading dot-anim">Loading</div></div>
<div class="panels"  id="panels"></div>

<script>
const TFS = ['1m','3m','5m','15m','1h'];
const charts = {};

Chart.defaults.color       = '#4a6070';
Chart.defaults.borderColor = '#1c2530';

const sigClass = s => (!s || s==='neutral') ? 'sig-neutral' : 'sig-'+s;
const sigLabel = s => ({buy:'BUY ZONE',sell:'SELL ZONE',oversold:'OVERSOLD',overbought:'OVERBOUGHT',neutral:'NEUTRAL'})[s]||'NEUTRAL';

// WT1 always blue, WT2 always red — fixed colors, no value-based switching
const WT1_COLOR = '#38bdf8';
const WT2_COLOR = '#f43f5e';

const colorWr  = v => v < -80 ? '#22d3a0' : v > -20 ? '#f43f5e' : '#94a3b8';

// ── DOM helpers ───────────────────────────────────────────────
function setEl(id, val, color) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = (val === null || val === undefined) ? '—' : val;
  if (color) el.style.color = color;
}
function setBadge(id, text, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.className   = 'sig-badge ' + cls;
}

// ── X-axis tick reduction helper ──────────────────────────────
// Show ~8 evenly spaced labels regardless of data length
function xTickCallback(val, index, ticks) {
  const total = ticks.length;
  const step  = Math.max(1, Math.floor(total / 8));
  if (index % step === 0 || index === total - 1) return this.getLabelForValue(val);
  return '';
}

// ── Build DOM ─────────────────────────────────────────────────
function buildPanels() {
  document.getElementById('summary').innerHTML = TFS.map(tf => `
    <div class="summary-card" id="sc-${tf}">
      <div class="tf-label">${tf.toUpperCase()}</div>
      <div class="price-val" id="sc-price-${tf}">—</div>
      <div class="indicator-row">
        <span class="stat-label">WT1</span>
        <span id="sc-wt1-${tf}" style="font-size:12px">—</span>
      </div>
      <div class="indicator-row">
        <span class="stat-label">WR</span>
        <span id="sc-wr-${tf}" style="font-size:12px">—</span>
      </div>
      <div class="indicator-row">
        <span class="stat-label">EMA</span>
        <span id="sc-wrema-${tf}" style="font-size:12px;color:#22d0ee">—</span>
      </div>
      <div class="indicator-row" style="margin-top:6px">
        <span class="sig-badge sig-neutral" id="sc-wtsig-${tf}">WT —</span>
        <span class="sig-badge sig-neutral" id="sc-wrbadge-${tf}">WR —</span>
      </div>
    </div>`).join('');

  document.getElementById('panels').innerHTML = TFS.map(tf => `
    <div class="panel" id="panel-${tf}">
      <div class="panel-header">
        <span class="panel-tf">${tf.toUpperCase()}</span>
        <div class="panel-stats">
          <div class="stat"><span class="stat-label">WT1</span><span class="stat-val" id="ph-wt1-${tf}" style="color:${WT1_COLOR}">—</span></div>
          <div class="stat"><span class="stat-label">WT2</span><span class="stat-val" id="ph-wt2-${tf}" style="color:${WT2_COLOR}">—</span></div>
          <div class="stat"><span class="stat-label">WR</span> <span class="stat-val" id="ph-wr-${tf}">—</span></div>
          <div class="stat"><span class="stat-label">EMA</span><span class="stat-val" id="ph-wrema-${tf}" style="color:#22d0ee">—</span></div>
          <div class="stat"><span class="stat-label">WT SIG</span><span class="sig-badge sig-neutral" id="ph-wtsig-${tf}">—</span></div>
          <div class="stat"><span class="stat-label">WR SIG</span><span class="sig-badge sig-neutral" id="ph-wrsig-${tf}">—</span></div>
        </div>
      </div>
      <div class="panel-charts">
        <!-- WaveTrend -->
        <div class="chart-section">
          <div class="chart-title">WaveTrend Oscillator [LazyBear]</div>
          <div class="chart-legend">
            <div class="legend-item"><div class="legend-dot" style="background:${WT1_COLOR}"></div>WT1</div>
            <div class="legend-item"><div class="legend-dot" style="background:${WT2_COLOR}"></div>WT2</div>
          </div>
          <div class="chart-wrap"><canvas id="wt-canvas-${tf}"></canvas></div>
        </div>
        <!-- The Willy -->
        <div class="chart-section">
          <div class="chart-title">The Willy&nbsp;[ WR 21 · EMA 13 ]</div>
          <div class="chart-legend">
            <div class="legend-item"><div class="legend-dot" style="background:#94a3b8"></div>WR(21)</div>
            <div class="legend-item"><div class="legend-dot" style="background:#22d0ee"></div>EMA(13)</div>
          </div>
          <div class="chart-wrap" style="position:relative">
            <div class="zone-ob">OB −20</div>
            <div class="zone-os">OS −80</div>
            <canvas id="wr-canvas-${tf}"></canvas>
          </div>
        </div>
      </div>
    </div>`).join('');

  function hLine(yVal, color, dash) {
    return { type:'line', yMin:yVal, yMax:yVal,
      borderColor:color, borderWidth:1, borderDash:dash||[] };
  }

  // Shared x-axis config (labels visible, reduced ticks)
  const xAxisCfg = {
    display: true,
    ticks: {
      color: '#4a6070',
      font: { size: 8 },
      maxRotation: 0,
      autoSkip: false,
      callback: xTickCallback,
    },
    grid: { color: 'rgba(28,37,48,.5)' }
  };

  TFS.forEach(tf => {
    charts['wt-'+tf] = new Chart(document.getElementById('wt-canvas-'+tf).getContext('2d'), {
      type: 'line',
      data: { labels:[], datasets:[
        { label:'WT1', data:[], borderColor: WT1_COLOR, borderWidth:1.5, pointRadius:0, tension:0.3, fill:false },
        { label:'WT2', data:[], borderColor: WT2_COLOR, borderWidth:1,   pointRadius:0, tension:0.3, borderDash:[3,2], fill:false },
      ]},
      options:{ animation:false, responsive:true, maintainAspectRatio:false,
        plugins:{
          legend:{ display:false },
          annotation:{ annotations:{
            ob1: hLine( 60, 'rgba(244,63,94,0.5)'),
            ob2: hLine( 53, 'rgba(244,63,94,0.25)', [3,3]),
            os1: hLine(-60, 'rgba(34,211,160,0.5)'),
            os2: hLine(-53, 'rgba(34,211,160,0.25)', [3,3]),
            zero: hLine(0,  'rgba(74,96,112,0.3)',   [2,2]),
          }}
        },
        scales:{
          x: xAxisCfg,
          y: { ticks:{ color:'#4a6070', font:{size:9}, maxTicksLimit:5 }, grid:{ color:'rgba(28,37,48,.8)' } }
        }
      }
    });

    charts['wr-'+tf] = new Chart(document.getElementById('wr-canvas-'+tf).getContext('2d'), {
      type: 'line',
      data: { labels:[], datasets:[
        { label:'WR',  data:[], borderColor:'#94a3b8', borderWidth:1,   pointRadius:0, tension:0.1, fill:false },
        { label:'EMA', data:[], borderColor:'#22d0ee', borderWidth:1.8, pointRadius:0, tension:0.4, fill:false },
      ]},
      options:{ animation:false, responsive:true, maintainAspectRatio:false,
        plugins:{
          legend:{ display:false },
          annotation:{ annotations:{
            ob:  hLine(-20, 'rgba(244,63,94,0.5)'),
            os:  hLine(-80, 'rgba(34,211,160,0.5)'),
            mid: hLine(-50, 'rgba(74,96,112,0.2)', [2,2]),
          }}
        },
        scales:{
          x: xAxisCfg,
          y: { min:-100, max:0,
            ticks:{ color:'#4a6070', font:{size:9}, maxTicksLimit:5, callback: v => v+'%' },
            grid:{ color:'rgba(28,37,48,.8)' }
          }
        }
      }
    });
  });
}

// ── Main update ───────────────────────────────────────────────
function updateCharts(data) {
  window._lastData = data;
  TFS.forEach(tf => {
    const d = data[tf];
    if (!d) return;

    const sc = document.getElementById('sc-'+tf);
    if (sc) sc.className = 'summary-card ' + (d.wt_signal !== 'neutral' ? d.wt_signal : d.willy_signal);

    setEl('sc-price-'+tf, '$' + d.price.toLocaleString());
    setEl('sc-wt1-'+tf,   d.cur_wt1, WT1_COLOR);
    setBadge('sc-wtsig-'+tf,   'WT '  + sigLabel(d.wt_signal),    sigClass(d.wt_signal));
    setEl('sc-wr-'+tf,    d.cur_willy,     colorWr(d.cur_willy));
    setEl('sc-wrema-'+tf, d.cur_willy_ema, '#22d0ee');
    setBadge('sc-wrbadge-'+tf, 'WR '  + sigLabel(d.willy_signal), sigClass(d.willy_signal));

    setEl('ph-wt1-'+tf, d.cur_wt1, WT1_COLOR);
    setEl('ph-wt2-'+tf, d.cur_wt2, WT2_COLOR);
    setBadge('ph-wtsig-'+tf, sigLabel(d.wt_signal), sigClass(d.wt_signal));

    setEl('ph-wr-'+tf,    d.cur_willy,     colorWr(d.cur_willy));
    setEl('ph-wrema-'+tf, d.cur_willy_ema, '#22d0ee');
    setBadge('ph-wrsig-'+tf, sigLabel(d.willy_signal), sigClass(d.willy_signal));

    const wtChart = charts['wt-'+tf];
    if (wtChart) {
      wtChart.data.labels           = d.labels;
      wtChart.data.datasets[0].data = d.wt1;
      wtChart.data.datasets[1].data = d.wt2;
      // WT1 always blue, WT2 always red — no color change based on value
      wtChart.data.datasets[0].borderColor = WT1_COLOR;
      wtChart.data.datasets[1].borderColor = WT2_COLOR;
      wtChart.update('none');
    }

    const wrChart = charts['wr-'+tf];
    if (wrChart) {
      wrChart.data.labels           = d.labels;
      wrChart.data.datasets[0].data = d.willy_raw;
      wrChart.data.datasets[1].data = d.willy_smooth;
      wrChart.data.datasets[0].borderColor = colorWr(d.cur_willy);
      wrChart.update('none');
    }
  });
}

function switchCoin(coin) {
  const sym = coin.split('/')[0];
  document.querySelectorAll('.coin-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('cb-' + sym);
  if (btn) btn.classList.add('active');

  document.getElementById('loading-coin-name').textContent = sym + ' / USDT';
  document.getElementById('loading-overlay').classList.add('visible');

  fetch('/switch_coin/' + coin.replace('/', '_')).then(() => {
    const fastPoll = setInterval(() => {
      fetch('/data').then(r => r.json()).then(d => {
        if (!d.loading && d.data && Object.keys(d.data).length > 0) {
          clearInterval(fastPoll);
          document.getElementById('loading-overlay').classList.remove('visible');
          updateCharts(d.data);
          document.getElementById('updated').textContent = 'Updated ' + d.last_updated;
        }
      }).catch(() => {});
    }, 500);
    setTimeout(() => {
      clearInterval(fastPoll);
      document.getElementById('loading-overlay').classList.remove('visible');
    }, 15000);
  });
}

buildPanels();

function poll() {
  fetch('/data').then(r => r.json()).then(d => {
    document.getElementById('updated').textContent = 'Updated ' + d.last_updated;
    if (d.error) document.getElementById('updated').textContent = '⚠ ' + d.error;
    if (d.data && Object.keys(d.data).length > 0) updateCharts(d.data);
  }).catch(() => {});
}

poll();
let pollInterval = setInterval(poll, 5000);

function setUpdateInterval(sec) {
  const safe = Math.max(5, sec);
  clearInterval(pollInterval);
  pollInterval = setInterval(poll, safe * 1000);
  fetch('/set_interval/' + safe);
  document.getElementById('interval-val').textContent = safe + 's';
}
</script>
</body>
</html>
'''

# ✅ start background loop (for Render / gunicorn)
threading.Thread(target=update_loop, daemon=True).start()

if __name__ == '__main__':
    print("🌐 WT + WR Dashboard starting...")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)