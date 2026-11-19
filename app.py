from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from pathlib import Path
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import re
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = Path(__file__).parent
DIST_DIR = BASE_DIR / 'frontend' / 'dist'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
app.config['STOCK_OPTION_EPOCHS'] = 25
app.config['OPTION_RESULTS_DIR'] = str(BASE_DIR / 'outputs' / 'comparison')
app.config['NEWS_RESULTS_DIR'] = str(BASE_DIR / 'outputs' / 'news_stock_live')
OPTION_CHARTS = {
    'overall_comparison.png': ('Overall pricing error', 'Compare RMSE and MAE on the same held-out contracts. Lower bars mean smaller errors.'),
    'actual_vs_predicted.png': ('Predicted vs. actual prices', 'Points closer to the dashed diagonal are more accurate. All four methods use the same test contracts.'),
    'error_across_moneyness.png': ('Error across moneyness', 'Compare absolute pricing errors as the stock-to-strike ratio changes.'),
    'moneyness_comparison.png': ('Accuracy by contract category', 'OTM: S/K < 0.97; ATM: 0.97 ≤ S/K < 1.03; ITM: S/K ≥ 1.03.'),
    'loss_curves.png': ('Neural network learning curves', 'Validation MSE during training. Direct ANN and LSTM learn standardized price/strike; residual ANN learns standardized error/strike, so their loss magnitudes are not directly comparable. Black-Scholes has no training curve.'),
}

# ─── Helpers ────────────────────────────────────────────────────────────────

def clean_symbol(raw: str) -> str:
    raw = raw.strip().upper()
    if not re.fullmatch(r'[A-Z0-9&-]+(?:\.NS)?', raw):
        raise ValueError("Enter an NSE symbol such as RELIANCE or TCS.NS.")
    if not raw.endswith(".NS"):
        raw = raw + ".NS"
    return raw


def safe_get(d: dict, *keys, default="N/A"):
    for k in keys:
        v = d.get(k)
        if v is not None and v != "":
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                continue
            return v
    return default


def format_large(num, default="N/A"):
    if num == "N/A" or num is None:
        return "N/A"
    try:
        num = float(num)
        if not np.isfinite(num):
            return "N/A"
        if num >= 1e12:
            return f"₹{num/1e12:.2f} L Cr."
        if num >= 1e7:
            return f"₹{num/1e7:.2f} Cr."
        if num >= 1e5:
            return f"₹{num/1e5:.2f} L"
        return f"₹{num:,.2f}"
    except Exception:
        return "N/A"


def prepare_sequences(scaled_data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])
        y.append([scaled_data[i][0], scaled_data[i][3]])  # Open, Close
    return np.array(X), np.array(y)


def inverse_pred(pred_arr, scaler):
    dummy = np.zeros((len(pred_arr), 5))
    dummy[:, 0] = pred_arr[:, 0]
    dummy[:, 3] = pred_arr[:, 1]
    return scaler.inverse_transform(dummy)


# ─── Prediction Engine ───────────────────────────────────────────────────────

def run_prediction(symbol: str):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    ticker = yf.Ticker(symbol)
    # yfinance treats `end` as exclusive, so end=today drops today's bar and
    # leaves the forecast a trading day behind. Use tomorrow to include the most
    # recent available session (today's close once Yahoo posts it).
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    df = yf.download(symbol, start="2021-01-01", end=end, progress=False)

    if df.empty or len(df) < 120:
        raise ValueError(f"Not enough data for {symbol}. Try a different stock.")

    # Flatten MultiIndex if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].replace(
        [np.inf, -np.inf], np.nan).dropna().sort_index()
    if len(df) < 120 or (df['Close'] <= 0).any():
        raise ValueError(f"Not enough valid price data for {symbol}.")
    try:
        info = ticker.info or {}
    except Exception:
        info = {}  # Optional fundamentals must not block the forecast.

    # ── Feature matrix ──────────────────────────────────────────────────────
    stock_data = df[["Open", "High", "Low", "Close", "Volume"]].values

    train_end = int(0.7 * len(stock_data))
    split = int(0.8 * len(stock_data))
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(stock_data[:train_end])
    test_scaled  = scaler.transform(stock_data[train_end:])
    scaled_data  = np.vstack((train_scaled, test_scaled))

    X, y = prepare_sequences(scaled_data)
    n_train, n_val_end = train_end - 60, split - 60
    X_train, X_val, X_test = X[:n_train], X[n_train:n_val_end], X[n_val_end:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_val_end], y[n_val_end:]

    # ── LSTM ─────────────────────────────────────────────────────────────────
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.1),
        LSTM(32),
        Dropout(0.1),
        Dense(2)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32,
                   validation_data=(X_val, y_val), shuffle=False,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=5,
                                            restore_best_weights=True)], verbose=0)

    # ── GRU ──────────────────────────────────────────────────────────────────
    gru_model = Sequential([
        GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.1),
        GRU(32),
        Dropout(0.1),
        Dense(2)
    ])
    gru_model.compile(optimizer='adam', loss='mse')
    gru_model.fit(X_train, y_train, epochs=50, batch_size=32,
                  validation_data=(X_val, y_val), shuffle=False,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=5,
                                           restore_best_weights=True)], verbose=0)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    lstm_pred_s = lstm_model.predict(X_test, verbose=0)
    gru_pred_s  = gru_model.predict(X_test, verbose=0)

    lstm_rmse = float(np.sqrt(mean_squared_error(y_test, lstm_pred_s)))
    gru_rmse  = float(np.sqrt(mean_squared_error(y_test, gru_pred_s)))

    lstm_val = mean_squared_error(y_val, lstm_model.predict(X_val, verbose=0))
    gru_val = mean_squared_error(y_val, gru_model.predict(X_val, verbose=0))
    best_name = 'LSTM' if lstm_val <= gru_val else 'GRU'
    best_model = lstm_model if best_name == 'LSTM' else gru_model
    best_pred_s = lstm_pred_s if best_name == 'LSTM' else gru_pred_s

    best_real = inverse_pred(best_pred_s, scaler)

    # Explainability: permutation feature importance over the OHLCV inputs. Each
    # channel is shuffled across the test sequences and the RMSE increase measured.
    feat_names = ['Open', 'High', 'Low', 'Close', 'Volume']
    base_rmse_fi = float(np.sqrt(mean_squared_error(y_test, best_pred_s)))
    rng_fi = np.random.default_rng(42)
    raw_fi = []
    for j in range(X_test.shape[2]):
        xp = X_test.copy()
        xp[:, :, j] = X_test[rng_fi.permutation(len(X_test)), :, j]
        rmse_j = float(np.sqrt(mean_squared_error(y_test, best_model.predict(xp, verbose=0))))
        raw_fi.append(max(rmse_j - base_rmse_fi, 0.0))
    total_fi = sum(raw_fi) or 1.0
    feature_importance = [{'feature': n, 'importance': round(100 * v / total_fi, 1)}
                          for n, v in sorted(zip(feat_names, raw_fi), key=lambda t: -t[1])]

    dummy_y = np.zeros((len(y_test), 5))
    dummy_y[:, 0] = y_test[:, 0]
    dummy_y[:, 3] = y_test[:, 1]
    y_real = scaler.inverse_transform(dummy_y)

    # ── Next-day forecast ─────────────────────────────────────────────────────
    last_60 = scaled_data[-60:].reshape(1, 60, 5)
    nxt_s   = best_model.predict(last_60, verbose=0)
    nxt_r   = inverse_pred(nxt_s, scaler)

    current_close    = float(df['Close'].iloc[-1])
    predicted_close  = float(nxt_r[0, 3])
    predicted_open   = float(nxt_r[0, 0])
    change_pct       = (predicted_close - current_close) / current_close * 100
    direction        = 'UP' if predicted_close >= current_close else 'DOWN'

    # ── Chart data ───────────────────────────────────────────────────────────
    hist_dates = df.index.strftime('%Y-%m-%d').tolist()
    hist_close = [round(float(x), 2) for x in df['Close'].values.flatten()]

    test_df    = df.iloc[split:]
    test_dates = test_df.index[-len(y_real):].strftime('%Y-%m-%d').tolist()
    actual_close = [round(float(x), 2) for x in y_real[:, 3]]
    pred_close   = [round(float(x), 2) for x in best_real[:, 3]]

    # ── Fundamentals ─────────────────────────────────────────────────────────
    def fmt_pct(v):
        if v == "N/A": return "N/A"
        return f"{float(v)*100:.2f}%"

    pe    = safe_get(info, 'trailingPE', 'forwardPE')
    pb    = safe_get(info, 'priceToBook')
    mkcap = safe_get(info, 'marketCap')
    eps   = safe_get(info, 'trailingEps')
    # Derive the ratio explicitly instead of guessing provider percentage units.
    div_rate = safe_get(info, 'dividendRate', 'trailingAnnualDividendRate')
    div = float(div_rate) / current_close if div_rate != "N/A" else "N/A"
    bv    = safe_get(info, 'bookValue')
    promo = safe_get(info, 'heldPercentInsiders')
    roe   = safe_get(info, 'returnOnEquity')
    roa   = safe_get(info, 'returnOnAssets')
    sector = safe_get(info, 'sector', 'industry', default='—')
    name   = safe_get(info, 'longName', 'shortName', default=symbol)

    # 52-week
    high52 = safe_get(info, 'fiftyTwoWeekHigh')
    low52  = safe_get(info, 'fiftyTwoWeekLow')
    day_high = safe_get(info, 'dayHigh')
    day_low  = safe_get(info, 'dayLow')
    prev_close = safe_get(info, 'previousClose')

    day_change    = "N/A"
    day_change_pct = "N/A"
    if prev_close != "N/A" and float(prev_close) != 0:
        dc = current_close - float(prev_close)
        day_change     = f"{'+' if dc >= 0 else ''}{dc:.2f}"
        day_change_pct = f"{'+' if dc >= 0 else ''}{dc/float(prev_close)*100:.2f}%"

    return {
        # Identity
        "symbol":    symbol,
        "name":      name,
        "sector":    sector,
        # Price
        "current_close":   round(current_close, 2),
        "day_change":      day_change,
        "day_change_pct":  day_change_pct,
        "day_high":        round(float(day_high), 2)  if day_high  != "N/A" else "N/A",
        "day_low":         round(float(day_low), 2)   if day_low   != "N/A" else "N/A",
        "week52_high":     round(float(high52), 2)    if high52    != "N/A" else "N/A",
        "week52_low":      round(float(low52), 2)     if low52     != "N/A" else "N/A",
        # Prediction
        "best_model":      best_name,
        "lstm_rmse":       round(lstm_rmse, 4),
        "gru_rmse":        round(gru_rmse, 4),
        "next_open":       round(predicted_open, 2),
        "next_close":      round(predicted_close, 2),
        "change_pct":      round(change_pct, 2),
        "direction":       direction,
        # Fundamentals
        "pe":    round(float(pe), 2)  if pe  != "N/A" else "N/A",
        "pb":    round(float(pb), 2)  if pb  != "N/A" else "N/A",
        "eps":   round(float(eps), 2) if eps != "N/A" else "N/A",
        "mkcap": format_large(mkcap),
        "div_yield": fmt_pct(div)  if div  != "N/A" else "N/A",
        "book_value": round(float(bv), 2) if bv != "N/A" else "N/A",
        "promo_hold": fmt_pct(promo) if promo != "N/A" else "N/A",
        "roe":  fmt_pct(roe)  if roe  != "N/A" else "N/A",
        "roa": fmt_pct(roa) if roa != "N/A" else "N/A",
        # Chart
        "hist_dates":    hist_dates,
        "hist_close":    hist_close,
        "test_dates":    test_dates,
        "actual_close":  actual_close,
        "pred_close":    pred_close,
        # Explainability
        "feature_importance": feature_importance,
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/options')
def options():
    directory = Path(app.config['OPTION_RESULTS_DIR'])
    metrics_path = directory / 'overall_metrics.csv'
    metrics = pd.read_csv(metrics_path).to_dict('records') if metrics_path.exists() else []
    charts = [{'filename': filename, 'title': title, 'description': description}
              for filename, (title, description) in OPTION_CHARTS.items()
              if (directory / filename).is_file()]
    return render_template('options.html', metrics=metrics, charts=charts)


@app.route('/options/charts/<filename>')
def option_chart(filename):
    if filename not in OPTION_CHARTS:
        abort(404)
    return send_from_directory(app.config['OPTION_RESULTS_DIR'], filename, max_age=0)


@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get('symbol'), str):
        return jsonify({'error': 'Please enter a stock symbol.'}), 400
    raw = body['symbol'].strip()
    if not raw:
        return jsonify({'error': 'Please enter a stock symbol.'}), 400

    try:
        symbol = clean_symbol(raw)
        result = run_prediction(symbol)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock-options', methods=['POST'])
def stock_options():
    from src.stock_options import read_contracts, compare_stock_options
    if request.content_length and request.content_length > app.config['MAX_CONTENT_LENGTH']:
        abort(413)
    body = request.form if request.mimetype == 'multipart/form-data' else request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get('symbol'), str):
        return jsonify(error='Please enter an NSE stock symbol.'), 400
    try:
        symbol = clean_symbol(body['symbol'])
        mode = body.get('mode', 'simulation')
        if mode not in ('simulation', 'observed'):
            raise ValueError('Choose simulation or observed-premium comparison.')
        try:
            rate, dividend = float(body.get('rate', .05)), float(body.get('dividend', 0))
        except (ValueError, TypeError):
            raise ValueError('Rate and dividend yield must be numbers.')
        if not np.isfinite([rate, dividend]).all() or not 0 <= rate <= .3 or not 0 <= dividend <= .3:
            raise ValueError('Rate and dividend yield must be between 0 and 0.30 (0%–30%).')
        contracts = None
        if mode == 'observed':
            uploaded = request.files.get('contracts')
            if uploaded is None or not uploaded.filename:
                raise ValueError('Upload historical call premiums for observed-data comparison.')
            contracts = read_contracts(uploaded.stream, symbol)
            start = (contracts.date.min() - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
            end = (contracts.date.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            start = (pd.Timestamp.today() - pd.Timedelta(days=3 * 365)).strftime('%Y-%m-%d')
            end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')  # end is exclusive
        history = yf.download(symbol, start=start, end=end, auto_adjust=True,
                              progress=False, threads=False)
        result = compare_stock_options(symbol, history, rate, dividend, contracts,
                                       epochs=app.config['STOCK_OPTION_EPOCHS'])
        return jsonify(result)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception('Stock option comparison failed')
        return jsonify(error='Could not complete the option comparison. Check the data connection and try again.'), 500


@app.errorhandler(413)
def upload_too_large(error):
    return jsonify(error='The option CSV must be smaller than 2 MB.'), 413


# ─── News API ────────────────────────────────────────────────────────────────

# Lightweight finance headline lexicon for fast, offline sentiment scoring.
# The deep attention models (see /api/news/benchmark) are the research result;
# this heuristic just gives the live feed a readable tone per headline.
_POS_WORDS = {
    'surge', 'surges', 'surged', 'jump', 'jumps', 'jumped', 'rise', 'rises', 'rose',
    'gain', 'gains', 'gained', 'rally', 'rallies', 'rallied', 'beat', 'beats', 'profit',
    'profits', 'growth', 'grow', 'grows', 'record', 'high', 'highs', 'upgrade', 'upgraded',
    'outperform', 'bullish', 'boost', 'boosts', 'strong', 'strength', 'win', 'wins',
    'buy', 'soar', 'soars', 'soared', 'top', 'tops', 'approval', 'approved', 'dividend',
    'expand', 'expands', 'expansion', 'positive', 'optimistic', 'recover', 'recovers',
}
_NEG_WORDS = {
    'fall', 'falls', 'fell', 'drop', 'drops', 'dropped', 'plunge', 'plunges', 'plunged',
    'slump', 'slumps', 'slumped', 'loss', 'losses', 'miss', 'misses', 'missed', 'cut',
    'cuts', 'downgrade', 'downgraded', 'underperform', 'bearish', 'weak', 'weakness',
    'sell', 'sinks', 'sink', 'crash', 'crashes', 'crashed', 'decline', 'declines',
    'declined', 'warn', 'warns', 'warning', 'fraud', 'probe', 'lawsuit', 'fine', 'fined',
    'slip', 'slips', 'slipped', 'low', 'lows', 'concern', 'concerns', 'risk', 'risks',
    'negative', 'tumble', 'tumbles', 'tumbled', 'default', 'debt', 'layoff', 'layoffs',
}
_WORD_RE = re.compile(r"[a-z']+")


def score_headline(text: str):
    tokens = _WORD_RE.findall(text.lower())
    pos = sum(1 for t in tokens if t in _POS_WORDS)
    neg = sum(1 for t in tokens if t in _NEG_WORDS)
    net = pos - neg
    if net > 0:
        label = 'positive'
    elif net < 0:
        label = 'negative'
    else:
        label = 'neutral'
    return label, net


# Cache news responses per symbol so a demo of several companies does not burn
# the Marketaux free-tier quota (~100 requests/day). {symbol: (epoch, payload)}.
_NEWS_CACHE = {}
_NEWS_TTL = 15 * 60  # seconds
# Keep each Marketaux lookup cheap: a few pages is plenty for a dashboard.
_NEWS_MAX_PAGES = 3


@app.route('/api/news', methods=['GET'])
def api_news():
    import time as _time
    raw = (request.args.get('symbol') or '').strip()
    if not raw:
        return jsonify(error='Please enter a stock symbol.'), 400
    try:
        symbol = clean_symbol(raw)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    cached = _NEWS_CACHE.get(symbol)
    if cached and (_time.time() - cached[0]) < _NEWS_TTL and not request.args.get('provider'):
        return jsonify({**cached[1], 'cached': True})

    from src.news import get_provider

    # Build an ordered preference list. Marketaux covers NSE/BSE, so it leads for
    # Indian tickers when its key is set; yfinance is the always-available fallback.
    requested = (request.args.get('provider') or '').strip().lower()
    if requested:
        order = [requested]
    else:
        order = []
        if os.environ.get('MARKETAUX_API_KEY'):
            order.append('marketaux')
        if os.environ.get('FINNHUB_API_KEY'):
            order.append('finnhub')
        if os.environ.get('ALPHAVANTAGE_API_KEY'):
            order.append('alphavantage')
        order.append('yfinance')

    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=30)
    items, used = [], None
    last_error = None
    for name in order:
        try:
            provider = get_provider(name)
            if hasattr(provider, 'max_pages'):
                provider.max_pages = _NEWS_MAX_PAGES  # conserve API quota
            fetched = provider.fetch(symbol, start, end)
        except Exception as exc:  # bad key, unknown provider, transient outage
            last_error = exc
            continue
        used = name
        if fetched:
            items = fetched
            break
    if used is None:
        app.logger.warning('All news providers failed: %s', last_error)
        return jsonify(error='Could not reach any news feed. Try again shortly.'), 502

    scored = []
    counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    net_total = 0
    for it in items:
        label, net = score_headline(it.headline)
        counts[label] += 1
        net_total += net
        scored.append({
            'date': it.date.isoformat() if hasattr(it.date, 'isoformat') else str(it.date),
            'headline': it.headline,
            'source': it.source,
            'sentiment': label,
            'score': net,
        })
    scored.sort(key=lambda r: r['date'], reverse=True)
    n = len(scored)
    payload = {
        'symbol': symbol,
        'provider': used,
        'count': n,
        'summary': {
            **counts,
            'net': net_total,
            'avg': round(net_total / n, 3) if n else 0,
            'bias': 'positive' if net_total > 0 else 'negative' if net_total < 0 else 'neutral',
        },
        # Explainability: which headlines push sentiment most in each direction.
        'drivers': {
            'positive': sorted([r for r in scored if r['score'] > 0], key=lambda r: -r['score'])[:3],
            'negative': sorted([r for r in scored if r['score'] < 0], key=lambda r: r['score'])[:3],
        },
        'items': scored,
    }
    # Only cache non-empty results, so a transient miss can be retried.
    if n:
        _NEWS_CACHE[symbol] = (_time.time(), payload)
    return jsonify({**payload, 'cached': False})


@app.route('/api/news/benchmark', methods=['GET'])
def api_news_benchmark():
    path = Path(app.config['NEWS_RESULTS_DIR']) / 'stock_news_metrics.csv'
    if not path.exists():
        return jsonify(models=[], chart=False)
    df = pd.read_csv(path)
    records = []
    for row in df.to_dict('records'):
        rec = {}
        for k, v in row.items():
            if isinstance(v, float) and not np.isfinite(v):
                rec[k] = None
            elif isinstance(v, float):
                rec[k] = round(v, 5)
            else:
                rec[k] = v
        records.append(rec)
    chart_exists = (Path(app.config['NEWS_RESULTS_DIR']) / 'stock_news_comparison.png').is_file()
    return jsonify(models=records, chart=chart_exists)


@app.route('/api/news/chart')
def api_news_chart():
    return send_from_directory(app.config['NEWS_RESULTS_DIR'],
                               'stock_news_comparison.png', max_age=0)


# ─── Single-page app (production build) ───────────────────────────────────────

@app.route('/app', defaults={'path': ''})
@app.route('/app/<path:path>')
def spa(path):
    """Serve the built Vite SPA. In dev, use the Vite server instead."""
    if not DIST_DIR.is_file() and not DIST_DIR.exists():
        return jsonify(error='Frontend not built. Run "npm run build" in frontend/.'), 404
    target = DIST_DIR / path
    if path and target.is_file():
        return send_from_directory(str(DIST_DIR), path, max_age=0)
    return send_from_directory(str(DIST_DIR), 'index.html', max_age=0)


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
