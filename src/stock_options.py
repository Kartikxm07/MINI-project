"""Stock-conditioned European call experiments and uploaded-premium evaluation.

Simulated targets are risk-neutral empirical-bootstrap Monte Carlo estimates,
not exchange quotes. Features at each valuation date use only earlier/history-to-
date prices. All contracts on a date belong to the same evaluation partition.
"""
from threading import Lock

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model import SimpleOptionANN, ResidualOptionANN, OptionLSTM
from src.pricing import black_scholes_call


TRAINING_LOCK = Lock()
CSV_COLUMNS = ['symbol', 'date', 'expiry', 'spot', 'strike', 'call_price']
MODEL_NAMES = ['Black-Scholes', 'Direct ANN', 'Residual ANN', 'Option LSTM']


def clean_history(history):
    if history.empty:
        raise ValueError('No stock history returned. Check the symbol and try again.')
    history = history.copy()
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    if 'Close' not in history:
        raise ValueError('Stock history must contain Close prices.')
    # Adjusted closes avoid treating corporate actions as enormous daily returns.
    close = pd.to_numeric(history['Close'], errors='coerce').dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated()].sort_index()
    if len(close) < 150 or not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError('At least 150 valid positive daily stock prices are required.')
    return close


def read_contracts(stream, symbol):
    try:
        data = pd.read_csv(stream, nrows=4001)
    except Exception as exc:
        raise ValueError('Could not read the option CSV. Use the supplied column format.') from exc
    if not set(CSV_COLUMNS).issubset(data.columns):
        raise ValueError('CSV requires: ' + ', '.join(CSV_COLUMNS))
    if not 60 <= len(data) <= 4000:
        raise ValueError('Upload 60–4,000 call observations across at least 20 valuation dates.')
    symbols = data.symbol.astype(str).str.strip().str.upper().str.removesuffix('.NS')
    if not symbols.eq(symbol.removesuffix('.NS')).all():
        raise ValueError('Every CSV row must match the searched stock symbol.')
    for column in ['date', 'expiry']:
        data[column] = pd.to_datetime(data[column], format='%Y-%m-%d', errors='coerce')
        if data[column].isna().any():
            raise ValueError(f'{column} must use YYYY-MM-DD dates.')
    for column in ['spot', 'strike', 'call_price']:
        data[column] = pd.to_numeric(data[column], errors='coerce')
        if not np.isfinite(data[column]).all():
            raise ValueError(f'{column} must contain finite numbers.')
    if (data[['spot', 'strike']] <= 0).any().any() or (data.call_price < 0).any():
        raise ValueError('Spot and strike must be positive; call_price must be nonnegative.')
    days = (data.expiry - data.date).dt.days
    if (days < 1).any() or (days > 366).any():
        raise ValueError('Each expiry must be 1–366 calendar days after its valuation date.')
    if data.duplicated(['date', 'expiry', 'strike']).any():
        raise ValueError('Duplicate date/expiry/strike contracts: provide one closing quote per contract per day.')
    if data.date.nunique() < 20:
        raise ValueError('At least 20 distinct valuation dates are required; a single option-chain snapshot is insufficient.')
    if (data.date > pd.Timestamp.now().normalize()).any():
        raise ValueError('Valuation dates cannot be in the future.')
    return data.sort_values(['date', 'expiry', 'strike']).reset_index(drop=True)


def make_dataset(close, rate=.05, dividend=.0, contracts=None, max_dates=150, paths=1024):
    """Return contract rows plus actual, past-only 20-session input sequences."""
    rng = np.random.default_rng(42)
    dates = (pd.DatetimeIndex(contracts.date.unique()) if contracts is not None
             else close.index[np.unique(np.linspace(126, len(close) - 1,
                                                   min(max_dates, len(close) - 126), dtype=int))])
    rows, sequences = [], []
    for date in dates:
        position = close.index.get_indexer([date])[0]
        if position < 126:
            raise ValueError(f'Need 126 preceding stock sessions for {date:%Y-%m-%d}; check CSV dates and stock history.')
        history = close.iloc[position - 126:position + 1].to_numpy()
        returns = np.diff(np.log(history))
        sigma = max(float(returns[-60:].std(ddof=1) * np.sqrt(252)), .001)
        sequence = np.column_stack([history[-20:] / history[-1] - 1, returns[-20:]])
        if contracts is None:
            # iid empirical log-return innovations, recentered to enforce E[S_T]
            # = S*exp((r-q)*T) before Monte Carlo sampling error.
            residuals = returns - returns.mean()
            correction = np.log(np.exp(residuals).mean())
            simulated = rng.choice(residuals, size=(paths, 63), replace=True) - correction
            for sessions in [21, 42, 63]:
                maturity = sessions / 252
                terminal = history[-1] * np.exp(simulated[:, :sessions].sum(axis=1)
                                               + (rate - dividend) * maturity)
                for ratio in [.9, 1., 1.1]:
                    strike = float(history[-1] * ratio)
                    payoff = np.exp(-rate * maturity) * np.maximum(terminal - strike, 0)
                    rows.append(dict(date=date, expiry=None, spot=float(history[-1]), strike=strike,
                                     T=maturity, sigma=sigma, target=float(payoff.mean()),
                                     mc_se=float(payoff.std(ddof=1) / np.sqrt(paths))))
                    sequences.append(sequence)
        else:
            for row in contracts[contracts.date.eq(date)].itertuples():
                rows.append(dict(date=date, expiry=row.expiry, spot=row.spot, strike=row.strike,
                                 T=(row.expiry - row.date).days / 365, sigma=sigma,
                                 target=row.call_price, mc_se=0.))
                sequences.append(sequence)
    frame = pd.DataFrame(rows).sort_values(['date', 'T', 'strike'])
    seq = np.asarray(sequences, dtype=np.float32)[frame.index]
    frame = frame.reset_index(drop=True)
    frame['r'], frame['q'] = rate, dividend
    frame['moneyness'] = frame.spot / frame.strike
    frame['baseline'] = black_scholes_call(frame.spot * np.exp(-dividend * frame['T']),
                                         frame.strike, frame['T'], rate, frame.sigma)
    frame['lower'] = np.maximum(0, frame.spot * np.exp(-dividend * frame['T'])
                               - frame.strike * np.exp(-rate * frame['T']))
    frame['upper'] = frame.spot * np.exp(-dividend * frame['T'])
    if contracts is not None:
        # Allow quote rounding, but reject mismatched spot/premium units or stale quotes.
        tolerance = np.maximum(.05, frame.spot * .001)
        invalid = (frame.target < frame.lower - tolerance) | (frame.target > frame.upper + tolerance)
        if invalid.any():
            raise ValueError(f'{int(invalid.sum())} premiums violate call price bounds under the entered rate/dividend assumptions. Check dates, spot, units and quotes.')
    return frame, seq


def date_partitions(frame):
    dates = frame.date.drop_duplicates().sort_values().to_numpy()
    if len(dates) < 20:
        raise ValueError('At least 20 valuation dates are required for chronological evaluation.')
    a, b = int(.7 * len(dates)), int(.85 * len(dates))
    return [np.flatnonzero(frame.date.isin(part)) for part in (dates[:a], dates[a:b], dates[b:])]


def fit_network(model, inputs, target, train_idx, val_idx, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    best_loss, best_state, stale, losses = float('inf'), None, 0, []
    for _ in range(epochs):
        model.train()
        order = torch.as_tensor(train_idx)[torch.randperm(len(train_idx))]
        for indices in order.split(128):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(*(x[indices] for x in inputs)), target[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(torch.nn.functional.mse_loss(model(*(x[val_idx] for x in inputs)), target[val_idx]))
        losses.append(value)
        if value < best_loss:
            best_loss, stale = value, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 6:
            break
    if best_state is None:
        raise ValueError('Training did not produce finite validation errors. Check the dataset.')
    model.load_state_dict(best_state)
    model.eval()
    return losses


def compare_stock_options(symbol, history, rate=.05, dividend=.0, contracts=None, epochs=25):
    close = clean_history(history)
    frame, sequences = make_dataset(close, rate, dividend, contracts)
    tr, va, te = date_partitions(frame)
    features = frame[['moneyness', 'T', 'r', 'q', 'sigma']].to_numpy()
    static = np.column_stack([features, frame.baseline / frame.strike])
    xscale = StandardScaler().fit(features[tr])
    sscale = StandardScaler().fit(static[tr])
    seqscale = StandardScaler().fit(sequences[tr].reshape(-1, 2))
    y = (frame.target / frame.strike).to_numpy()[:, None]
    residual = ((frame.target - frame.baseline) / frame.strike).to_numpy()[:, None]
    yscale, escale = StandardScaler().fit(y[tr]), StandardScaler().fit(residual[tr])
    tensor = lambda value: torch.tensor(value, dtype=torch.float32)
    x = tensor(xscale.transform(features))
    st = tensor(sscale.transform(static))
    seq = tensor(seqscale.transform(sequences.reshape(-1, 2)).reshape(sequences.shape))
    yt, et = tensor(yscale.transform(y)), tensor(escale.transform(residual))
    with TRAINING_LOCK:
        # Limit CPU oversubscription during a local web request; restore settings.
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(min(4, previous_threads))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                direct = SimpleOptionANN(input_dim=5)
                residual_model = ResidualOptionANN(input_dim=5)
                lstm = OptionLSTM(static_input_dim=6, lstm_hidden=32)
                curves = {
                    'Direct ANN': fit_network(direct, [x], yt, tr, va, epochs),
                    'Residual ANN': fit_network(residual_model, [x], et, tr, va, epochs),
                    'Option LSTM': fit_network(lstm, [seq, st], yt, tr, va, epochs),
                }
                with torch.no_grad():
                    predictions = [frame.baseline.to_numpy()[te],
                                   yscale.inverse_transform(direct(x[te]).numpy()).ravel() * frame.strike.to_numpy()[te],
                                   frame.baseline.to_numpy()[te] + escale.inverse_transform(residual_model(x[te]).numpy()).ravel() * frame.strike.to_numpy()[te],
                                   yscale.inverse_transform(lstm(seq[te], st[te]).numpy()).ravel() * frame.strike.to_numpy()[te]]
        finally:
            torch.set_num_threads(previous_threads)
    test = frame.iloc[te].copy()
    metrics = []
    for name, prediction in zip(MODEL_NAMES, predictions):
        prediction = np.clip(prediction, test.lower, test.upper)
        if not np.isfinite(prediction).all():
            raise ValueError('Non-finite predictions encountered. Check the data.')
        test[name] = prediction
        metrics.append(dict(model=name, rmse=float(np.sqrt(mean_squared_error(test.target, prediction))),
                            mae=float(mean_absolute_error(test.target, prediction)),
                            r2=float(r2_score(test.target, prediction))))
    records = []
    for _, row in test.iterrows():
        records.append(dict(date=row.date.strftime('%Y-%m-%d'),
                            expiry=row.expiry.strftime('%Y-%m-%d') if pd.notna(row.expiry) else None,
                            days=round(row['T'] * 365, 1), spot=float(row.spot), strike=float(row.strike),
                            volatility=float(row.sigma), target=float(row.target), mc_se=float(row.mc_se),
                            predictions={name: float(row[name]) for name in MODEL_NAMES}))
    split = {}
    for name, indices in zip(['train', 'validation', 'test'], [tr, va, te]):
        part = frame.iloc[indices]
        split[name] = dict(rows=len(part), dates=int(part.date.nunique()),
                           start=part.date.min().strftime('%Y-%m-%d'), end=part.date.max().strftime('%Y-%m-%d'))
    # Explainability: permutation factor importance for the Direct ANN. Each input
    # factor is shuffled on the test set and the RMSE increase measured — a larger
    # increase means the model relies on that factor more to price the option.
    factor_names = ['Moneyness (S/K)', 'Time to expiry', 'Risk-free rate', 'Dividend yield', 'Volatility']
    strike_te = frame.strike.to_numpy()[te]
    target_te = frame.target.to_numpy()[te]
    with torch.no_grad():
        x_te = x[te].clone()

        def _direct_rmse(mat):
            pred = yscale.inverse_transform(direct(mat).numpy()).ravel() * strike_te
            return float(np.sqrt(mean_squared_error(target_te, pred)))

        base_rmse = _direct_rmse(x_te)
        rng = np.random.default_rng(42)
        raw = []
        for j in range(x_te.shape[1]):
            perm = x_te.clone()
            perm[:, j] = perm[torch.as_tensor(rng.permutation(len(perm))), j]
            raw.append(max(_direct_rmse(perm) - base_rmse, 0.0))
    total = sum(raw) or 1.0
    factor_importance = [dict(factor=n, importance=round(100 * v / total, 1), rmse_increase=round(v, 4))
                         for n, v in sorted(zip(factor_names, raw), key=lambda t: -t[1])]

    return dict(symbol=symbol, mode='observed' if contracts is not None else 'simulation',
                source='User-uploaded closing call premiums' if contracts is not None else 'Stock-history-calibrated simulation',
                history_start=close.index[0].strftime('%Y-%m-%d'), history_end=close.index[-1].strftime('%Y-%m-%d'),
                rate=rate, dividend=dividend, metrics=metrics, contracts=records, split=split,
                curves=curves, factor_importance=factor_importance,
                simulation_paths=1024 if contracts is None else None)
