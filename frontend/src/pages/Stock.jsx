import { useState } from 'react'
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts'
import { predictStock } from '../api.js'
import { Spinner, ErrorBox } from '../components/Status.jsx'
import CompanyPicker from '../components/CompanyPicker.jsx'
import NumberTicker from '@/components/ui/number-ticker'

function buildChartData(r) {
  // Align actual vs predicted on the test window.
  return r.test_dates.map((d, i) => ({
    date: d,
    Actual: r.actual_close[i],
    Predicted: r.pred_close[i],
  }))
}

const FUNDAMENTALS = [
  ['pe', 'P/E'], ['pb', 'P/B'], ['eps', 'EPS'], ['mkcap', 'Market Cap'],
  ['div_yield', 'Div. Yield'], ['book_value', 'Book Value'],
  ['roe', 'ROE'], ['roa', 'ROA'], ['promo_hold', 'Insider Hold'],
]

export default function Stock() {
  const [symbol, setSymbol] = useState('RELIANCE')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [data, setData] = useState(null)

  async function go(sym) {
    setLoading(true); setError(''); setData(null)
    try {
      setData(await predictStock(sym))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function run(e) {
    e.preventDefault()
    go(symbol)
  }

  function pick(sym) {
    setSymbol(sym)
    go(sym)
  }

  return (
    <div className="page">
      <div className="page-head">
        <h1>Stock Prediction</h1>
        <p>Next-day forecast from LSTM &amp; GRU models trained live on the last ~5 years of prices.</p>
      </div>

      <form className="toolbar" onSubmit={run}>
        <input
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          placeholder="NSE symbol e.g. RELIANCE, TCS"
          aria-label="Stock symbol"
        />
        <button className="btn btn-primary" disabled={loading}>
          {loading ? 'Training…' : 'Predict'}
        </button>
      </form>

      <CompanyPicker onPick={pick} active={symbol} />

      {loading && <Spinner label="Training LSTM & GRU — this can take a minute…" />}
      {error && <ErrorBox message={error} />}

      {data && (
        <div className="results">
          <div className="result-head">
            <div>
              <h2>{data.name}</h2>
              <span className="muted">{data.symbol} · {data.sector}</span>
            </div>
            <div className="price">
              <NumberTicker value={data.current_close} />
              <span className={data.day_change_pct?.startsWith('-') ? 'neg' : 'pos'}>
                {data.day_change} ({data.day_change_pct})
              </span>
            </div>
          </div>

          <div className="stat-row">
            <div className="stat">
              <span className="stat-label">Predicted close</span>
              <span className="stat-value"><NumberTicker value={data.next_close} /></span>
            </div>
            <div className="stat">
              <span className="stat-label">Predicted open</span>
              <span className="stat-value"><NumberTicker value={data.next_open} /></span>
            </div>
            <div className={`stat ${data.direction === 'UP' ? 'up' : 'down'}`}>
              <span className="stat-label">Expected move</span>
              <span className="stat-value">
                {data.direction === 'UP' ? '▲' : '▼'} {data.change_pct}%
              </span>
            </div>
            <div className="stat">
              <span className="stat-label">Best model</span>
              <span className="stat-value">{data.best_model}</span>
            </div>
          </div>

          <div className="panel">
            <h3>Actual vs. predicted (held-out test window)</h3>
            <ResponsiveContainer width="100%" height={340}>
              <LineChart data={buildChartData(data)} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fill: 'var(--muted)', fontSize: 11 }} minTickGap={40} />
                <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} domain={['auto', 'auto']} width={56} />
                <Tooltip contentStyle={{ background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text)' }} />
                <Legend />
                <Line type="monotone" dataKey="Actual" stroke="#38bdf8" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="Predicted" stroke="#f472b6" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
            <p className="caption">
              Model error — LSTM RMSE {data.lstm_rmse} · GRU RMSE {data.gru_rmse} (scaled units).
            </p>
          </div>

          <div className="panel">
            <h3>Fundamentals</h3>
            <div className="fund-grid">
              {FUNDAMENTALS.map(([k, label]) => (
                <div className="fund" key={k}>
                  <span className="fund-label">{label}</span>
                  <span className="fund-value">{data[k]}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
