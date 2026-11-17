import { useState } from 'react'
import {
  ResponsiveContainer, BarChart, Bar, LineChart, Line, ScatterChart, Scatter,
  XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts'
import { compareOptions } from '../api.js'
import { Spinner, ErrorBox } from '../components/Status.jsx'
import CompanyPicker from '../components/CompanyPicker.jsx'

const MODEL_COLORS = {
  'Black-Scholes': '#94a3b8',
  'Direct ANN': '#38bdf8',
  'Residual ANN': '#a78bfa',
  'Option LSTM': '#f472b6',
}

function curvesToData(curves) {
  const names = Object.keys(curves || {})
  const maxLen = Math.max(0, ...names.map((n) => curves[n].length))
  const rows = []
  for (let i = 0; i < maxLen; i++) {
    const row = { epoch: i + 1 }
    names.forEach((n) => { row[n] = curves[n][i] ?? null })
    rows.push(row)
  }
  return { rows, names }
}

export default function Options() {
  const [form, setForm] = useState({ symbol: 'RELIANCE', mode: 'simulation', rate: '0.05', dividend: '0' })
  const [contracts, setContracts] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [data, setData] = useState(null)

  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value })

  async function go(overrides) {
    const payload = { ...form, ...overrides }
    setLoading(true); setError(''); setData(null)
    try {
      setData(await compareOptions({ ...payload, contracts: payload.mode === 'observed' ? contracts : null }))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function run(e) {
    e.preventDefault()
    go()
  }

  function pick(sym) {
    setForm({ ...form, symbol: sym })
    go({ symbol: sym })
  }

  const curveData = data ? curvesToData(data.curves) : null
  const scatterData = data
    ? data.contracts.map((c) => ({ actual: c.target, ...c.predictions }))
    : null

  return (
    <div className="page">
      <div className="page-head">
        <h1>Option Pricing</h1>
        <p>Neural option pricers vs. the Black–Scholes benchmark on European calls.</p>
      </div>

      <form className="toolbar wrap" onSubmit={run}>
        <input value={form.symbol} onChange={set('symbol')} placeholder="NSE symbol" aria-label="Symbol" />
        <select value={form.mode} onChange={set('mode')} aria-label="Mode">
          <option value="simulation">Simulation</option>
          <option value="observed">Observed premiums (CSV)</option>
        </select>
        <label className="inline">Rate
          <input type="number" step="0.01" min="0" max="0.3" value={form.rate} onChange={set('rate')} />
        </label>
        <label className="inline">Div. yield
          <input type="number" step="0.01" min="0" max="0.3" value={form.dividend} onChange={set('dividend')} />
        </label>
        {form.mode === 'observed' && (
          <input type="file" accept=".csv" onChange={(e) => setContracts(e.target.files[0] || null)} />
        )}
        <button className="btn btn-primary" disabled={loading}>
          {loading ? 'Pricing…' : 'Compare'}
        </button>
      </form>

      <CompanyPicker onPick={pick} active={form.symbol} />

      {loading && <Spinner label="Training pricers and running Monte-Carlo…" />}
      {error && <ErrorBox message={error} />}

      {data && (
        <div className="results">
          <div className="meta-row">
            <span className="chip">{data.mode}</span>
            <span className="muted">{data.source}</span>
            <span className="muted">· {data.history_start} → {data.history_end}</span>
          </div>

          <div className="panel">
            <h3>Pricing error by model</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data.metrics} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" />
                <XAxis dataKey="model" tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} width={56} />
                <Tooltip contentStyle={{ background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text)' }} />
                <Legend />
                <Bar dataKey="rmse" name="RMSE" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="mae" name="MAE" fill="#a78bfa" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="panel">
            <h3>Model accuracy</h3>
            <table className="metrics-table">
              <thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead>
              <tbody>
                {data.metrics.map((m) => (
                  <tr key={m.model}>
                    <td><span className="dot" style={{ background: MODEL_COLORS[m.model] }} />{m.model}</td>
                    <td>{m.rmse.toFixed(3)}</td>
                    <td>{m.mae.toFixed(3)}</td>
                    <td>{m.r2.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {curveData && curveData.names.length > 0 && (
            <div className="panel">
              <h3>Validation loss curves</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={curveData.rows} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                  <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                  <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} width={56} />
                  <Tooltip contentStyle={{ background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text)' }} />
                  <Legend />
                  {curveData.names.map((n) => (
                    <Line key={n} type="monotone" dataKey={n} stroke={MODEL_COLORS[n]} dot={false} strokeWidth={2} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="panel">
            <h3>Predicted vs. actual premium (test contracts)</h3>
            <ResponsiveContainer width="100%" height={320}>
              <ScatterChart margin={{ top: 8, right: 16, bottom: 10, left: 0 }}>
                <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" />
                <XAxis type="number" dataKey="actual" name="Actual" tick={{ fill: 'var(--muted)', fontSize: 11 }}
                  label={{ value: 'Actual', position: 'insideBottom', offset: -4, fill: 'var(--muted)' }} />
                <YAxis type="number" tick={{ fill: 'var(--muted)', fontSize: 11 }} width={56} />
                <ZAxis range={[36, 36]} />
                <Tooltip contentStyle={{ background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text)' }} />
                <Legend />
                {Object.keys(MODEL_COLORS).map((n) => (
                  <Scatter key={n} name={n} data={scatterData.map((d) => ({ actual: d.actual, [n]: d[n] }))}
                    dataKey={n} fill={MODEL_COLORS[n]} />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
            <p className="caption">Points on the diagonal are perfect; spread above/below shows over/under-pricing.</p>
          </div>
        </div>
      )}
    </div>
  )
}
