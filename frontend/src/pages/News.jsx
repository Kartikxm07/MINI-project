import { useEffect, useState } from 'react'
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts'
import { fetchNews, fetchNewsBenchmark } from '../api.js'
import { Spinner, ErrorBox, Empty } from '../components/Status.jsx'
import CompanyPicker from '../components/CompanyPicker.jsx'
import { Marquee } from '@/components/ui/marquee'

const DOT = { positive: '#34d399', negative: '#f87171', neutral: '#94a3b8' }

const SENT_CLASS = { positive: 'pos', negative: 'neg', neutral: 'neu' }

export default function News() {
  const [symbol, setSymbol] = useState('RELIANCE')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [feed, setFeed] = useState(null)
  const [bench, setBench] = useState(null)

  useEffect(() => {
    fetchNewsBenchmark().then(setBench).catch(() => setBench(null))
  }, [])

  async function go(sym) {
    setLoading(true); setError(''); setFeed(null)
    try {
      setFeed(await fetchNews(sym))
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

  const benchData = bench?.models?.map((m) => ({
    model: m.Model,
    RMSE: m.RMSE,
    DirAcc: m.DirAcc,
  }))

  return (
    <div className="page">
      <div className="page-head">
        <h1>News &amp; Attention</h1>
        <p>Live headline sentiment, plus a benchmark of attention models that fuse news with price signals.</p>
      </div>

      <form className="toolbar" onSubmit={run}>
        <input value={symbol} onChange={(e) => setSymbol(e.target.value)}
          placeholder="NSE symbol e.g. RELIANCE" aria-label="Symbol" />
        <button className="btn btn-primary" disabled={loading}>
          {loading ? 'Fetching…' : 'Get headlines'}
        </button>
      </form>

      <CompanyPicker onPick={pick} active={symbol} />

      {loading && <Spinner label="Fetching latest headlines…" />}
      {error && <ErrorBox message={error} />}

      {feed && feed.count > 0 && (
        <Marquee pauseOnHover speed={40} className="news-ticker">
          {feed.items.slice(0, 12).map((it, i) => (
            <span key={i} className="ticker-item">
              <span className="ticker-dot" style={{ background: DOT[it.sentiment] }} />
              {it.headline}
            </span>
          ))}
        </Marquee>
      )}

      {feed && (
        <div className="results">
          <div className="stat-row">
            <div className="stat pos"><span className="stat-label">Positive</span><span className="stat-value">{feed.summary.positive}</span></div>
            <div className="stat neu"><span className="stat-label">Neutral</span><span className="stat-value">{feed.summary.neutral}</span></div>
            <div className="stat down"><span className="stat-label">Negative</span><span className="stat-value">{feed.summary.negative}</span></div>
            <div className="stat"><span className="stat-label">Net bias</span><span className="stat-value">{feed.summary.bias}</span></div>
          </div>

          <div className="panel">
            <h3>Recent headlines · {feed.symbol} <span className="muted">({feed.count})</span>
              {feed.provider && <span className="chip" style={{ marginLeft: 8 }}>via {feed.provider}</span>}
            </h3>
            {feed.count === 0
              ? <Empty message="No recent headlines found for this symbol." />
              : (
                <ul className="news-list">
                  {feed.items.map((it, i) => (
                    <li key={i} className="news-item">
                      <span className={`tag ${SENT_CLASS[it.sentiment]}`}>{it.sentiment}</span>
                      <div className="news-body">
                        <p className="news-head">{it.headline}</p>
                        <span className="muted small">{it.date}{it.source ? ` · ${it.source}` : ''}</span>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            <p className="caption">
              Sentiment here is a fast keyword heuristic for the live feed. The models below are the
              research result — they learn sentiment end-to-end.
            </p>
          </div>
        </div>
      )}

      <div className="panel">
        <h3>Attention model benchmark</h3>
        <p className="muted">
          Do news embeddings improve next-day return prediction? These models were trained offline
          and compared against a numeric-only baseline.
        </p>
        {!bench || !bench.models?.length
          ? <Empty message="No benchmark results found. Run train_stock_news.py to generate them." />
          : (
            <>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={benchData} margin={{ top: 8, right: 16, bottom: 30, left: 0 }}>
                  <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" />
                  <XAxis dataKey="model" tick={{ fill: 'var(--muted)', fontSize: 10 }} angle={-12} textAnchor="end" height={50} interval={0} />
                  <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} width={56} />
                  <Tooltip contentStyle={{ background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text)' }} />
                  <Legend />
                  <Bar dataKey="RMSE" fill="#f472b6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="DirAcc" name="Directional accuracy" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <table className="metrics-table">
                <thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th><th>Dir. Acc</th></tr></thead>
                <tbody>
                  {bench.models.map((m) => (
                    <tr key={m.Model}>
                      <td>{m.Model}</td>
                      <td>{m.RMSE?.toFixed?.(4) ?? '—'}</td>
                      <td>{m.MAE?.toFixed?.(4) ?? '—'}</td>
                      <td>{m.R2?.toFixed?.(3) ?? '—'}</td>
                      <td>{m.DirAcc != null ? `${(m.DirAcc * 100).toFixed(1)}%` : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {bench.chart && (
                <img className="bench-chart" src="/api/news/chart" alt="News model comparison chart" />
              )}
            </>
          )}
      </div>
    </div>
  )
}
