// All calls go to same-origin paths. In dev, Vite proxies them to Flask (:5000).
// In production, Flask serves both the SPA and these endpoints.

async function parse(res) {
  let data = null
  try {
    data = await res.json()
  } catch {
    // non-JSON error (e.g. HTML 500 page)
  }
  if (!res.ok) {
    const msg = (data && data.error) || `Request failed (${res.status})`
    throw new Error(msg)
  }
  return data
}

export async function predictStock(symbol) {
  const res = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol }),
  })
  return parse(res)
}

export async function compareOptions({ symbol, mode, rate, dividend, contracts }) {
  let res
  if (mode === 'observed') {
    const fd = new FormData()
    fd.append('symbol', symbol)
    fd.append('mode', mode)
    fd.append('rate', rate)
    fd.append('dividend', dividend)
    if (contracts) fd.append('contracts', contracts)
    res = await fetch('/api/stock-options', { method: 'POST', body: fd })
  } else {
    res = await fetch('/api/stock-options', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, mode, rate, dividend }),
    })
  }
  return parse(res)
}

export async function fetchNews(symbol) {
  const res = await fetch(`/api/news?symbol=${encodeURIComponent(symbol)}`)
  return parse(res)
}

export async function fetchNewsBenchmark() {
  const res = await fetch('/api/news/benchmark')
  return parse(res)
}
