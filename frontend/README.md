# StockSight — Web Frontend

A React (Vite) single-page app for the StockSight models. It has a home page
and three dashboards: **Stock Prediction**, **Option Pricing**, and **News & Attention**.

The frontend only renders results — all models run in the Flask backend (`../app.py`),
which exposes JSON APIs (`/predict`, `/api/stock-options`, `/api/news`, `/api/news/benchmark`).

## Develop (two terminals)

**1. Backend** (from the project root, with your venv active):

```bash
python app.py            # serves the APIs on http://127.0.0.1:5000
```

**2. Frontend** (from this `frontend/` folder):

```bash
npm install              # first time only
npm run dev              # http://localhost:5173/app/
```

Vite proxies `/predict` and `/api/*` to the Flask server, so there is no CORS setup.

## Production (single server)

Build the SPA, then let Flask serve it:

```bash
npm run build            # outputs frontend/dist/
```

Start `python app.py` and open **http://127.0.0.1:5000/app/** — Flask serves the
built SPA and the APIs from one process.

## Notes

- Stock prediction trains LSTM + GRU live, so a request can take ~1 minute.
- News headlines try Marketaux (best NSE/BSE coverage) → Finnhub → Alpha Vantage →
  Yahoo Finance, using whichever keys are set in the project `.env`.
- Headline sentiment in the live feed is a fast keyword heuristic; the attention-model
  benchmark on the News page is the trained research result.
