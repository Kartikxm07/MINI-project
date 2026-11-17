import { COMPANIES } from '../companies.js'

// Clickable company chips. `onPick` receives the bare NSE symbol; pass `active`
// (the current symbol) to highlight the selected chip.
export default function CompanyPicker({ onPick, active, limit }) {
  const list = limit ? COMPANIES.slice(0, limit) : COMPANIES
  return (
    <div className="picker">
      <span className="picker-label">Try:</span>
      <div className="picker-chips">
        {list.map((c) => (
          <button
            key={c.symbol}
            type="button"
            title={c.name}
            className={`pick ${active?.toUpperCase() === c.symbol ? 'pick-active' : ''}`}
            onClick={() => onPick(c.symbol)}
          >
            {c.symbol}
          </button>
        ))}
      </div>
    </div>
  )
}
