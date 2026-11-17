import { Routes, Route, NavLink, Link } from 'react-router-dom'
import Home from './pages/Home.jsx'
import Stock from './pages/Stock.jsx'
import Options from './pages/Options.jsx'
import News from './pages/News.jsx'

function NavBar() {
  return (
    <header className="nav">
      <Link to="/" className="brand">
        <span className="brand-mark">◆</span> StockSight
      </Link>
      <nav className="nav-links">
        <NavLink to="/" end>Home</NavLink>
        <NavLink to="/stock">Stock</NavLink>
        <NavLink to="/options">Options</NavLink>
        <NavLink to="/news">News</NavLink>
      </nav>
    </header>
  )
}

export default function App() {
  return (
    <div className="app">
      <NavBar />
      <main className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/stock" element={<Stock />} />
          <Route path="/options" element={<Options />} />
          <Route path="/news" element={<News />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </main>
      <footer className="footer">
        StockSight · Deep-learning market research · Educational use only — not financial advice.
      </footer>
    </div>
  )
}
