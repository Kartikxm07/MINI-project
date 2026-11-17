import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { AuroraBackground } from '@/components/ui/aurora-background'
import BentoFeatures from '@/components/ui/bento-features'

export default function Home() {
  const navigate = useNavigate()
  return (
    <div className="home">
      <AuroraBackground className="h-auto min-h-[460px] rounded-3xl !bg-[#0b1120]">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.8, ease: 'easeInOut' }}
          className="relative flex flex-col gap-4 items-center justify-center px-4 text-center py-16"
        >
          <p className="eyebrow">Deep Learning for Financial Markets</p>
          <h1 className="text-4xl md:text-6xl font-bold text-white tracking-tight">StockSight</h1>
          <p className="lede">
            A research dashboard for Indian equity markets — sequence models for price
            forecasting, neural option pricers, and news-attention networks, behind one
            clean interface.
          </p>
          <div className="hero-actions">
            <button className="btn btn-primary" onClick={() => navigate('/stock')}>Start forecasting</button>
            <button className="btn btn-ghost" onClick={() => navigate('/news')}>Explore news models</button>
          </div>
        </motion.div>
      </AuroraBackground>

      <section style={{ marginTop: 40 }}>
        <BentoFeatures />
      </section>

      <section className="about">
        <h2>About this project</h2>
        <div className="about-grid">
          <div>
            <h4>What it does</h4>
            <p>
              StockSight predicts short-term stock movements, prices options with both
              classical and deep-learning methods, and measures whether financial news
              adds predictive signal through attention-based fusion.
            </p>
          </div>
          <div>
            <h4>How it works</h4>
            <p>
              A Flask backend runs the models (TensorFlow &amp; PyTorch) and serves JSON APIs.
              This React frontend renders the results — price charts, error comparisons,
              and a live sentiment feed — in real time.
            </p>
          </div>
          <div>
            <h4>Data sources</h4>
            <p>
              Market data comes from Yahoo Finance; news headlines from Marketaux, Finnhub
              and others. Everything runs locally on your machine.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
