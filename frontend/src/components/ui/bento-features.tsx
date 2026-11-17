import React from "react"
import { motion } from "framer-motion"
import { useNavigate } from "react-router-dom"
import {
  LineChart, Calculator, Newspaper, Globe, Cpu, Database, ShieldCheck, Zap,
} from "lucide-react"

const card =
  "bg-[#111a2e] border border-[#1e293b] rounded-xl p-6 flex flex-col hover:border-[#38bdf8] transition-colors cursor-pointer overflow-hidden"

function GlobalNetwork() {
  const pulses = [0, 1, 2, 3]
  return (
    <div className="flex items-center justify-center h-full relative">
      <Globe className="w-14 h-14 text-sky-400/90 z-10" />
      {pulses.map((p) => (
        <motion.div
          key={p}
          className="absolute w-14 h-14 border-2 border-sky-400/30 rounded-full"
          initial={{ scale: 0.6, opacity: 1 }}
          animate={{ scale: 3, opacity: 0 }}
          transition={{ duration: 3, repeat: Infinity, delay: p * 0.8, ease: "easeOut" }}
        />
      ))}
    </div>
  )
}

function MiniChart() {
  const bars = [40, 65, 50, 80, 60, 95, 75]
  return (
    <div className="flex items-end justify-center gap-2 h-full pb-2">
      {bars.map((h, i) => (
        <motion.div
          key={i}
          className="w-4 rounded-t bg-gradient-to-t from-sky-500/40 to-sky-400"
          initial={{ height: 0 }}
          animate={{ height: `${h}%` }}
          transition={{ duration: 0.8, delay: i * 0.08, repeat: Infinity, repeatType: "mirror", repeatDelay: 1.2 }}
        />
      ))}
    </div>
  )
}

export default function BentoFeatures() {
  const navigate = useNavigate()
  return (
    <section className="w-full">
      <motion.p
        className="text-slate-400 text-xs uppercase tracking-[0.2em] mb-6"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
      >
        Explore the dashboards
      </motion.p>

      <div className="grid grid-cols-1 md:grid-cols-6 gap-4 auto-rows-[200px]">
        {/* Stock Prediction — tall */}
        <motion.button
          onClick={() => navigate("/stock")}
          className={`md:col-span-2 md:row-span-2 ${card}`}
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          whileHover={{ scale: 1.02 }}
        >
          <div className="flex-1"><MiniChart /></div>
          <div className="mt-4 text-left">
            <h3 className="text-xl text-white font-semibold flex items-center gap-2">
              <LineChart className="w-5 h-5 text-sky-400" /> Stock Prediction
            </h3>
            <p className="text-slate-400 text-sm mt-1">
              Next-day LSTM &amp; GRU forecasts with fundamentals and live charts.
            </p>
          </div>
        </motion.button>

        {/* Option Pricing — wide */}
        <motion.button
          onClick={() => navigate("/options")}
          className={`md:col-span-4 ${card}`}
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.1 }}
          whileHover={{ scale: 0.99 }}
        >
          <div className="flex-1 flex items-center">
            <Calculator className="w-12 h-12 text-violet-400" />
          </div>
          <div className="mt-4 text-left">
            <h3 className="text-xl text-white font-semibold">Option Pricing</h3>
            <p className="text-slate-400 text-sm mt-1">
              Compare ANN, residual and LSTM pricers against the Black–Scholes benchmark.
            </p>
          </div>
        </motion.button>

        {/* News & Attention — tall */}
        <motion.button
          onClick={() => navigate("/news")}
          className={`md:col-span-2 md:row-span-2 ${card}`}
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 }}
          whileHover={{ scale: 1.02 }}
        >
          <div className="flex-1"><GlobalNetwork /></div>
          <div className="mt-auto text-left">
            <h3 className="text-xl text-white font-semibold flex items-center gap-2">
              <Newspaper className="w-5 h-5 text-sky-400" /> News &amp; Attention
            </h3>
            <p className="text-slate-400 text-sm mt-1">
              Live headline sentiment plus attention models that fuse news with prices.
            </p>
          </div>
        </motion.button>

        {/* Feature tiles */}
        <Feature icon={<Database className="w-5 h-5" />} title="Live NSE Data"
          desc="Prices and fundamentals straight from Yahoo Finance." delay={0.3} spanClass="md:col-span-2" />
        <Feature icon={<Cpu className="w-5 h-5" />} title="Deep Learning"
          desc="PyTorch & TensorFlow models trained on demand." delay={0.4} spanClass="md:col-span-2" />
        <Feature icon={<Zap className="w-5 h-5" />} title="Real-time Sentiment"
          desc="Headline tone scored as news breaks." delay={0.5} spanClass="md:col-span-1" />
        <Feature icon={<ShieldCheck className="w-5 h-5" />} title="Runs Locally"
          desc="Your keys and data never leave your machine." delay={0.6} spanClass="md:col-span-1" />
      </div>
    </section>
  )
}

function Feature({ icon, title, desc, delay, spanClass }) {
  return (
    <motion.div
      className={`${spanClass} ${card} cursor-default hover:border-[#1e293b]`}
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay }}
    >
      <div className="flex-1 flex items-center text-sky-400">{icon}</div>
      <div className="mt-3 text-left">
        <h3 className="text-base text-white font-semibold">{title}</h3>
        <p className="text-slate-400 text-sm mt-1">{desc}</p>
      </div>
    </motion.div>
  )
}
