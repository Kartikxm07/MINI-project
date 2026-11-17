import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

// Served under /app in production (Flask serves frontend/dist at /app).
// In dev, API calls are proxied to the Flask server on :5000 so there is no CORS.
export default defineConfig({
  base: '/app/',
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/predict': 'http://127.0.0.1:5000',
      '/api': 'http://127.0.0.1:5000',
      '/options/charts': 'http://127.0.0.1:5000',
    },
  },
})
