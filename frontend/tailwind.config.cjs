const { default: flattenColorPalette } = require('tailwindcss/lib/util/flattenColorPalette')

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  darkMode: 'class',
  // Preflight is disabled so Tailwind's global reset does not clobber the
  // existing hand-written CSS in index.css. Utilities still work; the minimal
  // border reset the components need is added manually in index.css.
  corePlugins: { preflight: false },
  theme: {
    extend: {
      animation: {
        aurora: 'aurora 60s linear infinite',
        marquee: 'marquee var(--duration, 30s) linear infinite',
        'marquee-reverse': 'marquee-reverse var(--duration, 30s) linear infinite',
      },
      keyframes: {
        aurora: {
          from: { backgroundPosition: '50% 50%, 50% 50%' },
          to: { backgroundPosition: '350% 50%, 350% 50%' },
        },
        marquee: {
          from: { transform: 'translateX(0)' },
          to: { transform: 'translateX(-50%)' },
        },
        'marquee-reverse': {
          from: { transform: 'translateX(-50%)' },
          to: { transform: 'translateX(0)' },
        },
      },
    },
  },
  plugins: [addVariablesForColors],
}

// Expose each Tailwind color as a global CSS variable (e.g. var(--blue-500)),
// which the Aurora background relies on.
function addVariablesForColors({ addBase, theme }) {
  const allColors = flattenColorPalette(theme('colors'))
  const newVars = Object.fromEntries(
    Object.entries(allColors).map(([key, val]) => [`--${key}`, val])
  )
  addBase({ ':root': newVars })
}
