/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      /* [CHANGE] Updated color palette to deep cinematic tones */
      colors: {
        dark: "#020617",        /* Slate-950 — deepest background */
        card: "#0f172a",        /* Slate-900 — card surfaces */
        accent: "#38BDF8",      /* Sky-400 — primary UI accent */
        danger: "#e11d48",      /* Rose-600 — cinematic danger red */
        success: "#10b981",     /* Emerald-500 — cinematic safe green */
      },
      /* [CHANGE] Added font families for Inter and JetBrains Mono */
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      /* [CHANGE] Custom animations for ambient motion */
      animation: {
        'breathe': 'breathe 3s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'glitch': 'glitch 0.3s ease-in-out',
      },
      keyframes: {
        breathe: {
          '0%, 100%': { opacity: '0.4' },
          '50%': { opacity: '0.8' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-8px)' },
        },
        glitch: {
          '0%': { transform: 'translate(0)' },
          '20%': { transform: 'translate(-3px, 3px)' },
          '40%': { transform: 'translate(3px, -3px)' },
          '60%': { transform: 'translate(-2px, -2px)' },
          '80%': { transform: 'translate(2px, 2px)' },
          '100%': { transform: 'translate(0)' },
        },
      },
    },
  },
  plugins: [],
}