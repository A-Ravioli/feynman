/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Futuristic color palette
        cyber: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
          950: '#082f49',
        },
        neon: {
          blue: '#00ffff',
          purple: '#8b5cf6',
          pink: '#ec4899',
          green: '#10b981',
          orange: '#f59e0b',
        },
        dark: {
          bg: '#0a0a0f',
          surface: '#1a1a2e',
          card: '#16213e',
          border: '#2d3748',
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Monaco', 'Cascadia Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-up': 'slide-up 0.3s ease-out',
        'slide-down': 'slide-down 0.3s ease-out',
        'fade-in': 'fade-in 0.5s ease-out',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'float': 'float 3s ease-in-out infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { opacity: '0.8', boxShadow: '0 0 20px rgba(59, 130, 246, 0.5)' },
          '50%': { opacity: '1', boxShadow: '0 0 40px rgba(59, 130, 246, 0.8)' },
        },
        'slide-up': {
          '0%': { transform: 'translateY(100%)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        'slide-down': {
          '0%': { transform: 'translateY(-100%)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'glow': {
          '0%': { textShadow: '0 0 5px rgba(59, 130, 246, 0.5)' },
          '100%': { textShadow: '0 0 20px rgba(59, 130, 246, 1)' },
        },
        'float': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'cyber-grid': 'linear-gradient(transparent 24px, rgba(59, 130, 246, 0.03) 25px, rgba(59, 130, 246, 0.03) 26px, transparent 27px, transparent 49px, rgba(59, 130, 246, 0.03) 50px), linear-gradient(90deg, transparent 24px, rgba(59, 130, 246, 0.03) 25px, rgba(59, 130, 246, 0.03) 26px, transparent 27px, transparent 49px, rgba(59, 130, 246, 0.03) 50px)',
      },
      backgroundSize: {
        'grid': '50px 50px',
      }
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}