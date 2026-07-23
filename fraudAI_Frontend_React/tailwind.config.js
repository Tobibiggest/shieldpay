/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background:  "hsl(var(--background) / <alpha-value>)",
        foreground:  "hsl(var(--foreground) / <alpha-value>)",
        border:      "hsl(var(--border) / <alpha-value>)",
        input:       "hsl(var(--input) / <alpha-value>)",
        ring:        "hsl(var(--ring) / <alpha-value>)",
        accent: {
          DEFAULT:   "hsl(var(--accent) / <alpha-value>)",
          foreground:"hsl(var(--accent-foreground) / <alpha-value>)",
        },
        muted: {
          DEFAULT:   "hsl(var(--muted) / <alpha-value>)",
          foreground:"hsl(var(--muted-foreground) / <alpha-value>)",
        },
        card: {
          DEFAULT:   "hsl(var(--card) / <alpha-value>)",
          foreground:"hsl(var(--card-foreground) / <alpha-value>)",
        },
        primary: {
          DEFAULT:   "hsl(var(--primary) / <alpha-value>)",
          foreground:"hsl(var(--primary-foreground) / <alpha-value>)",
        },
        secondary: {
          DEFAULT:   "hsl(var(--secondary) / <alpha-value>)",
          foreground:"hsl(var(--secondary-foreground) / <alpha-value>)",
        },
        destructive: {
          DEFAULT:   "hsl(var(--destructive) / <alpha-value>)",
          foreground:"hsl(var(--destructive-foreground) / <alpha-value>)",
        },
      },
      fontFamily: {
        sans:    ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["Playfair Display", "ui-serif", "Georgia", "serif"],
        mono:    ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      keyframes: {
        "reveal-down": {
          from: { clipPath: "inset(0 0 100% 0)", transform: "translateY(-20px)" },
          to:   { clipPath: "inset(0 0 0% 0)",   transform: "translateY(0)" },
        },
        scanline: {
          "0%":   { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(400%)" },
        },
        "fade-in-up": {
          from: { opacity: 0, transform: "translateY(16px)" },
          to:   { opacity: 1, transform: "translateY(0)" },
        },
      },
      animation: {
        "reveal":   "reveal-down 1.2s cubic-bezier(0.16,1,0.3,1) both",
        "scan":     "scanline 4s linear infinite",
        "fade-up":  "fade-in-up 0.8s cubic-bezier(0.16,1,0.3,1) both",
      },
    },
  },
  plugins: [],
}