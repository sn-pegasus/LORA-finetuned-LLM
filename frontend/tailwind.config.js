/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'chat-gpt': {
          'bg': '#343541',
          'bg-secondary': '#40414f',
          'bg-tertiary': '#565869',
          'text': '#ececf1',
          'text-secondary': '#8e8ea0',
          'border': '#565869',
          'accent': '#10a37f',
        },
      },
    },
  },
  plugins: [],
}

