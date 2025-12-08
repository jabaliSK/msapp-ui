/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#4F46E5', // Indigo 600
        secondary: '#1E293B', // Slate 800
        accent: '#0EA5E9', // Sky 500
      }
    },
  },
  plugins: [],
}