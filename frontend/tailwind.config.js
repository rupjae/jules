const palette = require('@catppuccin/palette');
module.exports = {
  darkMode: 'class',
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ctp: palette.flavors.macchiato.colors,
      },
    },
  },
  plugins: [],
};
