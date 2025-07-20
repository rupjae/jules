// Load the Catppuccin sheet only when the theme is selected.  Storybook picks
// up environment variables via cross-env / `start-storybook -e`.
if (process.env.NEXT_PUBLIC_THEME === 'catppuccin') {
  require('../src/styles/catppuccin-macchiato.css');
}

export const parameters = { darkMode: true };
