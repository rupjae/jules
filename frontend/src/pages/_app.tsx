import type { AppProps } from 'next/app';
// Include Catppuccin Macchiato CSS variables globally
import '../styles/catppuccin-macchiato.css';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';

// ---------------------------------------------------------------------------
// Optional theme inclusion
// ---------------------------------------------------------------------------
// The Catppuccin Macchiato palette is beautiful but not always desired in every
// deployment.  Rather than hard-coding the stylesheet we allow operators to opt
// in by setting an environment variable at build-time (or run-time for Vercel
// style environments).
//
//     NEXT_PUBLIC_THEME=catppuccin  npm run build
//
// If the variable is *not* set the application renders using the default MUI
// theme and no additional CSS payload is shipped, keeping bundles small.

// Default to Catppuccin Macchiato theme; always include its stylesheet
// eslint-disable-next-line @typescript-eslint/no-var-requires
require('../styles/catppuccin-macchiato.css');

// Create MUI theme using Catppuccin CSS variables
const theme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: 'var(--ctp-semantic-background)',
      paper: 'var(--ctp-semantic-surface)',
    },
    text: {
      primary: 'var(--ctp-semantic-text)',
      secondary: 'var(--ctp-semantic-subtext1)',
    },
    primary: {
      main: 'var(--ctp-semantic-primary)',
    },
    error: {
      main: 'var(--ctp-semantic-error)',
    },
  },
});

export default function JulesApp({ Component, pageProps }: AppProps) {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Component {...pageProps} />
    </ThemeProvider>
  );
}
