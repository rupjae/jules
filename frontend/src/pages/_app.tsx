import type { AppProps } from 'next/app';
// Include Catppuccin Macchiato CSS variables globally
// Import Catppuccin Macchiato CSS variables (if used elsewhere)
import '../styles/catppuccin-macchiato.css';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import {
  CtpSemanticBackground,
  CtpSemanticText,
  CtpSemanticSurface,
  CtpSemanticPrimary,
  CtpSemanticError,
  CtpColorSubtext1,
} from '../styles/catppuccin-macchiato';

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
// Create MUI theme using Catppuccin Macchiato palette constants
const theme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: CtpSemanticBackground,
      paper: CtpSemanticSurface,
    },
    text: {
      primary: CtpSemanticText,
      secondary: CtpColorSubtext1,
    },
    primary: {
      main: CtpSemanticPrimary,
    },
    error: {
      main: CtpSemanticError,
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
