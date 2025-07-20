import type { AppProps } from 'next/app';

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

if (process.env.NEXT_PUBLIC_THEME === 'catppuccin') {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  require('../styles/catppuccin-macchiato.css');
}

export default function JulesApp({ Component, pageProps }: AppProps) {
  return <Component {...pageProps} />;
}

