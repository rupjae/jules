import { extendTheme } from '@chakra-ui/react';
import * as tokens from '../styles/catppuccin-macchiato';

export const catppuccinTheme = extendTheme({
  colors: {
    ctp: {
      rosewater: tokens.CtpColorRosewater,
      flamingo: tokens.CtpColorFlamingo,
      pink: tokens.CtpColorPink,
      mauve: tokens.CtpColorMauve,
      red: tokens.CtpColorRed,
      maroon: tokens.CtpColorMaroon,
      peach: tokens.CtpColorPeach,
      yellow: tokens.CtpColorYellow,
      green: tokens.CtpColorGreen,
      teal: tokens.CtpColorTeal,
      sky: tokens.CtpColorSky,
      sapphire: tokens.CtpColorSapphire,
      blue: tokens.CtpColorBlue,
      lavender: tokens.CtpColorLavender,
      text: tokens.CtpColorText,
      subtext1: tokens.CtpColorSubtext1,
      subtext0: tokens.CtpColorSubtext0,
      overlay2: tokens.CtpColorOverlay2,
      overlay1: tokens.CtpColorOverlay1,
      overlay0: tokens.CtpColorOverlay0,
      surface2: tokens.CtpColorSurface2,
      surface1: tokens.CtpColorSurface1,
      surface0: tokens.CtpColorSurface0,
      base: tokens.CtpColorBase,
      mantle: tokens.CtpColorMantle,
      crust: tokens.CtpColorCrust,
    },
  },
  styles: {
    global: {
      body: {
        bg: tokens.CtpSemanticBackground,
        color: tokens.CtpSemanticText,
      },
    },
  },
});
export default catppuccinTheme;
