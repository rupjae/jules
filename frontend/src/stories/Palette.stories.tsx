import React from 'react';
import * as tokens from '../styles/catppuccin-macchiato';

export default {
  title: 'Color Palette',
};

export const AllColors = () => (
  <div style={{ display: 'flex', flexWrap: 'wrap' }}>
    {Object.entries(tokens).map(([name, value]) => (
      <div key={name} style={{ margin: 8 }}>
        <div style={{ background: value as string, width: 80, height: 40 }} />
        <div>{name}</div>
      </div>
    ))}
  </div>
);
