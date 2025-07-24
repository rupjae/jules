import { test, expect } from 'vitest';
import fs from 'fs';
import path from 'path';
import getContrastRatio from 'color-contrast';

const css = fs.readFileSync(
  path.join(__dirname, '../../../src/styles/catppuccin-macchiato.css'),
  'utf8'
);

test('css variables generated', () => {
  expect(css).toMatch(/--ctp-color-rosewater/);
});

test('text vs crust contrast AA', () => {
  const text = '#cad3f5';
  const crust = '#181926';
  const ratio = getContrastRatio(text, crust);
  expect(ratio).toBeGreaterThanOrEqual(4.5);
});
