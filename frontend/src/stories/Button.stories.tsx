import React from 'react';
import { Button } from '@mui/material';

export default {
  title: 'Sample/Button',
};

export const Primary = () => (
  <Button variant="contained" style={{ background: 'var(--ctp-semantic-primary)', color: 'var(--ctp-semantic-text)' }}>
    Button
  </Button>
);
