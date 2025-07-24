import { describe, it, expect, afterEach } from 'vitest';
import { render, fireEvent, screen } from '@testing-library/react';
import InfoPacketToggle from '../components/InfoPacketToggle';

// Simple JSDOM localStorage shim for the test environment.
afterEach(() => {
  localStorage.clear();
});

describe('InfoPacketToggle', () => {
  it('persists state to localStorage', () => {
    render(<InfoPacketToggle />);

    const checkbox = screen.getByRole('checkbox') as HTMLInputElement;
    expect(checkbox.checked).toBe(false);

    fireEvent.click(checkbox);
    expect(checkbox.checked).toBe(true);
    expect(localStorage.getItem('showInfoPacket')).toBe('true');
  });
});

