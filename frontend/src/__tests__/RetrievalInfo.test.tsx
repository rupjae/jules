import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import RetrievalInfo from '../components/RetrievalInfo';

describe('RetrievalInfo component', () => {
  it('renders decision and hides packet when need_search is false', () => {
    render(<RetrievalInfo need_search={false} info_packet={null} />);

    expect(screen.getByText(/Retrieval considered helpful: No/i)).toBeInTheDocument();
    // accordion details should not render packet text
    expect(screen.queryByRole('textbox')).toBeNull();
  });

  it('shows packet when need_search is true', () => {
    const pkt = '• fact A\n• fact B';
    render(<RetrievalInfo need_search={true} info_packet={pkt} />);

    expect(screen.getByText(/Retrieval considered helpful: Yes/i)).toBeInTheDocument();
    expect(screen.getByText(pkt)).toBeInTheDocument();
  });
});

