import { useEffect, useState, useRef } from 'react';
import {
  Box,
  Button,
  CircularProgress,
  Paper,
  TextField,
  Typography,
} from '@mui/material';

interface Message {
  sender: 'user' | 'assistant';
  content: string;
}

// When served from the same origin, no absolute URL is needed
const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? '';

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  const sendMessage = () => {
    if (!input.trim()) return;
    const newMessages: Message[] = [...messages, { sender: 'user', content: input }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    const url = `${backendUrl}/api/chat?message=${encodeURIComponent(input)}`;
    const es = new EventSource(url, { withCredentials: false });

    eventSourceRef.current = es;

    es.onopen = () => {
      // EventSource established
    };

    let assistantBuffer = '';

    es.onmessage = (ev) => {
      assistantBuffer += ev.data;
      setMessages([
        ...newMessages,
        { sender: 'assistant', content: assistantBuffer },
      ]);
    };

    es.onerror = () => {
      es.close();
      setLoading(false);
    };
  };

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  return (
    <Box sx={{ maxWidth: 600, margin: 'auto', mt: 4 }}>
      <Paper sx={{ p: 2, height: 500, overflowY: 'auto' }}>
        {messages.map((m, idx) => (
          <Typography
            key={idx}
            sx={{
              mb: 1,
              textAlign: m.sender === 'user' ? 'right' : 'left',
            }}
          >
            <strong>{m.sender === 'user' ? 'You' : 'Jules'}:</strong> {m.content}
          </Typography>
        ))}
        {loading && <CircularProgress size={20} />}
      </Paper>
      <Box sx={{ display: 'flex', mt: 2 }}>
        <TextField
          fullWidth
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              sendMessage();
            }
          }}
          placeholder="Type your message..."
        />
        <Button onClick={sendMessage} variant="contained" sx={{ ml: 1 }}>
          Send
        </Button>
      </Box>
    </Box>
  );
}
