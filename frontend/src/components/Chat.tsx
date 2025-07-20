import { useEffect, useState, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
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

// ---------------------------------------------------------------------------
// Backend base URL resolution
// ---------------------------------------------------------------------------
// In production the frontend and backend are usually served from the same
// origin (e.g., behind one reverse proxy) so we default to *relative* paths.
// During local development however `next dev` runs on :3000 while the FastAPI
// server sits on :8000.  Falling back to an empty string therefore breaks the
// connection – requests are sent to the *frontend* origin which naturally
// returns a 404.  We fix the issue by detecting the common dev setup and
// defaulting to "http://localhost:8000" when NEXT_PUBLIC_BACKEND_URL is left
// undefined.

let backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
if (!backendUrl) {
  // Heuristic: if the page was loaded from localhost:3000 assume FastAPI on 8000
  if (typeof window !== 'undefined' && window.location.port === '3000') {
    backendUrl = 'http://localhost:8000';
  } else {
    backendUrl = '';
  }
}

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  // threadId: generated once (server or client) and persisted in localStorage on the client
  const [threadId, setThreadId] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('julesThreadId') ?? uuidv4();
    }
    return uuidv4();
  });
  // persist threadId when it changes
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('julesThreadId', threadId);
    }
  }, [threadId]);
  // start a new conversation
  const handleNewConversation = () => {
    const newId = uuidv4();
    setThreadId(newId);
    setMessages([]);
    if (typeof window !== 'undefined') {
      localStorage.setItem('julesThreadId', newId);
    }
  };
  // message history loaded from backend
  const [tokenCount, setTokenCount] = useState<number>(0);
  // load previous conversation on mount
  // load previous conversation on mount
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const loadHistory = async () => {
      try {
        const url = `${backendUrl}/api/chat/history?thread_id=${threadId}`;
        const res = await fetch(url);
        if (res.ok) {
          const data: Message[] = await res.json();
          setMessages(data);
        }
      } catch {
        // ignore
      }
    };
    loadHistory();
  }, [threadId]);

  // update token count whenever messages change (approximate: chars/4)
  useEffect(() => {
    const approx = Math.round(
      messages.reduce((sum, m) => sum + m.content.length, 0) / 4
    );
    setTokenCount(approx);
  }, [messages]);
  const eventSourceRef = useRef<EventSource | null>(null);
  // ref to scrollable messages container
  const containerRef = useRef<HTMLDivElement>(null);

  const sendMessage = () => {
    if (!input.trim()) return;
    const newMessages: Message[] = [...messages, { sender: 'user', content: input }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    const url =
      `${backendUrl}/api/chat?thread_id=${threadId}` +
      `&message=${encodeURIComponent(input)}`;
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
  // scroll to bottom whenever messages change
  useEffect(() => {
    const el = containerRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages]);

  return (
    <Box sx={{ maxWidth: 600, margin: 'auto', mt: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="caption">Memory tokens: {tokenCount}</Typography>
        <Button size="small" onClick={handleNewConversation}>New Conversation</Button>
      </Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="caption">Session ID: {threadId}</Typography>
      </Box>
      <Paper ref={containerRef} sx={{ p: 2, height: 500, overflowY: 'auto' }}>
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
          name="chatMessage"
          autoComplete="off"
          value={input}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setInput(e.target.value)}
          onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
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
