import { useEffect, useState, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  Box,
  Button,
  CircularProgress,
  Paper,
  TextField,
  Typography,
  Collapse,
} from '@mui/material';

import InfoPacketToggle from './InfoPacketToggle';
import { useChatStream } from '../lib/useChatStream';

interface Message {
  sender: 'user' | 'assistant';
  content: string;
  infoPacket?: string | null;
  searchDecision?: boolean | null;
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

  // Index of the assistant “placeholder” message that is currently being
  // filled by the streaming response. The value is set right after the
  // placeholder is pushed to the `messages` array and cleared once the
  // backend notifies that the message has finished (via the `info_packet`
  // custom SSE event).
  const [activeAssistantIdx, setActiveAssistantIdx] = useState<number | null>(
    null
  );
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
    setActiveAssistantIdx(null);
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
  // ref to scrollable messages container
  const containerRef = useRef<HTMLDivElement>(null);

  // input that triggers streaming; kept separate to satisfy React hook rules
  const [streamInput, setStreamInput] = useState<string | null>(null);

  // Hook – connects when *streamInput* is non-null
  useChatStream(
    streamInput ?? '',
    (token) => {
      // Ignore tokens when no assistant message is active (should not happen)
      if (activeAssistantIdx === null) return;

      setMessages((prev) => {
        // Guard against race conditions where the messages array might have
        // grown (e.g. a "new conversation" was triggered) after the stream
        // started.
        if (activeAssistantIdx! >= prev.length) return prev;

        const copy = [...prev];
        copy[activeAssistantIdx!] = {
          ...copy[activeAssistantIdx!],
          content: copy[activeAssistantIdx!].content + token,
        };
        return copy;
      });
    },
    (meta) => {
      // Attach the info packet to the message that has just finished
      setMessages((prev) => {
        if (activeAssistantIdx === null || activeAssistantIdx >= prev.length)
          return prev;

        const copy = [...prev];
        copy[activeAssistantIdx] = {
          ...copy[activeAssistantIdx],
          infoPacket: meta.infoPacket,
          searchDecision: meta.searchDecision,
        };
        return copy;
      });

      // Streaming for this message has completed
      setActiveAssistantIdx(null);
      setLoading(false);
      setStreamInput(null);
    },
    backendUrl,
    threadId
  );

  const sendMessage = () => {
    if (!input.trim()) return;
    if (!input.trim()) return;

    setMessages((prev) => {
      const withNew: Message[] = [
        ...prev,
        { sender: 'user', content: input },
        { sender: 'assistant', content: '' },
      ];
      // Index of the placeholder we just pushed is the last element
      setActiveAssistantIdx(withNew.length - 1);
      return withNew;
    });

    setInput('');
    setLoading(true);

    // Fire the stream after state has been queued – we purposefully don’t wait
    // for React to finish the render cycle because the SSE handshake runs
    // independently.
    setStreamInput(input);
  };

  // no explicit cleanup necessary – handled inside useChatStream
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
          <div key={idx}>
            <Typography
              sx={{
                mb: 1,
                textAlign: m.sender === 'user' ? 'right' : 'left',
              }}
            >
              <strong>{m.sender === 'user' ? 'You' : 'Jules'}:</strong>{' '}
              {m.content}
            </Typography>
            {typeof window !== 'undefined' &&
              localStorage.getItem('showInfoPacket') === 'true' &&
              (m.infoPacket || typeof m.searchDecision === 'boolean') && (
                <Collapse in>
                  <Paper
                    elevation={1}
                    sx={{ p: 1, my: 1, bgcolor: 'grey.100' }}
                  >
                    {typeof m.searchDecision === 'boolean' && (
                      <Typography variant="caption" sx={{ fontStyle: 'italic' }}>
                        Retrieval considered helpful: {m.searchDecision ? 'Yes' : 'No'}
                      </Typography>
                    )}
                    {m.infoPacket && (
                      <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                        {m.infoPacket}
                      </pre>
                    )}
                  </Paper>
                </Collapse>
              )}
          </div>
        ))}
        {loading && <CircularProgress size={20} />}
      </Paper>
      <Box sx={{ mt: 1 }}>
        <InfoPacketToggle />
      </Box>
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
