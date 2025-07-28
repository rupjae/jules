// Lightweight chat window used by the unit-tests added in PR #34 follow-up.
// The component intentionally omits most of the rich UI found in
// `components/Chat.tsx` – it only renders the pieces required by
// `ChatWindow.retrieval.test.tsx` so we avoid large-scale refactors.

import { useState, useEffect } from 'react';
import { Box, Switch, Collapse, Paper, Typography } from '@mui/material';

import RetrievalInfo from '../../components/RetrievalInfo';
import { ChatContext, RetrievalInfo as RetrievalInfoType } from '../../context/chat';

interface Props {
  // *message* is provided by the test to trigger the SSE setup.  In the real
  // application this would come from the input box.
  message?: string;
  backendUrl?: string;
}

export default function ChatWindow({ message = 'hi', backendUrl = '' }: Props) {
  const [show, setShow] = useState(false);
  const [retrievalInfo, setRetrievalInfo] = useState<RetrievalInfoType | undefined>(undefined);

  // ------------------------------------------------------------------
  // SSE – connects once when the component mounts.  The tests stub the
  // global EventSource so we can synchronously dispatch events.
  // ------------------------------------------------------------------

  useEffect(() => {
    const url = `${backendUrl}/api/chat/stream?prompt=${encodeURIComponent(message)}&show_retrieval=true`;
    const es = new EventSource(url);

    es.addEventListener('retrieval_info', (e: MessageEvent) => {
      try {
        const parsed = JSON.parse(e.data) as RetrievalInfoType;
        setRetrievalInfo(parsed);
      } catch {
        // ignore – malformed
      }
    });

    return () => es.close();
  }, [backendUrl, message]);

  return (
    <ChatContext.Provider value={{ retrievalInfo, setRetrievalInfo }}>
      <Box>
        <label style={{ cursor: 'pointer' }}>
          Show Retrieval Info <Switch checked={show} onChange={(e) => setShow(e.target.checked)} />
        </label>

        <Collapse in={show && !!retrievalInfo}>
          {retrievalInfo && (
            <Paper elevation={1} sx={{ p: 1, mt: 1, bgcolor: 'grey.100' }}>
              <RetrievalInfo
                need_search={retrievalInfo.need_search}
                info_packet={retrievalInfo.info_packet}
              />
            </Paper>
          )}
        </Collapse>
      </Box>
    </ChatContext.Provider>
  );
}

