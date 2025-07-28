import { createContext, useContext } from 'react';

// Minimal shape – extended when the *retrieval_info* SSE event arrives.

export interface RetrievalInfo {
  need_search: boolean;
  info_packet: string | null;
}

interface ChatContextShape {
  retrievalInfo?: RetrievalInfo;
  setRetrievalInfo?: (ri: RetrievalInfo) => void;
}

export const ChatContext = createContext<ChatContextShape>({});

export function useChatContext() {
  return useContext(ChatContext);
}

