# Retrieval-aware Architecture (v1.0.0)

## Executive overview

Traditional Jules prompts embedded the entire memory into one system prompt.  
That approach is brittle and expensive once the knowledge base grows.  Version 1.0.0 introduces a **two-stage Retrieval-Augmented Generation (RAG)** pipeline:

```
User ─▶ RetrievalAgent (gpt-4o-mini)
├─ need_search? ──╴ no ╶──▶ JulesAgent (gpt-4o) ─▶ Reply
╰─ yes ╮
╰▶ ChromaSearch (k=5) ─▶ summarise ≤150 tokens (info-packet)
╰───────────────▶ JulesAgent
```

Key benefits:

* **Cost-efficient** – small model decides whether we need external context.
* **Bounded context window** – summarisation caps additional tokens at 150.
* **Transparency** – the UI can display the *info-packet* via a toggle so power-users can audit the source material.
* **Single source of truth** – `config/agents.toml` holds all tunables (models, `k_hits`, token limits).

Terminology:

| Term | Description |
|------|-------------|
| RetrievalAgent | Lightweight GPT-4o-mini tool that (a) decides if search is needed and (b) produces a bullet-point summary of the Chroma passages. |
| Info-packet | ≤150-token bullet list that carries distilled context into the final LLM prompt. |
| LangGraph node | Each logical step in the pipeline (decision, search, LLM). Implemented in `backend/app/graphs/next_gen.py`. |
| SSE events | `/api/chat/stream?prompt=<text>` (GET) pushes incremental `data:` tokens plus a final `info_packet` event so the frontend can render background notes. |

### Quick client example

```ts
// Native browser EventSource usage (no polyfill required)
const prompt = "Summarise the last board meeting and cite the source";

const es = new EventSource(
  `/api/chat/stream?prompt=${encodeURIComponent(prompt)}`
);

es.onmessage = (e) => console.log("token", e.data);

es.addEventListener("info_packet", (e: MessageEvent) => {
  const notes = e.data === "null" ? null : e.data;
  console.log("info_packet", notes);
});
```

The POST-based code snippet found in early drafts is now obsolete – the backend
rejects body payloads and only supports **GET** requests for maximum
compatibility with the native `EventSource` API.

See `docs/arch/next_gen_graph.md` for a Mermaid sequence diagram.

---

### Retrieval Info SSE

Starting with PR #34 the backend emits an **additional SSE event** that carries
the *search decision* and optional *info-packet* so the UI can display it in a
collapsible panel.

```
Event: retrieval_info
Payload:
{
  "need_search": bool,
  "info_packet": str | null
}
```

Activate the event by adding the `show_retrieval=true` flag (any of
`1`, `true`, `yes` works) to the `/api/chat/stream` query string. When the flag
is absent or evaluates to *false* the backend omits the event entirely to avoid
unnecessary traffic.

The React frontend surfaces a “Show Retrieval Info” toggle that toggles this
flag, listens for the event, and renders both the decision and the packet.
