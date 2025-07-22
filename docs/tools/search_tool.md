# ChromaSearchTool

This tool calls the `/api/chat/search` endpoint over HTTP.

Set `JULES_API_BASE` to point at the backend if not `http://localhost:8000`.

Example wiring with LangGraph:
```python
from jules.tools import ChromaSearchTool
search = ChromaSearchTool()
```
