# Movie Intelligence & Follow-up Assistant

An AI-powered RAG system and Agentic Assistant for processing subtitle (`.srt`) files across movie datasets to answer plot questions, retrieve dialogue with precise citations, and execute external actions via MCP.

---

## Key Features

1. **Subtitle RAG Engine with LLM Scene Enrichment**
   - **SRT Parsing & Chunking**: Timed chunking (60s window, 20s overlap) preserving timestamps and subtitle metadata.
   - **LLM Chunk Enrichment (Scene Summarization)**: Generates 1-2 sentence scene context summaries for each chunk to capture implicit narrative/plot actions not present in raw spoken dialogue.
   - **Persistent Disk Caching**: Summaries are cached in `data/cache/scene_summaries.json` so every chunk is summarized only once ($0 API cost on repeat runs).
   - **Hybrid Search**: Reciprocal Rank Fusion (RRF) combining **ChromaDB Cosine Vector Embeddings** (`all-MiniLM-L6-v2`) and **BM25 Keyword Matching**.
   - **Grounded Answers & Citations**: Provides precise source citations `[Movie Title, Timestamp Range]`.

2. **Agentic Decision Routing**
   - **`INFORMATIONAL`**: Directs plot and dialogue questions to the RAG Engine.
   - **`ACTION_EMAIL`**: Triggers RAG retrieval and dispatches scene breakdowns/dialogue via MCP Email Tool.
   - **`AMBIGUOUS`**: Identifies missing parameters (e.g., missing recipient email, vague query) and asks for clarification.

3. **Model Context Protocol (MCP) Tool Integration**
   - Implements standard MCP Email Tool (`send_email`) logging all email dispatches to `data/mcp_email_dispatch_log.json`.

4. **User Interface**
   - Interactive Streamlit app (`app.py`).

---

## Project Structure

```
Movie Intel/
├── app.py                     # Streamlit User Interface
├── data/
│   ├── cache/                 # Persistent disk cache for LLM scene summaries
│   └── movies/                # Subtitle .srt files
├── src/
│   ├── agent.py               # Master Agent Orchestrator
│   ├── chunker.py             # Subtitle window chunker (60s / 20s overlap)
│   ├── enricher.py            # LLM Chunk Enrichment (Scene Summarizer)
│   ├── mcp_email_tool.py      # Model Context Protocol Email Tool
│   ├── parser.py              # SRT parser
│   ├── rag_engine.py          # Grounded RAG QA Engine with citations
│   ├── router.py              # Agentic Decision Router
│   └── vector_store.py        # Hybrid Search (BM25 + ChromaDB RRF)
└── README.md
```

---

## Quick Start

### 1. Requirements & Setup
Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

*(Optional)* Set your Gemini API key in `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Run Master Agent CLI Test
```bash
python src/agent.py
```

### 3. Launch Streamlit Web UI
```bash
streamlit run app.py
```
