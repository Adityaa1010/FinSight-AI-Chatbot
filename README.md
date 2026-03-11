# FinSight AI 🏦
### RAG-Powered AI Banking Assistant

> Instantly answers customer banking queries by retrieving answers from real bank policy PDFs — powered by LangChain, Groq LLaMA3, ChromaDB, and HuggingFace Embeddings.

---

## 📌 What is FinSight AI?

FinSight AI is a production-style **Retrieval-Augmented Generation (RAG)** chatbot built for the banking domain. Instead of relying on an LLM's general knowledge, it retrieves answers **directly from real bank documents** (SBI policy PDFs), ensuring accurate, grounded, and hallucination-free responses.

---

## 🧠 RAG Pipeline — 6 Steps

```
📄 PDF Document
      ↓
  1. LOAD       → PyPDFLoader reads every page of the PDF
      ↓
  2. CHUNK      → Split into 600-char overlapping chunks
      ↓
  3. EMBED      → HuggingFace MiniLM converts chunks to vectors (runs locally)
      ↓
  4. STORE      → ChromaDB saves vectors to disk (reused on next run)
      ↓
  5. RETRIEVE   → Top-4 most relevant chunks fetched per query
      ↓
  6. GENERATE   → Groq LLaMA3-70B generates grounded answer
      ↓
💬 Answer + Source Attribution
```

---

## 🗂️ Project Structure

```
FinSight-AI/
├── app.py                          # Streamlit UI
├── agent.py                        # Chatbot agent logic
├── rag_engine.py                   # Full 6-step RAG pipeline
├── utils.py                        # Helper functions
├── requirements.txt                # All dependencies
├── .env.example                    # Environment variable template
├── .env                            # Your API keys (never commit this)
├── data/
│   └── banking_knowledge_base.pdf  # SBI bank knowledge base
└── chroma_db/                      # Auto-created vector DB (after first run)
```

---

## ⚙️ Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Framework | LangChain 1.x | RAG pipeline orchestration |
| LLM | Groq LLaMA3-70B | Answer generation |
| Embeddings | HuggingFace MiniLM-L6-v2 | Text → vectors (local) |
| Vector DB | ChromaDB | Store & retrieve embeddings |
| PDF Loader | PyPDFLoader | Parse PDF documents |
| UI | Streamlit | Chat interface |
| Env | python-dotenv | API key management |

---

## 🚀 Setup & Run

### 1. Clone the project
```bash
git clone https://github.com/yourusername/finsight-ai.git
cd finsight-ai
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
```bash
cp .env.example .env
```
Open `.env` and fill in your keys:
```
GROQ_API_KEY=gsk_your_groq_key_here
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

| Key | Where to get it | Cost |
|---|---|---|
| `GROQ_API_KEY` | https://console.groq.com | Free |
| `HUGGINGFACEHUB_API_TOKEN` | https://huggingface.co/settings/tokens | Free |

### 5. Run the app
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

> ⚡ First run takes ~30 seconds to download the embedding model and build ChromaDB.
> Subsequent runs are instant — ChromaDB is reused from disk.

---

## 💬 Sample Questions to Ask

- *"What types of deposit accounts are available?"*
- *"How do I submit a grievance?"*
- *"What documents are required to open an account?"*
- *"How are depositor rights protected?"*
- *"What is the process for loan repayment?"*

---

## 🔑 Key Features

- ✅ **Zero hallucinations** — answers grounded strictly in retrieved PDF context
- ✅ **Source attribution** — shows exactly which PDF pages were used
- ✅ **Persistent vector DB** — ChromaDB reused across restarts (no re-embedding)
- ✅ **Modular pipeline** — each RAG step is a separate, testable function
- ✅ **Clean UI** — Streamlit chat interface with sidebar quick-questions
- ✅ **Production-style code** — separation of concerns across 4 files

---

## 🧩 File Responsibilities

| File | Role |
|---|---|
| `rag_engine.py` | Core RAG pipeline — all 6 steps clearly separated |
| `agent.py` | Wraps pipeline, handles errors, maintains chat history |
| `utils.py` | Shared helpers — logging, env vars, source formatting |
| `app.py` | Streamlit UI — chat display, sidebar, session state |

---

## 📄 Resume Description

**FinSight AI** | Python, LangChain, ChromaDB, Groq, HuggingFace, Streamlit
- Engineered a RAG-powered chatbot using LLaMA3-70B and ChromaDB vector search to resolve banking queries from real SBI policy PDFs with zero hallucinations.
- Architected a modular 6-step pipeline (Load → Chunk → Embed → Store → Retrieve → Generate) with persistent vector storage, reducing response latency by reusing pre-built embeddings.
- Deployed an interactive Streamlit interface with real-time source attribution and semantic search across banking documents, simulating a production-grade AI assistant.

---

## 🙋 FAQ

**Q: Do I need a GPU?**
A: No. Embeddings run on CPU. Groq handles LLM inference in the cloud for free.

**Q: Can I use a different PDF?**
A: Yes. Replace `data/banking_knowledge_base.pdf` with any PDF, delete `chroma_db/`, and restart.

**Q: Can I use a different LLM?**
A: Yes. Change `GROQ_MODEL` in `rag_engine.py`. Available Groq models: `llama3-8b-8192`, `llama3-70b-8192`, `mixtral-8x7b-32768`, `gemma2-9b-it`.

**Q: Why Groq instead of OpenAI?**
A: Groq is completely free, requires no credit card, and is faster than OpenAI for inference.

---

<p align="center">Built with ❤️ using LangChain · Groq · ChromaDB · Streamlit</p>
