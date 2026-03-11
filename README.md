# 🏦 AI Banking Chatbot — RAG Pipeline

A production-style AI chatbot for banking queries, built with:
- **LangChain** — RAG pipeline orchestration
- **HuggingFace** — Embeddings + LLM (Mistral-7B)
- **Chroma** — Local vector database
- **Streamlit** — Clean chat UI

---

## 🗂️ Project Structure

```
banking_chatbot/
├── app.py                        # Streamlit UI
├── agent.py                      # Chatbot agent logic
├── rag_engine.py                 # Full RAG pipeline (Steps 1–6)
├── utils.py                      # Helper functions
├── requirements.txt              # All dependencies
├── .env.example                  # Environment variable template
├── data/
│   └── banking_knowledge_base.pdf   # Sample banking PDF
└── chroma_db/                    # Auto-created vector DB (after first run)
```

---

## ⚙️ Setup Instructions (Mac/Linux)

### 1. Clone / Download the project
```bash
cd banking_chatbot
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your HuggingFace API token
```bash
cp .env.example .env
```
Open `.env` and replace `your_huggingface_token_here` with your actual token.

> Get your free token at: https://huggingface.co/settings/tokens

### 5. Run the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 🧠 RAG Pipeline Explained

```
rag_engine.py
│
├── Step 1: load_documents()         → Load PDF with PyPDFLoader
├── Step 2: split_documents()        → Split into 500-char chunks
├── Step 3: create_embeddings()      → HuggingFace MiniLM embeddings
├── Step 4: store_in_vectordb()      → Save to Chroma vector DB
├── Step 5: create_retriever()       → Top-3 similarity search
└── Step 6: create_llm() +           → Mistral-7B via HuggingFace API
           build_qa_chain()          → RetrievalQA chain
```

---

## 💬 Sample Questions to Ask

- *"What types of accounts do you offer?"*
- *"How do I apply for a home loan?"*
- *"What are the ATM withdrawal limits?"*
- *"How do I reset my online banking password?"*
- *"What documents do I need to open an account?"*
- *"How do I report a fraudulent transaction?"*

---

## 🛠️ Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| RAG Framework | LangChain | Pipeline orchestration |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Text → vectors |
| Vector DB | Chroma | Store & retrieve embeddings |
| LLM | Mistral-7B-Instruct | Answer generation |
| PDF Loader | PyPDF | Parse PDF documents |
| UI | Streamlit | Chat interface |

---

## 📝 Notes

- The first startup takes ~30 seconds (model download + PDF embedding)
- Subsequent runs are faster (Chroma DB is cached to disk)
- The `chroma_db/` folder is auto-created on first run
- You can replace the PDF in `data/` with any banking document
