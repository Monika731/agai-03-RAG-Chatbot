# Coursera AI/ML RAG Chatbot
An intelligent chatbot that answers questions about AI & Machine Learning courses on Coursera, built using a full RAG pipeline.

---

## Project Description

This project scrapes Coursera's AI/ML course pages, generates synthetic Q/A pairs using Groq LLaMA 3.3 70B, and provides a Streamlit chatbot with hybrid retrieval — in-memory semantic search on Q/A pairs first, with ChromaDB vector search as fallback on full course documents.

**Target Website:** [Coursera AI/ML Courses](https://www.coursera.org/courses?query=machine+learning)

---

## Architecture

```
User Query
    │
    ▼
Hybrid Retriever
    ├── 1. In-memory cosine similarity on Q/A pairs (sentence-transformers)
    │       └── If distance < 0.35 → use Q/A answer directly
    └── 2. Fallback: ChromaDB vector search on full course documents
            └── Retrieve top-3 relevant chunks
    │
    ▼
Groq LLaMA 3.3 70B → Final Answer with source citation
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Web Scraping | `requests` + `BeautifulSoup4` |
| LLM | Groq API (LLaMA 3.3 70B Versatile) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB (course documents only) |
| Q/A Search | In-memory cosine similarity (sentence-transformers) |
| Hybrid Retrieval | Custom cosine distance thresholding |
| UI | Streamlit |

---

## Repository Structure

```
rag-chatbot/
├── data/
│   ├── raw/                  # Scraped .txt files + all_courses.json
│   ├── processed/            # qa_dataset.csv + qa_dataset.json
│   └── chroma_db/            # Persistent ChromaDB (auto-generated)
├── src/
│   ├── scraper.py            # Phase 1: Coursera scraper
│   ├── qa_generator.py       # Phase 2: Groq Q/A generation
│   ├── vector_store.py       # Phase 3: ChromaDB indexing (course docs only)
│   ├── retriever.py          # Phase 4: In-memory Q/A search + ChromaDB fallback
│   └── chatbot.py            # Phase 4: Chatbot with memory
├── app.py                    # Phase 5: Streamlit UI
├── run_pipeline.py           # Run all phases at once
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## How to Run Locally

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/rag-chatbot-yourname.git
cd rag-chatbot-yourname
pip install -r requirements.txt
```

### 2. Set API Key

```bash
cp .env.example .env
# Edit .env and add your Groq API key
# Get a free key at: https://console.groq.com
```

### 3. Run the Pipeline

**Option A — All at once:**
```bash
python run_pipeline.py
```

**Option B — Step by step:**
```bash
python -m src.scraper          # Phase 1: Scrape ~25 Coursera pages
python -m src.qa_generator     # Phase 2: Generate up to 50 Q/A pairs
python -m src.vector_store     # Phase 3: Build ChromaDB index (course docs only)
```

### 4. Launch the Chatbot

```bash
streamlit run app.py
```

---

## Sample Questions to Ask

- "What is the best course to start learning machine learning?"
- "How long does the Deep Learning Specialization take?"
- "What skills will I gain from Andrew Ng's ML course?"
- "Which course teaches Python for AI and data science?"
- "Is there a course on generative AI and LLMs?"

---

## Features

- **In-memory Q/A Search**: 50 Q/A pairs loaded and searched with cosine similarity at runtime — no vector DB overhead
- **ChromaDB Fallback**: Full course documents indexed in ChromaDB for broader vector search
- **Chat Memory**: Retains last 3 exchanges for context-aware answers
- **Source Citation**: Every answer shows which page/Q/A pair was used
- **Streamlit UI**: Clean header layout with stats, sample questions, and clear chat

---

## Team

- Solo Project

---

## Limitations & Future Improvements

- Coursera pages are JavaScript-heavy; some content is only accessible via JSON-LD and meta tags
- Could add Selenium for full JS rendering of dynamic sections
- Could expand to 50+ course pages for richer coverage
- Could add re-ranking with a cross-encoder model
