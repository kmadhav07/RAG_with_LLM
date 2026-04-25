---
title: E5 RAG App
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

<div align="center">

# 🔍 E5-RAG — Semantic PDF Search & AI Analysis

### Retrieval-Augmented Generation App Using E5-base-v2 Embeddings + Google Gemini

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face%20Spaces-blue?style=for-the-badge)](https://huggingface.co/spaces/DumbMaddy/E5-RAG-App)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-E5--base--v2-yellow?style=flat-square)](https://huggingface.co/intfloat/e5-base-v2)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Gemini](https://img.shields.io/badge/Google-Gemini%20AI-4285F4?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)

**A semantic document retrieval application that uses E5-base-v2 embeddings to find the most relevant PDFs for your query. Optionally integrates Google Gemini for AI-powered summaries and interactive follow-up chat.**

[🚀 Try the Live Demo](https://huggingface.co/spaces/DumbMaddy/E5-RAG-App) · [📄 View on Hugging Face](https://huggingface.co/spaces/DumbMaddy/E5-RAG-App)

</div>

---

## 🌐 Live Demo

> **Try it instantly — no setup required!**
>
> 👉 **[https://huggingface.co/spaces/DumbMaddy/E5-RAG-App](https://huggingface.co/spaces/DumbMaddy/E5-RAG-App)**

Hosted on **Hugging Face Spaces** via Docker. Comes with **11 pre-bundled PDF documents** spanning machine learning, deep learning, Python programming, and more — ready to search immediately.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **E5-base-v2 Embeddings** | State-of-the-art text embeddings from `intfloat/e5-base-v2` (~438 MB model) |
| 📄 **PDF Upload & Processing** | Upload multiple PDFs — text is extracted and embedded automatically |
| 📚 **11 Default Documents** | Pre-bundled PDFs covering ML, DL, Python, control theory, and more |
| 🔎 **Semantic Search** | Cosine similarity ranking of documents against your natural language query |
| 🤖 **Gemini AI Integration** | Optional AI-powered summary of results + interactive follow-up chat |
| 🔄 **Dual Modes** | Toggle between **Without Gemini** (pure retrieval) and **With Gemini** (RAG + chat) |
| 📊 **Similarity Scores** | Visual progress bars showing cosine similarity for each result |
| 💬 **Follow-up Chat** | Ask follow-up questions about your documents with Gemini context |
| 📱 **Responsive UI** | Clean, modern design with Inter font — works on all devices |
| 🗂️ **Selective Loading** | Choose specific default documents via checkboxes — no need to load all 11 |

---

## 🏗️ System Architecture

```
Browser (Frontend)
  │
  │  Query + PDFs (upload or select defaults)
  │  Toggle: Without Gemini / With Gemini
  ▼
Flask Server (Backend)
  │
  ├── 1. PDF Text Extraction (PyPDF2)
  │     └─ Extract text from all pages
  │
  ├── 2. Text Tokenization & Truncation
  │     └─ Truncate to first 512 tokens per document
  │
  ├── 3. E5 Embedding Generation
  │     ├─ Query:    "query: <user question>"     → 768-dim vector
  │     └─ Document: "passage: <doc text>"         → 768-dim vector
  │
  ├── 4. Cosine Similarity Ranking
  │     └─ Rank all documents by similarity to query
  │
  ├── 5. Return Top-K Results (k=5)
  │
  └── 6. [Optional] Gemini AI
        ├─ Generate summary of why top docs match
        └─ Handle follow-up chat questions
```

### RAG Pipeline Flow

```
┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Query   │───▶│ Tokenize  │───▶│ E5 Embed │───▶│ Cosine   │───▶│ Top-K    │
│  Input   │    │ (512 max) │    │ (768-dim)│    │ Similarity│   │ Results  │
└──────────┘    └───────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                                      │
                                                            [With Gemini?]
                                                                      │
                                                    ┌─────────────────▼──────┐
                                                    │  Gemini AI Summary     │
                                                    │  + Follow-up Chat      │
                                                    └────────────────────────┘
```

---

## 🧬 How It Works

### 1. PDF Text Extraction

Documents are processed using **PyPDF2** — text is extracted from every page and concatenated into a single string per document.

### 2. E5 Embedding Model

The app uses **[intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)** — a state-of-the-art text embedding model from Microsoft Research.

| Property | Value |
|----------|-------|
| Model | `intfloat/e5-base-v2` |
| Embedding Dimension | 768 |
| Max Sequence Length | 512 tokens |
| Model Size | ~438 MB |
| Pooling | Mean pooling over last hidden states |

**Key Design Choice — E5 Prefixes:**
- Queries are prefixed with `"query: "` before embedding
- Documents are prefixed with `"passage: "` before embedding
- This asymmetric prefixing is critical for E5 models to work correctly

### 3. Document Truncation

Each document is truncated to exactly **512 tokens** (the model's max sequence length) using the model's own tokenizer. This ensures:
- Consistent embedding quality
- No silent truncation inside the model
- Deterministic behavior

### 4. Cosine Similarity Ranking

Query and document embeddings are compared using **cosine similarity** via scikit-learn. Documents are ranked from most to least similar, and the **top 5** are returned.

### 5. Gemini AI Integration (Optional)

When enabled, the app uses **Google Gemini** to:
- **Summarize** why the top documents match the query (max 200 words)
- **Chat** — answer follow-up questions using the document content as context

The app tries multiple Gemini models in order: `gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-1.5-flash` → `gemini-1.5-pro`

---

## 📁 Project Structure

```
E5-RAG-App/
├── app.py                  # Flask backend (497 lines)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker config for HF Spaces
├── README.md               # This file
├── templates/
│   └── index.html          # Frontend UI (~44 KB)
├── default_pdfs/           # 11 pre-bundled PDF documents
│   ├── 103079.pdf
│   ├── 1404.4548v2.pdf
│   ├── 2005.11401v4.pdf
│   ├── 2111.03796v1.pdf
│   ├── 2204.08129v2.pdf
│   ├── From_Data_to_Dynamics_....pdf
│   ├── Identification_and_Classification_....pdf
│   ├── Introduction_to_Python_Programming_-_WEB.pdf
│   ├── PaythonProgramming.pdf
│   ├── narendramodi_gamechanger.PDF
│   └── ssrn-3669801.pdf
└── uploads/                # Temporary storage for uploaded PDFs
```

---

## 🚀 Quick Start

### Option 1: Live Demo (No Setup)

👉 **[https://huggingface.co/spaces/DumbMaddy/E5-RAG-App](https://huggingface.co/spaces/DumbMaddy/E5-RAG-App)**

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://huggingface.co/spaces/DumbMaddy/E5-RAG-App
cd E5-RAG-App

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Set your own Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Run the application
python app.py
```

Open **http://localhost:7860** in your browser.

> **Note:** The E5 model (~438 MB) will be downloaded on first run and cached locally.

### Option 3: Docker

```bash
docker build -t e5-rag-app .
docker run -p 7860:7860 -e GEMINI_API_KEY="your-key" e5-rag-app
```

Open **http://localhost:7860**.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main web UI |
| `GET` | `/list-defaults` | List available default PDF filenames |
| `POST` | `/load-defaults` | Search selected default PDFs with a query |
| `POST` | `/upload` | Upload PDFs and search with a query |
| `POST` | `/chat` | Follow-up chat with Gemini (requires prior search) |
| `GET` | `/serve-pdf/<source>/<filename>` | Serve a PDF file for viewing |

### POST `/load-defaults` — Request Body

```json
{
  "query": "What are neural networks?",
  "selected_files": ["2005.11401v4.pdf", "Introduction_to_Python_Programming_-_WEB.pdf"],
  "use_gemini": true
}
```

### POST `/upload` — Multipart Form

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Search query |
| `pdfs` | file[] | One or more PDF files |
| `use_gemini` | string | `"true"` or `"false"` |

### Example Response

```json
{
  "success": true,
  "query": "What are neural networks?",
  "total_documents": 5,
  "top_documents": [
    {
      "filename": "2005.11401v4.pdf",
      "similarity": 0.8734,
      "preview": "First 500 characters of document..."
    }
  ],
  "gemini_enabled": true,
  "gemini_summary": "These documents are relevant because...",
  "chat_context": "..."
}
```

### POST `/chat` — Follow-up Questions

```json
{
  "question": "Can you explain backpropagation from these documents?",
  "context": "<context from previous search>"
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Embedding Model | `intfloat/e5-base-v2` (HuggingFace Transformers) |
| LLM (Optional) | Google Gemini (2.5-flash / 2.0-flash / 1.5-flash / 1.5-pro) |
| Web Framework | Flask 3.0 |
| PDF Processing | PyPDF2 3.0 |
| Similarity Search | scikit-learn (cosine similarity) |
| Tensor Operations | PyTorch ≥ 2.2 |
| Frontend | HTML5, Vanilla CSS, JavaScript |
| Typography | Google Fonts — Inter |
| Production Server | Gunicorn (4 threads, 300s timeout) |
| Containerization | Docker (Python 3.11-slim) |
| Deployment | Hugging Face Spaces |

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | hardcoded fallback | Google Gemini API key (set via env var) |
| `PORT` | `7860` | Server port |
| `MAX_CONTENT_LENGTH` | 100 MB | Maximum total upload size |
| `MODEL_NAME` | `intfloat/e5-base-v2` | Embedding model to use |
| `DEFAULT_QUERY` | `"What are the key concepts..."` | Pre-filled query for demo |

---

## 📚 Pre-Bundled Default Documents

The app ships with **11 diverse PDF documents** for instant testing:

| # | Document | Topic |
|---|----------|-------|
| 1 | `Introduction_to_Python_Programming_-_WEB.pdf` | Python programming textbook |
| 2 | `PaythonProgramming.pdf` | Python programming guide |
| 3 | `2005.11401v4.pdf` | Machine learning research paper |
| 4 | `2111.03796v1.pdf` | Deep learning / AI research |
| 5 | `2204.08129v2.pdf` | Computer science research |
| 6 | `1404.4548v2.pdf` | Academic research paper |
| 7 | `103079.pdf` | Technical document |
| 8 | `From_Data_to_Dynamics_....pdf` | Control theory / MIMO channels |
| 9 | `Identification_and_Classification_....pdf` | Animal kingdom classification |
| 10 | `narendramodi_gamechanger.PDF` | Political / biographical |
| 11 | `ssrn-3669801.pdf` | SSRN research paper |

---

## 🐳 Docker Configuration

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates/ templates/
COPY default_pdfs/ default_pdfs/

EXPOSE 7860

# Production server with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860",
     "--timeout", "300", "--workers", "1",
     "--threads", "4", "app:app"]
```

> **Note:** Uses single worker with 4 threads since the E5 model is loaded into memory once and shared. The 300s timeout accommodates large PDF processing.

---

## 🔄 Modes of Operation

### Mode 1: Without Gemini (Pure Retrieval)

- Upload PDFs or select from defaults
- Enter a natural language query
- Get **top 5** documents ranked by semantic similarity
- See cosine similarity scores and text previews

### Mode 2: With Gemini (RAG + Chat)

Everything from Mode 1, **plus:**
- AI-generated summary explaining why the top documents match
- Interactive **follow-up chat** — ask deeper questions about the documents
- Gemini uses the top documents as context for grounded answers

---

## 📬 Contact

- **Email:** kmadhav0726@gmail.com
- **Phone:** 9693600978
- **Hugging Face:** [DumbMaddy](https://huggingface.co/DumbMaddy)

---

<div align="center">

**⭐ Star this project if you found it helpful!**

Made with ❤️ using E5 Embeddings, Gemini AI & Flask

</div>
