# 🚀 AI Call Analyzer Pro (Enterprise Grade)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-call-analyzer-362.streamlit.app/)

A production-ready AI system designed to analyze sales and customer conversations using a state-of-the-art stack: **Hugging Face**, **Groq (Llama 3 + Whisper)**, **FAISS (RAG)**, and **Streamlit**.

## 🌟 Key Features
- **⚡ Ultra-Fast Transcription**: Powered by Groq Whisper (transcribes audio in milliseconds).
- **🧠 Hybrid Intelligence**: Combines rule-based speed with LLM-based (Llama 3) accuracy for Intent Detection and Insight Generation.
- **🔍 RAG-Enhanced Insights**: Retrieves similar past conversations from a FAISS vector database to provide deep context to the AI.
- **Genuine Data**: Uses the `DialogSum` dataset from Hugging Face for real-world testing.
- **📊 Analytics Dashboard**: Built-in SQLite persistence with visualizations for sentiment trends and intent distribution.
- **⏱️ Performance Metrics**: Transparent tracking of latency for every stage of the pipeline.

## 🛠️ Tech Stack
- **NLP**: Hugging Face Transformers (Sentiment Analysis)
- **STT**: Groq Whisper-v3
- **LLM**: Groq Llama 3.1 (70B)
- **Vector DB**: FAISS
- **Embeddings**: Sentence-Transformers (MiniLM-L6-v2)
- **Database**: SQLite3
- **Frontend**: Streamlit

## 🚀 Quick Start

### 1. Requirements
Ensure you have Python 3.9+ installed.

### 2. Setup
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. configuration
- Enter your **Groq API Key** in the sidebar.
- (Optional) Click **"Index Reference Dataset"** to populate the RAG system with 100 sample conversations from Hugging Face.

## 📂 Project Structure
- `app.py`: Main dashboard and UI logic.
- `modules/`:
    - `stt.py`: Audio transcription logic.
    - `sentiment.py`: Hugging Face NLP pipeline.
    - `intent.py`: Hybrid intent detection.
    - `vector_db.py`: FAISS index management.
    - `llm_engine.py`: Groq Llama 3 integration.
    - `database.py`: SQLite persistence layer.
- `data/`: Local storage for the database and vector index.
