# ⚖️ Vietnamese Legal Contract NLP & RAG Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat&logo=huggingface)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Natural Language Processing (NLP) pipeline and Retrieval-Augmented Generation (RAG) chatbot designed specifically for Vietnamese legal contracts. 

This project transforms unstructured legal documents (e.g., labor, rental, and sales contracts) into structured, queryable data, leveraging state-of-the-art Vietnamese language models.

## 🚀 Key Features

* **Advanced Text Preprocessing:** Clause-level segmentation, IOB Noun-Phrase (NP) chunking, and dependency parsing using `Underthesea` and `VnCoreNLP`.
* **Legal Information Extraction:** * **Custom NER:** Fine-tuned recognition of domain-specific entities (`PARTY`, `MONEY`, `DATE`, `PENALTY`, `LAW`).
  * **Semantic Role Labeling (SRL):** Identifies agent, predicate, and recipient roles within legal clauses.
  * **Intent Classification:** Categorizes clauses into functional intents (*Obligation, Prohibition, Right, Termination Condition*).
* **Interactive RAG Chatbot:** A semantic search system backed by `ChromaDB` and LLMs, allowing users to ask natural language questions about the contract and receive heavily contextualized answers with exact source citations.

## 🧠 System Architecture

The system is broken down into three core micro-pipelines:
1. **Syntax Analyzer (`src/preprocessing/`):** Breaks raw `.txt` files into independent grammatical clauses.
2. **Semantic Extractor (`src/extraction/`):** Powered by `PhoBERT`, this module extracts structured JSON representations of "who must do what, by when, and under what conditions."
3. **Q&A Engine (`src/qa/`):** Embeds the extracted clauses using Multilingual Sentence Transformers, stores them in a Vector Database, and orchestrates the RAG pipeline via LangChain.

## 🛠️ Tech Stack
* **Core NLP:** `underthesea`, `spacy`, Hugging Face `transformers` (PhoBERT)
* **Vector Store & LLM:** `chromadb`, `langchain`, `sentence-transformers`, Google Gemini API
* **Backend & API:** `FastAPI`, `Uvicorn`, `Pydantic`
* **Frontend Demo:** `Streamlit`

## ⚙️ Quick Start

### 1. Installation
Clone the repository and install the dependencies using Poetry or pip:
```bash
git clone [https://github.com/yourusername/vietnamese-legal-nlp.git](https://github.com/yourusername/vietnamese-legal-nlp.git)
cd vietnamese-legal-nlp
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 2. Environment Variables
Copy the template and add your LLM API keys:
```bash
cp .env.example .env
```

### 3. Running the Pipeline
You can run the full extraction pipeline on a raw text contract:
```bash
python scripts/build_vector_store.py --input data/raw/hop_dong_thue_nha.txt
```

### 4. Launching the UI & API
To interact with the RAG Chatbot locally:
```bash
# Terminal 1: Start the FastAPI backend
make run-api 

# Terminal 2: Start the Streamlit frontend
make run-ui
```

## 📝 Disclaimer
*This repository is an extended, production-oriented implementation inspired by a university NLP assignment at Ho Chi Minh City University of Technology (HCMUT). The legal models herein are for educational/demonstration purposes and do not constitute professional legal advice.*