# ⚖️ Vietnamese Legal Contract NLP & RAG Pipeline

**Author:** Nguyễn Văn Đức  
**Project:** End-to-End Information Extraction and Semantic Analysis for Legal Documents.

This project is a sophisticated NLP pipeline designed to process, analyze, and query Vietnamese legal contracts. It transforms raw documents (PDF, DOCX, TXT) into a structured knowledge base using fine-tuned PhoBERT models and provides an interactive RAG (Retrieval-Augmented Generation) chatbot for legal Q&A.

---

## 🎥 Demo Video
![Demo](demo/demo.gif)

---

## 🧠 System Architecture

The system is composed of three main modules:
1.  **Task 1 (Preprocessing):** Clause segmentation, IOB Noun-Phrase chunking, and Dependency parsing.
2.  **Task 2 (Semantic Extraction):** Custom NER (PARTY, MONEY, DATE...), Semantic Role Labeling (SRL), and Intent Classification (Obligation, Prohibition...).
3.  **Task 3 (Legal RAG):** Intelligent retrieval using semantic similarity and metadata filtering, powered by Google Gemini 3.1 Flash.

---

## 🚀 Setup Instructions

### 1. Requirements & Installation
Ensure you have Python 3.10+ and a virtual environment ready.

```bash
git clone https://github.com/nguyenduc13475/Vietnamese-Legal-NLP.git
cd vietnamese-legal-nlp
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 📥 Download Pre-trained Models
Due to file size limits, the fine-tuned PhoBERT models are hosted on Google Drive. 
**MANDATORY:** Download the models and extract them into the `models/` directory.

- **Download Link:** [Google Drive - Legal Models](https://drive.google.com/drive/folders/1pyLIdZcm0L1XqaMwQ5BbFy4sJOMjXpUa)
- **Structure after extraction:**
  ```text
  models/
  ├── intent_regression/
  ├── intent_transformer/
  ├── ner/
  ├── segmenter/
  └── srl/
  ```

### 3. Configuration
Rename `.env.example` to `.env` and add your Google Gemini API Key.
```env
GOOGLE_API_KEY="your_api_key_here"
HF_HUB_OFFLINE=1
```

---

## 🛠️ How to Run

### 🖥️ Running the Backend API
The FastAPI backend handles all NLP heavy-lifting and Vector DB operations.
```bash
make run-api
# or
uvicorn api.main:app --reload --port 8000
```

### 🌐 Running the User Interface (Streamlit)
The UI allows you to manage documents, visualize extraction results, and chat with the RAG system.
```bash
make run-ui
# or
streamlit run ui/app.py
```

### ⚙️ Running the CLI Pipeline
To process a contract directly via command line:
```bash
python main.py --input data/raw/your_contract.txt --output_dir output
```

### 🗄️ Rebuilding the Vector Database
If you add new processed files to `data/processed/` and want to re-index them:
```bash
python scripts/build_vector_store.py --input data/processed/
```

---

## 📊 Evaluation Results
Reports for NER, SRL, and Intent Classification can be generated using:
```bash
make eval-ner
make eval-srl
make eval-intent
make generate-report
```
The final report will be available at `report/FINAL_REPORT.md`.

## 📜 Disclaimer
*This tool is for educational purposes. Always consult with a qualified legal professional for actual legal analysis.*