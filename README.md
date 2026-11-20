# Enterprise AI Knowledge Assistant

An AI-powered document question-answering system built using **Retrieval-Augmented Generation (RAG)**. It allows users to upload documents and ask questions, generating context-aware answers with relevant source references.

## 🚀 Project Overview

The Enterprise AI Knowledge Assistant helps users retrieve information from uploaded documents using semantic search and Large Language Models (LLMs).

Instead of manually searching through lengthy documents, users can interact with their files through a conversational AI interface.

## ✨ Key Features

- **Document Upload:** Supports PDF and TXT documents.
- **Document Processing:** Extracts and processes document content.
- **Semantic Search:** Retrieves relevant information using vector embeddings.
- **Vector Database:** Uses FAISS for efficient similarity search.
- **AI-Powered Responses:** Generates contextual answers using Groq LLM.
- **Source References:** Displays relevant document sources alongside answers.
- **Duplicate Detection:** Helps avoid processing duplicate documents.
- **Interactive Interface:** User-friendly Streamlit application.
- **REST API:** FastAPI backend for API-based interactions.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| LLM Integration | Groq API |
| RAG Framework | LangChain |
| Embeddings | Hugging Face |
| Vector Database | FAISS |
| Backend | FastAPI |
| Frontend | Streamlit |
| Document Processing | PDF, TXT |

## 🏗️ Architecture

```text
          User
           |
           v
    Streamlit Interface
           |
           v
    Document Upload
           |
           v
   Document Processing
           |
           v
   Text Chunking
           |
           v
 Hugging Face Embeddings
           |
           v
       FAISS Index
           |
           v
      User Query
           |
           v
    Semantic Retrieval
           |
           v
       Groq LLM
           |
           v
  Contextual AI Response
           |
           v
   Answer + Sources
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/arjunajai221-a11y/Enterprise_AI_Knowledge_Assistant.git

cd Enterprise_AI_Knowledge_Assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create a `.env` file in the project root directory:

```env
GROQ_API_KEY=your_groq_api_key
```

Replace `your_groq_api_key` with your own Groq API key.

**Important:** Never upload your `.env` file or expose your API key on GitHub.

### 5. Run the Streamlit Application

```bash
python -m streamlit run streamlit_app.py
```

### 6. Run the FastAPI Backend

```bash
uvicorn main:app --reload
```

## 🧪 Testing and Results

The application was tested using a batch of 20 questions.

| Metric | Result |
|---|---:|
| Total Questions | 20 |
| Responses Generated | 20 |
| Errors | 0 |
| Responses with Sources | 20 |
| Average Response Latency | 1.73 seconds |
| Maximum Response Latency | 6.98 seconds |

*These results represent the observed test run. Response generation and source availability do not independently establish factual accuracy.*

## 🎯 Project Objectives

- Build a practical Retrieval-Augmented Generation application.
- Implement semantic document retrieval.
- Integrate open-source embedding models.
- Connect an LLM through the Groq API.
- Develop an interactive document-based AI assistant.
- Provide source-grounded responses.

## 👨‍💻 Developer

**Arjun Ajai**

B.Sc. Computer Science Graduate | Data Science & AI

- GitHub: https://github.com/arjunajai221-a11y
- Project Repository: https://github.com/arjunajai221-a11y/Enterprise_AI_Knowledge_Assistant

---

*Built with Python, RAG, LangChain, FAISS, Hugging Face, Groq, FastAPI and Streamlit.*
