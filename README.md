# Ashraq Communication Strategy RAG Agent

This is a sophisticated Retrieval-Augmented Generation (RAG) agent designed to serve as an expert on the Ashraq Communication Strategy document. It provides accurate, context-aware answers, complete with visual source citations from the original document.

The system features a unique dual-LLM architecture: it leverages **Google's Gemini 2.5 Pro** for high-fidelity knowledge extraction from the source PDF and **OpenAI's GPT-4o** for the conversational Q&A and reasoning.

[![Streamlit App Demo](https://img.shields.io/badge/Interface-Streamlit-ff69b4.svg)](https://streamlit.io)
[![VectorDB](https://img.shields.io/badge/VectorDB-Chroma-blue.svg)](https://www.trychroma.com/)
[![LLMs](https://img.shields.io/badge/LLMs-OpenAI%20%7C%20Google%20Gemini-green.svg)](https://openai.com/)

---

## 🚀 Key Features

-   **Automated High-Quality Knowledge Base**: Uses a powerful vision model (Gemini 2.5 Pro) to analyze the PDF's visual structure and text, creating a highly detailed and accurate JSON knowledge base. This is far superior to simple text extraction.
-   **Intelligent Q&A**: Employs GPT-4o to understand user queries and generate precise answers based *only* on the retrieved document context.
-   **Visual Source Citations**: Not only tells you the source page number but also displays an image of the actual page, providing essential context for diagrams, tables, and charts.
-   **Robust & Reproducible Setup**: A single script handles the entire data pipeline—from knowledge base generation to vector indexing—ensuring a consistent and error-free setup.
-   **Interactive UI**: A clean and user-friendly chat interface built with Streamlit.

---

## 🏗️ How It Works: The Dual-LLM Architecture

The project is designed around a clear, two-stage process to ensure the highest quality results.

1.  **Stage 1: Knowledge Base Ingestion (`ingest.py`)**
    This is a one-time setup process you run locally.
    -   **PDF Analysis (Gemini)**: If a structured `data.json` file is not found, the script sends images of each PDF page to the Gemini 2.5 Pro model with a highly specific prompt. Gemini analyzes the layout, text, and structure to generate a detailed JSON representation of the document.
    -   **Indexing (OpenAI)**: The script then takes the high-quality content from `data.json` and uses OpenAI's `text-embedding-3-large` model to create vector embeddings. These are stored locally in a persistent ChromaDB vector store.

2.  **Stage 2: Interactive Q&A (`app.py`)**
    This is the live application you interact with.
    -   **Retrieval**: When you ask a question, the application converts it into a vector and queries the ChromaDB to find the most semantically relevant pages from the document.
    -   **Generation (OpenAI)**: The retrieved context is passed to the GPT-4o model along with your question. The model generates a conversational answer based strictly on the provided information and cites the source pages.
    -   **Display**: The Streamlit app displays the text answer and a collapsible section showing the source page images.

---

## 🛠️ Setup and Installation

### Prerequisites

-   Python 3.8+
-   An **OpenAI API Key**.
-   A **Google Gemini API Key**.

### Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd ashraq-rag-agent
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate it (on macOS/Linux)
    source venv/bin/activate

    # Or on Windows
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Your API Keys**
    Create a file named `.env` by copying the example, then add your keys:
    ```bash
    cp .env.example .env
    ```
    Now, edit the `.env` file and add your secret keys. It should look like this:
    ```env
    OPENAI_API_KEY="sk-..."
    GOOGLE_API_KEY="AIza..."
    ```

---

## 🚀 How to Run

The entire setup process is handled by a single script.

### Step 1: Place Your Document

Place the PDF document you wish to query (`task-mohamed-rag.pdf`) inside the **`data/`** folder.

### Step 2: Build the Knowledge Base and Index

This command performs the one-time setup. It will use the Gemini API to create the `data.json` knowledge base and then use the OpenAI API to create the local vector database.

```bash
python ingest.py
```
> **Note:** This process may take a few moments the first time it runs, as it is communicating with the Gemini API to analyze the entire document. Subsequent runs will be much faster as they will use the existing `data.json`.

### Step 3: Launch the Streamlit Application

Once the ingestion is complete, you can start the chat application:
```bash
streamlit run app.py
```
Your browser will automatically open, and you can start asking questions.

---

## 💡 Example Questions

-   "What is Narrow Content Scope?"
-   "What are the main competitive advantages outlined in the document?"
-   "Show me the methodology for the communication strategy."
-   "What were the key findings from the benchmarking analysis?"
-   "What challenges are impacting visibility and engagement?"

## 📁 Project Structure

```
.
├── data/
│   └── task-mohamed-rag.pdf # Place the source PDF here.
├── chroma_db/               # (Ignored by Git) Local vector DB, created by ingest.py.
├── images/                  # (Ignored by Git) Page images, created by app.py.
├── src/
│   ├── __init__.py
│   ├── data_processor.py
│   └── knowledge_base_generator.py # Gemini-powered JSON generation logic.
├── .env                     # (Ignored by Git) Your secret API keys.
├── .env.example
├── .gitignore
├── app.py                   # The main Streamlit application.
├── config.py                # Application configuration.
├── data.json                # (Ignored by Git) Generated by ingest.py.
├── ingest.py                # Master script for knowledge base generation and indexing.
├── requirements.txt         # Project dependencies.
└── README.md                # This file.
```

## ⚙️ Troubleshooting

If you encounter issues with the database or want to re-build it from scratch with a new version of the PDF, use the `--reset` flag. This will delete the existing vector store before running the ingestion process.

```bash
# This will delete the old database and build a new one
python ingest.py --reset
```-   **Bring Your Own Document**: The user provides the source PDF, ensuring private documents are not stored in the public repository.

## 🏗️ How It Works: The "Smart Start" Architecture

The application is designed for maximum ease of use. When you run `streamlit run app.py`:

1.  **It Checks for a Knowledge Base:** The application first looks for a local vector database (`chroma_db/`).
2.  **Automatic Ingestion (If Needed):** If the database does not exist, the application will automatically find the PDF in the `data/` folder and run a one-time ingestion process in the background. You will see status updates directly in the app. This process:
    a. Converts each page of the PDF into a high-quality PNG image.
    b. Extracts the text from each page.
    c. Uses OpenAI's embedding model to create vectors for each page's content.
    d. Builds a local, persistent ChromaDB vector store.
3.  **Launches the RAG Agent:** Once the knowledge base is ready (either found or newly created), the interactive chat interface loads, and you can begin asking questions.

## 🛠️ Setup and Installation

### Prerequisites

-   Python 3.8+
-   An OpenAI API Key.

### Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd ashraq-rag-agent
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate it (on macOS/Linux)
    source venv/bin/activate

    # Or on Windows
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Your API Key**
    Create a file named `.env` in the root of the project and add your OpenAI API key:
    ```
    OPENAI_API_KEY="sk-..."
    ```

## 🚀 How to Run

### Step 1: Add Your Document

Place the PDF document you wish to query inside the **`data/`** folder. For this task, you should use the PDF provided in the interview email.

*(Note: The `data` folder is included in the `.gitignore` to ensure your private documents are never accidentally committed.)*

### Step 2: Launch the Streamlit Application

That's it! There is **no separate indexing step.** Simply run the main application:

```bash
streamlit run app.py
```

Your browser will automatically open. If this is the first time you're running the app with this PDF, it will perform the one-time data processing and indexing, showing you the progress. Afterward, the chat interface will appear, ready for your questions.

## 💡 Example Questions

-   "What are the main competitive advantages outlined in the document?"
-   "Show me the methodology for the communication strategy."
-   "What were the key findings from the benchmarking analysis?"
-   "What challenges are impacting visibility and engagement?"

## 📁 Project Structure

```
.
├── data/                   # (Ignored by Git) Place your source PDF here.
├── images/                 # (Ignored by Git) Page images, created automatically.
├── chroma_db/              # (Ignored by Git) Local vector database, created automatically.
├── .env                    # Your secret API key (Ignored by Git).
├── app.py                  # The main Streamlit application with auto-ingestion logic.
├── config.py               # Application configuration.
├── requirements.txt        # Project dependencies.
└── README.md               # This file.
