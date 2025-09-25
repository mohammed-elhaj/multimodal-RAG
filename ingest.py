#!/usr/bin/env python3
"""
Data indexing script for the Ashraq RAG Agent using LangChain and OpenAI.
"""
import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.data_processor import DataProcessor
from config import DATA_FILE, CHROMA_DB_PATH, COLLECTION_NAME

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main indexing function."""
    logging.info("Starting indexing process...")
    load_dotenv()

    # Verify API key is set
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("FATAL: OPENAI_API_KEY environment variable not set.")
        return

    try:
        # 1. Load and process source data using the existing DataProcessor
        logging.info(f"Loading data from {DATA_FILE}...")
        data_processor = DataProcessor()
        data = data_processor.load_source_data(str(DATA_FILE))
        documents, metadatas, ids = data_processor.prepare_documents(data)
        logging.info(f"Loaded {len(documents)} documents.")

        # 2. Convert to LangChain Document objects
        langchain_docs = [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(documents, metadatas)
        ]

        # 3. Initialize OpenAI Embeddings model
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

        # 4. Create and persist the Chroma vector store
        # LangChain's Chroma.from_documents handles embedding and storage in one step.
        logging.info(f"Creating and persisting vector store at {CHROMA_DB_PATH}...")
        db = Chroma.from_documents(
            documents=langchain_docs,
            embedding=embeddings_model,
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DB_PATH)
        )

        logging.info("✅ Indexing complete!")
        logging.info(f"{len(langchain_docs)} documents were successfully indexed into ChromaDB.")
        logging.info(f"Collection '{COLLECTION_NAME}' is ready in '{CHROMA_DB_PATH}'.")

    except Exception as e:
        logging.error(f"An error occurred during indexing: {e}", exc_info=True)

if __name__ == "__main__":
    main()