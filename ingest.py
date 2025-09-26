import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# New import for the generator
from src.knowledge_base_generator import KnowledgeBaseGenerator
from src.data_processor import DataProcessor
from config import DATA_FILE, CHROMA_DB_PATH, COLLECTION_NAME, PROJECT_ROOT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main indexing function."""
    logging.info("Starting master ingestion process...")
    load_dotenv()

    # --- Step 0: Generate data.json if it doesn't exist ---
    pdf_path = PROJECT_ROOT / "data" / "task-mohamed-rag.pdf"

    if not DATA_FILE.exists():
        logging.warning(f"{DATA_FILE} not found. Attempting to generate from PDF...")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logging.error("FATAL: GOOGLE_API_KEY is required to generate the knowledge base, but it's not set.")
            return

        generator = KnowledgeBaseGenerator(api_key=google_api_key)
        success = generator.generate(pdf_path=pdf_path, output_path=DATA_FILE)

        if not success:
            logging.error("Failed to generate the knowledge base. Aborting ingestion.")
            return
    else:
        logging.info(f"Found existing knowledge base at {DATA_FILE}. Skipping generation.")

    # --- Step 1: Indexing data.json into ChromaDB ---
    logging.info("Proceeding with indexing into ChromaDB...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logging.error("FATAL: OPENAI_API_KEY environment variable not set for indexing.")
        return

    try:
        # Load and process source data
        logging.info(f"Loading data from {DATA_FILE}...")
        data_processor = DataProcessor()
        data = data_processor.load_source_data(str(DATA_FILE))
        documents, metadatas, ids = data_processor.prepare_documents(data)
        logging.info(f"Loaded {len(documents)} documents for indexing.")

        # Convert to LangChain Document objects
        langchain_docs = [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(documents, metadatas)
        ]

        # Initialize OpenAI Embeddings model
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

        # Create and persist the Chroma vector store
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
