"""Configuration settings for the Ashraq RAG Agent."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_FILE = PROJECT_ROOT / "data.json"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# ChromaDB settings
COLLECTION_NAME = "ashraq_strategy"

# Retrieval settings
DEFAULT_N_RESULTS = 3

# Streamlit settings
PAGE_TITLE = "Ashraq Communication Strategy RAG Agent"
PAGE_ICON = "🤖"

# Environment variables
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"