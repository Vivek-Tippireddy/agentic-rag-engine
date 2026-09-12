import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", r"C:\Users\tippi\OneDrive\Desktop\Agentic RAG for NLP poster\data")
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
SQLITE_DB_PATH = str(BASE_DIR / "financial_data.db")

# API Keys & Tokens
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Default System Configuration
DEFAULT_MODEL_PROVIDER = "gemini"  # Options: 'gemini', 'huggingface', 'openai', 'ollama'
DEFAULT_MODEL_NAME = "gemini-2.5-flash"

# Default Verifier & Critic Settings
DEFAULT_VERIFIER_THRESHOLD = 0.75  # Minimum score (0.0 to 1.0) to pass verification
DEFAULT_MAX_RETRY_LOOPS = 2       # Max critique-revision cycles

# Active Source Toggles (Can be overridden dynamically via Streamlit UI)
DEFAULT_SOURCE_TOGGLES = {
    "vector_db": True,
    "sql_api": True,
    "web_search": True,
    "knowledge_graph": True
}
