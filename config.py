
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables for local development.
load_dotenv()


def get_setting(name, default=None):
    """Read Streamlit secrets first, then local environment variables."""
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


# OpenRouter configuration
OPENROUTER_API_KEY = get_setting("OPENROUTER_API_KEY")
OPENROUTER_MODEL = get_setting("OPENROUTER_MODEL")

# Project directories
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
DATA_DIR = BASE_DIR / "data"

# Persistent DuckDB analytics database
DUCKDB_PATH = DATA_DIR / "nexus_analytics.duckdb"

# Create directories without deleting existing data.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
