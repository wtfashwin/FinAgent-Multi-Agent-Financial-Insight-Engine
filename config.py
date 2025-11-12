import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # Project paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    CHROMA_DB_PATH = BASE_DIR / "chroma_db"
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    CHROMA_DB_PATH.mkdir(exist_ok=True)
    
    # LLM API Keys (Free options)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    
    # LLM Configuration (Free models)
    LLM_PROVIDER = "groq"  # groq, together, openai
    LLM_MODEL = "mixtral-8x7b-32768"  # Groq's free model
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 2048
    
    # Alternative free models
    FREE_MODELS = {
        "groq": "mixtral-8x7b-32768",  # Fast and free
        "together": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "openai": "gpt-3.5-turbo",  # Limited free tier
    }
    
    # Embeddings (Free local model)
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and accurate
    EMBEDDING_DIMENSION = 384
    
    # ChromaDB Settings
    CHROMA_COLLECTION_NAME = "financial_transactions"
    
    # API Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # Streamlit Settings
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))
    
    # Fraud Detection Thresholds
    FRAUD_THRESHOLD = 0.7
    HIGH_RISK_THRESHOLD = 0.5
    
    # Data Processing
    CHUNK_SIZE = 1000
    OVERLAP_SIZE = 200
    
    @classmethod
    def get_llm_config(cls):
        """Get LLM configuration based on provider"""
        return {
            "provider": cls.LLM_PROVIDER,
            "model": cls.FREE_MODELS.get(cls.LLM_PROVIDER),
            "temperature": cls.LLM_TEMPERATURE,
            "max_tokens": cls.LLM_MAX_TOKENS,
        }
    
    @classmethod
    def validate_api_keys(cls):
        """Check if at least one API key is configured"""
        api_keys = [
            cls.GROQ_API_KEY,
            cls.TOGETHER_API_KEY,
            cls.OPENAI_API_KEY,
        ]
        
        if not any(api_keys):
            print(" WARNING: No LLM API keys configured!")
            print("Please set at least one API key in .env file")
            print("  - Groq: https://console.groq.com")
            return False
        return True

# Validate on import
Config.validate_api_keys()