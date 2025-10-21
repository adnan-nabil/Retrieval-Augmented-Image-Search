from pydantic_settings import BaseSettings
from pydantic import Field   
from typing import List
from functools import lru_cache
import torch


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    All values can be overridden via environment variables or .env file.
    """

    # === API Configuration ===
    API_TITLE: str
    API_VERSION: str 
    API_PREFIX: str 

    # === Authentication ===
    API_KEY_ENABLED: bool 
    API_KEYS: str 

    # === Model Configuration ===
    MODEL_NAME: str 
    RANKER_MODEL: str
    QDRANT_URL: str
    QDRANT_COLLECTION_NAME: str 
    VECTOR_DIMENSION: int 
    DEVICE: str = Field("cuda" if torch.cuda.is_available() else "cpu", description="Computation device")

    # === Logging ===
    LOG_LEVEL: str 
    LOG_TO_FILE: bool 
    LOG_FILE_PATH: str 

    class Config:
        env_file = ".env"
        case_sensitive = True

    # === Computed Properties ===
    @property
    def api_keys_list(self) -> List[str]:
        """Parse comma-separated API keys into a list"""
        return [key.strip() for key in self.API_KEYS.split(",") if key.strip()]
    

    
    # @property
    # def allowed_extensions_set(self) -> Set[str]:
    #     """Parse comma-separated extensions into a set"""
    #     return {ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(",") if ext.strip()}

    
    # @property
    # def max_file_size_bytes(self) -> int:
    #     """Convert MB to bytes"""
    #     return self.MAX_FILE_SIZE_MB * 1024 * 1024




@lru_cache()
def get_settings() -> Settings:
    
    #settings are loaded only once and cached globally.
    return Settings()



# Convenience export for direct imports
settings = get_settings()
