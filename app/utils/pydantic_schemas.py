from pydantic import BaseModel, Field
from typing import List, Optional

# --- QDRANT POINT IDENTIFICATION ---

def hash_product_id_and_index(product_id: str, image_index: str) -> int:
    """
    Creates a consistent, unique, numerical ID for a vector point in Qdrant
    based on the product ID and its image index.
    This hash must be the same algorithm used in generate_embeddings.py.
    """
    # Combine product ID and index into a single string
    unique_key = f"{product_id}_{image_index}"
    # Use a large integer hash to avoid collisions (Python's hash() is consistent within a session)
    # Modulo is used to ensure the ID fits within Qdrant's 63-bit integer limit
    return hash(unique_key) % (2**63 - 1)


class BaseProductRequest(BaseModel):
    """Base class for any request involving a product and a specific image index."""
    product_id: str = Field(..., description="The unique identifier for the product.")
    query: Optional[str] = Field(..., description="Serial number of the image eg 1,2,3 or 4.")

class AddProduct(BaseModel):
    product_id: str = Field(..., description="The unique ID of the product.")

class SearchResult(BaseModel):
    """Defines the structure of a single vector search result."""
    product_name: str = Field(..., description="product_name")
    product_id: str = Field(..., description="product_id.")
    url: Optional[str] = Field(None, description="url of image in original db.")
    image_similarity: float = Field(..., description="Image search score from Qdrant.")
    rerank_score: Optional[float] = Field(None, description="Brand name relevance score.")

class SearchResponse(BaseModel):
    """The final response structure for any search endpoint."""
    results: List[SearchResult] = Field(..., description="A list of similar products found.")
    #query_type: str = Field(..., description="The type of query executed (e.g., 'image_upload').")

class StatusResponse(BaseModel):
    """Generic response for status updates (Create, Update, Delete)."""
    status: str
    message: str
    qdrant_point_id: Optional[int] = None

class SearchRequest(BaseModel):
    text_query: Optional[str] = None    
