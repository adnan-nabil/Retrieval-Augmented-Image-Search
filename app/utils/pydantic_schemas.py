from pydantic import BaseModel, Field
from typing import List, Optional


class BaseProductRequest(BaseModel):
    """Base class for any request involving a product and a specific image index."""
    product_id: str = Field(..., description="The unique identifier for the product.")
    query: Optional[str] = Field(..., description="Serial number of the image eg 1,2,3 or 4.")



class ProductID(BaseModel):
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
    status: Optional[str] = None
    message: Optional[str] = None
    qdrant_point_id: Optional[int] = None
    deleted_count: Optional[int] = None
    added_count: Optional[int] = None    



class AddImageLinksRequest(BaseModel):
    product_id: str
    image_urls: List[str]    