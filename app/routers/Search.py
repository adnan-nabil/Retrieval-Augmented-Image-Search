from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from typing import Optional
import io
from utils.pydantic_schemas import SearchResponse, SearchResult
from utils.dboperations import DBOperations

router = APIRouter(
    prefix="/search",
    tags=["Visual Search"],
)


@router.post("/image_search", response_model=SearchResponse, status_code=200)

async def search_by_image_upload(
    file: UploadFile = File(..., description="Upload Image"),
    text_query: Optional[str] = None 
):
    """
    Args:
        file (UploadFile): The image file uploaded by the user.
        text_query (Optional[str]): An optional text query to refine search results.
    """
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # 1. Read the image file content
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
    except Exception:
        raise HTTPException(400, detail="Corrupted or invalid image")
    
    
    try:
        retriever = DBOperations()
        qdrant_results = retriever.find_similar(image, text_query)
        
        # 3. Format the results using the Pydantic schema [SeachResult]
        results = [
            SearchResult(
                product_name=hit.payload.get('product_name'),
                product_id=hit.payload.get('product_id'),
                url=hit.payload.get('image_url'),
                image_similarity=hit.score,
                rerank_score=(hit.payload.get('rerank_score') 
                              if text_query and 'rerank_score' in hit.payload else None)
            )for hit in qdrant_results
        ]
        
        return SearchResponse(results=results)
        
    except Exception as e:
        print(f"Error processing visual search: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during search: {e}")
