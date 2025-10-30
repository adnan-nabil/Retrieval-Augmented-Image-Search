import io
import sys
import json
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Security
from PIL import Image
from pathlib import Path
from typing import Optional, Dict
from utils.pydantic_schemas import SearchResponse, SearchResult
from utils.dboperations import DBOperations
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/search_by_image",
    tags=["Visual Search"],
)
auth_scheme = HTTPBearer()
project_root = Path(__file__).resolve().parent.parent.parent
tenant_file = project_root / 'tenants.json'

try:
    with open(tenant_file, 'r') as f:
        TENANT_CONFIGS = json.load(f)
        print(f"Tenant configurations loaded successfully.Len = {len(TENANT_CONFIGS)}")
except FileNotFoundError:
    logger.critical("FATAL ERROR: tenants.json config file not found.")
    print("___server is shutting down____")
    sys.exit(1)
    TENANT_CONFIGS = {}
except json.JSONDecodeError:
    logger.critical("FATAL ERROR: tenants.json is not valid JSON.")
    TENANT_CONFIGS = {}

async def get_shop_info(
    token: HTTPAuthorizationCredentials = Security(auth_scheme)
) -> Dict:
    """
    Validates the API key (Bearer token) by checking the tenants.json file.
    """
    api_key = token.credentials
    shop_info = TENANT_CONFIGS.get(api_key)
    
    if not shop_info:
        raise HTTPException(status_code=401, detail="Invalid API Key")
        
    return shop_info



@router.post("/", response_model=SearchResponse, status_code=200)
async def search_by_image_upload(
    file: UploadFile = File(..., description="Upload Image"),
    text_query: Optional[str] = None,
    shop_info: Dict = Depends(get_shop_info) 
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
        retriever = DBOperations(tenant_info=shop_info)
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
        logger.error(f"Error processing visual search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during search: {e}")
