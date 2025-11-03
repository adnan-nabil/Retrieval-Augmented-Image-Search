import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict
from utils.dboperations import DBOperations
from dependencies.auth import get_shop_info
from utils.pydantic_schemas import ProductID, StatusResponse

logger = logging.getLogger(__name__)

# Create a new router for delete operations
router = APIRouter(
    prefix="/delete_products",
    tags=["delete Product"],
)

@router.delete("/", response_model=StatusResponse, status_code=200)
async def delete_product(
    request: ProductID,
    shop_info: Dict = Depends(get_shop_info)
):
    """
    Deletes all image vector embeddings associated with a specific
    product ID from the Qdrant database.
    """
    
    try:
        # 1. Initialize DBOperations with tenant info
        db = DBOperations(tenant_info=shop_info)
        
        # 2. Call delete method
        deleted_count = db.delete_product_vectors(
            product_id=request.product_id
        )
        
        if deleted_count > 0:
            message = f"Successfully deleted {deleted_count} image vectors for product ID {request.product_id}."
        else:
            message = f"No image vectors found to delete for product ID {request.product_id}."
            
        return StatusResponse(
            deleted_count=deleted_count,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Error during vector deletion for {request.product_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting vector data: {e}")
