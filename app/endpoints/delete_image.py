import logging
from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict

from utils.dboperations import DBOperations
from dependencies.auth import get_shop_info
from utils.pydantic_schemas import ImageLinksRequest, StatusResponse

logger = logging.getLogger(__name__)

# Create a new router
router = APIRouter(
    prefix="/delete_image",
    tags=["Delete Images to Product"] 
)

@router.delete("/", response_model=StatusResponse, status_code=200)
async def delete_single_image(
    request: ImageLinksRequest = Body(...), 
    shop_info: Dict = Depends(get_shop_info)
):
    """
    Deletes a single image vector from the database.
    
    This endpoint calculates the unique ID from the product_id and image_url
    and deletes that specific point from Qdrant.
    """
    
    try:
        db = DBOperations(tenant_info=shop_info)
        
        logger.info(f"Received request to delete image: {request.image_url} from product: {request.product_id}")
        
        # Call the new function in DBOperations
        deleted_point = db.delete_image_vector(
            product_id=request.product_id,
            image_url=request.image_url
        )
            
        return StatusResponse(
            status="success",
            message=f"Successfully deleted image. Product ID: {request.product_id}",
            qdrant_point_id=deleted_point
        )

    except Exception as e:
        logger.error(f"Error deleting image {request.image_url} for {request.product_id}: {e}", exc_info=True)
        # Check for a specific "not found" style error if needed,
        # but Qdrant's delete is often silent if the point doesn't exist.
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
