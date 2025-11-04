import logging
import asyncio
import aiohttp
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List

from utils.dboperations import DBOperations
from dependencies.auth import get_shop_info
from utils.pydantic_schemas import AddImageLinksRequest, StatusResponse

logger = logging.getLogger(__name__)

# Create a new router
router = APIRouter(
    prefix="/add_new_images",
    tags=["Add New Images to Product"]
)

@router.post("/", response_model=StatusResponse, status_code=201)
async def add_image_links_to_product(
    request: AddImageLinksRequest, 
    shop_info: Dict = Depends(get_shop_info)
):
    """
    Receives a product_id and a list of new image URLs.
    It will:
    1. Validate the product_id exists by fetching its name.
    2. Download all images from the URLs in parallel.
    3. Upsert embeddings for all successfully downloaded images in a SINGLE BATCH.
    """
    
    try:
        db = DBOperations(tenant_info=shop_info)
        product_id = request.product_id

        # check if the product is also available in main product DB
        product_data = db.get_product_data_from_mysql(product_id)
        
        if not product_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Product not found with ID: {product_id}"
            )
        
        product_name = product_data.get('name', 'no_name')

        if not request.image_urls:
             raise HTTPException(
                status_code=400, 
                detail="No image_urls provided in the request."
            )
        
        logger.info(f"Processing {len(request.image_urls)} new URLs for product {product_id}...")
        
        async with aiohttp.ClientSession() as session:
            tasks = [db._download_image_async(session, url) for url in request.image_urls]
            downloaded_images = await asyncio.gather(*tasks)

        image_data_list = []
        for img, url in zip(downloaded_images, request.image_urls):
            if img:
                image_data_list.append({
                    "product_id": product_id,
                    "product_name": product_name,
                    "image_url": url,
                    "image": img  
                })
            else:
                logger.warning(f"Failed to download new image: {url}")
        
        if not image_data_list:
            raise HTTPException(
                status_code=400, 
                detail="All provided URLs failed to download or were invalid images."
            )

        upserted_ids = db.upsert_image_batch(image_data_list)
            
        return StatusResponse(
            status="success",
            message=f"Successfully processed and added {len(upserted_ids)} new images."
        )

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions
        raise http_exc
    except Exception as e:
        logger.error(f"Error adding image links for {request.product_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
