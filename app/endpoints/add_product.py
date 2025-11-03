import aiohttp
import asyncio
import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict
from utils.dboperations import DBOperations
from dependencies.auth import get_shop_info
from utils.pydantic_schemas import ProductID

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/add_product",
    tags=["Add Product"],
)

@router.post("/", status_code=201)
async def create_product(
    request: ProductID,
    shop_info: Dict = Depends(get_shop_info)
):
    """
    Fetches a product's data from MySQL, downloads its images,
    and upserts the embeddings into the vector database.
    """
    db = DBOperations(tenant_info=shop_info)
    
    product_data = db.get_product_data_from_mysql(request.product_id)
    
    if not product_data:
        raise HTTPException(
            status_code=404, 
            detail=f"Product not found or not associated with this tenant."
        )

    image_urls = db.extract_urls_from_product(product_data)
    product_name = product_data.get('name', 'no_name')
    
    if not image_urls:
        raise HTTPException(
            status_code=400, 
            detail="No valid image URLs found for this product."
        )

    
    async with aiohttp.ClientSession() as session:
        tasks = [db._download_image_async(session, url) for url in image_urls]
        downloaded_images = await asyncio.gather(*tasks)


    successful_downloads = []
    for img, url in zip(downloaded_images, image_urls):
        if img:
            successful_downloads.append((img, url))
        else:
            logger.warning(f"Failed to download or verify image: {url}")
            
    if not successful_downloads:
        raise HTTPException(
            status_code=400, 
            detail="No valid images could be processed from the provided URLs."
        )

    
    upserted_ids = []
    try:
        for image, url in successful_downloads:
            point_id = db.upsert_product_vector(
                product_id=request.product_id,
                product_name=product_name,
                image_url=url,
                image=image
            )
            upserted_ids.append(point_id)
            
        return {
            "status": "success",
            "message": f"Successfully processed product {request.product_id}",
            "total_images_processed": len(upserted_ids),
            "point_ids": upserted_ids
        }
        
    except Exception as e:
        logger.error(f"Error during vector upsert for {request.product_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error saving vector data.")