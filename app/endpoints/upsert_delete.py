from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
from PIL import Image
import io
import logging

from app.utils.pydantic_schemas import (
    UpsertProductRequest, 
    DeleteProductRequest, 
    DeleteProductAllRequest,
    StatusResponse
)
from app.models.qdrant_client_logic import QdrantSearchClient
from app.models.clip_model import encode_image

logger = logging.getLogger("__name__")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

router = APIRouter(
    prefix="/products",
    tags=["Product CRUD"],
)

# Initialize Qdrant Client globally for the router
qdrant_client = QdrantSearchClient()

# --- Endpoint 2 & 3: CREATE/UPDATE (Upsert) ---



@router.post("/upsert", response_model=StatusResponse, status_code=200)
async def upsert_product_image(
    pid_index: UpsertProductRequest = Depends(),
    image_file: UploadFile = File(..., description="The new or updated image file."),
):
    """
    Handles both creation of new product vectors and updating of existing ones (UPSERT).
    - If the product_id/image_index combination exists, it is UPDATED.
    - If it's new, it is CREATED.
    """
    
    product_id = pid_index.product_id
    image_index = pid_index.image_index
        
    # 1. Read the image file content    
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await image_file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")    
    except Exception as e:
        logger.error(f"Failed to read image. product_id={product_id}, image_index={image_index}: {e}")
        raise HTTPException(400, detail="Corrupted or invalid image")
    
    #Generate embedding
    try:
        embeddings = encode_image(image)
    except Exception as e:
        logger.error(f"Failed to generate embedding for product_id={product_id}, image_index={image_index}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process image embedding")


    # 2. Perform the upsert operation
    try:
        point_id = qdrant_client.upsert_product_vector(
            product_id=product_id, 
            image_index=image_index, 
            vector=embeddings
        )
        logger.info(f"Upserted vector: product_id={product_id}, image_index={image_index}, point_id={point_id}")
        
        # 3. Return status
        return StatusResponse(
            status="success",
            message=f"Product vector (ID: {product_id}, Index: {image_index}) successfully created/updated.",
            qdrant_point_id=point_id
        )

    except Exception as e:
        logger.error(f"Failed to upsert vector for product_id={product_id}, image_index={image_index}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upsert vector: {e}")

# --- Endpoint 4: DELETE ---








@router.delete("/delete-single", response_model=StatusResponse, status_code=200)
async def delete_single_image(
    request: DeleteProductRequest
):
    """
    Deletes a single vector corresponding to a specific product ID and image index.
    """
    try:
        qdrant_client.delete_product_vectors(
            product_id=request.product_id, 
            image_index=request.image_index
        )
        logger.info(f"Deleted vector(s) for product_id={request.product_id}, image_index={request.image_index}")
        
        return StatusResponse(
            status="success",
            message=f"Vector for Product ID {request.product_id}, Index {request.image_index} successfully deleted."
        )

    except Exception as e:
        logger.error(f"Failed to delete vector for product_id={request.product_id}, image_index={request.image_index}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete vector: {e}")







@router.delete("/delete-all", response_model=StatusResponse, status_code=200)
async def delete_all_product_images(
    request: DeleteProductAllRequest
):
    """
    Deletes ALL vectors associated with a given product ID (all its images).
    """
    try:
        qdrant_client.delete_product_vectors(
            product_id=request.product_id, 
            image_index=None # Passing None tells the client to delete by filter
        )
        
        return StatusResponse(
            status="success",
            message=f"ALL vectors for Product ID {request.product_id} successfully deleted."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete product vectors: {e}")
