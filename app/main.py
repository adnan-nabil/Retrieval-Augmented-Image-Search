from fastapi import FastAPI
from config import settings
from endpoints import Search_product, add_product, delete_product, add_images, delete_image


app = FastAPI(
    title=settings.API_TITLE,
    description="Product Search through image",
    version=settings.API_VERSION
)


# --- Include Routers ---
app.include_router(Search_product.router)
app.include_router(add_product.router)
app.include_router(delete_product.router)
app.include_router(add_images.router)
app.include_router(delete_image.router)


# --- Health Check Endpoint ---
@app.get("/", tags=["Health"])
def read_root():
    """Health check endpoint to ensure the API is running."""
    return {"status": "ok", "service": "visual_search_api", "message": "API is running and ready to handle requests."}

# --- Uvicorn Server Start (Optional, for easy testing) ---
# if __name__ == "__main__":
#     # NOTE: This block is usually removed when deploying with gunicorn or standard Uvicorn commands
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
