from transformers import AutoImageProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import torch
import mysql.connector
from PIL import Image, UnidentifiedImageError
import io
import hashlib
from typing import List, Dict, Optional
import logging
import numpy as np
import json
import asyncio  
import aiohttp  


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductEmbeddingPipeline:
    def __init__(self, mysql_config: Dict, qdrant_url: str, collection_name: str):
        """
        Initialize the pipeline
        """
        self.mysql_config = mysql_config
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(url=qdrant_url)
        
        logger.info("Loading DINOv2 model...")
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")
        
    def create_collection(self, vector_size: int = 768):
        """Create Qdrant collection if it doesn't exist"""
        try:
            self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Created collection '{self.collection_name}'")
    
    def fetch_products_from_mysql(self, batch_size: int = 100, offset: int = 0) -> List[Dict]:
        """
        Fetch products from MySQL database
        """
        # This remains synchronous as mysql.connector is a blocking library.
        # For full async, one would use a library like aiomysql.
        conn = mysql.connector.connect(**self.mysql_config)
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT 
            id, name, image_path, image_path1, 
            image_path2, image_path3, image_paths
        FROM products
        WHERE user_id = 188
        LIMIT %s OFFSET %s
        """
        
        cursor.execute(query, (batch_size, offset))
        products = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return products
    
    async def _download_image_async(self, session: aiohttp.ClientSession, url: str) -> Optional[Image.Image]:
        """
        Asynchronously download a single image and verify it.
        """
        try:
            async with session.get(url, timeout=50) as response:
                response.raise_for_status() 

                content_type = response.headers.get('Content-Type')
                if not content_type or not content_type.startswith('image/'):
                    logger.warning(f"Skipping: {url} not reported as image. Content-Type: {content_type}")
                    return None

                # 2. Read the full image data
                image_data = await response.read()

                # 3. Verify content with PIL
                try:
                    # PIL operations are CPU-bound and fast, so we do them synchronously
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    return image
                except UnidentifiedImageError:
                    logger.error(f"Failed {url}: Server claimed image, but PIL could not identify.")
                    return None
                except Exception as pil_e:
                    logger.error(f"Failed to open image {url} with PIL: {pil_e}")
                    return None

        except Exception as e:
            # Catches aiohttp.ClientError, asyncio.TimeoutError, etc.
            logger.error(f"Failed to download {url}: {e}")
            return None

    def generate_embeddings_batch(self, images: List[Image.Image]) -> List[List[float]]:
        """
        Generate embeddings for a BATCH of images using DINOv2
        """
        # This is CPU/GPU-bound, so it remains synchronous
        if not images:
            return []
        
        with torch.no_grad():
            inputs = self.processor(
                images=images, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized_embeddings = embeddings / norms
            
            return normalized_embeddings.tolist()

    def generate_unique_id(self, product_id: str, img_url: str) -> str:
        """Generate unique ID for each image embedding"""
        combined = f"{product_id}_{img_url}"
        return hashlib.md5(combined.encode()).hexdigest()
    

    async def process_products_batch(self, products: List[Dict]) -> List[PointStruct]:
        """
        Process a BATCH of products: asynchronously download all images, 
        generate all embeddings, and create points.
        """
        
        urls_to_download = []      # Stores all unique URLs to fetch
        metadata_by_url = {}       # Stores metadata for each URL
        
        logger.info(f"Collecting and deduplicating URLs for {len(products)} products...")
        
        
        for product in products:
            product_id = str(product['id'])
            product_name = product.get('name', 'no_name').lower().strip()
            
            all_urls = []
            single_image_columns = ['image_path', 'image_path1', 'image_path2', 'image_path3']
            for col in single_image_columns:
                if col in product:
                    all_urls.append(product[col])
            
            json_paths_str = product.get('image_paths')
            if json_paths_str:
                try:
                    parsed_paths = json.loads(json_paths_str)
                    if isinstance(parsed_paths, list):
                        all_urls.extend(parsed_paths)
                    elif isinstance(parsed_paths, str):
                        all_urls.append(parsed_paths)
                except (json.JSONDecodeError, TypeError):
                    if isinstance(json_paths_str, str):
                        all_urls.append(json_paths_str)
                        
            unique_urls = {url.strip() for url in all_urls if url and url.strip()}

            for img_url in unique_urls:
                # We store metadata by URL to re-associate it after concurrent download
                if img_url not in metadata_by_url:
                    urls_to_download.append(img_url)
                    metadata_by_url[img_url] = {
                        'product_id': product_id,
                        'product_name': product_name,
                        'image_url': img_url
                    }
        
        if not urls_to_download:
            logger.info("No valid, unique images found in this batch.")
            return []
            
        
        logger.info(f"Downloading {len(urls_to_download)} images concurrently...")
        images_to_process = []       # Stores PIL.Image objects
        metadata_for_images = []   # Stores corresponding info
        
        async with aiohttp.ClientSession() as session:
            # Create a list of tasks
            tasks = [self._download_image_async(session, url) for url in urls_to_download]
            # Run them all in parallel
            downloaded_images = await asyncio.gather(*tasks)

        for img_url, image in zip(urls_to_download, downloaded_images):
            if image:
                images_to_process.append(image)
                metadata_for_images.append(metadata_by_url[img_url])

        if not images_to_process:
            logger.info("No images successfully downloaded in this batch.")
            return []

        # Synchronous
        logger.info(f"Generating embeddings for {len(images_to_process)} images...")
        try:
            embeddings = self.generate_embeddings_batch(images_to_process)
        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch: {e}", exc_info=True)
            return []

        logger.info(f"Creating {len(embeddings)} PointStructs...")
        points_batch = [
            PointStruct(
                id=self.generate_unique_id(meta['product_id'], meta['image_url']),
                vector=embedding,
                payload={
                    'product_id': meta['product_id'],
                    'product_name': meta['product_name'],
                    'image_url': meta['image_url'],
                }
            ) 
            for meta, embedding in zip(metadata_for_images, embeddings)
        ]
        
        return points_batch

    async def run_pipeline(self, batch_size: int = 100, total_products: Optional[int] = None):
        """
        Run the complete pipeline asynchronously.
        """
        self.create_collection()
        
        offset = 0
        processed_products_count = 0
        
        while True:
            if total_products and processed_products_count >= total_products:
                logger.info(f"Processed product limit of {total_products} reached.")
                break
                
            products = self.fetch_products_from_mysql(batch_size, offset)
            
            if not products:
                logger.info("No more products to fetch from MySQL.")
                break
            
            if total_products:
                remaining_needed = total_products - processed_products_count
                if len(products) > remaining_needed:
                    products = products[:remaining_needed]
            
            try:
                #async batch processor
                points_batch = await self.process_products_batch(products)
                
            except Exception as e:
                logger.error(f"Critical error processing batch at offset {offset}: {e}", exc_info=True)
                offset += batch_size
                continue
                
            processed_products_count += len(products)
            
            # Step 3: Upsert to Qdrant (Synchronous)
            if points_batch:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points_batch
                )
                logger.info(f"Stored {len(points_batch)} embeddings in Qdrant for this batch")
            
            offset += batch_size
            logger.info(f"Processed {processed_products_count} products so far...")
        
        logger.info(f"Pipeline complete! Processed {processed_products_count} products")
        

if __name__ == "__main__":
    
    mysql_config = {
        'host': 'localhost',
        'port': 3307,
        'user': 'root',
        'password': '',
        'database': 'gadget_bodda'
    }
    
    
    pipeline = ProductEmbeddingPipeline(
        mysql_config=mysql_config,
        qdrant_url="http://localhost:6333",
        collection_name="gadget_bodda"    
    )
    
    
    logger.info("Starting pipeline...")
    asyncio.run(pipeline.run_pipeline(batch_size=200))
    logger.info("Pipeline finished.")