from transformers import AutoImageProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import torch
import mysql.connector
import requests
from PIL import Image
import io
import hashlib
from typing import List, Dict, Optional
import logging
import numpy as np
import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductEmbeddingPipeline:
    def __init__(self, mysql_config: Dict, qdrant_url: str, collection_name: str):
        """
        Initialize the pipeline
        
        Args:
            mysql_config: Dictionary with MySQL connection details
            qdrant_url: Qdrant server URL (e.g., "http://localhost:6333")
            collection_name: Name of the Qdrant collection
        """
        self.mysql_config = mysql_config
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url)
        
        # Initialize DINOv2 model
        logger.info("Loading DINOv2 model...")
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base') # base=768, giant=1536
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        self.model.eval()
        
        # Move to GPU if available
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
        
        Returns:
            List of product dictionaries with all columns
        """
        conn = mysql.connector.connect(**self.mysql_config)
        cursor = conn.cursor(dictionary=True)
        
        # Adjust this query based on your actual table structure
        query = """
        SELECT 
            id, 
            name,
            image_path, 
            image_path1, 
            image_path2, 
            image_path3,
            image_paths
        FROM products
        WHERE user_id = 188
        LIMIT %s OFFSET %s
        """
        
        cursor.execute(query, (batch_size, offset))
        products = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return products
    
    def download_image_to_memory(self, url: str) -> Optional[Image.Image]:
        """
        Download image directly to memory without saving to disk
        
        Args:
            url: Image URL
            
        Returns:
            PIL Image object or None if failed
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
    
    def generate_embedding(self, image: Image.Image) -> List[float]:
        """
        Generate embedding for an image using DINOv2
        
        Args:
            image: PIL Image
            
        Returns:
            Embedding vector as list of floats
        """
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy()[0]
            
        norm = np.linalg.norm(embedding)
        if norm == 0: 
            # Handle the rare case of a zero vector
            return [0.0] * len(embedding)
        
        normalized_embedding = embedding / norm    
            
        return normalized_embedding.tolist()
    
    def generate_unique_id(self, product_id: str, img_url: str) -> str:
        """Generate unique ID for each image embedding"""
        combined = f"{product_id}_{img_url}"
        return hashlib.md5(combined.encode()).hexdigest()
    
  
    def process_product(self, product: Dict) -> List[PointStruct]:
        """
        Process a single product: download images, generate embeddings.
        Returns a list of PointStructs for Qdrant.
        
        Args:
            product: Dictionary containing product data from MySQL
        """
        product_id = str(product['id'])
       
        product_name = product.get('name', 'no_name').lower().strip()
        
        all_urls = []
        
        # 1. Get URLs from all single-image columns
        single_image_columns = [
            'image_path', 'image_path1', 'image_path2', 'image_path3'
        ]
        for col in single_image_columns:
            if col in product:
                all_urls.append(product[col])

        # 2. Get URLs from the 'image_paths' JSON column
        json_paths_str = product.get('image_paths')
        
        if json_paths_str:
            try:
                # Try to parse the string as JSON
                parsed_paths = json.loads(json_paths_str)
                if isinstance(parsed_paths, list):
                    all_urls.extend(parsed_paths)
                    
                elif isinstance(parsed_paths, str):
                    all_urls.append(parsed_paths)
                    
            except (json.JSONDecodeError, TypeError):
                if isinstance(json_paths_str, str):
                    all_urls.append(json_paths_str)

        unique_urls = {
            url.strip() for url in all_urls if url and url.strip()
        }
        

        points = []
        
        for img_url in unique_urls:
            
            # This log is now cleaner
            logger.info(f"Processing {product_name} (ID: {product_id}) - URL: {img_url}")
            
            # Download image to memory
            image = self.download_image_to_memory(img_url)
            if image is None:
                continue
            
            # Generate embedding
            try:
                embedding = self.generate_embedding(image)
            except Exception as e:
                logger.error(f"Failed to generate embedding for {img_url}: {e}")
                continue
            
            # ---Payload ---
            
            payload = {
                'product_id': product_id,
                'product_name': product_name,
                'image_url': img_url,
            }
            
            # Generate unique point ID (still safe to do)
            point_id = self.generate_unique_id(product_id, img_url)
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            points.append(point)
        
        return points

   
    def run_pipeline(self, batch_size: int = 50, total_products: Optional[int] = None):
        """
        Run the complete pipeline
        
        Args:
            batch_size: Number of products to fetch from MySQL at once
            total_products: Total number of products to process (None = all)
        """
        # Create collection
        self.create_collection()
        
        offset = 0
        processed = 0
        
        while True:
            # Fetch batch of products
            products = self.fetch_products_from_mysql(batch_size, offset)
            
            if not products:
                logger.info("No more products to fetch from MySQL.")
                break
            
            # --- NEW ---
            # List to hold all points for this MySQL batch
            points_batch = [] 
            
            # Process each product
            for product in products:
                try:
                    # --- CHANGED ---
                    # Call the renamed method and get the list of points
                    product_points = self.process_product(product)
                    
                    if product_points:
                        # --- NEW ---
                        # Add the generated points to the batch list
                        points_batch.extend(product_points)
                        
                    processed += 1
                    
                    if total_products and processed >= total_products:
                        logger.info(f"Reached limit of {total_products} products")
                        break # Break inner loop
                        
                except Exception as e:
                    logger.error(f"Error processing product {product.get('id')}: {e}")
                    continue
            
            # --- NEW ---
            # After processing all products in the MySQL batch,
            # upload all collected points to Qdrant in a single call.
            if points_batch:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points_batch
                )
                logger.info(f"Stored {len(points_batch)} embeddings in Qdrant for this batch")
            
            # Check again in case the inner loop was broken
            if total_products and processed >= total_products:
                break # Break outer loop
                
            offset += batch_size
            logger.info(f"Processed {processed} products so far...")
        
        logger.info(f"Pipeline complete! Processed {processed} products")
        
        