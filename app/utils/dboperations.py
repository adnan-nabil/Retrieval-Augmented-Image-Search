from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from typing import List, Optional, Dict
from PIL import Image, UnidentifiedImageError
from config import settings
from utils.image_encoder import encoder
from utils.ranking_encoder import reranker
import hashlib
import aiohttp
import io
import mysql.connector
import json



print("All models initialized and ready.")

class DBOperations:
    def __init__(self, tenant_info: Dict):
        self.tenant_info = tenant_info
        self.tenant_user_id = tenant_info.get('tenant_user_id')
        self.client = QdrantClient(url=tenant_info.get('qdrant_url'))
        self.collection_name = tenant_info['qdrant_collection']
        self.vector_dimension = settings.VECTOR_DIMENSION
        self.k = 30 # how many similar points I want
        self.mysql_config = {
            'host': tenant_info['db_host'],
            'port': tenant_info['db_port'],
            'user': tenant_info['db_user'],
            'password': tenant_info['db_pass'],
            'database': tenant_info['db_name']
        }
        

    def find_similar(self, image: Image.Image, text_query: Optional[str]) -> List[models.ScoredPoint]:
        """
        Generates a vector for the input image and searches Qdrant for top K results.
        Then rank them as brand name/model name etc. using a cross-encoder model.
        """
        # 1. Encode the search image
        image_embedding = encoder.encode_image(image)
        
        # 2. Perform the search
        search_results = self.client.search(
            collection_name = self.collection_name,
            query_vector = image_embedding,
            limit = self.k, 
            with_payload = True
        )
        
        # pass it to the cross-encoder for ranking by text_query
        search_results_with_rank = reranker.rerank(search_results, text_query)
        
        return search_results_with_rank
    
    def generate_unique_id(self, product_id: str, img_url: str) -> str:
        """Generate unique ID for each image embedding"""
        combined = f"{product_id}_{img_url}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def upsert_product_vector(self, product_id: str, product_name: str, image_url: str, image: Image.Image) -> str:
        """
        Creates/updates a vector, MATCHING the pipeline's logic.
        """
        # 1. Generate unique ID (MATCHES PIPELINE)
        point_id = self.generate_unique_id(product_id, image_url)
        
        vector = encoder.encode_image(image)
        
        payload = {
            "product_id": str(product_id), # Match pipeline's str() cast
            "product_name": product_name.lower().strip(), # Match pipeline's formatting
            "image_url": image_url 
        }
        
        # 4. Create PointStruct
        points = [
            models.PointStruct(
                id=point_id, 
                vector=vector,
                payload=payload
            )
        ]
        
        # 5. Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )
        
        print(f"Upsert successful for ID {point_id}.")
        return point_id

    async def _download_image_async(self, session: aiohttp.ClientSession, url: str) -> Optional[Image.Image]:
        """
        Asynchronously download a single image and verify it.
        """
        try:
            async with session.get(url, timeout=60) as response:
                response.raise_for_status() 

                content_type = response.headers.get('Content-Type')
                if not content_type or not content_type.startswith('image/'):
                    raise UnidentifiedImageError("URL does not point to an image.") 

                image_data = await response.read()

                try:
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    return image
                except Exception:
                    raise UnidentifiedImageError("Downloaded data is not a valid image.") 

        except Exception:
            raise 
    
    def get_product_data_from_mysql(self, product_id: str) -> Optional[Dict]:
        """
        Fetches the data for a single product from MySQL.
        """
        conn = mysql.connector.connect(**self.mysql_config)
        cursor = conn.cursor(dictionary=True)
        
        base_query = """
        SELECT 
            id, name, image_path, image_path1, 
            image_path2, image_path3, image_paths
        FROM products
        WHERE id = %s
        """
        params = [product_id]
        
        if self.tenant_user_id:
            base_query += " AND user_id = %s"
            params.append(self.tenant_user_id)
        
        cursor.execute(base_query, tuple(params))
        product = cursor.fetchone()
        
        cursor.close()
        conn.close()
        return product

    def extract_urls_from_product(self, product: Dict) -> List[str]:
        """
        Parses a product dictionary to get all unique, valid image URLs.
        """
        all_urls = []
        single_image_columns = ['image_path', 'image_path1', 'image_path2', 'image_path3']
        for col in single_image_columns:
            if col in product and product[col]:
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
                    
        # Return a list of unique, non-empty URLs
        return list({url.strip() for url in all_urls if url and url.strip()})    
    

    def delete_product_vectors(self, product_id: str) -> int:
        """
        Deletes all vector points from Qdrant that match the given product_id.
        
        Args:
            product_id: The ID of the product to delete.
            
        Returns:
            The count of deleted points.
        """
        print(f"Attempting to delete points for product_id: {product_id} from {self.collection_name}")
        
        # 1. Create a filter to target the payload field
        deletion_filter = Filter(
            must=[
                FieldCondition(
                    key="product_id",  # The field in your payload
                    match=MatchValue(value=str(product_id)) # Ensure it's a string
                )
            ]
        )
        
        # 2. Execute the delete operation
        operation_info = self.client.delete_points(
            collection_name=self.collection_name,
            points_selector=deletion_filter,
            wait=True  # Wait for the operation to complete
        )
        
        deleted_count = operation_info.result.deleted
        print(f"Delete operation status: {operation_info.status}. Deleted {deleted_count} points.")
        
        return deleted_count
