from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from typing import List, Optional
from PIL import Image
from config import settings
from utils.pydantic_schemas import hash_product_id_and_index
from utils.image_encoder import encoder
from utils.ranking_encoder import reranker



print("All models initialized and ready.")

class DBOperations:
    def __init__(self):
        self.url = settings.QDRANT_URL
        self.client = QdrantClient(url=self.url)
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.vector_dimension = settings.VECTOR_DIMENSION
        self.k = 100 # how many similar points I want
        

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
    
    
    
    def upsert_product_vector(self, product_id: str, image_index: str, image: Image.Image) -> int:
        """
        Creates a new vector or updates an existing vector for a specific image.
        """
        # 1. Generate unique ID for Qdrant
        point_id = hash_product_id_and_index(product_id, image_index)
        
        # 2. Generate vector embedding
        vector = self.encoder.encode_image(image)
        
        # 3. Define payload
        payload = {
            "product_id": product_id,
            "filename": f"{product_id}_{image_index}.jpg", # Mocking filename structure
            "image_index": image_index,
            "qdrant_point_id": point_id
        }
        
        # 4. Create PointStruct
        points = [
            models.PointStruct(
                id=point_id, 
                vector=vector,
                payload=payload
            )
        ]
        
        # 5. Upsert to Qdrant (will update if ID exists, insert if new)
        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )
        
        print(f"Upsert successful for ID {point_id}. Status: {operation_info.status}")
        return point_id

    def delete_product_vectors(self, product_id: str, image_index: str = None):
        """
        Deletes one specific vector (by ID and index) or all vectors (by product_id).
        """
        if image_index:
            # Case 1: Delete specific image vector
            point_id = hash_product_id_and_index(product_id, image_index)
            # Qdrant delete by ID
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[point_id])
            )
            print(f"Deleted specific point ID: {point_id}")
            return [point_id]

        else:
            # Case 2: Delete ALL vectors for the product (using filter)
            delete_filter = Filter(
                must=[FieldCondition(key="product_id", match=MatchValue(value=product_id))]
            )
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=delete_filter),
                wait=True
            )
            print(f"Deleted all points for Product ID: {product_id}")
            return []
