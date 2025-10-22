
from ops import ProductEmbeddingPipeline



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
    
    
    pipeline.run_pipeline(batch_size=200)
    
    # Or process limited number for testing
    # pipeline.run_pipeline(batch_size=10, total_products=100)