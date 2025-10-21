
from ops import ProductEmbeddingPipeline



if __name__ == "__main__":
    
    mysql_config = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': '',
        'database': 'shamim_db'
    }
    
    
    pipeline = ProductEmbeddingPipeline(
        mysql_config=mysql_config,
        qdrant_url="http://localhost:6333",
        collection_name="gadget_bodda_dino"  # Ensure this matches your MySQL database name
    )
    
    
    pipeline.run_pipeline(batch_size=50)
    
    # Or process limited number for testing
    # pipeline.run_pipeline(batch_size=10, total_products=100)