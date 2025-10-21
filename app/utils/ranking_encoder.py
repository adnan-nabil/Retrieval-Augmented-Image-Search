from sentence_transformers import CrossEncoder
from typing import List
from qdrant_client import models 
from config import settings

class CrossEncoderReranker:
    """
    Encapsulates the Cross-Encoder model for re-ranking search results.
    The model is loaded once upon initialization and stored in the instance.
    """
    def __init__(self):
        self.model_name = settings.RANKER_MODEL   
        try:
            # Load the model 
            self.model = CrossEncoder(self.model_name, max_length=512)
            print("RANKER LOADED SUCCESSFULLY...")
            
        except Exception as e:
            raise RuntimeError(f"FAILED TO LOAD RANKER... '{self.model_name}': {e}")

    def rerank(self, primary_search_results: List[models.ScoredPoint], text_query: str) -> List[models.ScoredPoint]:
        """
        Re-ranks a list of candidate points based on a text query.

        Args:
            primary_search_results (List[models.ScoredPoint]): The list of search results from Qdrant.
            text_query (str): The text to compare against.

        Returns:
            List[models.ScoredPoint]: The same list, sorted by the new rerank_score.
        """
        # if nothing from client
        if not text_query or not primary_search_results:
            return primary_search_results

        # Create pairs of [query, product_description] for the model to score
        sentence_pairs = []
        for point in primary_search_results:
            product_name = f"{point.payload.get('product_name', '')}"
            sentence_pairs.append([text_query.strip(), product_name.strip()])

        # ranking
        scores = self.model.predict(sentence_pairs)

        # Add the new relevance score to each candidate's payload
        for point, score in zip(primary_search_results, scores):
            point.payload['rerank_score'] = float(score)

        # Sort the candidates by the new rerank_score in descending order
        reranked_results = sorted(
            primary_search_results,
            key=lambda p: p.payload.get('rerank_score', 0.0),
            reverse=True
        )

        return reranked_results
    
# make an instance to be used everywhere and once at a time    
reranker = CrossEncoderReranker()    