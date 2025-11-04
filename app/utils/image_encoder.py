import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from typing import List
import numpy as np
from config import settings

class DinoEmbeddingModel:
    """
    A class to encapsulate the DINOv2 model for generating image embeddings.

    This class handles the loading of the model and processor and provides a method
    to encode PIL images into fixed-dimension vector embeddings.

    Attributes:
        model_name (str): The name or path of the pretrained model.
        device (str): The device ('cuda', 'cpu') to run the model on.
        model (AutoModel): The loaded transformer model.
        processor (AutoImageProcessor): The loaded image processor.
    """

    def __init__(self):
 
        self.model_name = settings.MODEL_NAME
        self.device = settings.DEVICE
        print(f"Loading DINO model.....")
          
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print("DINO LOADED SUCCESSFULLY...")
            
        except Exception as e:
            raise RuntimeError(f"FAILED TO LOAD '{self.model_name}': {e}")
        

    def encode_image(self, image: Image.Image) -> List[float]:
        """
        Args:
            image (Image.Image): The input image to encode.

        Returns:
            List[float]: A normalized vector embedding of the image.
        """
            
        if not self.model or not self.processor:
            raise RuntimeError("Model is not loaded. Initialization might have failed.")

        # Ensure the image is in RGB format, as expected by most vision models
        image = image.convert("RGB")
        
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy()[0]
        
        norm = np.linalg.norm(embedding)
        if norm == 0: 
            # Handle the rare case of a zero vector
            return [0.0] * len(embedding)    

        normalized_embedding = embedding / norm
        return normalized_embedding.tolist()
        
    def encode_image_batch(self, images: List[Image.Image]) -> List[List[float]]:
        """
        Args:
            images (List[Image.Image]): A list of input images to encode.

        Returns:
            List[List[float]]: A list of normalized vector embeddings.
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model is not loaded. Initialization might have failed.")
            
        if not images:
            return []

        rgb_images = [img.convert("RGB") for img in images]
        
        with torch.no_grad():
            inputs = self.processor(
                images=rgb_images, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            
            # Move inputs to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized_embeddings = embeddings / norms
        
        return normalized_embeddings.tolist()




encoder = DinoEmbeddingModel()

