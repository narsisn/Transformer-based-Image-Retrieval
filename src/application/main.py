import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import default_data_collator

from .utils import VisionDataset, TextDataset



class SNAPDemo:
    def __init__(self, vision_encoder,
                 batch_size: int = 32, max_len: int = 64, device='cuda'):
        """ Initializes CLIPDemo
            it has the following functionalities:
                image_search: Search images based on text query
               
            Args:
            vision_encoder: Fine-tuned vision encoder
            device (torch.device): Running device
            batch_size (int): Size of mini-batches used to embeddings
            max_length (int): Tokenizer max length
        
        """
        self.vision_encoder = vision_encoder.eval().to(device)
        self.batch_size = batch_size
        self.device = device
        self.max_len = max_len
        self.text_embeddings_ = None
        self.image_embeddings_ = None
        

    def compute_image_embeddings(self, image_paths: list):
        """ Compute image embeddings for a list of image paths
            Args:
                image_paths (list[str]): An image database
        """
        self.image_paths = image_paths

        datalodear = DataLoader(VisionDataset(
            image_paths=self.image_paths), batch_size=self.batch_size)
        embeddings = []
        with torch.no_grad():
            for images in tqdm(datalodear, desc='computing image embeddings'):
                  
                image_embedding = self.vision_encoder(
                            pixel_values=images.to(self.device)).pooler_output
                    
                embeddings.append(image_embedding)
        self.image_embeddings_ =  torch.cat(embeddings)

    def image_query_embedding(self, image):
        image = VisionDataset.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            image_embedding = self.vision_encoder(
                image.to(self.device)).pooler_output
        return image_embedding

    def most_similars(self, embeddings_1, embeddings_2):
        values, indices = torch.cosine_similarity(
            embeddings_1, embeddings_2).sort(descending=True)
        return values.cpu(), indices.cpu()

    def image_search(self,  image_path: str, top_k=10):
        """ Search images based on text query
            Args:
                image_path (str): image query 
                top_k (int): number of relevant images 
        """
     
        print(image_path)
        image = Image.open(image_path)
        image_embedding = self.image_query_embedding(image)
        _, indices = self.most_similars(self.image_embeddings_, image_embedding)

        matches = np.array(self.image_paths)[indices][:top_k]
        _, axes = plt.subplots(2, int(top_k/2), figsize=(15, 5))
        for match, ax in zip(matches, axes.flatten()):
            ax.imshow(Image.open(match).resize((224, 224)))
            ax.axis("off")
        plt.show()
    def image_query(self,  image_path: str, top_k=10):
        """ Search images based on text query
            Args:
                image_path (str): image query 
                top_k (int): number of relevant images 
        """
     
        print(image_path)
        image = Image.open(image_path)
        image_embedding = self.image_query_embedding(image)
        _, indices = self.most_similars(self.image_embeddings_, image_embedding)

        matches = np.array(self.image_paths)[indices][:top_k]
        return matches
        
  