from pathlib import Path
import sys
sys.path.append(str(Path('.').absolute().parent))

import os 
import torch 
import glob
import pandas as pd

from src import SNAPDemo
from transformers import CLIPVisionModel, RobertaModel, AutoTokenizer,CLIPConfig


class evalute:
    def __init__(self, csv_path="", model_path="" ,image_path=""):
        self.csv_path = csv_path
        self.model_path = model_path
        self.image_path = image_path
    
    def load_model(self):

        config = CLIPConfig.from_pretrained(self.model_path)
        vision_encoder = CLIPVisionModel.from_pretrained(self.model_path, config=config.vision_config)
        return vision_encoder

    def extract_features(self, model):
        # load the images 
        demo = SNAPDemo(model)
        images = glob.glob( self.image_path , recursive=True)
        # image embedding 
        demo.compute_image_embeddings(images)
        return demo
    
    def calc_acc(self,demo):
        simData = pd.read_csv(self.csv_path)
        for index, row in simData.iterrows():
            print(row['pin'])
            # retrived_images = demo.image_query(row['iamge..'],10)

        return acc


def main():

    # dataset file path 
    csv_path = 'csv_files/similar_images.csv'

    # image file path
    image_path = '~/Documents/codes/pinterest_similar_data_crawler/images/Women/***/*.jpg'
    image_path = glob.glob(image_path, recursive=True)
    print(image_path)
    exit(0)

    # define pretrained model version
    model_source = 'openai'
    model_version = 'clip-vit-large-patch14'
    model_path = model_source + '/'+model_version

    # initial the class 
    evl = evalute(csv_path,model_path,image_path)

    # load pretrained model
    model = evl.load_model()

    # Image Embedding 
    demo = extract_features(model)

    # claculate the accuracy 
    acc = evl.calc_acc(demo)

    print(f'Accuracy = {acc :.4f}')

if __name__ == "__main__":
    main()

