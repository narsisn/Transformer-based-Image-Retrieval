from pathlib import Path
import sys
sys.path.append(str(Path('.').absolute().parent))

import os 
import torch 
import glob
import pandas as pd
from statistics import mean


from src import SNAPDemo
from transformers import CLIPVisionModel, RobertaModel, AutoTokenizer,CLIPConfig


class evalute:
    def __init__(self, csv_path="", model_path="" ,image_path="",gender="",pro_cat="",main_cat="",k_n=10):
        self.csv_path = csv_path
        self.model_path = model_path
        self.image_path = image_path
        self.gender = gender
        self.pro_cat = pro_cat
        self.main_cat = main_cat
        self.k_n = k_n
    
    def load_model(self):

        config = CLIPConfig.from_pretrained(self.model_path)
        vision_encoder = CLIPVisionModel.from_pretrained(self.model_path, config=config.vision_config)
        return vision_encoder

    def acc_per_image(self,retrived_images,pin,path):
        img_cnt = len(glob.glob(path+'/*.jpg' ,recursive=True))
        correct_cnt = sum(pin in s for s in list(retrived_images))
        acc = correct_cnt/self.k_n
        return acc 

    def extract_features(self, model):
        # load the images 
        demo = SNAPDemo(model,device="cuda")
        images = glob.glob( self.image_path , recursive=True)
        # image embedding 
        demo.compute_image_embeddings(images)
        return demo
    
    def calc_acc(self,demo):
        path_prefix = '/home/yazahra/Documents/codes/pinterest_similar_data_crawler/'
        simData = pd.read_csv(self.csv_path)
        acc_list = []
        for index, row in simData.iterrows():
            if row['gender'] == self.gender and row['product_category'] == self.pro_cat and row['main_category'] == self.main_cat:
                image_name = row['image_list'].replace('[','').replace(']','').replace("'",'').split(',')[0]
                image_dir = row['image_path'] + '/'
                try:
                    if 'jpg' in image_name:
                        retrived_images = demo.image_query(path_prefix + image_dir + image_name,self.k_n)
                        acc = self.acc_per_image(retrived_images,row['pin'],path_prefix + image_dir)
                        acc_list.append(acc)
                except:
                    continue
        return mean(acc_list)

def main():

    # dataset file path 
    csv_path = 'csv_files/similar_images.csv'

    # gender and type
    gender = 'Men'
    pro_cat = 'Clothing'
    main_cat = 'Activewear'
    k_n = 10

    # image file path
    image_path = '/home/yazahra/Documents/codes/pinterest_similar_data_crawler/images/' + gender + '/' + pro_cat + '/' +  main_cat + '/**/*.jpg'

    # define pretrained model version
    model_source = 'openai'
    model_version = 'clip-vit-large-patch14'
    model_path = model_source + '/'+model_version

    # initial the class 
    evl = evalute(csv_path,model_path,image_path,gender,pro_cat,main_cat,k_n)

    # load pretrained model
    model = evl.load_model()

    # Image Embedding 
    demo = evl.extract_features(model)
    
    # claculate the accuracy 
    acc = evl.calc_acc(demo)

    # find . -maxdepth 1 -type f -size 0 -delete
    print(f'Accuracy = {acc :.4f}')

if __name__ == "__main__":
    main()

