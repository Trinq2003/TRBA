import os
import numpy as np
import cv2
import pandas as pd
import csv

import torch
from torch.utils.data import Dataset
from data.utils import resize_image, normalize_tensor_image, padding

class HWDataset(Dataset):
    '''
    Handwriting English Dataset
    '''
    def __init__(
        self, 
        root_dir: str,
        max_size: int,
        min_size: int,        
        max_len: int,
        split_type='A',
        mode='train',
    ):
        """Initialization of HW Dataset

        Args:
            root_dir (str): Relative root directory of images
            label_file (str): Relative path to txt file containing all labels and corresponding image names
            mode (str, optional): Mode of dataset. Defaults to 'train'.
            preprocess (bool, optional): Whether to perform preprocessing. Defaults to True.
            max_len (int, optional): Maximum length of the label. Defaults to 100.
        """

        self.root_dir = os.path.join(root_dir, "images")
        self.label_file = os.path.join(root_dir, f'IAM_splitting/{split_type}/{mode}.csv')
        
        # self.labels = pd.read_csv(
        #                 self.label_file, sep="\t", 
        #                 header=0, encoding="utf-8", 
        #                 na_filter=False, engine="python", 
        #                 usecols=[1,2])
        self.data_dict = pd.read_csv(
                        self.label_file, sep="\t",
                        header=0, encoding="utf-8", 
                        na_filter=False, engine="python").set_index('No').to_dict(orient='index')
        
        self.mode = mode
        self.max_len = max_len
        self.max_size = max_size
        self.min_size = min_size
        
    def preprocess(self, image: torch.tensor):
        _, H, W = image.size()
        image = resize_image(image, min_size=self.min_size, max_size=self.max_size)
        image = normalize_tensor_image(image)
        image = padding(image, min_size=self.min_size, max_size=self.max_size)
        return image
            
    def __len__(self):
        with open(self.label_file, 'r') as file:
            reader = csv.reader(file)
            num_of_samples = len(list(reader))-1
        return num_of_samples
    def __getitem__(self, idx):
        img_name = self.data_dict[idx+1]['Image']
        image = cv2.imread(os.path.join(self.root_dir, img_name))
        try:
            image = image.astype("float32")
        except:
            print(f"[ERROR] Image {img_name} (index {idx+1}) is not found. Return null image.")
            return torch.ones(3, self.min_size, self.max_size)*255, ''
        
        if image.ndim==2:
            image = image[np.newaxis]
        image = image.transpose((2,0,1))
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image = self.preprocess(image)
            
        # Label getter
        label = self.data_dict[idx+1]['Label']
        # label = self.labels.set_index('Image').T.to_dict('list')[img_name][0]
        return image, label
   
        
            
            
        
        
    