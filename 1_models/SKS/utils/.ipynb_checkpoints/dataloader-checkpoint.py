import os
import torch
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from .imageprocess import image_transformer, image_processor

class CustomDataLoader():
    
    def __init__(self, img_dir=None, label_dir=None, train=True, row_index=None, device=None):
        """
        Arguments:
        img_dir  : Where train dataset is located, e.g. /loc/of/your/path/trainset
        label_dir: Where label csv file is located, e.g. /loc/of/your/path/LABEL.csv
        train    : Used to specify trainset. For validation and testset, set this value as False.
        row_idx  : Mainly used for train/val split. Among all images it specifies which row should be processed.
            Usage INPUT:np.array()
                  e.g. np.array([1, 5, 10, 102, ...])
        use_cuda : Decide whether to use cuda. If cuda is not available, it will be set False.
        """
        assert os.path.exists(img_dir) and os.path.exists(label_dir), "Path not exists."
        assert row_index, "Index should be given."
        
        self.img_dir = img_dir
        self.label_dir = label_dir
        
        label_df = pd.read_csv(label_dir)
        
        self.label_index = label_df.iloc[row_index, 0].values
        self.label_values = label_df.iloc[row_index, 1:].values
        
        # Transformation
        self.train = train

        # Set device
        self.device = device
        
        
    def __len__(self):
        return self.label_index.__len__()
    
    
    def __getitem__(self, item_index):
        """
        This method returns [label, image] given an index.
        Returned values are of torch.tensor type.
        """
        idx = item_index
        label = self.label_values[idx]
        
        image_idx = str(self.label_index[idx]).zfill(5) + '.png'
        image_path = os.path.join(self.img_dir, image_idx)
        assert os.path.exists(image_path), f"Given image path not exists: {image_path}"
        
        #_image = cv2.imread(image_path)
        #pil_image = Image.open(image_path).convert('RGB')
        pil_image = Image.open(image_path)
        fin_image = image_transformer(pil_image, self.train)

        # To torch.tensor type
        label_item = torch.tensor(label).float().to(self.device)
        image_item = fin_image.float().to(self.device)
        
        return image_item, label_item