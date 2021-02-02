import os
import torch
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from .imageprocess import image_transformer, image_processor

class CustomDataLoader():
    def __init__(self, img_dir, label_dir, transform=True, device=None):
        """
        Arguments:
        img_dir  : Where train dataset is located, e.g. /loc/of/your/path/trainset
        label_dir: Where label csv file is located, e.g. /loc/of/your/path/LABEL.csv
        use_cuda : Decide whether to use cuda. If cuda is not available, it will be set False.
        """
        assert os.path.exists(img_dir) and os.path.exists(label_dir), "Path not exists."
        
        self.img_dir = img_dir
        self.label_dir = label_dir
        
        label_df = pd.read_csv(label_dir)
        self.label_index  = label_df.iloc[:, 0].values
        self.label_values = label_df.iloc[:, 1:].values
        
        self.transform = transform
        
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
        
        _image = cv2.imread(image_path)
        
        if self.transform:
            image = np.array(image_processor(_image))
        else:
            image = np.array(_image)
        
        pil_image = Image.fromarray(image)
        fin_image = image_transformer(pil_image)

        # To torch.tensor type
        label_item = torch.tensor(label).float().to(self.device)
        image_item = fin_image.float().to(self.device)
        
        return image_item, label_item