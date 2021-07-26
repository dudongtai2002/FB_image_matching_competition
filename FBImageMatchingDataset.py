import pandas as pd
import glob
import os
import numpy as np
import torch
from PIL import Image
from torch import is_tensor, empty, as_tensor
from torch.utils.data.dataset import Dataset



class FBImageMatchingDataset(Dataset):

    def __init__(self, root_dir, left_index = 0, right_index = None, transforms = None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample image.

        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_suffix = ".jpg"
        image_files = [filename for filename in glob.glob(os.path.join(root_dir,'*'+self.image_suffix))]
        print("detect {} jpg images under directory {}".format(len(image_files), root_dir))
        right_index = right_index if right_index != None else len(image_files)
        self.image_files = image_files[left_index:right_index]
        print("built dataset with {} entries".format(len(self.image_files)))

    def __len__(self):
        return len(self.image_files)
    
    def get_image_id(self, idx):
        img_path = self.image_files[idx]
        return img_path[len(self.root_dir):-len(self.image_suffix)]

    def __getitem__(self, idx):
        # The following logic is based on integral indices automated generated by pytorch dataloader

        # Args:
        #  idx(int) : the index number of the image, depends on the given range when initializing this dataset
        
        # not sure the below part of code is useful
        if is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_files[idx]

        # Open the image using Pillow
        image = Image.open(img_name)       

        # apply transforms to the image
        if self.transforms:
            image = self.transforms(image)
            
        return image, as_tensor(idx)