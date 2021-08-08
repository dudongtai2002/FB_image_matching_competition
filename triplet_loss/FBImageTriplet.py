import pandas as pd
import glob
import os
import numpy as np
import random
import torch
from PIL import Image
from torch import is_tensor, empty, as_tensor
from torch.utils.data.dataset import Dataset
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance



class FBImgMatchingDataSetTriplet(Dataset):

    def __init__(self, query_dir, reference_dir, training_dir, ground_truth_csv, transforms = None):
        """
        Args:
            query_dir (string): Directory of the query image, a.k.a anchor image
            reference_dir (string): Directory of the reference image, a.k.a positive image
            training_dir (string): Directory of the reference image, a.k.a negative image
            ground_truth_csv: Path of the given ground truth csv, contains the mapping from query image to reference image
            transform (callable, optional): Optional transform to be applied on a sample image.

        """
        self.transforms = transforms
        self.query_dir = query_dir
        self.reference_dir = reference_dir
        self.training_dir = training_dir
        self.image_suffix = ".jpg"
        
        query_image_files = [filename for filename in glob.glob(os.path.join(query_dir,'*'+self.image_suffix))]
        print("detect {} jpg images under query directory {}".format(len(query_image_files), query_dir))
        reference_image_files = [filename for filename in glob.glob(os.path.join(reference_dir,'*'+self.image_suffix))]
        print("detect {} jpg images under reference directory {}".format(len(reference_image_files), reference_dir))
        training_image_files = [filename for filename in glob.glob(os.path.join(training_dir,'*'+self.image_suffix))]
        print("detect {} jpg images under directory {}".format(len(training_image_files), training_dir))
        
        self.negative_sample_files = training_image_files # use training images as negative samples
        
        # read in ground truth as dataframe
        ground_truth = pd.read_csv(ground_truth_csv)
        self.ground_truth = ground_truth.loc[ground_truth['reference_id'].notnull()] #filter out null rows
        print("detect {} number of ground truth pairs".format(len(self.ground_truth)))
        

    def __len__(self):
        return len(self.ground_truth)
    
    def get_image_id(self, idx):
        return self.ground_truth.iloc[idx]['query_id'], self.ground_truth.iloc[idx]['reference_id']

    def __getitem__(self, idx):
        # Args:
        #  idx(int) : the index number of the image, depends on the given range when initializing this dataset
        
        if is_tensor(idx):
            idx = idx.tolist()

        query_img_id,ref_image_id = self.get_image_id(idx)
        query_image_path = self.query_dir + query_img_id + self.image_suffix
        ref_image_path = self.reference_dir + ref_image_id + self.image_suffix
        
        # Open the image using Pillow
        query_image = Image.open(query_image_path)
        ref_image = Image.open(ref_image_path)
        
        # sample 10 negative images
        negative_image_paths = random.sample(self.negative_sample_files, 10) #return list of paths
        negative_images = [Image.open(img_path) for img_path in negative_image_paths]

        # apply transforms to the image
        if self.transforms:
            query_image = self.transforms(query_image)
            ref_image = self.transforms(ref_image)
            negative_images = [self.transforms(img) for img in negative_images]
        
        # stack negative samples:
        negative_image_stack = torch.stack(negative_images) # size: (10, 3, length, width)
        
        return query_image, ref_image, negative_image_stack, as_tensor(idx)
    
    
    
    
""" 
This code was imported from tbmoon's 'facenet' repository:
https://github.com/tbmoon/facenet/blob/master/utils.py
"""    
    
class TripletLoss(Function):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.pdist.forward(anchor, positive)
        neg_dist = self.pdist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)

        return loss