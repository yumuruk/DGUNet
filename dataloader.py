import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

import torch
 
class UJEDDataset(Dataset):
    def __init__(self, root, transform=None, is_test = False):
        self.input_files, self.gt_files, self.t_p_files, self.B_p_files, self.label_files = self.get_file_paths(root, is_test)
        if is_test==False:
            self.len = min(len(self.input_files), len(self.gt_files), len(self.B_p_files), len(self.t_p_files), len(self.label_files))
        else:
            self.len = min(len(self.input_files), len(self.t_p_files), len(self.B_p_files), len(self.label_files))

        self.max_objects = 100
        self.transform = transform
        self.transform = T.Compose(transform)


    
    def __getitem__(self, index):
    
        input_image = Image.open(self.input_files[index % self.len])
        
        if len(self.gt_files) > 0:
            gt_image = Image.open(self.gt_files[index % self.len])
            gt_image = self.transform(gt_image)
        else:
            gt_image = self.transform(input_image.copy())
        
        t_p = Image.open(self.t_p_files[index % self.len])
        B_p = Image.open(self.B_p_files[index % self.len])
        
        input_image = self.transform(input_image)
        t_p = self.transform(t_p)
        B_p = self.transform(B_p)

        file_names = os.path.basename(self.input_files[index % self.len])

        label_file_path = self.label_files[index % self.len]
        with open(label_file_path, 'r') as f:
            lines = f.readlines()

            labels = [] 
            for line in lines:
                if line.strip():
                    float_values = [float(value) for value in line.strip().split()]
                    labels.append(float_values)
                
            if not labels:
                labels_tensor = torch.zeros((0, 5), dtype=torch.float32)
            else:
                labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        labels = torch.zeros(labels_tensor.shape[0], 6)
        labels[:, 1:] = labels_tensor

        return {"inp": input_image,"gt": gt_image,"t": t_p,"B": B_p,"labels": labels_tensor,"fn": file_names}
    
    def __len__(self):
        return self.len 

    def get_file_paths(self, root, is_test):
        if is_test == True:
            print(f"[INFO] Test mode: No ground truth images will be loaded from {root}")
            input_files = sorted(glob.glob(os.path.join(root, 'images') + "/*.*"))
            t_p_files = sorted(glob.glob(os.path.join(root, 't_prior') + "/*.*"))
            B_p_files = sorted(glob.glob(os.path.join(root, 'B_prior') + "/*.*"))
            label_files = sorted(glob.glob(os.path.join(root, 'labels') + "/*.*"))
            gt_files = []
        else:
            input_files = sorted(glob.glob(os.path.join(root, 'images') + "/*.*"))
            gt_files = sorted(glob.glob(os.path.join(root, 'gt') + "/*.*"))
            t_p_files = sorted(glob.glob(os.path.join(root, 't_prior') + "/*.*"))
            B_p_files = sorted(glob.glob(os.path.join(root, 'B_prior') + "/*.*"))
            label_files = sorted(glob.glob(os.path.join(root, 'labels') + "/*.*"))
        return input_files, gt_files, t_p_files, B_p_files, label_files
    

 