import cv2
import numpy as np
import os
import random

import torch
from torchvision import transforms as T
from torch.utils import data
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, path, mode = None):

        self.top_label_path = path + "/labels/"
        self.top_img_path = path + "/images/"
        self.mode = mode


        self.ids = [x.split('.')[0] for x in sorted(os.listdir(self.top_img_path))] # sorted names of all images
        self.num_images = len(self.ids)


        self.object_list = self.get_objects(self.ids) # object_list is a list containg tuples where each tuple containes name of image 
        # with index of line in label, this is suppose if there are multiple fiiferent lines containing different observation
        # but for our project we can save evrything in one line, there object list would be tuple containing image name and 0 as line number.

        self.labels = {}
        last_id = ""
        for obj in self.object_list:
            id = obj[0]
            line_num = obj[1]
            label = self.get_label(id, line_num)
            if id != last_id:
                self.labels[id] = {}
                last_id = id

            self.labels[id][str(line_num)] = label # self.label is a dict with key as image 

        # hold one image at a time
        self.curr_id = ""
        self.curr_img = None


    # should return (Input, Label)
    def __getitem__(self, index):
        id = self.object_list[index][0]
        line_num = self.object_list[index][1]

        if id != self.curr_id:
            self.curr_id = id
            self.curr_img = Image.open(self.top_img_path + '%s.jpg'%id)

        label = self.labels[id][str(line_num)]

        obj = Normalize(self.curr_img, self.mode)
 

        return obj.image, label

    def __len__(self):
        return len(self.object_list)

    def get_objects(self, ids):
        objects = []
        for id in ids:
            with open(self.top_label_path + '%s.txt'%id) as file:
                for line_num,line in enumerate(file):
                    line = line[:-1].split(' ')    
                    objects.append((id, line_num))

        return objects


    def get_label(self, id, line_num):
        lines = open(self.top_label_path + '%s.txt'%id).read().splitlines()
        label = self.format_label(lines[line_num])

        return label


    def format_label(self, line):
        line = line[:-1].split(' ')
        my_label = torch.zeros([1,9])
                  
        for i in range(len(line)):
            my_label[0,i] = float(line[i])

        label = {
                'labels': my_label
                }

        return label

class Normalize:
    
    def __init__(self, img, mode):
        
        self.mode = mode
        self.image = self.format_img(img)

    def format_img(self, img):
        if self.mode == 'train':
            transform = T.Compose([T.Resize((224, 224)),
                                    T.ToTensor(),
                                    
            ])
            
        elif self.mode == 'val':
            
            transform = T.Compose([T.Resize((224, 224)),
                                    T.ToTensor(),
                                   
            ])

        elif self.mode == 'test':
            
            transform = T.Compose([T.Resize((224, 224)),
                                    T.ToTensor(),
                                   
            ])
            
        image = transform(img)

        return image

