import math
import torch
import cv2
import numpy as np
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os 
mean_all = []
var_all = []
path = 'test_data/images'
ids = [x.split('.')[0] for x in os.listdir(path)]
def get_mean_std(ids):
    count = 0
    for i in range(len(ids)):
        mean = []
        var = []
        np_img = np.array(Image.open(path + '/%s.jpg' % ids[i]), dtype=float)
        # image_path = path + ids[i] + '.jpg'
        # img = cv2.imread(image_path)
        # print(img.dtype)
        # np_img = np.array(img)
        tensor = torch.Tensor(np_img)
        old_value = tensor
        new_tensor = tensor.view(tensor.size(2),-1) 
        new_tensor = new_tensor.to(device)
        value = new_tensor
        mean.append(torch.mean(new_tensor, dim = 1))
        var.append(torch.var(new_tensor, dim = 1))

        mean_all.append(mean)
        var_all.append(var)
        count +=1
        #print(i, '\t', torch.mean(new_tensor, dim = 1).data)
    b_channel_mean = 0
    r_channel_mean = 0
    g_channel_mean = 0
    b_channel_var = 0
    r_channel_var = 0
    g_channel_var = 0
    
    
    for i in range(len(mean_all)):
        mean_array = mean_all[i][0].detach().cpu().data.numpy()
        b_channel_mean += mean_array[0]
        r_channel_mean += mean_array[1]
        g_channel_mean += mean_array[2]

        var_array = var_all[i][0].detach().cpu().data.numpy()
        b_channel_var += var_array[0]
        r_channel_var += var_array[1]
        g_channel_var += var_array[2]



    mean_b, mean_r, mean_g = b_channel_mean / count, r_channel_mean / count, g_channel_mean / count
    std_b, std_r, std_g = math.sqrt(b_channel_var / count), math.sqrt(r_channel_var / count) , math.sqrt(g_channel_var / count)

    b_mean, r_mean, g_mean = mean_b / 255, mean_r / 255, mean_g / 255
    b_std, r_std, g_std = std_b / 255, std_r / 255, std_g / 255 
    
    return b_mean, r_mean, g_mean, b_std, r_std, g_std

b_mean, r_mean, g_mean, b_std, r_std, g_std = get_mean_std(ids)
print(f"bm:{b_mean}, rm:{r_mean}, gm:{g_mean}, bs:{b_std}, rs:{r_std}, gs:{g_std}")




        




