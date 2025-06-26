# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import omegaconf
import hydra
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt


from r3m import load_r3m

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

r3m = load_r3m("resnet50") # resnet18, resnet34
r3m.eval()
r3m.to(device)

## DEFINE PREPROCESSING
transforms = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()]) # ToTensor() divides by 255

## ENCODE IMAGE
image_1 = Image.open("./test_timestep_4.jpg")
image_2 = Image.open("./168test_100.jpg")



image_3 = Image.open("./19test_51.jpg")
image_4 = Image.open("./test_timestep_34.jpg")

#image = cv2.imread(r"C:/Users/ghassen.jridi/Desktop/r3m/robot_arm.JPG")
img_1 = np.array(image_1)
img_2 = np.array(image_2)

img_3 = np.array(image_3)
img_4 = np.array(image_4)
#print(img_1.shape)
#print(img_2.shape)




# Define transformations



#print(img)
#img = img.astype(int)
#image = np.random.randint(0, 255, (500, 500, 3))
#first figure
preprocessed_image_1 = transforms(Image.fromarray(img_1.astype(np.uint8))).reshape(-1, 3, 224, 224)
preprocessed_image_2 = transforms(Image.fromarray(img_2.astype(np.uint8))).reshape(-1, 3, 224, 224)
#preprocessed_image = transforms(Image.fromarray(image.astype(np.uint8))).reshape(-1, 3, 224, 224)
preprocessed_image_1.to(device) 
preprocessed_image_2.to(device) 


#second figure
preprocessed_image_3 = transforms(Image.fromarray(img_3.astype(np.uint8))).reshape(-1, 3, 224, 224)
preprocessed_image_4 = transforms(Image.fromarray(img_4.astype(np.uint8))).reshape(-1, 3, 224, 224)
preprocessed_image_3.to(device) 
preprocessed_image_4.to(device) 


with torch.no_grad():

  embedding1 = r3m(preprocessed_image_1 * 255.0)
  embedding2 = r3m(preprocessed_image_2 * 255.0) ## R3M expects image input to be [0-255]

  embedding3 = r3m(preprocessed_image_3 * 255.0) 
  embedding4 = r3m(preprocessed_image_4 * 255.0) 

#print(embedding1)
#shape = embedding1.size()
#print("embedding1 shape:", shape)

# Transpose the tensor
#transposed_embedding1 = torch.transpose(embedding1, 0, 1)  # You can specify the dimensions to swap
#transposed_embedding2 = torch.transpose(embedding2, 0, 1)
#x_high_precision = torch.linspace(0, len(transposed_embedding1) - 1, 2048)

# Print the original and transposed tensors
#print("Original Tensor:")
#print(embedding1)
#print("\nTransposed Tensor:")
#print(transposed_embedding1)
#shape_transposed_embedding1 = transposed_embedding1.size()
#print("transposed_embedding1 shape:", shape_transposed_embedding1)


a = embedding1
b = embedding2

c = embedding3
d = embedding4
#a = torch.log(embedding1)
#b = torch.log(embedding2)
#a = torch.log(transposed_embedding1)
#b = torch.log(transposed_embedding2)
#print(a)
"""
fig = plt.figure(figsize=(8, 12))
height_ratios = [2, 2]  # Adjust as needed

# Create a 1x2 grid with specified width ratios
gs = fig.add_gridspec(2, 1, height_ratios=height_ratios)

ax1 = fig.add_subplot(gs[0, 0])
ax1.stem(x_high_precision.numpy(), a.numpy(), linefmt='b-', markerfmt='bo', basefmt='k-', label='Latent 1')
#plt.subplot(2, 2, 1)
#plt.plot(x_high_precision, a.numpy(), marker='o', color = 'orange',label='Latent 1')
#plt.plot(b.numpy(), marker='s', label='Latent 2')
ax1.set_title('logarithmic Latent Representations 1')
ax1.set_xlabel('Data Points')
ax1.set_ylabel('Latent Values')
ax1.legend()

ax2 = fig.add_subplot(gs[1, 0])
ax2.stem(x_high_precision.numpy(), b.numpy(), linefmt='r-', markerfmt='ro', basefmt='k-', label='Latent 2')
ax2.set_title('logarithmic Latent Representations 1')
ax2.set_xlabel('Data Points')
ax2.set_ylabel('Latent Values')
ax2.legend()

#plt.subplot(2, 2, 3)
#plt.plot(x_high_precision, b.numpy(), marker='s', color = 'blue', label='Latent 2')
#plt.title('Logarithmic Latent Representation 2')
#plt.legend()



plt.tight_layout()
# Save the plot
plt.savefig('latent_representations_plot.png')

# Show the plot
plt.show()
#plt.tight_layout()
"""
np_array1 = a.detach().cpu().data.numpy()
np_array2 = b.detach().cpu().data.numpy()
np_array3 = c.detach().cpu().data.numpy()
np_array4 = d.detach().cpu().data.numpy()
  
np_diff1 = np.abs(np_array2 - np_array3)
np_diff1 = np.log(np_diff1)
np_diff1 = np_diff1[:, :2025]
print(np_diff1.shape)
#diff = torch.log(diff)
#np_diff = diff.detach().cpu().data.numpy
#print("np_diff", shape_diff)
image_matrix1 = np_diff1[:45 * 45].reshape((45, 45))

#print(embedding1)
#print(np.count_nonzero(np_array1))
#print(np.count_nonzero(np_array2))
#print(diff.sum(dim = 1)) 
plt.imshow(image_matrix1, cmap='viridis')  # You can choose a different colormap based on your preference
plt.colorbar()  # Add a colorbar to show the intensity values
plt.title('Latent Vector Visualization')
plt.show()
plt.savefig('figure_1.png')
"""
with open ("file.txt", "w") as f: 
    for i in range(np_diff.shape[1]):
        content1 = str(np_diff[0,i])+'\n'
  #      content1 = str(np_array1[:])+'\n'
 #       content2 = str(np_array2[:])
        f.write(content1)
  #      f.write(content2)

 
#print(diff)  
#print(embedding.shape) # [1, 2048]
"""
#np_array3 = c.detach().cpu().data.numpy()
#np_array4 = d.detach().cpu().data.numpy()
  
np_diff2 = np.abs(np_array3 - np_array4)
np_diff2 = np.log(np_diff2)
np_diff2 = np_diff2[:, :2025]
print(np_diff2.shape)

#diff = torch.log(diff)
#np_diff = diff.detach().cpu().data.numpy
#print("np_diff", shape_diff)
#image_matrix2 = np_diff2[:45 * 45].reshape((45, 45))

#print(embedding1)
#print(np.count_nonzero(np_array1))
#print(np.count_nonzero(np_array2))
#print(diff.sum(dim = 1)) 
#plt.imshow(image_matrix2, cmap='viridis')  # You can choose a different colormap based on your preference
#plt.colorbar()  # Add a colorbar to show the intensity values
#plt.title('Latent Vector Visualization')
#plt.show()
#plt.savefig('figure_2.png')

