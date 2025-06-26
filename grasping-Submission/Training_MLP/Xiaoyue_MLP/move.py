import os
import random
import shutil

source_dir = 'test_data'
train_dir = 'train_set'
val_dir = 'val_set'

image_path = 'images'
label_path = 'labels'

train_image = os.path.join(train_dir, image_path)
train_label = os.path.join(train_dir, label_path)
val_image = os.path.join(val_dir, image_path)
val_label = os.path.join(val_dir, label_path)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(train_image, exist_ok=True)
os.makedirs(train_label, exist_ok=True)

os.makedirs(val_dir, exist_ok=True)
os.makedirs(val_image, exist_ok=True)
os.makedirs(val_label, exist_ok=True)

image_files = os.listdir(os.path.join(source_dir, 'images'))
label_files = os.listdir(os.path.join(source_dir, 'labels'))

random.shuffle(image_files)


train_size = int(0.8 * len(image_files))
val_size = len(image_files) - train_size

for i in range(train_size):
    image_file = image_files[i]
    label_file = image_file.replace('.jpg', '.txt')  # 假设标签文件和图片文件同名，只是扩展名不同
    shutil.copy(os.path.join(source_dir, 'images', image_file), os.path.join(train_dir, 'images', image_file))
    shutil.copy(os.path.join(source_dir, 'labels', label_file), os.path.join(train_dir, 'labels', label_file))

for i in range(train_size, train_size + val_size):
    image_file = image_files[i]
    label_file = image_file.replace('.jpg', '.txt')
    shutil.copy(os.path.join(source_dir, 'images', image_file), os.path.join(val_dir, 'images', image_file))
    shutil.copy(os.path.join(source_dir, 'labels', label_file), os.path.join(val_dir, 'labels', label_file))
