import os
import pickle

import numpy as np
from PIL import Image


class DummyDataset:

    def __init__(self, root_image, root_label, transform=None, mode='train', limit_files=None, split={'train': 0.6, 'val': 0.2, 'test': 0.2}, **kwargs):

        assert mode in ["train", "val", "test"], "wrong mode for dataset given"
        split_values = [v for k, v in split.items()]
        assert sum(split_values) == 1.0
        self.root_path_image = root_image
        self.root_path_label = root_label
        self.split = split
        self.limit_files = limit_files
        self.images, self.labels = self.make_dataset(
            directory_image=self.root_path_image,
            directory_labels=self.root_path_label,
            mode=mode,
        )
        self.transform = transform


    def select_split(self, images, labels, mode):
        """
        Depending on the mode of the dataset, deterministically split it.

        :param images, a list containing paths to all images in the dataset
        :param labels, a list containing one label per image

        :returns (images, labels), where only the indices for the
            corresponding data split are selected.
        """
        print(f"Mode: {mode}")
        print(f"Total number of images: {len(images)}")
        print(f"Total number of labels: {len(labels)}")

        fraction_train = self.split['train']
        fraction_val = self.split['val']
        num_samples = len(images)
        num_train = int(num_samples * fraction_train)
        num_valid = int(num_samples * fraction_val)

        # np.random.seed(0)
        # rand_perm = np.random.permutation(num_samples)
        # TODO: Using list() should already ensure that there is no overlap between datasets
        rand_perm = list(range(num_samples))

        if mode == 'train':
            idx = rand_perm[:num_train]
            # TODO: uncomment to print idx
            # print(f"train_idx:{idx}")
        elif mode == 'val':
            idx = rand_perm[num_train:num_train + num_valid]
            # TODO: uncomment to print idx
            # print(f"valis_idx:{idx}")
        elif mode == 'test':
            idx = rand_perm[num_train + num_valid:]
            # TODO: uncomment to print idx
            # print(f"test_idx:{idx}")

        if self.limit_files:
            idx = idx[:self.limit_files]

        # print(f"Selected indices for '{mode}': {idx}")
        # print(f"Number of selected indices for '{mode}': {len(idx)}")


        if isinstance(images, list):
            # print(f"Sample images for '{mode}':")
            # for i in idx[:5]:
            #     print(images[i])
            selected_images = [images[i] for i in idx]
            return selected_images, list(np.array(labels)[idx])
        else:
            # print(f"Sample images for '{mode}':")
            # for i in idx[:5]:
            #     print(images[i])

            return images[idx], list(np.array(labels)[idx])

    def make_dataset(self, directory_image, directory_labels, mode):

        images, labels = [], []
        fnames = [x.split('.')[0] for x in os.listdir(directory_image)]

        for file in (fnames):
            labels.append(self.load_labels(directory_labels, file))
            images.append(self.load_image_as_numpy(directory_image, file))



        images, labels = self.select_split(images, labels, mode)

        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        length = None
        length = len(self.images)
        return length

    @staticmethod
    def load_image_as_numpy(image_path, file):
        """Load image from image_path as numpy array"""
        img = np.array(Image.open(image_path + '/%s.jpg'%file), dtype=float)
        return Image.fromarray(img.astype(np.uint8))

    @staticmethod
    def load_labels(label_path, file):
        """Load label from label_path as list"""
        read_list = []
        with open(label_path +'/%s.txt'%file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                read_list.append(float(line.strip()))
        return read_list

    def __getitem__(self, index):
        data_dict = None

        label = self.labels[index]
        image = self.images[index]
        # image = self.load_image_as_numpy(path_image)
        # label = self.load_labels(path_label)
        if self.transform is not None:
            image = self.transform(image)
        data_dict = {
            "image": image,
            "label": label,
        }

        return data_dict

class TestDataset:

    def __init__(self, root_image, root_label, transform=None, limit_files=None):

        self.root_path_image = root_image
        self.root_path_label = root_label
        self.limit_files = limit_files
        self.images, self.labels = self.make_dataset(
            directory_image=self.root_path_image,
            directory_labels=self.root_path_label,
        )
        self.transform = transform


    def make_dataset(self, directory_image, directory_labels):

        images, labels = [], []
        fnames = [x.split('.')[0] for x in os.listdir(directory_image)]

        for file in (fnames):
            labels.append(self.load_labels(directory_labels, file))
            images.append(self.load_image_as_numpy(directory_image, file))


        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        length = None
        length = len(self.images)
        return length

    @staticmethod
    def load_image_as_numpy(image_path, file):
        """Load image from image_path as numpy array"""
        img = np.array(Image.open(image_path + '/%s.jpg'%file), dtype=float)
        return Image.fromarray(img.astype(np.uint8))

    @staticmethod
    def load_labels(label_path, file):
        """Load label from label_path as list"""
        read_list = []
        with open(label_path +'/%s.txt'%file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                read_list.append(float(line.strip()))
        return np.array(read_list)

    def __getitem__(self, index):
        data_dict = None

        label = self.labels[index]
        image = self.images[index]
        # image = self.load_image_as_numpy(path_image)
        # label = self.load_labels(path_label)
        if self.transform is not None:
            image = self.transform(image)
        data_dict = {
            "image": image,
            "label": label,
        }

        return data_dict