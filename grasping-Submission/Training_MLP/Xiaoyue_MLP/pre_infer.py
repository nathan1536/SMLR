import torch
from dataset import TestDataset
import random
from model import MLP
from model import R3M_FT
from model import load_r3m
import numpy as np
from PIL import Image
import torchvision.transforms as T

transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.7638367389230166, 0.7876362596562249, 0.7742304502748976], std=[0.2136013180208868, 0.2112356498668464, 0.210953909872434])
    # T.Normalize(mean=[0.9130278582751299, 0.6987045922599552, 0.7275872380701353], std=[0.1856895204305658, 0.24154157828503725, 0.23048761183159028])
])
class Inference:
    """
        This Inference class is specified for the inferring non-modified r3m + MLP
        call infer() to print the labels and the corresponding predictions
    """
    def __init__(self, ckpt):
        self.model = MLP(in_channels=2048, out_channels=9)
        self.r3m = load_r3m("resnet50")
        self.device = torch.device('cuda:0')
        self.model.load_state_dict(torch.load(ckpt, map_location='cuda:0'))
        self.r3m.to(self.device)
        self.model.to(self.device)
        self.r3m.eval()
        self.model.eval()

    def infer(self, path_img, path_lb):

        image = np.array(Image.open(path_img), dtype=float)
        image = Image.fromarray(image.astype(np.uint8))
        image = transforms(image)
        image = image.reshape(-1, 3, 224, 224)
        embedding = self.r3m(image * 255)
        prediction = self.model(embedding)
        read_list = []
        with open(path_lb, 'r') as file:
            lines = file.readlines()
            for line in lines:
                read_list.append(float(line.strip()))

        # print(f"Embedding: {embedding}")
        print(f"Prediction: {prediction}")
        print(f"Label: {read_list}")

        return None

    def infer_(self, dataset, number_samples=5):
        samples_list = [{"image": sample["image"], "label": sample["label"]} for sample in dataset]
        select_samples = random.sample(samples_list, number_samples)
        for sample in select_samples:
            image, label = sample['image'], sample['label']
            image = image.reshape(-1, 3, 224, 224)
            embedding = self.r3m(image * 255)
            prediction = self.model(embedding)

            print(f"Label: {label}")
            # print(f"Embedding: {embedding}")
            print(f"Prediction: {prediction}")

        return None

class Inference_FT:
    """
        This Inference class is specified for the inferring fine-tuned r3m + MLP
        call infer() to print the labels and the corresponding predictions
    """
    def __init__(self, ckpt_r3m, ckpt_mlp):
        self.mlp = MLP(in_channels=2048, out_channels=9)
        self.r3m = load_r3m("resnet50")
        self.r3m = R3M_FT(self.r3m)
        self.device = torch.device('cuda:0')
        self.r3m.load_state_dict(torch.load(ckpt_r3m, map_location='cuda:0'))
        self.mlp.load_state_dict(torch.load(ckpt_mlp, map_location='cuda:0'))
        self.r3m.to(self.device)
        self.mlp.to(self.device)
        self.r3m.eval()
        self.mlp.eval()



    def infer(self, path_img, path_lb):

        image = np.array(Image.open(path_img), dtype=float)
        image = Image.fromarray(image.astype(np.uint8))
        image = transforms(image)
        image = image.reshape(-1, 3, 224, 224)
        embedding = self.r3m(image * 255)
        prediction = self.mlp(embedding)
        read_list = []
        with open(path_lb, 'r') as file:
            lines = file.readlines()
            for line in lines:
                read_list.append(float(line.strip()))

        # print(f"Embedding: {embedding}")
        print(f"Prediction: {prediction}")
        print(f"Label: {read_list}")

        return None

    def infer_(self, dataset, number_samples=5):
        samples_list = [{"image": sample["image"], "label": sample["label"]} for sample in dataset]
        select_samples = random.sample(samples_list, number_samples)
        for sample in select_samples:
            image, label = sample['image'], sample['label']
            image = image.reshape(-1, 3, 224, 224)
            embedding = self.r3m(image * 255)
            prediction = self.mlp(embedding)

            print(f"Label: {label}")
            # print(f"Embedding: {embedding}")
            print(f"Prediction: {prediction}")

        return None

# if __name__ == "__main__":
#     # TODO: specify your image_path and the corresponding label_path here. This pre_infer.py file only takes one pair of data as input.
#     path_img = 'test_set/images/5test_40.jpg'
#     path_lb = 'test_set/labels/5test_40.txt'
#
#     # TODO: uncomment to infer non-modified r3m + mlp
#     # infer = Inference(ckpt='r3m_MLP_ckpt/model_best_B64.ckpt')
#     # infer.infer(path_img, path_lb)
#
#     # TODO: uncomment to infer fine-tuned r3m + mlp
#     infer = Inference_FT(ckpt_r3m='r3m_MLP_ckpt/model_best_r3m4.ckpt', ckpt_mlp='r3m_MLP_ckpt/model_best_mlp4.ckpt')
#     infer.infer(path_img, path_lb)

if __name__ == "__main__":
    # TODO: specify your image_path and the corresponding label_path here. This pre_infer.py file only takes one pair of data as input.
    config = {
        'image_path': 'test_set/images',
        'label_path': 'test_set/labels',
    }

    testset = TestDataset(root_image=config['image_path'], root_label=config['label_path'], transform=transforms)

    # TODO: uncomment to infer non-modified r3m + mlp
    # infer = Inference(ckpt='r3m_MLP_ckpt/model_best_L1.ckpt')
    # infer.infer_(dataset=testset, number_samples=50)

    # TODO: uncomment to infer fine-tuned r3m + mlp
    infer = Inference_FT(ckpt_r3m='r3m_MLP_ckpt/model_best_r3m4.ckpt', ckpt_mlp='r3m_MLP_ckpt/model_best_mlp4.ckpt')
    infer.infer_(dataset=testset, number_samples=50)
