import torch
from dataset import DummyDataset
import random
from model import MLP
from model import R3M_FT
from model import load_r3m
import numpy as np
class Inference:
    """
        This Inference class is specified for the inferring non-modified r3m + MLP
        call infer() to print the labels and the corresponding predictions
    """
    def __init__(self, ckpt):
        self.model = MLP(in_channels=2048, out_channels=9)
        self.r3m = load_r3m("resnet50")

        # TODO: I'm using a GPU, so the device here is always 'cuda'. idk if setting device to 'cpu' also works.
        self.device = torch.device('cuda:0')
        self.model.load_state_dict(torch.load(ckpt, map_location='cuda:0'))

        self.r3m.to(self.device)
        self.model.to(self.device)
        self.r3m.eval()
        self.model.eval()

    def infer(self, dataset, number_samples=5):
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

        # TODO: I'm using a GPU, so the device here is always 'cuda'. idk if setting device to 'cpu' also works.
        self.device = torch.device('cuda:0')
        self.r3m.load_state_dict(torch.load(ckpt_r3m, map_location='cuda:0'))
        self.mlp.load_state_dict(torch.load(ckpt_mlp, map_location='cuda:0'))

        self.r3m.to(self.device)
        self.mlp.to(self.device)
        self.r3m.eval()
        self.mlp.eval()

    def infer(self, dataset, number_samples=5):
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