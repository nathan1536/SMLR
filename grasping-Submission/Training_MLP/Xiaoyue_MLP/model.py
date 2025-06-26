import omegaconf
import hydra
from torch import nn
import torchvision.models as models
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import os
import copy
import gdown
from os.path import expanduser

def remove_language_head(state_dict):
    keys = state_dict.keys()
    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    for key in list(keys):
        if ("lang_enc" in key) or ("lang_rew" in key):
            del state_dict[key]
    return state_dict


def cleanup_config(cfg):
    config = copy.deepcopy(cfg)
    keys = config.agent.keys()
    for key in list(keys):
        if key not in VALID_ARGS:
            del config.agent[key]
    config.agent["_target_"] = "r3m.R3M"
    config["device"] = device

    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    config.agent["langweight"] = 0
    return config.agent

def load_r3m(modelid):
    home = os.path.join(expanduser("~"), ".r3m")
    if modelid == "resnet50":
        foldername = "r3m_50"
        modelurl = 'https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA'
        configurl = 'https://drive.google.com/uc?id=10jY2VxrrhfOdNPmsFdES568hjjIoBJx8'
    elif modelid == "resnet34":
        foldername = "r3m_34"
        modelurl = 'https://drive.google.com/uc?id=15bXD3QRhspIRacOKyWPw5y2HpoWUCEnE'
        configurl = 'https://drive.google.com/uc?id=1RY0NS-Tl4G7M1Ik_lOym0b5VIBxX9dqW'
    elif modelid == "resnet18":
        foldername = "r3m_18"
        modelurl = 'https://drive.google.com/uc?id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-'
        configurl = 'https://drive.google.com/uc?id=1nitbHQ-GRorxc7vMUiEHjHWP5N11Jvc6'
    else:
        raise NameError('Invalid Model ID')

    if not os.path.exists(os.path.join(home, foldername)):
        os.makedirs(os.path.join(home, foldername))
    modelpath = os.path.join(home, foldername, "model.pt")
    configpath = os.path.join(home, foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep)
    r3m_state_dict = remove_language_head(torch.load(modelpath, map_location=torch.device(device))['r3m'])
    rep.load_state_dict(r3m_state_dict)
    return rep

VALID_ARGS = ["_target_", "device", "lr", "hidden_dim", "size", "l2weight", "l1weight", "langweight", "tcnweight", "l2dist", "bs"]
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# r3m = load_r3m("resnet50") # resnet18, resnet34
# r3m.eval()
# r3m.to(device)
#
# ## DEFINE PREPROCESSING
# transforms = T.Compose([T.Resize(256),
#     T.CenterCrop(224),
#     T.ToTensor()]) # ToTensor() divides by 255
#
# ## ENCODE IMAGE
# image = np.random.randint(0, 255, (500, 500, 3))
# preprocessed_image = transforms(Image.fromarray(image.astype(np.uint8))).reshape(-1, 3, 224, 224)
# preprocessed_image.to(device)
# with torch.no_grad():
#   embedding = r3m(preprocessed_image * 255.0) ## R3M expects image input to be [0-255]
# print(embedding.shape) # [1, 2048]

"""Parts before all inherited from the original r3m file, should not make modifications."""

class MLP(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: number of input channels to the first linear layer
        :param out_channels: number of output channels for linear layers
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=512),
            # nn.ReLU(),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=512),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=512),
            # nn.ReLU(),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=out_channels)
        )

    def forward(self, x):

        return self.model(x)

class R3M_FT(nn.Module):
    def __init__(self, r3m):
        """:param r3m: the original r3m model(called with load_r3m())"""
        super().__init__()
        self.model = r3m
        for name, param in self.model.named_parameters():
            if 'layer4' not in name:  # Freeze if not part of the selected layer
                param.requires_grad = False

    def forward(self, x):

        return self.model(x)


class MODEL(nn.Module):  # combine the r3m and mlp as a whole model, may be useful
    def __init__(self, r3m, mlp):
        """
        :param r3m: r3m model
        :param mlp: mlp model
        """
        super().__init__()
        self.r3m = r3m
        self.mlp = mlp

    def forward(self, x):
        x = self.r3m(x)
        x = self.mlp(x)
        return x


class mlp(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: number of input channels to the first linear layer
        :param out_channels: number of output channels for linear layers
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=512),
            # nn.ReLU(),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=512),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=512),
            # nn.ReLU(),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=out_channels)
        )

    def forward(self, x):

        return self.model(x)

if __name__ == "__main__":
    r3m = load_r3m("resnet50")
    mlp = mlp(512, 9)
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.9130278582751299, 0.6987045922599552, 0.7275872380701353],
                    std=[0.1856895204305658, 0.24154157828503725, 0.23048761183159028])
    ])
    device = torch.device('cuda:0')
    # mlp.load_state_dict(torch.load('r3m_MLP_ckpt/model_best_L1.ckpt', map_location='cuda:0'))
    r3m.to(device)
    mlp.to(device)
    r3m.eval()
    mlp.eval()
    image = np.array(Image.open('test_set/images/5test_40.jpg'), dtype=float)



    image = Image.fromarray(image.astype(np.uint8))
    image = transforms(image)
    print(type(image))
    image = image.reshape(-1, 3, 224, 224)
    batch = torch.zeros(4,3,224,224)
    for i in range(4):
        batch[i, :, :, :] = image

    embedding = r3m(batch * 255)
    print(embedding.shape)
    # embedding = embedding.view(4, -1, 512)
    #print(embedding.shape)
    prediction = mlp(embedding)
    print(prediction.shape)

