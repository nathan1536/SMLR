import omegaconf
import hydra
import torch
import os
import copy
import gdown
from os.path import expanduser

import torchvision.models as models
from torch.nn.modules.linear import Identity

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


if __name__ == "__main__":
    """
        This part is used to show the r3m model structure. The difference between r3m model and the resnet50 is that,
        r3m doesn't have a fc layer.
        Modify the iterating part to freeze specified layer
    """
    VALID_ARGS = ["_target_", "device", "lr", "hidden_dim", "size", "l2weight", "l1weight", "langweight", "tcnweight", "l2dist", "bs"]
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    r3m = load_r3m("resnet50")
    # model = models.resnet50(pretrained=False)
    # model.fc = Identity()
    # model.load_state_dict(r3m.state_dict())

    frozen_params = []
    trainable_params = []

    # TODO: Print layer parameter or freeze specified layers. Same as R3M_FT in model.py
    for name, param in r3m.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter shape: {param.shape}")

        if 'layer4' not in name:  # Freeze if not part of the FC layer
            param.requires_grad = False
            frozen_params.append(name)
        else:
            trainable_params.append(name)

    print("Frozen Parameters:")
    print(frozen_params)
    print("\nTrainable Parameters (Layer):")
    print(trainable_params)