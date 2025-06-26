# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from r3m.models.models_r3m import R3M

import os 
from os.path import expanduser
import omegaconf
import hydra
import torch
import copy

VALID_ARGS = ["_target_", "device", "lr", "hidden_dim", "size", "l2weight", "l1weight", "langweight", "tcnweight", "l2dist", "bs"]
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

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

def remove_language_head(state_dict):
    keys = state_dict.keys()
    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    for key in list(keys):
        if ("lang_enc" in key) or ("lang_rew" in key):
            del state_dict[key]
    return state_dict

def load_r3m(modelid):
    home = os.path.join(expanduser("~"), ".r3m")
    # weights and config are predownloaded in weights folder and just loaded here
    try :
         
        if modelid == "resnet50":
            modelpath = '../repos/r3m/r3m/model_weights/r3m_50.pt'
            configpath = '../repos/r3m/r3m/model_configs/r3m50_config.yaml'
            
        elif modelid == "resnet34":
            modelpath ='model_weights/r3m_34.pt'
            configpath = 'model_configs/r3m34_config.yaml'
            
        elif modelid == "resnet18":
            modelpath = 'model_weights/r3m_18.pt'
            configpath = 'model_configs/r3m18_config.yaml'
    
    except NameError:
        print('Invalid Model ID, valid IDs are resnet50 resnet34 resnet18 ')
        
    
    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep)
    r3m_state_dict = remove_language_head(torch.load(modelpath, map_location=torch.device(device))['r3m'])
    rep.load_state_dict(r3m_state_dict)
    return rep


