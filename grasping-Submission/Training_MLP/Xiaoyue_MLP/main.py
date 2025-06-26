import trainer
from dataset import DummyDataset

from torch.utils.data import DataLoader
from infer import Inference
from infer import Inference_FT
import torchvision.transforms as T

transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.9130278582751299, 0.6987045922599552, 0.7275872380701353], std=[0.1856895204305658, 0.24154157828503725, 0.23048761183159028])
])

config = {
    'device': 'cuda:0',                    # change this to cpu if you do not have a GPU
    'mode': 'val_ft',                       # change mode 'train' or 'val' or 'finetune' or 'val_ft' or 'test_idx'
    'batch_size': 64,
    'image_path': 'dummy_dataset/images',
    'label_path': 'dummy_dataset/labels',
    'image_path_train': 'train_set/images',
    'label_path_train': 'train_set/labels',
    'image_path_val': 'val_set/images',
    'label_path_val': 'val_set/labels',
    'ckpt': 'r3m_MLP_ckpt/model_best_Dropout.ckpt', # change this parameter if set mode to 'train'. To specified the name of saving ckpt
    'resume_ckpt': None,
    'learning_rate': 0.001,
    'max_epochs': 25,
    'print_every_n': 20,
    'validate_every_n': 25,
    'tensorboard_dir': 'Finetune4',  # specify a dir name for saving the tensorboard logs
}

if config['mode'] == 'train':
    trainer.main(config)

elif config['mode'] == 'finetune':
    trainer.main_ft(config)

elif config['mode'] == 'val':
    testset = DummyDataset(root_image=config['image_path'], root_label=config['label_path'], transform=transforms, mode='test')
    infer = Inference(ckpt=config['ckpt'])
    infer.infer(dataset=testset, number_samples=50)

elif config['mode'] == 'val_ft':
    testset = DummyDataset(root_image=config['image_path'], root_label=config['label_path'], transform=transforms, mode='test')
    infer = Inference_FT(ckpt_r3m='r3m_MLP_ckpt/model_best_r3m4.ckpt', ckpt_mlp='r3m_MLP_ckpt/model_best_mlp4.ckpt')
    infer.infer(dataset=testset, number_samples=50)

elif config['mode'] == 'test_idx':
    # TODO: Only used to print the idx of each set. If printing, uncomment the print() function in dataset.py
    trainset = DummyDataset(root_image=config['image_path'], root_label=config['label_path'], transform=transforms, mode='train')
    valset = DummyDataset(root_image=config['image_path'], root_label=config['label_path'], transform=transforms, mode='val')
    testset = DummyDataset(root_image=config['image_path'], root_label=config['label_path'], transform=transforms, mode='test')

else:
    print("you have selected an invalid mode")

