from pathlib import Path

import numpy as np
import torch
from dataset import DummyDataset
from dataset import TestDataset
from torch.utils.tensorboard import SummaryWriter
# from dataloader import DataLoader
from torch.utils.data import DataLoader
from model import load_r3m
from model import MLP
from model import R3M_FT
from model import MODEL
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

## DEFINE PREPROCESSING
# transform = T.Compose([T.Resize(256),
#     T.CenterCrop(224),
#     T.ToTensor()])
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.7638367389230166, 0.7876362596562249, 0.7742304502748976], std=[0.2136013180208868, 0.2112356498668464, 0.210953909872434])
])
# transform = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor()
# ])
def main(config):
    # declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # create dataloaders
    trainset = DummyDataset(root_image=config['image_path'], root_label=config['label_path'], transform=transform, limit_files=None, mode='train')
    trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)

    valset = DummyDataset(root_image=config['image_path'], root_label=config['label_path'], transform=transform, limit_files=None, mode='val')
    valloader = DataLoader(valset, batch_size=config['batch_size'], shuffle=False)

    # instantiate model
    r3m = load_r3m("resnet50")
    model = MLP(in_channels=2048, out_channels=9)

    # load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # move model to specified device
    model.to(device)

    # create folder for saving checkpoints
    Path('r3m_MLP_ckpt').mkdir(exist_ok=True, parents=True)

    # start training
    train(r3m, model, trainloader, valloader, device, config)

def main_ft(config):
    # declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # create dataloaders
    trainset = TestDataset(root_image=config['image_path_train'], root_label=config['label_path_train'], transform=transform, limit_files=None)
    trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)

    valset = TestDataset(root_image=config['image_path_val'], root_label=config['label_path_val'], transform=transform, limit_files=None)
    valloader = DataLoader(valset, batch_size=config['batch_size'], shuffle=False)

    # instantiate model
    r3m = load_r3m("resnet50")
    r3m_ft = R3M_FT(r3m)
    mlp = MLP(in_channels=2048, out_channels=9)
    model = MODEL(r3m_ft, mlp)


    # load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # move model to specified device
    model.to(device)

    # create folder for saving checkpoints
    Path('r3m_MLP_ckpt').mkdir(exist_ok=True, parents=True)

    # start training
    train_ft(model, trainloader, valloader, device, config)

def train(r3m, model, trainloader, valloader, device, config):

    tb_writer = SummaryWriter(log_dir=f"runs/{config['tensorboard_dir']}")

    # TODO: declare loss and move to specified device
    # loss_criterion = torch.nn.MSELoss()
    loss_criterion = torch.nn.L1Loss()
    # loss_criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    loss_criterion.to(device)

    # TODO: declare optimizer (learning rate provided in config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # set model to train, important if your network has e.g. dropout or batchnorm layers
    model.train()

    # keep track of best validation accuracy achieved so that we can save the weights
    best_loss = None

    # keep track of running average of train loss for printing
    train_loss_running = 0.
    epoch_loss = 0.
    epoch_val_loss = 0.

    for epoch in range(config['max_epochs']):
        for i, batch in enumerate(trainloader):
            # move batch to device
            # print(f"batch_shape:{np.array(batch['label']).shape}")
            batch['image'] = batch['image'].to(device)
            print(batch['image'].shape)
            batch['label'] = batch['label'].to(torch.float)
            batch['label'] = batch['label'].to(device)


            # zero out previously accumulated gradients
            optimizer.zero_grad()

            # forward pass
            r3m.eval()
            with torch.no_grad():
                embedding = r3m(batch['image'] * 255.0)
            prediction = model(embedding)
            prediction = prediction.to(torch.float)
            # print(f"prediction_size:{prediction.shape}")

            # compute total loss = sum of loss for whole prediction + losses for partial predictions
            loss_total = loss_criterion(prediction, batch['label'])

            # compute gradients on loss_total (backward pass)
            loss_total.backward()

            # update network params
            optimizer.step()

            # loss logging
            train_loss_running += loss_total.item()
            iteration = epoch * len(trainloader) + i
            train_losses.append(train_loss_running / config['print_every_n'])

            # if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
            #     train_losses.append(train_loss_running / config['print_every_n'])
            #     print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / config["print_every_n"]:.3f}')
            #     tb_writer.add_scalar('Train Loss', train_loss_running / config['print_every_n'], iteration)
            #     train_loss_running = 0.

            # validation evaluation and logging
            # if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
            #
            #     # set model to eval, important if your network has e.g. dropout or batchnorm layers
            #     model.eval()
            #
            #     loss_total_val = 0
            #     # forward pass and evaluation for entire validation set
            #     for batch_val in valloader:
            #         batch_val['image'] = batch_val['image'].to(device)
            #         batch_val['label'] = batch_val['label'].to(torch.float)
            #         batch_val['label'] = batch_val['label'].to(device)
            #
            #         r3m = load_r3m("resnet50")  # resnet18, resnet34
            #         r3m.eval()
            #         with torch.no_grad():
            #             embedding = r3m(batch_val['image'] * 255.0)
            #         with torch.no_grad():
            #             prediction = model(embedding)
            #         loss = loss_criterion(prediction, batch_val['label'])
            #         loss_total_val += loss.item()
            #     val_losses.append(loss_total_val / len(valloader))
            #     tb_writer.add_scalar('Validation Loss', loss_total_val / len(valloader), iteration)
            #
            #     print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_total_val / len(valloader):.3f}')
            #
            #     if not best_loss or (loss_total_val / len(valloader)) < best_loss:
            #         torch.save(model.state_dict(), f'r3m_MLP_ckpt/model_best.ckpt')
            #         best_loss = loss_total_val / len(valloader)
            #
            #     # set model back to train
            #     model.train()
        print(f"epoch={epoch}")
        epoch_loss = train_loss_running / len(trainloader)
        tb_writer.add_scalar('Epoch Loss', epoch_loss, epoch + 1)
        train_loss_running = 0.
        epoch_loss = 0
        # set model to eval, important if your network has e.g. dropout or batchnorm layers
        model.eval()

        loss_total_val = 0
        # forward pass and evaluation for entire validation set
        for batch_val in valloader:
            batch_val['image'] = batch_val['image'].to(device)
            batch_val['label'] = batch_val['label'].to(torch.float)
            batch_val['label'] = batch_val['label'].to(device)

            r3m.eval()
            with torch.no_grad():
                embedding = r3m(batch_val['image'] * 255.0)
            with torch.no_grad():
                prediction = model(embedding)
            loss = loss_criterion(prediction, batch_val['label'])
            loss_total_val += loss.item()
            epoch_val_loss = loss_total_val / len(valloader)
        val_losses.append(loss_total_val / len(valloader))
        tb_writer.add_scalar('Validation Loss', epoch_val_loss, epoch + 1)

        if not best_loss or (loss_total_val / len(valloader)) < best_loss:
            torch.save(model.state_dict(), f"{config['ckpt']}")
            best_loss = loss_total_val / len(valloader)

        # set model back to train
        model.train()

    # plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    # plt.plot(range(0, len(val_losses) * config['validate_every_n'], config['validate_every_n']), val_losses,
    #          label='Validation Loss')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    # plt.savefig('runs/figure_1.png')


def train_ft(model, trainloader, valloader, device, config):
    tb_writer = SummaryWriter(log_dir=f"runs/{config['tensorboard_dir']}")

    # TODO: declare loss and move to specified device
    # loss_criterion = torch.nn.MSELoss()
    loss_criterion = torch.nn.L1Loss()

    train_losses = []
    val_losses = []
    loss_criterion.to(device)

    # TODO: declare optimizer (learning rate provided in config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # set model to train, important if your network has e.g. dropout or batchnorm layers
    model.train()

    # keep track of best validation accuracy achieved so that we can save the weights
    best_loss = None

    # keep track of running average of train loss for printing
    train_loss_running = 0.
    epoch_loss = 0.
    epoch_val_loss = 0.

    for epoch in range(config['max_epochs']):
        for i, batch in enumerate(trainloader):
            # move batch to device
            # print(f"batch_shape:{np.array(batch['label']).shape}")
            batch['image'] = batch['image'].to(device)
            batch['label'] = batch['label'].to(torch.float)
            batch['label'] = batch['label'].to(device)


            # zero out previously accumulated gradients
            optimizer.zero_grad()

            # forward pass
            prediction = model(batch['image'] * 255.0)
            prediction = prediction.to(torch.float)
            # print(f"prediction_size:{prediction.shape}")

            # compute total loss = sum of loss for whole prediction + losses for partial predictions
            loss_total = loss_criterion(prediction, batch['label'])

            # compute gradients on loss_total (backward pass)
            loss_total.backward()

            # update network params
            optimizer.step()

            # loss logging
            train_loss_running += loss_total.item()
            iteration = epoch * len(trainloader) + i
            # train_losses.append(train_loss_running / config['print_every_n'])


        print(f"epoch={epoch}")
        epoch_loss = train_loss_running / len(trainloader)
        tb_writer.add_scalar('Epoch Loss', epoch_loss, epoch + 1)
        train_loss_running = 0.
        epoch_loss = 0
        # set model to eval, important if your network has e.g. dropout or batchnorm layers
        model.eval()

        loss_total_val = 0
        # forward pass and evaluation for entire validation set
        for batch_val in valloader:
            batch_val['image'] = batch_val['image'].to(device)
            batch_val['label'] = batch_val['label'].to(torch.float)
            batch_val['label'] = batch_val['label'].to(device)

            with torch.no_grad():
                prediction = model(batch_val['image'] * 255.0)
                prediction = prediction.to(torch.float)
            loss = loss_criterion(prediction, batch_val['label'])
            loss_total_val += loss.item()
            epoch_val_loss = loss_total_val / len(valloader)
        # val_losses.append(loss_total_val / len(valloader))
        tb_writer.add_scalar('Validation Loss', epoch_val_loss, epoch + 1)

        if not best_loss or (loss_total_val / len(valloader)) < best_loss:
            # saving the state_dict() separately
            torch.save(model.r3m.state_dict(), f'r3m_MLP_ckpt/model_best_r3m4.ckpt')
            torch.save(model.mlp.state_dict(), f'r3m_MLP_ckpt/model_best_mlp4.ckpt')
            best_loss = loss_total_val / len(valloader)

        # set model back to train
        model.train()