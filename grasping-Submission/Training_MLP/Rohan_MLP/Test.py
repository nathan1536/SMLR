import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torch.utils import data
from Model import MLP
from torch.nn.modules.linear import Identity
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from Dataset import Dataset
import numpy as np

import os


def get_accuracy(ground_truth, mlp_output):

    mlp_output = mlp_output.detach().cpu().numpy()
    ground_truth = ground_truth.detach().cpu().numpy()
    err = (abs(ground_truth - mlp_output) / (abs(ground_truth) + 1e-10)) * 100

    mean_err = np.mean(err, axis=0)
    mean_err = np.reshape(mean_err,(9,1))
    all = np.array([[100], [100], [100], [100], [100], [100], [100], [100], [100]])

    acc_vector = all - mean_err
    acc_tensor = torch.from_numpy(acc_vector)
    acc_tensor = torch.reshape(acc_tensor, (9,1))
 
    return acc_tensor


def main():


    batch_size = 32
    device = 'cuda'

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 12}
    
    # Note test data folder should have folders images and labels
    test_path = os.path.abspath(os.path.dirname(__file__)) + '/test_random_box_data' # add path to test data here
    test_dataset = Dataset(test_path, mode = 'test')
    test_dataloader = data.DataLoader(test_dataset, **params)
    
    network50 = torchvision.models.vgg16(pretrained=False)
    features = network50.features

    model = MLP(in_channels=128, out_channels=9, features = features).cuda()
    model.load_state_dict(torch.load('model_weights/epoch_50.pkl')['model_state_dict'])

    Loss = nn.L1Loss().cuda()
    runs_folder = 'test_runs/'
    writer = SummaryWriter(log_dir=runs_folder)

    model.eval()
 
    with torch.no_grad():  # Disable gradient tracking during test
        for i, (test_local_batch, test_local_labels) in enumerate(test_dataloader):
            test_labels = test_local_labels['labels'].cuda()
            test_labels = torch.squeeze(test_labels, dim = 1)
            test_local_batch = test_local_batch.cuda()

            # Forward pass
            test_output = model(test_local_batch)
            test_loss = Loss(test_output, test_labels)

            batch_acc = get_accuracy(test_labels, test_output)

            batch_acc = torch.squeeze(batch_acc, dim = 1)
            batch_acc = batch_acc.detach().cpu().numpy()
        
            scalar_dict = {
                    'j1': batch_acc[0],
                    'j2': batch_acc[1],
                    'j3': batch_acc[2],
                    'j4': batch_acc[3],
                    'j5': batch_acc[4],
                    'j6': batch_acc[5],
                    'j7': batch_acc[6],
                    'dx': batch_acc[7],
                    'dz': batch_acc[8]
                }
            
            for name, value in scalar_dict.items():
                writer.add_scalar(name, value, i+1)
            
            print(f"Batch Acuracy [{i+1}/{int(len(test_dataset))}], Joint 1 : {batch_acc[0]:.4f}")
            print(f"Batch Acuracy [{i+1}/{int(len(test_dataset))}], Joint 2 : {batch_acc[1]:.4f}") 
            print(f"Batch Acuracy [{i+1}/{int(len(test_dataset))}], Joint 3 : {batch_acc[2]:.4f}") 
            print(f"Batch Acuracy [{i+1}/{int(len(test_dataset))}], Joint 4 : {batch_acc[3]:.4f}") 
            print(f"Batch Acuracy [{i+1}/{int(len(test_dataset))}], Joint 5 : {batch_acc[4]:.4f}") 
            print(f"Batch Acuracy [{i+1}/{int(len(test_dataset))}], Joint 6 : {batch_acc[5]:.4f}") 
            print(f"Batch Acuracy [{i+1}/{int(len(test_dataset))}], Joint 7 : {batch_acc[6]:.4f}") 
            print(f"Batch Acuracy [{i+1}/{int(len(test_dataset))}], Delta X : {batch_acc[7]:.4f}") 
            print(f"Batch Acuracy [{i+1}/{int(len(test_dataset))}], Delta Z : {batch_acc[8]:.4f}")   
            

            # Calculate test batch loss
            batch_test_loss = test_loss.item()
            print(f"Batch[{i+1}/{int(len(test_dataset))}, Batch Loss: {batch_test_loss:.4f}")
            writer.add_scalar('Test Batch Loss', batch_test_loss, i+1)
            print("====================")
            print ("Done with batch:", i+1)
            print("====================")

    

if __name__=='__main__':
    main()
