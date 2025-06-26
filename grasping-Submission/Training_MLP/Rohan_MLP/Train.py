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

    # hyper parameters
    epochs = 50
    batch_size = 32
    device = 'cuda'

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 12}
    
    acc_folder = 'mlp_acc/'
    writer_2 = SummaryWriter(log_dir=acc_folder)
    
    # Note train data folder should have folders images and labels
    train_path = os.path.abspath(os.path.dirname(__file__)) + '/train_random_box_data' # add path to train data here
    train_dataset = Dataset(train_path, mode = 'train')
    train_dataloader = data.DataLoader(train_dataset, **params)
    
    
    # Note val data folder should have folders images and labels
    val_path = os.path.abspath(os.path.dirname(__file__)) + '/val_random_box_data' # add path to val data here
    val_dataset = Dataset(val_path, mode = 'val')
    val_dataloader = data.DataLoader(val_dataset, **params)

 
    network50 = torchvision.models.vgg16(pretrained=False)
    features = network50.features

    for param in features.parameters():
        param.requires_grad = True

    model = MLP(in_channels=128, out_channels=9, features = features).cuda()
    

    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
    scheduler = lr_scheduler.StepLR(opt_SGD, step_size=2, gamma=0.1)
    Loss = nn.L1Loss().cuda()
    runs_folder = 'runs/'
    writer = SummaryWriter(log_dir=runs_folder)
 
    first_epoch = 0
    
    total_num_train_batches = int(len(train_dataset) / batch_size)
    
    best_val_loss = np.inf

    for epoch in range(first_epoch+1, epochs+1):
        curr_batch = 0
        passes = 0
        running_loss = 0.0
        running_acc = torch.zeros(9,1)

        for local_batch, local_labels in train_dataloader:

            labels = local_labels['labels'].cuda()
            labels = torch.squeeze(labels, dim = 1)
            
            local_batch = local_batch.cuda()

            # clear gradients before forward pass
            opt_SGD.zero_grad()
            
            model.train()
            output = model(local_batch)
            output = output.to(torch.float)
            loss = Loss(output, labels)
            acc = get_accuracy(labels, output)
    
            loss.backward()
            opt_SGD.step()
            
            running_loss += loss.item()
            running_acc += acc

            passes += 1
            curr_batch += 1
        
        epoch_acc =  running_acc / int(len(train_dataset))
        epoch_acc = torch.squeeze(epoch_acc, dim = 1)
        epoch_acc = epoch_acc.detach().cpu().numpy()
        
        scalar_dict = {
                'j1': epoch_acc[0],
                'j2': epoch_acc[1],
                'j3': epoch_acc[2],
                'j4': epoch_acc[3],
                'j5': epoch_acc[4],
                'j6': epoch_acc[5],
                'j7': epoch_acc[6],
                'dx': epoch_acc[7],
                'dz': epoch_acc[8]
            }
        for name, value in scalar_dict.items():
            writer_2.add_scalar(name, value, epoch)
        
        print(f"Epoch Acuracy [{epoch}/{epochs}], Joint 1 : {epoch_acc[0]:.4f}")
        print(f"Epoch Acuracy [{epoch}/{epochs}], Joint 2 : {epoch_acc[1]:.4f}") 
        print(f"Epoch Acuracy [{epoch}/{epochs}], Joint 3 : {epoch_acc[2]:.4f}") 
        print(f"Epoch Acuracy [{epoch}/{epochs}], Joint 4 : {epoch_acc[3]:.4f}") 
        print(f"Epoch Acuracy [{epoch}/{epochs}], Joint 5 : {epoch_acc[4]:.4f}") 
        print(f"Epoch Acuracy [{epoch}/{epochs}], Joint 6 : {epoch_acc[5]:.4f}") 
        print(f"Epoch Acuracy [{epoch}/{epochs}], Joint 7 : {epoch_acc[6]:.4f}") 
        print(f"Epoch Acuracy [{epoch}/{epochs}], Delta X : {epoch_acc[7]:.4f}") 
        print(f"Epoch Acuracy [{epoch}/{epochs}], Delta Z : {epoch_acc[8]:.4f}")   

        epoch_loss = running_loss / int(len(train_dataset))
        print(f"Epoch [{epoch}/{epochs}], Epoch Loss: {epoch_loss:.4f}")
        # Update learning rate scheduler
        scheduler.step()
        # Log loss to tensorboard
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        
        model.eval()

        # forward pass and evaluation for entire validation set
        running_val_loss = 0.0
        with torch.no_grad():  # Disable gradient tracking during validation
            for val_local_batch, val_local_labels in val_dataloader:
                val_labels = val_local_labels['labels'].cuda()
                val_labels = labels = torch.squeeze(val_labels, dim = 1)
                val_local_batch = val_local_batch.cuda()

                # Forward pass
                val_output = model(val_local_batch)
                val_loss = Loss(val_output, val_labels)
                running_val_loss += val_loss.item()

        # Calculate average validation loss
        epoch_val_loss = running_val_loss / int(len(val_dataset))
        print(f"Epoch [{epoch}/{epochs}], Epoch Validation Loss: {epoch_val_loss:.4f}")
        writer.add_scalar('Validation Loss', epoch_val_loss, epoch)
        
        
        print("====================")
        print ("Done with epoch %s!" % epoch)

        # save model weights 
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            if epoch % 5 == 0:
                name = 'model_weights/' + 'epoch_%s.pkl' % epoch
                
                print ("Saving weights as %s ..." % name)
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt_SGD.state_dict(),
                        'loss': loss
                        }, name)
                
        print("====================")
        
        # set model back to train
        model.train()
      

if __name__=='__main__':
    main()
