"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import Cityscapes
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from argparse import ArgumentParser
from helper import *
import collections
from model import Model
def get_arg_parser():
    parser = ArgumentParser()

    # Detect the running environment
    run_env = os.getenv('RUN_ENV', 'Computer-Vision')

    if run_env == 'Computer-Vision':
        default_data_path = "City_Scapes/"
        default_batch_size = 4
        default_num_epochs = 1
        default_resize = (64,64)

    else:
        default_data_path = "/gpfs/work5/0/jhstue005/JHS_data/CityScapes"
        default_batch_size = 64
        default_num_epochs = 20
        default_resize = (1024, 2048)

    parser.add_argument("--data_path", type=str, default=default_data_path, help="Path to the data")
    parser.add_argument("--batch_size", type=int, default=default_batch_size, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=default_num_epochs, help="Number of epochs for training")
    parser.add_argument("--resize", type=int, default=default_resize, help="Image format that is being worked with ")
    return parser

def main(args):
    """define your model, trainingsloop optimitzer etc. here"""
    transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.ToTensor(),
    ])

    # The Cityscapes dataset returns the target as PIL Image for 'semantic' target_type
    target_transform = transforms.Compose([
        transforms.Resize(args.resize, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])

    # Define the datasets
    training_data = Cityscapes(root=args.data_path, split='train', mode='fine', target_type='semantic',
                               transform=transform, target_transform=target_transform)
    val_data = Cityscapes(root=args.data_path, split='val', mode='fine', target_type='semantic',
                             transform=transform, target_transform=target_transform)

    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

    """apply checks"""
    #print('train_loader mean,std',calculate_mean_std(train_loader))
    #print('val_loader mean,std', calculate_mean_std(val_loader))

    # visualize samples to confirm functionality.
    #visualize_samples(train_loader, 4)
    #visualize_samples(val_loader, 4)

    data_iterator = iter(train_loader)
    first_batch = next(data_iterator)
    inputs, targets = first_batch

    print(inputs.shape, targets.shape)

    # define model
    model = Model() #.cuda()
    criterion = DiceLoss(eps=1.0, activation=None)
    optimizer = optim.Adam(model.parameters())
    num_epochs = args.num_epochs

    epoch_data = collections.defaultdict(list)
    # training/validation loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print('train iteration:', i)

        epoch_loss = running_loss/len(train_loader)

        epoch_data['loss'].append(epoch_loss)

        model.eval()
        running_loss = 0.0
        for inputs,labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

        validation_loss = running_loss / len(val_loader)
        epoch_data['validation_loss'].append(validation_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Validation Loss: {validation_loss:.4f}")

    #torch.save(model.state_dict(), 'saved_model.pth')
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss criterion': criterion.state_dict(),
        'epoch_data': dict(epoch_data),
        'num_epochs': num_epochs,
    }

    torch.save(state, 'model_and_training_state.pth')


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    print(args,parser)
    main(args)
