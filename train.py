"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""

import torch.optim as optim
from torchvision.datasets import Cityscapes
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from argparse import ArgumentParser
from helper import *
import collections
from model import Model
import wandb
from torch.utils.data import random_split
from torchvision.transforms import Lambda

def get_arg_parser():
    parser = ArgumentParser()

    # Detect the running environment
    if 'SLURM_JOB_ID' in os.environ:
        # We're likely running on a server with SLURM
        default_data_path = "/gpfs/work5/0/jhstue005/JHS_data/CityScapes"
        default_batch_size = 16
        default_num_epochs = 5
        default_resize = (256, 512)
    else:
        # We're likely running in a local environment
        default_data_path = "City_Scapes/"
        default_batch_size = 4
        default_num_epochs = 1
        default_resize = (32, 32)


    parser.add_argument("--data_path", type=str, default=default_data_path, help="Path to the data")
    parser.add_argument("--batch_size", type=int, default=default_batch_size, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=default_num_epochs, help="Number of epochs for training")
    parser.add_argument("--resize", type=int, default=default_resize, help="Image format that is being worked with ")
    return parser

def main(args):
    LEARNING_RATE = 5e-5
    VAL_SIZE = 0.2
    NUM_CLASSES = 34

    wandb.init(
        # set the wandb project where this run will be logged
        project="5LSM0",
        # track hyperparameters and run metadata
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "Initial architecture",
            "dataset": "CityScapes",
            "epochs": args.num_epochs,
            "optimizer":"Adam",
            "batch_size":args.batch_size,
            "Image size": args.resize,
        }
    )
    """define your model, trainingsloop optimitzer etc. here"""
    transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.ToTensor(),
    ])

    # The Cityscapes dataset returns the target as PIL Image for 'semantic' target_type
    """
    target_transform = transforms.Compose([
        transforms.Resize(args.resize, interpolation=transforms.InterpolationMode.NEAREST),
       
    ])
    """
    target_transform = transforms.Compose([
        transforms.Resize(args.resize, interpolation=transforms.InterpolationMode.NEAREST),
        Lambda(edge_and_distance_transform),
        OneHotEncode(num_classes=NUM_CLASSES),
        #transforms.ToTensor(),
    ])

    full_training_data = Cityscapes(root=args.data_path, split='train', mode='fine', target_type='semantic',
                                    transform=transform, target_transform=target_transform)
    total_size = len(full_training_data)
    train_size = int(total_size * (1- VAL_SIZE))
    val_size = total_size - train_size
    training_data, validation_data = random_split(full_training_data, [train_size, val_size])

    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=1)

    model = Model()
    model.cuda() if torch.cuda.is_available() else model.cpu()
    initialize_weights(model)
    criterion = WeightedJaccardLoss(num_classes = NUM_CLASSES)
    #criterion = CombinedLoss(num_classes=NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    num_epochs = args.num_epochs

    epoch_data = collections.defaultdict(list)
    # training/validation loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            print('-------------- \n start train_loader iteration \n --------------')
            print(inputs.size(),labels.size())
            #img = Image.fromarray(labels[0,0,:,:])
            #img.show()

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                inputs = inputs.cpu()
                labels = labels.cpu()
            print('iteration:',i)

            optimizer.zero_grad()
            outputs = model(inputs)

            #print(labels.size(), inputs.size(), outputs.size())
            #print(len(torch.unique(labels)))
            #predicted_classes = torch.argmax(outputs, dim=1, keepdim=True)
            #print("Size of predicted_classes:", predicted_classes.size())
            #labels_one_hot = one_hot_encoding(labels, NUM_CLASSES)

            loss = criterion(outputs,labels)
            loss.backward()
            #print_gradients(model)
            optimizer.step()
            running_loss += loss.item()

        print('Epoch:',epoch)
        epoch_loss = running_loss/len(train_loader)
        epoch_data['loss'].append(epoch_loss)

        model.eval()
        running_loss = 0.0
        for inputs,labels in val_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                inputs = inputs.cpu()
                labels = labels.cpu()
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

    wandb.log(state)

    try:
        slurm_job_id = os.environ.get('SLURM_JOB_ID', 'default_job_id')
        torch.save(state, f'model_{slurm_job_id}.pth')
    except:
        torch.save(state, f'model.pth')


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    print(args,parser)
    main(args)


