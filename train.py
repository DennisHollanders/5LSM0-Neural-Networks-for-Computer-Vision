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
try:
    import pretty_errors
except ImportError:
    pass

def get_arg_parser():
    parser = ArgumentParser()
    loss = 'Jaccard'
    distance_transform_weight = True
    learning_rate = 5e-5
    val_size = 0.2
    num_classes = 19

    # Detect the running environment
    if 'SLURM_JOB_ID' in os.environ:
        default_data_path = "/gpfs/work5/0/jhstue005/JHS_data/CityScapes"
        default_batch_size = 16
        default_num_epochs = 20
        default_resize = (256, 512)
        default_pin_memory = True
    else:
        default_data_path = "City_Scapes/"
        default_batch_size = 4
        default_num_epochs = 1
        default_resize = (32, 32)
        default_pin_memory = False

    parser.add_argument("--data_path", type=str, default=default_data_path, help="Path to the data")
    parser.add_argument("--batch_size", type=int, default=default_batch_size, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=default_num_epochs, help="Number of epochs for training")
    parser.add_argument("--resize", type=tuple, default=default_resize, help="Image format that is being worked with ")
    parser.add_argument("--loss", type=str, default= loss, help="Loss function applied to the model")
    parser.add_argument("--distance_transform_weight", type=bool, default=distance_transform_weight, help="Adds boundary information to loss function")
    parser.add_argument("--learning_rate", type=float, default=learning_rate, help="Stepsize of the optimizer")
    parser.add_argument("--val_size", type=float, default=val_size, help="Size of validation set, size trainset = 1- val_size")
    parser.add_argument("--num_classes", type=int, default=num_classes, help="Number of classes to be predicted")
    parser.add_argument("--pin_memory",type=bool,default=default_pin_memory,help="variable to smoothen transfer to gpu")
    return parser

def main(args):
    if 'SLURM_JOB_ID' in os.environ:
        wandb.init(
            # set the wandb project where this run will be logged
            project="5LSM0",
            # track hyperparameters and run metadata
            config={
                "architecture": "Initial architecture",
                "dataset": "CityScapes",
                "optimizer": "Adam",
                "learning_rate": args.learning_rate,
                "epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "Image size": args.resize,
                'num_epochs': args.num_epochs,
                'Applied loss': args.loss,
                'Weights applied': args.distance_transform_weight,
                'Validation size': args.val_size,
            }
        )
    """define your model, trainingsloop optimitzer etc. here"""
    transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    target_transform = transforms.Compose([
        transforms.Resize(args.resize, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        #Lambda(reshape_targets),
        #Lambda(edge_and_distance_transform),
        #OneHotEncode(num_classes=args.num_classes),
    ])

    full_training_data = Cityscapes(root=args.data_path, split='train', mode='fine', target_type='semantic',
                                    transform=transform, target_transform=target_transform)
    total_size = len(full_training_data)
    train_size = int(total_size * (1- args.val_size))
    val_size = total_size - train_size
    training_data, validation_data = random_split(full_training_data, [train_size, val_size])

    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory= args.pin_memory)
    val_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=args.pin_memory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()
    model.to(device)
    #initialize_weights(model)
    criterion = Loss_Functions(args.num_classes,args.loss,args.distance_transform_weight,ignore_index=255)
    #criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    epoch_data = collections.defaultdict(list)
    # training/validation loop
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            labels = (labels * 255).long().squeeze()
            labels = utils.map_id_to_train_id(labels).to(device)
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            print(outputs.shape)
            #predicted = torch.argmax(outputs, 1)

            loss = criterion(outputs, labels)  # add .long.squeeze
            loss.backward()
            # print_gradients(model)
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{args.num_epochs}, Loss: {epoch_loss:.4f}')
        epoch_data['loss'].append(epoch_loss)

        model.eval()
        running_loss = 0.0
        for inputs, labels in val_loader:
            labels = (labels * 255).long().squeeze()
            labels = utils.map_id_to_train_id(labels).to(device)

            inputs = inputs.to(device)
            outputs = model(inputs)
            #predicted = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

        validation_loss = running_loss / len(val_loader)
        epoch_data['validation_loss'].append(validation_loss)
        print( f"Epoch [{epoch + 1}/{args.num_epochs}], Train Loss: {epoch_loss:.4f}, Validation Loss: {validation_loss:.4f}")
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss criterion': criterion.state_dict(),
        'epoch_data': dict(epoch_data),
    }

    try:
        slurm_job_id = os.environ.get('SLURM_JOB_ID', 'default_job_id')
        torch.save(state, f'model_{slurm_job_id}.pth')
        wandb.log(state)
    except:
        torch.save(state, f'model.pth')


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    #print(args,parser)
    main(args)


