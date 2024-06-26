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
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import itertools
try:
    import pretty_errors
except ImportError:
    pass


# Define the ranges for your hyperparameters
learning_rates = [5e-4]
batch_sizes = [8]
CEbalances = [0]
weight_balances = [2]
loss = ['Jaccard','Dice']

# Create a list of all possible combinations of hyperparameters
hyperparameter_combinations = list(itertools.product(
    learning_rates, batch_sizes, CEbalances, weight_balances, loss))
print(len(hyperparameter_combinations))

def get_arg_parser(hparams):
    #Tuneable hyperparams
    learning_rate, default_batch_size, default_CEbalance, default_weight_balance, loss = hparams
    val_size = 0.2
    num_classes = 19
    parser = ArgumentParser()

    # Detect the running environment
    # Used for actual
    if 'SLURM_JOB_ID' in os.environ:
        default_data_path = "/gpfs/work5/0/jhstue005/JHS_data/CityScapes"
        #default_batch_size = hyperparameters.
        default_num_epochs = 20
        default_resize = (256, 512)
        default_pin_memory = True
    # Used for local debugging
    else:
        default_data_path = "City_Scapes/"
        default_batch_size = 4
        default_num_epochs = 2
        default_resize = (32, 32)
        default_pin_memory = False

    parser.add_argument("--data_path", type=str, default=default_data_path, help="Path to the data")
    parser.add_argument("--batch_size", type=int, default=default_batch_size, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=default_num_epochs, help="Number of epochs for training")
    parser.add_argument("--resize", type=tuple, default=default_resize, help="Image format that is being worked with ")
    parser.add_argument("--loss", type=str, default= loss, help="Loss function applied to the model")
    parser.add_argument("--learning_rate", type=float, default=learning_rate, help="Stepsize of the optimizer")
    parser.add_argument("--val_size", type=float, default=val_size, help="Size of validation set, size trainset = 1- val_size")
    parser.add_argument("--num_classes", type=int, default=num_classes, help="Number of classes to be predicted")
    parser.add_argument("--pin_memory",type=bool,default=default_pin_memory,help="Variable to smoothen transfer to gpu")
    parser.add_argument("--CEbalance",type=float, default=default_CEbalance,help="Defines the ratio between CEbalance and jaccard")
    parser.add_argument("--weight_balance", type=float, default=default_weight_balance,help="Defines the ratio between distransform and class imbalance weights")
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
                'Validation size': args.val_size,
                'Weight balance': args.weight_balance,
                'Ce balance': args.CEbalance,
            }
        )
    """define your model, trainingsloop optimitzer etc. here"""
    transform = transforms.Compose([
        transforms.Resize(args.resize),
        # add possible data Augmentations
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    target_transform = transforms.Compose([
        transforms.Resize(args.resize, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        Lambda(reshape_targets), #change from 30 classes to 19
        Lambda(edge_and_distance_transform), #add distance transform and edge map
    ])
    #prepare training and validation loaders
    full_training_data = Cityscapes(root=args.data_path, split='train', mode='fine', target_type='semantic',
                                    transform=transform, target_transform=target_transform)
    total_size = len(full_training_data)
    train_size = int(total_size * (1- args.val_size))
    val_size = total_size - train_size
    training_data, validation_data = random_split(full_training_data, [train_size, val_size])
    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory= args.pin_memory)
    val_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=args.pin_memory)

    #initialize the model and weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()
    model.to(device)
    initialize_weights(model)

    # define loss functions and learning rates
    criterion = Loss_Functions(args.num_classes,args.loss,args.weight_balance , args.CEbalance,ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,  betas=(0.95, 0.999), eps=1e-08, weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    # Create containers to track progress
    epoch_data = collections.defaultdict(list)
    train_correct_counts = collections.defaultdict(int)
    train_total_counts = collections.defaultdict(int)
    val_correct_counts = collections.defaultdict(int)
    val_total_counts = collections.defaultdict(int)

    # training/validation loop
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        # run train loop
        for inputs, labels in train_loader:

            labels = labels.to(device).long().squeeze()
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = nn.functional.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            distance_transform_map, ground_truth_labels = labels[:,2,:,:],labels[:,0,:,:]
            preds = torch.argmax(outputs, dim=1)

            # calculate accuracy per class
            for cls in range(args.num_classes):
                train_correct_counts[cls] += ((preds == cls) & (ground_truth_labels == cls)).sum().item()
                train_total_counts[cls] += (ground_truth_labels == cls).sum().item()

        # print performance of training loop
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{args.num_epochs}, Loss: {epoch_loss:.4f}')
        print('train total count:',train_total_counts,'train correct count',train_correct_counts)
        epoch_data['loss'].append(epoch_loss)
        train_class_accuracies = {cls: (train_correct_counts[cls] / train_total_counts[cls] * 100 if train_total_counts[cls] > 0 else 0) for cls in range(args.num_classes)}
        train_class_accuracies = {key: round(value,2) for key,value in train_class_accuracies.items()}
        # Print class-specific training accuracies
        print(f'Train Class Accuracies: {train_class_accuracies}')

        # run validation loop
        model.eval()
        running_loss = 0.0
        for inputs, labels in val_loader:
            labels = labels.to(device).long().squeeze()
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = nn.functional.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            distance_transform_map, ground_truth_labels = labels[:, 2, :, :], labels[:, 0, :, :]
            preds = torch.argmax(outputs, dim=1)

            # calculate validation performance
            for cls in range(args.num_classes):
                val_correct_counts[cls] += ((preds == cls) & (ground_truth_labels == cls)).sum().item()
                val_total_counts[cls] += (ground_truth_labels == cls).sum().item()
        # print validation performance
        validation_loss = running_loss / len(val_loader)
        epoch_data['validation_loss'].append(validation_loss)
        print( f"Epoch [{epoch + 1}/{args.num_epochs}], Train Loss: {epoch_loss:.4f}, Validation Loss: {validation_loss:.4f}")
        val_class_accuracies = calculate_accuracy(preds, ground_truth_labels, args.num_classes)
        val_class_accuracies = {cls: (val_correct_counts[cls] / val_total_counts[cls] * 100 if val_total_counts[cls] > 0 else 0) for cls in range(args.num_classes)}
        val_class_accuracies = {key: round(value, 2) for key, value in val_class_accuracies.items()}

        # update weights for class imbalance
        criterion.update_class_weights(val_class_accuracies, smoothing_factor=0)

        # update learning rate
        scheduler.step()

        # Print class-specific training accuracies
        print(f'Validation Class Accuracies: {val_class_accuracies}')
        print('Mean performance:', sum(val_class_accuracies.values()+1) / len(val_class_accuracies))
        print(f'New class imbalance weights: {criterion.class_imbalance_weights}')
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']} \n")

    # log to wandb and save model
    additional_info = {
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_criterion_state_dict': criterion.state_dict(),
        'epoch_data': dict(epoch_data),}
    try:
        slurm_job_id = os.environ.get('SLURM_JOB_ID', 'default_job_id')
        torch.save(model.state_dict(), f'model_{slurm_job_id}.pth')
        torch.save(additional_info, f'addinfo_model_{slurm_job_id}.pth')
        state = {'state_dict': model.state_dict(),'additional_info': additional_info}
        wandb.log(state)
    except:
        torch.save(model.state_dict(), f'model.pth')
        torch.save(additional_info, f'addinfo_model.pth')

if __name__ == "__main__":
    # loop for hyperparametere search. Excute main and args based on the created list of combinations
    for hparams in hyperparameter_combinations:
        parser = get_arg_parser(hparams)
        args = parser.parse_args()
        # print current hyperparameter combinations
        print(args,parser)
        main(args)