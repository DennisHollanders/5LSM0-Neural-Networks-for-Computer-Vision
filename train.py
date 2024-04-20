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
CEbalances = [0]              #,0.5, 0.9]
weight_balances = [2]          #,0.5, 0.9]
loss = ['Jaccard' ] # ,'Dice']

# Create a list of all possible combinations of hyperparameters
hyperparameter_combinations = list(itertools.product(
    learning_rates, batch_sizes, CEbalances, weight_balances, loss))
print(len(hyperparameter_combinations))
def get_arg_parser(hparams):
    #Tuneable hyperparams
    learning_rate, default_batch_size, default_CEbalance, default_weight_balance, loss = hparams
    #loss =  hyperparameters.
    #learning_rate = hyperparameters.
    #default_CEbalance = hyperparameters.
    #default_weight_balance = hyperparameters
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
    #parser.add_argument("--weight_applied", type=str, default=weight_applied, help="Information which weights to add to loss function")
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
               # 'Weights applied': args.weight_applied,
                'Validation size': args.val_size,
                'Weight balance': args.weight_balance,
                'Ce balance': args.CEbalance,
            }
        )
    """define your model, trainingsloop optimitzer etc. here"""
    transform = transforms.Compose([
        transforms.Resize(args.resize),
        #RandomFog(),
        #RandomBrightness(brightness_factor=(0.5, 1.5)),
        #RandomContrast(contrast_factor=(0.5, 1.5)),
        #transforms.
        #v2.RandomIoUCrop([min_scale, max_scale, ...])
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        #transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    target_transform = transforms.Compose([
        transforms.Resize(args.resize, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        Lambda(reshape_targets),
        Lambda(edge_and_distance_transform),
    ])

    full_training_data = Cityscapes(root=args.data_path, split='train', mode='fine', target_type='semantic',
                                    transform=transform, target_transform=target_transform)
    total_size = len(full_training_data)
    train_size = int(total_size * (1- args.val_size))
    val_size = total_size - train_size

    # evaluate random split with class imbalance situations !
    training_data, validation_data = random_split(full_training_data, [train_size, val_size])

    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory= args.pin_memory)
    val_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=args.pin_memory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()
    model.to(device)
    initialize_weights(model)
    criterion = Loss_Functions(args.num_classes,args.loss,args.weight_balance , args.CEbalance,ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,  betas=(0.95, 0.999), eps=1e-08, weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    epoch_data = collections.defaultdict(list)

    train_correct_counts = collections.defaultdict(int)
    train_total_counts = collections.defaultdict(int)
    val_correct_counts = collections.defaultdict(int)
    val_total_counts = collections.defaultdict(int)

    # training/validation loop
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        iou_sum = 0.0
        edge_accuracy_sum = 0.0
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
            #print(distance_transform_map.shape,ground_truth_labels.shape,labels.shape,inputs.shape, outputs.shape)
            # comparison
            preds = torch.argmax(outputs, dim=1)
            #print(preds.shape,outputs.shape)
            for cls in range(args.num_classes):
                train_correct_counts[cls] += ((preds == cls) & (ground_truth_labels == cls)).sum().item()
                train_total_counts[cls] += (ground_truth_labels == cls).sum().item()

            #iou_sum += calculate_iou(preds, ground_truth_labels, num_classes=args.num_classes)
            #edge_accuracy_sum += calculate_accuracy_near_edges(distance_transform_map, preds, ground_truth_labels, threshold=10)
        #print_gradients(model)
        epoch_loss = running_loss / len(train_loader)
        #epoch_edge = edge_accuracy_sum / len(train_loader)
        #epoch_iou = iou_sum / len(train_loader)
        # epoch_edge_accuracy = edge_accuracy_sum / num_batches
        print(f'Epoch {epoch + 1}/{args.num_epochs}, Loss: {epoch_loss:.4f}')
        print('train total count:',train_total_counts,'train correct count',train_correct_counts)
        #print(f"edge: {epoch_edge:.4f}, iou: {epoch_iou:.4f} ")
        epoch_data['loss'].append(epoch_loss)
        train_class_accuracies = {
            cls: (train_correct_counts[cls] / train_total_counts[cls] * 100 if train_total_counts[cls] > 0 else 0) for
            cls in range(args.num_classes)}
        train_class_accuracies = {key: round(value,2) for key,value in train_class_accuracies.items()}

        # Print class-specific training accuracies
        print(f'Train Class Accuracies: {train_class_accuracies}')

        #epoch_data['edge'].append(epoch_edge)
        #epoch_data['iou'].append(epoch_iou)

        model.eval()
        running_loss = 0.0
        iou_sum = 0.0
        edge_accuracy_sum = 0.0
        for inputs, labels in val_loader:
            labels = labels.to(device).long().squeeze()
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = nn.functional.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            distance_transform_map, ground_truth_labels = labels[:, 2, :, :], labels[:, 0, :, :]
            # comparison
            preds = torch.argmax(outputs, dim=1)
            #iou_sum += calculate_iou(preds, ground_truth_labels, num_classes=args.num_classes)
            #edge_accuracy_sum += calculate_accuracy_near_edges(distance_transform_map, preds, ground_truth_labels,threshold=10)
            for cls in range(args.num_classes):
                val_correct_counts[cls] += ((preds == cls) & (ground_truth_labels == cls)).sum().item()
                val_total_counts[cls] += (ground_truth_labels == cls).sum().item()

        #epoch_edge = edge_accuracy_sum / len(train_loader)
        #epoch_iou = iou_sum / len(train_loader)
        validation_loss = running_loss / len(val_loader)
        epoch_data['validation_loss'].append(validation_loss)
        #epoch_data['validation_edge'].append(epoch_edge)
        #epoch_data['validation_iou'].append(epoch_iou)
        print( f"Epoch [{epoch + 1}/{args.num_epochs}], Train Loss: {epoch_loss:.4f}, Validation Loss: {validation_loss:.4f}")
        #print( f"Validation edge: {epoch_edge:.4f}, Validation iou: {epoch_iou:.4f} \n")
        val_class_accuracies = calculate_iou(preds, ground_truth_labels, args.num_classes)
        criterion.update_class_weights(val_class_accuracies)

        val_class_accuracies = {
            cls: (val_correct_counts[cls] / val_total_counts[cls] * 100 if val_total_counts[cls] > 0 else 0) for
            cls in range(args.num_classes)}
        val_class_accuracies = {key: round(value, 2) for key, value in val_class_accuracies.items()}
        scheduler.step()

        # Print class-specific training accuracies
        print(f'Validation Class Accuracies: {val_class_accuracies}')
        print(f'New class imbalance weights: {criterion.class_imbalance_weights}')

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr} \n")

    additional_info = {
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_criterion_state_dict': criterion.state_dict(),
        'epoch_data': dict(epoch_data),
    }

    try:
        slurm_job_id = os.environ.get('SLURM_JOB_ID', 'default_job_id')
        torch.save(model.state_dict(), f'model_{slurm_job_id}.pth')
        torch.save(additional_info, f'addinfo_model_{slurm_job_id}.pth')
        state = {
            'state_dict': model.state_dict(),
            'additional_info': additional_info
        }
        wandb.log(state)
    except:
        torch.save(model.state_dict(), f'model.pth')
        torch.save(additional_info, f'addinfo_model.pth')

if __name__ == "__main__":
    for hparams in hyperparameter_combinations:
        parser = get_arg_parser(hparams)
        args = parser.parse_args()
        print(args,parser)
        main(args)