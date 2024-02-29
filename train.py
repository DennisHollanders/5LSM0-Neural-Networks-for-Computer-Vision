"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import collections
from model import Model
from torchvision.datasets import Cityscapes
from torchvision import transforms
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.function as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

NUMBER_OF_CLASSES = 25
def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()

def visualize_predictions(model, test_loader):
    model.eval()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            if idx >= 4:  # Visualize the first 4 images
                break
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(inputs[0])
            axes[0].set_title('Input Image')
            axes[0].axis('off')

            axes[1].imshow(labels[0], cmap='jet', vmin=0, vmax=NUMBER_OF_CLASSES)  # Assuming there are 20 classes
            axes[1].set_title('True Segmentation')
            axes[1].axis('off')

            axes[2].imshow(preds[0], cmap='jet', vmin=0, vmax=NUMBER_OF_CLASSES)  # Assuming there are 20 classes
            axes[2].set_title('Obtained Segmentation')
            axes[2].axis('off')

            plt.show()
def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    #dataset
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize()])
    train_dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic',
                               transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = Cityscapes(args.data_path, split='val', mode='fine', target_type='semantic', transform=data_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    test_dataset = Cityscapes(args.data_path, split='test', mode='fine', target_type='semantic',
                              transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # define model
    model = Model().cuda()

    # define optimizer and loss function (don't forget to ignore class index 255)
    criterion = nn.CrossEntropyLoss()  # Changed from nn.CrosEntropyLoss() to nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    epoch_data = collections.defaultdict(list)

    num_epochs = 2
    # training/validation loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

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

    # Save model
    save_model(model, os.path.join(args.data_path, 'saved_model.pth'))

    # Plot losses
    plot_losses(epoch_data['loss'], epoch_data['validation_loss'])

    # Visualize predictions
    visualize_predictions(model, test_loader)

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
