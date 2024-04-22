from torchvision.datasets import Cityscapes
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from helper import *
from model import Model
from torchvision.transforms import Lambda
from utils import *
import torch.nn as nn

try:
    import pretty_errors
except ImportError:
    pass

# choose model to evaluate
model_path = 'models/model_5994545.pth'
model_additional = 'models/model_additional_5869017.pth'

def plot_losses(epoch_data):
    """
    Plot los trajectory over epochs
    """
    train_losses = epoch_data['loss']
    val_losses = epoch_data['validation_loss']
    plt.figure(figsize=(15, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()

def main():
    """
    Initialize the model and dataloaders
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model state
    model = Model()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # Prepare the dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((256,512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((256,512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        Lambda(reshape_targets),
        Lambda(edge_and_distance_transform),
    ])

    full_dataset = Cityscapes(root='City_Scapes/', split='val', mode='fine', target_type='semantic',
                         transform=transform, target_transform=target_transform)
    subset_size = int(0.2 * len(full_dataset))
    # take small subset of the data
    indices = torch.randperm(len(full_dataset))[:subset_size]
    dataset_subset = Subset(full_dataset, indices)
    loader = DataLoader(dataset_subset, batch_size=4, shuffle=False)

    # Visualize predictions
    visualize_segmentation(model, loader, device)


def visualize_segmentation(model, dataloader, device, num_examples=1):
    model.eval()
    # create accuracy containers
    total_class_correct = np.zeros(19)
    total_class_pixels = np.zeros(19)
    edge_accuracies = []

    # plots the models performance
    visualize_predictions(model, dataloader)

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets_full = targets
            targets = targets[:,0,:,:].to(device)
            outputs = model(images)
            predictions = torch.argmax(nn.functional.softmax(outputs, dim=1), dim=1)

            # Compute class-specific accuracy
            for cls in range(19):
                cls_mask = targets == cls
                cls_correct = (predictions == cls) & cls_mask
                total_class_correct[cls] += cls_correct.sum().item()
                total_class_pixels[cls] += cls_mask.sum().item()

            # Compute accuracy near edges if necessary
            edge_accuracy = calculate_accuracy_near_edges(predictions, targets_full)
            edge_accuracies.append(edge_accuracy)

            # plots the target channels
            if i < num_examples:
                visualize_edge_masks(targets_full,4)

    # Calculate per-class accuracies
    class_accuracies = (total_class_correct / total_class_pixels) * 100
    mean_performance = np.nanmean(class_accuracies)  # Handling classes with zero pixels in samples

    # Calculate Class Imbalance Score (CI)
    sorted_accuracies = np.sort(class_accuracies)
    ci_score = np.mean(sorted_accuracies[:10])  # Average of ten worst-performing classes

    # Calculate Edge Performance (EP)
    ep_score = np.mean(edge_accuracies)

    print(f"Mean Performance per Class: {class_accuracies}")
    print(f"Overall Mean Performance: {mean_performance:.2f}%")
    print(f"Class Imbalance Score (CI): {ci_score:.2f}%")
    print(f"Edge Performance (EP): {ep_score:.2f}%")

def visualize_predictions(model, dataloader, num_examples=5):
    """
    Visualizes segmentation results from a given model using a dataloader.

    Args:
        model (torch.nn.Module): The segmentation model to visualize.
        dataloader (torch.utils.data.DataLoader): Dataloader providing image-mask pairs.
        num_examples (int, optional): Number of examples to visualize. Defaults to 5.

    Returns:
        None
    """
    colors, class_names = names_colors_classes()
    model.eval()

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= 1:
                break

            if masks.shape[1] > 1:
                print('edge mask should be visualized')
                #visualize_edge_masks(masks, num_examples)

            plt.figure(figsize=(15 , 20))
            print('shape =', masks.shape)
            #masks = (masks[:,0, :, :] * 255).long().squeeze()
            #masks = utils.map_id_to_train_id(masks) #.unsqueeze(1)
            print('after',masks.shape)
            masks = masks[:, 0, :, :]

            # prep tensor to numpy to be plotted
            outputs = model(images)
            outputs = nn.functional.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, 1)
            images = images.numpy()
            masks = masks.numpy()
            predicted = predicted.numpy()


            for j in range(4):
                image = renormalize_image(images[j].transpose(1, 2, 0))
                mask_rgb = mask_to_rgb(masks[j], colors)
                pred_mask_rgb = mask_to_rgb(predicted[j], colors)

                # Calculate error mask only for non-ignored pixels
                error_mask = ((mask_rgb != pred_mask_rgb) & (mask_rgb != 255)).any(axis=2).astype(np.uint8) *255
                #error_mask = (mask_rgb != pred_mask_rgb).astype(np.uint8) * 255

                # Original Image
                plt.subplot(num_examples, 4, j * 4 + 1)
                plt.imshow(image)
                plt.title('Image')
                plt.axis('off')

                # Ground Truth Mask
                plt.subplot(num_examples, 4, j * 4 + 2)
                plt.imshow(mask_rgb)
                plt.title('Ground Truth Mask')
                plt.axis('off')

                # Model's Prediction
                plt.subplot(num_examples, 4, j * 4 + 3)
                plt.imshow(pred_mask_rgb)
                plt.title("Model's Prediction")
                plt.axis('off')

                # Error Mask
                plt.subplot(num_examples, 4, j * 4 + 4)
                plt.imshow(error_mask, cmap='binary')
                plt.title("Error Highlight")
                plt.axis('off')

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.tight_layout(pad=0)
            plt.show()


def renormalize_image(image):
    """
    Renormalizes the image to its original range.

    Args:
        image (numpy.ndarray): Image tensor to renormalize.

    Returns:
        numpy.ndarray: Renormalized image tensor.
    """
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    renormalized_image = image * std + mean
    return renormalized_image
def mask_to_rgb(mask, class_to_color):
    """
    Converts a numpy mask with multiple classes indicated by integers to a color RGB mask.

    Parameters:
        mask (numpy.ndarray): The input mask where each integer represents a class.
        class_to_color (dict): A dictionary mapping class integers to RGB color tuples.

    Returns:
        numpy.ndarray: RGB mask where each pixel is represented as an RGB tuple.
    """

    # Ensure mask is 2D
    mask = mask.squeeze()
    height, width = mask.shape

    # Initialize an empty RGB mask
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Iterate over each class and assign corresponding RGB color
    for class_idx, color in class_to_color.items():
        # Mask pixels belonging to the current class
        class_pixels = mask == class_idx
        # Assign RGB color to the corresponding pixels
        rgb_mask[class_pixels] = color

    return rgb_mask


def visualize_edge_masks(masks, num_examples):
    """
    Creates a plot of the target channel maps
    """
    plt.figure(figsize=(20, 15))
    colors, class_names = names_colors_classes()

    for j in range(min(num_examples, masks.shape[0])):
        # extract maps
        ground_truth_mask = masks[j, 0, :, :]
        edge_mask = masks[j, 1, :, :]
        distance_transform = masks[j, 2, :, :]
        mask_rgb = mask_to_rgb(ground_truth_mask, colors)

        # Ground Truth Mask
        plt.subplot(num_examples, 3, j * 3 + 1)
        plt.imshow(mask_rgb)
        plt.title('Ground Truth Mask')
        plt.axis('off')

        # Edge Mask
        plt.subplot(num_examples, 3, j * 3 + 2)
        plt.imshow(edge_mask, cmap='gray')
        plt.title('Edge Mask')
        plt.axis('off')

        # Distance Transform Map
        plt.subplot(num_examples, 3, j * 3 + 3)
        plt.imshow(distance_transform,cmap='viridis')
        plt.title("Distance Transform Map")
        plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    plt.show()


def calculate_accuracy_near_edges(predictions, targets, threshold=0.8):
    """
    Calculate the accuracy of predictions near the edges of the targets, using the distance transform map.
    """
    # Ensure that the distance transform map is accessed correctly, which is assumed to be in the third channel
    distance_transform = targets[:, 2, :, :]  # Accessing the third channel

    # Create a mask where the distance transform values are greater than the specified threshold
    near_edge_mask = distance_transform > threshold

    # Calculate the accuracy only on these near-edge pixels
    correct_near_edge = (predictions[near_edge_mask] == targets[:, 0, :, :][near_edge_mask]).sum()
    total_near_edge = near_edge_mask.sum()
    return (correct_near_edge / total_near_edge).item() * 100 if total_near_edge > 0 else 0



if __name__ == "__main__":
    main()

