from torchvision.datasets import Cityscapes
from torchvision import transforms
from torch.utils.data import DataLoader
from helper import *
from model import Model
from torchvision.transforms import Lambda
from utils import *

# try:
#     import pretty_errors
# except ImportError:
#     pass

model_path = 'models/model_5697465.pth'

def plot_losses(epoch_data):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)

    # Extract and plot the training and validation losses
    epoch_data = state['epoch_data']
    #plot_losses(epoch_data)

    # Prepare the dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((256,512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((256,512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        # Lambda(reshape_targets),
        # Lambda(edge_and_distance_transform),
        # OneHotEncode(num_classes=args.num_classes),
    ])

    dataset = Cityscapes(root='City_Scapes/', split='val', mode='fine', target_type='semantic',
                         transform=transform, target_transform=target_transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Visualize predictions
    visualize_segmentation(model,loader)


def visualize_segmentation(model, dataloader, num_examples=5):
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
            print('before', masks.shape)
            masks = (masks * 255).long().squeeze()  # *255 because the id are normalized between 0-1
            masks = utils.map_id_to_train_id(masks)
            print('after',masks.shape)
            outputs = model(images)
            #outputs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, 1)

            images = images.numpy()
            masks = masks.numpy()

            predicted = predicted.numpy()

            for j in range(4):
                image = renormalize_image(images[j].transpose(1, 2, 0))
                mask_rgb = mask_to_rgb(masks[j], colors)
                pred_mask_rgb = mask_to_rgb(predicted[j], colors)

                # Original Image
                plt.subplot(num_examples, 3, j*3 + 1)
                plt.imshow(image)
                plt.title('Image')
                plt.axis('off')

                # Ground Truth Mask
                plt.subplot(num_examples, 3, j*3 + 2)
                plt.imshow(mask_rgb)
                plt.title('Ground Truth Mask')
                plt.axis('off')

                # Model's Prediction
                plt.subplot(num_examples, 3, j*3 + 3)
                plt.imshow(pred_mask_rgb)
                plt.title("Model's Prediction")
                plt.axis('off')

            plt.tight_layout()
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
    print('in mask to rgb', mask.shape)

    # Ensure mask is 2D
    mask = mask.squeeze()
    print('in mask to rgb reshaped', mask.shape)

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

if __name__ == "__main__":
    main()
