from torchvision.datasets import Cityscapes
from torchvision import transforms
from torch.utils.data import DataLoader
from helper import *
from model import Model

model_path = 'model_5530539.pth'
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your trained model

    model = Model()  # Replace with your actual model class
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    model.to(device)

    # Prepare the dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])
    dataset = Cityscapes(root='City_Scapes/', split='val', mode='fine', target_type='semantic',
                         transform=transform, target_transform=target_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Visualize predictions
    visualize_predictions(loader, model, num_images=4, device=device)  # Adjust num_images as desired

def visualize_predictions(loader, model, num_images, device):
    fig, axs = plt.subplots(3, num_images, figsize=(num_images * 4, 12))  # Adjusted for 3 rows
    colormap = plt.get_cmap('viridis')  # Using 'viridis' colormap for better visibility

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for i, (img, true_segmentation) in enumerate(loader):
            if i >= num_images:
                break

            img = img.to(device)
            outputs = model(img)
            _, preds = torch.max(outputs, 1)  # Get the predicted classes

            # Convert to numpy for visualization
            img_np = img.cpu().squeeze().numpy()
            true_segmentation_np = true_segmentation.cpu().squeeze().numpy()
            preds_np = preds.cpu().squeeze().numpy()

            img_np = np.transpose(img_np, (1, 2, 0))  # Convert to HxWxC format for plotting
            true_color_segmentation = true_segmentation_np# apply_colormap(true_segmentation_np, colormap)
            preds_color_segmentation = preds_np #apply_colormap(preds_np, colormap)

            # Original Image
            axs[0, i].imshow(img_np)
            axs[0, i].set_title('Original Image')
            axs[0, i].axis('off')

            # True Segmentation
            axs[1, i].imshow(true_color_segmentation)
            axs[1, i].set_title('True Segmentation')
            axs[1, i].axis('off')

            # Model's Prediction
            axs[2, i].imshow(preds_color_segmentation)
            axs[2, i].set_title("Model's Prediction")
            axs[2, i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
