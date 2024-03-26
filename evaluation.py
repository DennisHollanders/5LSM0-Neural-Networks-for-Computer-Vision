from torchvision.datasets import Cityscapes
from torchvision import transforms
from torch.utils.data import DataLoader
from helper import *
from model import Model
from torchvision.transforms import Lambda
import utils
try:
    import pretty_errors
except ImportError:
    pass

model_path = 'models/model_5661125.pth'

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
        #Add utils part to reshape targets:
        transforms.reshape
        transforms.Resize((256, 512)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),

        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    target_transform = transforms.Compose([
        transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST),
        Lambda(edge_and_distance_transform),
        OneHotEncode(num_classes=34),
        # transforms.ToTensor(),
    ])
    dataset = Cityscapes(root='City_Scapes/', split='val', mode='fine', target_type='semantic',
                         transform=transform, target_transform=target_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Visualize predictions
    visualize_predictions(loader, model, num_images=4, device=device)

def visualize_predictions(loader, model, num_images, device):
    fig, axs = plt.subplots(3, num_images, figsize=(num_images * 4, 12))

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            if i >= num_images:
                break
            #print(np.unique(labels))
            print(labels.shape, '1')
            labels = labels.long().squeeze()
            labels = utils.map_id_to_train_id(labels).to(device)
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            print(labels.shape,'e')

            # Convert to numpy for visualization and transpose dimensions
            inputs_np = inputs.cpu().squeeze().numpy()
            # Transpose the dimensions from (Channels, Height, Width) to (Height, Width, Channels)
            inputs_np = np.transpose(inputs_np, (1, 2, 0))

            labels = torch.argmax(labels[:34], dim=1)
            print('label size:', labels.shape)

            labels_np = labels.cpu().squeeze() *255 #.numpy()
            preds_np = preds.cpu().squeeze().numpy()
            print(inputs_np.shape,labels_np.shape, preds_np.shape)
            #print('inputs', inputs_np,'labels', labels_np,'preds', preds_np)
            print(np.unique(labels_np), print(np.unique(preds_np)))
            # Original Image
            axs[0, i].imshow(inputs_np)
            axs[0, i].set_title('Original Image')
            axs[0, i].axis('off')

            # True Segmentation
            axs[1, i].imshow(labels_np)
            axs[1, i].set_title('True Segmentation')
            axs[1, i].axis('off')

            # Model's Prediction
            axs[2, i].imshow(preds_np)
            axs[2, i].set_title("Model's Prediction")
            axs[2, i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
