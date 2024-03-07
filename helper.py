import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
#from scipy.ndimage import distance_transform_edt


def visualize_samples(loader, num_images):
    fig, axs = plt.subplots(2, num_images, figsize=(12, 6))
    colormap = plt.get_cmap('jet')
    for i, (img, segmentation) in enumerate(loader):
        if num_images >0:
            img = img[i].numpy()
            segmentation = segmentation[i].numpy()
            img = np.transpose(img, (1, 2, 0))
            segmentation = np.transpose(segmentation, (1, 2, 0))

            color_segmentation = apply_colormap(segmentation, colormap)

            # Visualize the image and segmentation
            axs[0, i].imshow(img)
            axs[0, i].set_title('Image')
            axs[0, i].axis('off')
            axs[1, i].imshow(segmentation)
            axs[1, i].set_title('Label')
            axs[1, i].axis('off')
            num_images -=1
        else:
            break
    plt.tight_layout()
    plt.show()

"""
def apply_colormap(segmentation, colormap):
    normed_data = segmentation.astype(np.float32) / segmentation.max()
    mapped_color = colormap(normed_data)
    return mapped_color #(mapped_color[:,:,3] * 255).astype(np.uint8)


def calculate_mean_std(loader):
    channel_sum, channel_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        data = data.float()
        if data.dim() == 4:
            batch_size, num_channels = data.shape[0], data.shape[1]
        else:
            raise ValueError("Incorrect shape")
        channel_sum += torch.mean(data) * batch_size
        channel_squared_sum += torch.mean(data**2) * batch_size
        num_batches += batch_size
    mean = channel_sum / num_batches
    std = (channel_squared_sum / num_batches - mean**2)**0.5
    return mean, std
"""
class Jaccard(nn.Module):
    def __init__(self, epsilon =10e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):

        # Ensure the predictions and targets are the same size.
        assert predictions.size() == targets.size(), "The size of predictions and targets does not match."

        # Flatten the tensors to simplify the intersection and union calculations.
        predictions = predictions.view(*predictions.size()[:2], -1)  # [batch_size, num_classes, H*W]
        targets = targets.view(*targets.size()[:2], -1)  # [batch_size, num_classes, H*W]

        # Calculate intersection and union.
        intersection = torch.sum(predictions * targets, dim=2)  # [batch_size, num_classes]
        union = torch.sum(predictions, dim=2) + torch.sum(targets, dim=2) - intersection  # [batch_size, num_classes]

        # Calculate IoU and then the Jaccard loss.
        iou = intersection / (union + self.epsilon)
        loss = 1 - iou

        # Average the loss over the batch and classes.
        return loss.mean()

"""
class AdaptedJaccardLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        assert predictions.size() == targets.size(), "The size of predictions and targets does not match."

        # Compute distance transform for the targets
        # Note: This requires moving data to CPU and numpy, which can be inefficient. Consider optimizing for GPU if necessary.
        batch_size, num_classes, height, width = targets.size()
        distance_transforms = torch.zeros_like(targets)
        for b in range(batch_size):
            for c in range(num_classes):
                target_np = targets[b, c].cpu().numpy()
                # Compute the distance transform
                dist_transform = scipy.ndimage.distance_transform_edt(1 - target_np)
                distance_transforms[b, c] = torch.tensor(dist_transform).to(targets.device)

        # Flatten and prepare for weighted loss calculation
        predictions = predictions.view(*predictions.size()[:2], -1)  # [batch_size, num_classes, H*W]
        targets = targets.view(*targets.size()[:2], -1)  # [batch_size, num_classes, H*W]
        distance_transforms = distance_transforms.view(*distance_transforms.size()[:2],
                                                       -1)  # [batch_size, num_classes, H*W]

        # Calculate weighted intersection and union
        intersection = torch.sum(predictions * targets * distance_transforms, dim=2)  # [batch_size, num_classes]
        union = torch.sum((predictions + targets - predictions * targets) * distance_transforms,
                          dim=2)  # [batch_size, num_classes]

        # Calculate IoU and then the Jaccard loss
        iou = intersection / (union + self.epsilon)
        loss = 1 - iou

        return loss.mean()


class ExtendTargetTransform(transforms):
    def __init__(self):
        super().__init__()

    def __call__(self, image, target):
        # Check if the target has only one channel
        if target.shape[0] == 1:  # Assuming target shape is [C, H, W]
            target_np = target.squeeze().numpy().astype(np.uint8)

            # Edge detection
            edges = cv2.Canny(target_np, 100, 200)
            edges = torch.from_numpy(edges).to(torch.float32) / 255.0  # Normalize to [0, 1]

            # Distance transform
            dist_transform = distance_transform_edt(1 - edges.numpy())
            dist_transform = torch.from_numpy(dist_transform).to(torch.float32)
            dist_transform /= dist_transform.max()  # Normalize to [0, 1]

            # Stack the original target, edges, and distance transform
            extended_target = torch.stack([target.squeeze(), edges, dist_transform], dim=0)
        else:
            extended_target = target

        return image, extended_target
"""
def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()


    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score
class DiceLoss(nn.Module):
    __name__ = 'dice_loss'
    #from: https://www.kaggle.com/code/dhananjay3/image-segmentation-from-scratch-in-pytorch
    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(y_pr, y_gt, beta=1.,
                           eps=self.eps, threshold=None,
                           activation=self.activation)