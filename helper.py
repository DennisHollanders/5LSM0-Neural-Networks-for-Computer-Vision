import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
#from scipy.ndimage import

class OneHotEncode(torch.nn.Module):
    def __init__(self, num_classes):
        super(OneHotEncode, self).__init__()
        self.num_classes = num_classes

    def forward(self, label):
        # Assuming label is a PIL Image with mode 'L' and needs to be converted to a tensor
        label = transforms.PILToTensor()(label)  # Convert to tensor [1, H, W]
        label = label.squeeze(0).to(torch.int64)  # Remove channel dim, ensure dtype is int64 [H, W]
        one_hot = torch.zeros(self.num_classes, label.size(0), label.size(1), dtype=torch.float32, device=label.device)
        one_hot = one_hot.scatter_(0, label.unsqueeze(0), 1)  # Convert to one-hot [C, H, W]
        return one_hot


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(JaccardLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Intersection is the sum of the element-wise product
        intersection = (inputs * targets).sum()

        # The size of the union is the sum of all predictions plus
        # the sum of all targets minus the size of the intersection
        total = inputs.sum() + targets.sum()
        union = total - intersection

        # Jaccard index
        jaccard = (intersection + smooth) / (union + smooth)

        # Jaccard loss
        return 1 - jaccard

def print_gradients(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            print(
                f"{name} gradient: max {parameter.grad.data.abs().max()} | mean {parameter.grad.data.abs().mean()}")
        else:
            print(f"{name} has no gradient")

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


def apply_canny_edge_detection(image_tensor, low_threshold=50, high_threshold=150):
    """
    Apply Canny edge detection to a PyTorch tensor (image).
    Assumes the input tensor is in CxHxW format and normalized [0, 1].

    Args:
    - image_tensor (torch.Tensor): The input image tensor.
    - low_threshold (int): Low threshold for Canny.
    - high_threshold (int): High threshold for Canny.

    Returns:
    - torch.Tensor: The edge-detected image as a tensor.
    """
    # Convert the tensor to a numpy array and scale to [0, 255]
    image_np = image_tensor.numpy().transpose(1, 2, 0) * 255
    image_np = image_np.astype(np.uint8)

    # Convert to grayscale if it's not already
    if image_np.shape[2] == 3:
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image_np

    # Apply Canny edge detection
    edges = cv2.Canny(image_gray, low_threshold, high_threshold)

    # Convert back to tensor
    edges_tensor = torch.from_numpy(edges).to(torch.float32) / 255.0
    edges_tensor = edges_tensor.unsqueeze(0)  # Add a channel dimension

    return edges_tensor

def canny_edge_transform(image):
    image_tensor = transforms.ToTensor()(image)
    edge_tensor = apply_canny_edge_detection(image_tensor)
    return torch.cat((image_tensor, edge_tensor), dim=0)