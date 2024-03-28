import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
#from scipy.ndimage import
from PIL import Image
import utils

def reshape_targets(targets):
    labels = (targets * 255).long().squeeze()
    labels = utils.map_id_to_train_id(labels)
    return labels
class Loss_Functions(nn.Module):
    def __init__(self, num_classes,loss,Weight, ignore_index=255, weight=None, size_average=True):
        super(Loss_Functions, self).__init__()
        self.num_classes = num_classes
        self.loss_type = loss
        self.smooth = 1
        self.weight = Weight
        self.ignore_index =ignore_index

    def dice_loss(self, pred_flat, target_flat, distance_transform_map):
        """
        intersection = (pred_flat * target_flat).sum()
        dice_coef = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice_coef
        """
        # Weighting the intersection and the sums with the distance transform map
        weighted_intersection = (pred_flat * target_flat * distance_transform_map).sum()
        weighted_pred_sum = (pred_flat * distance_transform_map).sum()
        weighted_target_sum = (target_flat * distance_transform_map).sum()

        dice_coef = (2. * weighted_intersection + self.smooth) / (weighted_pred_sum + weighted_target_sum + self.smooth)
        return 1 - dice_coef

    def jaccard_loss(self, pred_flat, target_flat,distance_transform_map):
        """
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        jaccard_coef = (intersection + self.smooth) / (union + self.smooth)
        return 1 - jaccard_coef
        """
        print(pred_flat.shape,target_flat.shape,distance_transform_map.shape)
        # Applying weights to intersection and union calculations
        weighted_intersection = (pred_flat * target_flat * distance_transform_map).sum()
        weighted_union = (pred_flat * distance_transform_map).sum() + (target_flat * distance_transform_map).sum() - weighted_intersection
        jaccard_coef = (weighted_intersection + self.smooth) / (weighted_union + self.smooth)
        return 1 - jaccard_coef

    def forward(self, pred, target):
        # if self.weight is not None:
        #     self.weight = self.weight.to(pred.device)
        #print('target shape in loss function',target.shape)
        if target.shape[1] > 1:
            target_segmentation = target[:,0,:,:]
            Distance_transform_weight = target[:,1,:,:]
        C = pred.shape[1]
        total_loss = 0.0
        for c in range(C):
            if c != self.ignore_index:
                pred_flat = pred[:, c].contiguous().reshape(-1) # or apply view
                target_flat = (target_segmentation == c).float().reshape(-1)
                Distance_transform_weight_flat = Distance_transform_weight.reshape(-1)

                if self.loss_type == 'Dice':
                    loss = self.dice_loss(pred_flat, target_flat,Distance_transform_weight_flat)
                elif self.loss_type == 'Jaccard':
                    loss = self.jaccard_loss(pred_flat, target_flat,Distance_transform_weight_flat)
                else:
                    raise ValueError("Unsupported loss type. Use 'dice' or 'jaccard'.")
                total_loss += loss
        return total_loss / C

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
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias,0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

def detect_edges(tensor_np):
    """Finds transitions in a PIL image and returns a PIL image of edges."""
    diff_x = np.abs(np.diff(tensor_np, axis=1))
    diff_y = np.abs(np.diff(tensor_np, axis=0))

    edges_x = diff_x > 1
    edges_y = diff_y > 1

    edges = np.zeros_like(tensor_np)
    edges[:, :-1] |= edges_x
    edges[:-1, :] |= edges_y
    return edges


def edge_and_distance_transform(tensor, num_levels_below_30_percent=5):
    # If the tensor is not on CPU or not a numpy array, convert it
    if tensor.is_cuda:
        tensor = tensor.cpu()

    tensor_np = tensor.numpy().squeeze().astype(np.uint8)
    # Detect edges and calculate the distance transform
    edge_detected_image = detect_edges(tensor_np)

    edge_transform = edge_detected_image * 255
    distance_transform_map = calculate_distance_transform(edge_transform, num_levels_below_30_percent)

    # Ensure all images are of the same dtype
    edge_transform = edge_transform.astype(np.uint8)
    distance_transform_map = distance_transform_map.astype(np.uint8)

    # stack all maps ( edge transform might be redundant dependant on final execution)
    combined_image_array = np.stack((tensor_np, edge_transform, distance_transform_map), axis=0)
    return torch.from_numpy(combined_image_array)


def calculate_distance_transform(edge_detected_image, num_levels_below_30_percent):
    h, w = edge_detected_image.shape[:2]
    max_distance = np.sqrt(h ** 2 + w ** 2)

    inverted_image = np.abs(edge_detected_image - 255)
    distance_transform = cv2.distanceTransform(inverted_image, cv2.DIST_L2, 5)
    normalized_distance_transform = distance_transform / distance_transform.max()

    # Invert the distances so closer pixels have higher values
    inverted_distance_transform = 1 - normalized_distance_transform

    distance_transform = (inverted_distance_transform * 255).astype(np.uint8)
    Quantization = False
    if Quantization:
        threshold_30_percent = 0.05 * max_distance
        normalized_distance_transform = distance_transform / max_distance * 255

        quantized_distance_transform = np.zeros_like(normalized_distance_transform)

        quantized_distance_transform[normalized_distance_transform > threshold_30_percent] = 255
        for level in range(1, num_levels_below_30_percent + 1):
            lower_bound = (level - 1) / num_levels_below_30_percent * threshold_30_percent
            upper_bound = level / num_levels_below_30_percent * threshold_30_percent
            quantized_distance_transform[
                (normalized_distance_transform > lower_bound) & (normalized_distance_transform <= upper_bound)] = np.round(
                lower_bound / threshold_30_percent * 255)

        return quantized_distance_transform
    else:
        return distance_transform