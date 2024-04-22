import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import cv2
import random
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
    def __init__(self, num_classes, loss, balance_weight, ce_weight, ignore_index=255, alpha=None, gamma=2.0):
        super(Loss_Functions, self).__init__()
        self.num_classes = num_classes
        self.loss_type = loss
        self.ignore_index = ignore_index
        self.smooth = 1
        self.epsilon = 1e-4
        self.class_imbalance_weights = torch.ones(num_classes)
        self.balance_weight = balance_weight
        self.ce_weight = ce_weight
        self.dice_jaccard_weight = 1.0 - ce_weight
        self.gamma = gamma if isinstance(gamma, torch.Tensor) else torch.tensor([gamma] * num_classes)
        self.alpha = alpha if alpha is not None else torch.ones(num_classes)
    def update_class_weights(self, accuracy_dict, smoothing_factor):
        """
        Function that checks class performance accuracy and updates the alpha and gamma factors for the focal loss to it
        """
        # Constants for the quadratic gamma function
        a = 1 / 2500
        b = -2 / 25
        c = 5
        for cls, acc in accuracy_dict.items():
            # Calculate new gamma value using the quadratic function
            new_gamma = a * (acc ** 2) + b * acc + c
            self.gamma[cls] = new_gamma
            # Update class weights based on accuracy
            if acc > 0:
                new_alpha = 1 - (acc / 100)
                self.alpha[cls] = new_alpha
            else:
                new_alpha = 1
            # Smoothing the transition of class weights
            self.class_imbalance_weights[cls] = new_alpha
    def focal_loss(self, preds, targets, alpha, gamma):
        """
        Calculate focal loss for each class. Gamma and alpha parameters are dynamically updated in update_class_weight_function
        """
        ce_loss = torch.nn.functional.cross_entropy(preds, targets, reduction='none', ignore_index=self.ignore_index)
        p_t = torch.exp(-ce_loss)
        focal_loss = alpha * ((1 - p_t) ** gamma) * ce_loss
        return focal_loss.mean()

    def dice_loss(self, pred_flat, target_flat, weight_applied_flat):
        """
        Calculates the Dice loss including applied weights to it
        """
        weighted_intersection = (pred_flat * target_flat * weight_applied_flat).sum()
        weighted_pred_sum = (pred_flat * weight_applied_flat).sum()
        weighted_target_sum = (target_flat * weight_applied_flat).sum()
        dice_coef = (2. * weighted_intersection + self.smooth) / (weighted_pred_sum + weighted_target_sum + self.smooth)
        return 1 - dice_coef

    def jaccard_loss(self, pred_flat, target_flat,weight_applied_flat):
        """
        Calculates the jaccard loss including applied weights to it
        """
        weighted_intersection = (pred_flat * target_flat * weight_applied_flat).sum()
        weighted_union = (pred_flat * weight_applied_flat).sum() + (target_flat * weight_applied_flat).sum() - weighted_intersection
        jaccard_coef = (weighted_intersection + self.smooth) / (weighted_union + self.smooth)
        return 1 - jaccard_coef

    def forward(self, pred, target):
        """
        Calculate the loss based upon two weighting parameters

        :params:
        self.balance_weight = weight between class imbalance and edge-based weight
        self.ce_weight = weight between focal loss and jaccard loss
        """
        # Check if the required distance transform is in the predictions
        if target.shape[1] > 1:
            target_segmentation = target[:,0,:,:]
            distance_transform_map = (target[:,1,:,:].reshape(-1) /255 )+self.epsilon
        else:
            raise ValueError("Segmentation targets only have a single channel, add distance transform and edge map")
        C = pred.shape[1]
        total_loss = 0.0
        fl_loss = 0.0
        # Calculate the loss for each class independently
        for c in range(C):
            # ignore the ignore class
            if c != self.ignore_index:
                pred_flat = pred[:, c].contiguous().reshape(-1)
                target_flat = (target_segmentation == c).float().reshape(-1)

                # if weight balance > 1 or <0 apply no weight to jaccard otherwise balance between Class imbalance and edge distance
                if 0 <= self.balance_weight <= 1:
                    weight_applied_flat = ((distance_transform_map * self.balance_weight) + self.class_imbalance_weights[c]*(1-self.balance_weight))/2
                else:
                    weight_applied_flat = torch.ones_like(distance_transform_map)
                fl_loss += self.focal_loss(pred, target_segmentation, self.alpha[c], self.gamma[c])

                # Dice/Jaccard loss
                if self.loss_type == 'Dice':
                    total_loss += self.dice_loss(pred_flat, target_flat, weight_applied_flat)
                elif self.loss_type == 'Jaccard':
                    total_loss += self.jaccard_loss(pred_flat, target_flat, weight_applied_flat)
                else:
                    raise ValueError("Unsupported loss type. Use 'dice' or 'jaccard'.")

        combined_loss = ((self.dice_jaccard_weight * (total_loss / (C - 1))) + ((fl_loss/(C-1)) * self.ce_weight)) / 2
        return combined_loss


def print_gradients(model):
    """
    Function that prints the gradients to inspect vanishing gradient problem
    """
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            print(
                f"{name} gradient: max {parameter.grad.data.abs().max()} | mean {parameter.grad.data.abs().mean()}")
        else:
            print(f"{name} has no gradient")

def initialize_weights(model):
    """"
    Initialize the weights use kaiming he for relu activations
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias,0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

def detect_edges(tensor_np):
    """
    Finds transitions in an image and returns a PIL image of edges.
    """
    diff_x = np.abs(np.diff(tensor_np, axis=1))
    diff_y = np.abs(np.diff(tensor_np, axis=0))

    edges_x = diff_x > 1
    edges_y = diff_y > 1

    edges = np.zeros_like(tensor_np)
    edges[:, :-1] |= edges_x
    edges[:-1, :] |= edges_y
    return edges


def edge_and_distance_transform(tensor, num_levels_below_30_percent=5):
    """
    Calculates the edge map used for the boundary based losses.

    :params:
    num_level_below_30_percent: variable used for quantization of distance transform map
    """
    # If the tensor is not on CPU or not a numpy array, convert it
    if tensor.is_cuda:
        tensor = tensor.cpu()

    tensor_np = tensor.numpy().squeeze().astype(np.uint8)
    # Detect edges and calculate the distance transform
    edge_detected_image = detect_edges(tensor_np)
    edge_transform = edge_detected_image * 255

    # calculates the distance transform map based on the edge map
    distance_transform_map = calculate_distance_transform(edge_transform, num_levels_below_30_percent)
    # Ensure all images are of the same dtype
    edge_transform = edge_transform.astype(np.uint8)
    distance_transform_map = distance_transform_map.astype(np.uint8)
    # stack all maps ( edge transform might be redundant dependant on final execution)
    combined_image_array = np.stack((tensor_np, edge_transform, distance_transform_map), axis=0)
    return torch.from_numpy(combined_image_array)


def calculate_distance_transform(edge_detected_image, num_levels_below_30_percent):
    """
    Calculates distance transform map, with possibility for quantization (did not yield desireable results)
    """
    # quantization dependant on image shape
    h, w = edge_detected_image.shape[:2]
    max_distance = np.sqrt(h ** 2 + w ** 2)

    # invert edge map
    inverted_image = np.abs(edge_detected_image - 255)

    # apply distance transform
    distance_transform = cv2.distanceTransform(inverted_image, cv2.DIST_L2, 5)

    #Normalize distance transform map
    normalized_distance_transform = distance_transform / distance_transform.max()

    # Invert the distances so closer pixels have higher values
    inverted_distance_transform = 1 - normalized_distance_transform

    distance_transform = (inverted_distance_transform * 255).astype(np.uint8)
    Quantization = False
    if Quantization:
        threshold_30_percent = 0.2 * max_distance
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


def calculate_accuracy(preds, labels, num_classes):
    """
    Calculates the accuracy of prediction used to dynamically tweak the class imbalance
    """
    correct_percentages = {}
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        correct = (pred_inds & target_inds).sum().item()
        total = target_inds.sum().item()
        if total == 0:
            correct_percentages[cls] = 0
        else:
            correct_percentages[cls] = correct / total * 100
    return correct_percentages
