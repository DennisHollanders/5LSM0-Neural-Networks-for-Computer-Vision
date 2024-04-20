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
        self.alpha = alpha if alpha is not None else torch.ones(num_classes)
        self.gamma = gamma
    def update_class_weights(self, accuracy_dict, smoothing_factor =0.1):
        # Update weights inversely proportional to the correctly classified percentages
        for cls, acc in accuracy_dict.items():
            if acc > 0:
                new_weight = 1 - (acc / 100)
            else:
                new_weight = 1
            self.class_imbalance_weights[cls] = new_weight #(1 - smoothing_factor) * self.class_imbalance_weights[cls] + smoothing_factor * new_weight

            # Normalizing weights to prevent scaling issues
            #total_weight = self.class_imbalance_weights.sum()
            #self.class_imbalance_weights /= total_weight

    def focal_loss(self, preds, targets, alpha, gamma):
        """Calculate focal loss for each class"""
        ce_loss = torch.nn.functional.cross_entropy(preds, targets, reduction='none', ignore_index=self.ignore_index)
        p_t = torch.exp(-ce_loss)
        focal_loss = alpha * ((1 - p_t) ** gamma) * ce_loss
        return focal_loss.mean()

    def dice_loss(self, pred_flat, target_flat, weight_applied_flat):
        # Weighting the intersection and the sums with the distance transform map
        weighted_intersection = (pred_flat * target_flat * weight_applied_flat).sum()
        weighted_pred_sum = (pred_flat * weight_applied_flat).sum()
        weighted_target_sum = (target_flat * weight_applied_flat).sum()
        dice_coef = (2. * weighted_intersection + self.smooth) / (weighted_pred_sum + weighted_target_sum + self.smooth)
        return 1 - dice_coef

    def jaccard_loss(self, pred_flat, target_flat,weight_applied_flat):
        #print(pred_flat.shape,target_flat.shape,distance_transform_map.shape)
        # Applying weights to intersection and union calculations
        weighted_intersection = (pred_flat * target_flat * weight_applied_flat).sum()
        weighted_union = (pred_flat * weight_applied_flat).sum() + (target_flat * weight_applied_flat).sum() - weighted_intersection
        jaccard_coef = (weighted_intersection + self.smooth) / (weighted_union + self.smooth)
        return 1 - jaccard_coef

    def forward(self, pred, target):
        # if self.weight is not None:
        #     self.weight = self.weight.to(pred.device)
        #print('target shape in loss function',target.shape)
        #print(self.weight)
        if target.shape[1] > 1:
            target_segmentation = target[:,0,:,:]
            distance_transform_map = (target[:,1,:,:].reshape(-1) /255 )+self.epsilon
            #print('min max distance transform:', min(distance_transform_map), max(distance_transform_map))
        else:
            raise ValueError("Segmentation targets only have a single channel, add distance transform and edge map")
        C = pred.shape[1]
        total_loss = 0.0
        ce_loss = 0.0
        for c in range(C):
            if c != self.ignore_index:
                pred_flat = pred[:, c].contiguous().reshape(-1)
                target_flat = (target_segmentation == c).float().reshape(-1)
                if 0 <= self.balance_weight <= 1:
                    weight_applied_flat = ((distance_transform_map * self.balance_weight) + self.class_imbalance_weights[c]*(1-self.balance_weight))/2
                else:
                    weight_applied_flat = 1
                alpha_t = self.alpha[c]
                fl_loss = self.focal_loss(pred, target_segmentation, alpha_t, self.gamma)
                ce_loss += fl_loss

                # Dice/Jaccard loss
                if self.loss_type == 'Dice':
                    loss = self.dice_loss(pred_flat, target_flat, weight_applied_flat)
                elif self.loss_type == 'Jaccard':
                    loss = self.jaccard_loss(pred_flat, target_flat, weight_applied_flat)
                else:
                    raise ValueError("Unsupported loss type. Use 'dice' or 'jaccard'.")
                total_loss += loss
        #print('weight_applied:',np.unique(weight_applied_flat))
        #print('Dice/Jaccard:',(total_loss / (C - 1)))
        #print('ce_loss', ce_loss/(C-1))

        combined_loss = ((self.dice_jaccard_weight * (total_loss / (C - 1))) + ((ce_loss/(C-1)) * self.ce_weight)) / 2
        #print('combined loss', combined_loss)
        return combined_loss


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


def calculate_iou(preds, labels, num_classes):
    ious = []
    correct_percentages = {}
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    #print('in iou',preds.shape, labels.shape)
    """
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        #print(pred_inds.shape, target_inds.shape)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            continue

        iou = float(intersection) / float(max(union, 1))
        ious.append(iou)

    mean_iou = sum(ious) / len(ious) if ious else 1.0
    return mean_iou
    """
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



def calculate_accuracy_near_edges(distance_transform_map, predicted_labels, ground_truth_labels, threshold=10):
    # Identify pixels within the specified distance from the nearest edge
    near_edge_pixels = distance_transform_map <= threshold

    # Select the corresponding predictions and ground truth labels for those pixels
    predicted_near_edges = predicted_labels[near_edge_pixels]
    ground_truth_near_edges = ground_truth_labels[near_edge_pixels]

    # Calculate the number of correctly classified pixels
    correct_predictions = (predicted_near_edges == ground_truth_near_edges).sum()

    # Calculate accuracy as the ratio of correctly classified pixels to the total number of near-edge pixels
    accuracy = correct_predictions / len(predicted_near_edges) if len(predicted_near_edges) > 0 else 0

    return accuracy

class RandomFog:
    def __call__(self, img):
        if random.random() < 0.2:
            return F.adjust_gamma(img, gamma=random.uniform(0.5, 1), gain=0.5)
        return img

class RandomBrightness:
    def __init__(self, brightness_factor=(0.5, 2.0)):
        self.brightness_factor = brightness_factor

    def __call__(self, img):
        brightness_factor = random.uniform(*self.brightness_factor)
        return F.adjust_brightness(img, brightness_factor)

class RandomContrast:
    def __init__(self, contrast_factor=(0.5, 2.0)):
        self.contrast_factor = contrast_factor

    def __call__(self, img):
        contrast_factor = random.uniform(*self.contrast_factor)
        return F.adjust_contrast(img, contrast_factor)
