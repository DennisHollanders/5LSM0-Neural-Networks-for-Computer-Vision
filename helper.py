import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
#from scipy.ndimage import
from PIL import Image
class WeightedJaccardLoss(nn.Module):
    def __init__(self, num_classes):
        super(WeightedJaccardLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1):
        """
        inputs: predicted probabilities for each class, shape (N, H, W) assuming N is batch size
        targets: ground truth, with the first channel for semantic segmentation and
                 the third channel for the distance transform map, shape (N, C, H, W)
        smooth: smoothing constant to avoid division by zero
        """
        # Assuming targets are in the format [N, C, H, W] where C=3 with the first channel for segmentation
        # and the third for the distance transform map

        #print('-------------- \n Started losses \n ---------------')
        #print(targets.shape)
        #print('num_classes', self.num_classes)
        segmentation_targets = targets[:,:self.num_classes, :, :]
        distance_transform_weights = targets[:, -1, :, :].unsqueeze(1)
        #print('segmentation, distance map, inputs:',segmentation_targets.shape,distance_transform_weights.shape, inputs.shape)
        segmentation = segmentation_targets * distance_transform_weights
        #print('segmentation shape',segmentation.shape)

        # Flatten label, prediction, and weight tensors
        inputs_flat_weight = (inputs * distance_transform_weights).reshape(-1)
        targets_flat_weight = (segmentation_targets * distance_transform_weights).reshape(-1)
        #weights_flattened = distance_transform_weights.reshape(-1)

        #print('input,targets,weights', inputs_flat_weight.shape,targets_flat_weight.shape,)

        # Intersection is the sum of the element-wise product of inputs and targets, modulated by weights
        intersection = (inputs_flat_weight * targets_flat_weight).sum()

        # Calculate weighted sums for inputs and targets
        weighted_inputs_sum = inputs_flat_weight.sum()
        weighted_targets_sum = targets_flat_weight.sum()

        # The size of the union is the sum of weighted inputs and targets minus the size of the weighted intersection
        total = weighted_inputs_sum + weighted_targets_sum
        union = total - intersection

        # Weighted Jaccard index
        jaccard = (intersection + smooth) / (union + smooth)
        categorical_loss = -torch.sum(targets_flat_weight * torch.log(inputs_flat_weight + 1e-6), dim=0).mean()
        jaccard_loss = 1-jaccard
        # Weighted Jaccard loss
        return categorical_loss # (categorical_loss + jaccard_loss) / 2

    def categorical_cross_entropy_loss(predictions, labels, epsilon=1e-10):
        """
        Calculates the categorical cross entropy loss.

        Parameters:
        - predictions: Tensor of shape [B, C, W, H]. Predicted probabilities for each class.
        - labels: Tensor of shape [B, W, H]. Ground truth labels.
        - epsilon: Small value to avoid log(0).

        Returns:
        - loss: Scalar tensor. The average loss across all pixels and batch.
        """
        # Convert labels to one-hot encoding
        B, C, W, H = predictions.shape
        labels_one_hot = F.one_hot(labels, num_classes=C).permute(0, 3, 1, 2).float()

        # Apply epsilon to predictions to avoid log(0)
        predictions = torch.clamp(predictions, epsilon, 1. - epsilon)

        # Calculate pixel-wise loss
        pixel_wise_loss = -torch.sum(labels_one_hot * torch.log(predictions), dim=1)

        # Calculate the average loss
        loss = torch.mean(pixel_wise_loss)

        return loss


class OneHotEncode(torch.nn.Module):
    def __init__(self, num_classes):
        super(OneHotEncode, self).__init__()
        self.num_classes = num_classes

    def forward(self, target):
        #print('------------------------------------ \n START ONEHOT ENCODING\n ------------------------------------')
        #print('target size:',target.shape)
        # Convert target to a tensor if it's a NumPy array
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        #    print('if numpy yes to torch ')
        #print('target.shape[0]=',target.shape[0])
        # Assume target is a tensor here, with shape [C, H, W]
        if len(target.shape) == 3:
            #print('shape =3 ')
            label = target[0].long()  # Extract segmentation labels; assuming they are in the first channel
            one_hot = torch.nn.functional.one_hot(label, num_classes=self.num_classes)  # One-hot encoding
            #print('Shape after encoding:',one_hot.shape)
            one_hot = one_hot.permute(2, 0, 1).float()  # Reorder dimensions to [C, H, W]
            #print('target shape after permute',one_hot.shape)

            # Concatenate the two additional channels from the original target tensor
            additional_channels = target[1:].float()  # Extract and convert to float for consistency
            final_output = torch.cat([one_hot, additional_channels], dim=0)  # Concatenate along the channel dimension

            #print('Final output shape:', final_output.shape)
            return final_output

            return final_output
        else:
            return print('Error incorrect shape inserted in OneHot')

class DiceLoss(nn.Module):
    def __init__(self, num_classes, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1):

        targets = targets[:, :self.num_classes, :, :]

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

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

def detect_edges(image):
    """Finds transitions in a PIL image and returns a PIL image of edges."""
    image_np = np.array(image)
    diff_x = np.abs(np.diff(image_np, axis=1))
    diff_y = np.abs(np.diff(image_np, axis=0))

    edges_x = diff_x > 1
    edges_y = diff_y > 1

    edges = np.zeros_like(image_np)
    edges[:, :-1] |= edges_x
    edges[:-1, :] |= edges_y

    edges_pil = Image.fromarray(edges.astype(np.uint8))
    return edges_pil


def edge_and_distance_transform(image, num_levels_below_30_percent=5):
    """Creates an image with 3 channels: original, detected edges, and the distance transform map."""
    original_image_array = np.array(image)

    # Ensure the original image array is 2D (grayscale)
    if original_image_array.ndim != 2:
        raise ValueError("The original image must be a single-channel (grayscale) image.")

    # Detect edges and calculate the distance transform
    edge_detected_image = detect_edges(image)
    edge_transform = np.array(edge_detected_image) * 255
    distance_transform_map = calculate_distance_transform(edge_transform, num_levels_below_30_percent)

    # Ensure all images are of the same dtype
    original_image_array = original_image_array.astype(np.uint8)
    edge_transform = edge_transform.astype(np.uint8)
    distance_transform_map = distance_transform_map.astype(np.uint8)

    # Combine into a 3-channel image
    combined_image_array = np.stack((original_image_array, edge_transform, distance_transform_map), axis=0)

    # Convert back to PIL Image
    #combined_image_pil = Image.fromarray(combined_image_array)

    #print('combined image array shape',combined_image_array.shape)
    return combined_image_array


def calculate_distance_transform(edge_detected_image, num_levels_below_30_percent):
    h, w = edge_detected_image.shape[:2]
    max_distance = np.sqrt(h ** 2 + w ** 2)

    inverted_image = np.abs(edge_detected_image - 255)
    distance_transform = cv2.distanceTransform(inverted_image, cv2.DIST_L2, 5)
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