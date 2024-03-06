import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import Cityscapes
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from argparse import ArgumentParser
from helper import *
import collections
from model import Model