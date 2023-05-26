import os
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
import sys
import cv2
import torchvision
import torch.optim as optim
from tqdm import tqdm
import json
import random


from model import vgg

model_name = "vgg_chen"
net = vgg(model_name=model_name, num_classes=20, init_weights=True)



params = list(net.parameters())
num_params = sum(p.numel() for p in params)
print(f"Number of parameters: {num_params}")
