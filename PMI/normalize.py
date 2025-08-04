import numpy as np
import torch
import torchvision.transforms as transforms

labels = np.load('../CIFAR-10-C/labels.npy')
selected_indices_0 = labels == 0
selected_indices_1 = labels == 1

# Load images
images_a_0 = np.load('../CIFAR-10-C/brightness.npy')[selected_indices_0]  # Adjust path as needed
images_a_1 = np.load('../CIFAR-10-C/brightness.npy')[selected_indices_1]  # Adjust path as needed
images_b_0 = np.load('../CIFAR-10-C/contrast.npy')[selected_indices_0]  # Adjust path as needed
images_b_1 = np.load('../CIFAR-10-C/contrast.npy')[selected_indices_1]  # Adjust path as needed
images = np.concatenate([images_a_0, images_a_1, images_b_0, images_b_1])
images_tensor = torch.tensor(images, dtype=torch.float32) / 255.0  # Scale to [0, 1]

# Compute mean and standard deviation
mean = images_tensor.mean(dim=(0, 1, 2))  # Compute mean across height, width, and samples
std = images_tensor.std(dim=(0, 1, 2))    # Compute std across height, width, and samples

print("Calculated Mean:", mean.tolist())
print("Calculated Std:", std.tolist())

# Update transforms with calculated values
normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())
preprocess = transforms.Compose([transforms.ToTensor(), normalize])