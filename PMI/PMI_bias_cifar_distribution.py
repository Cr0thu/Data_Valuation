from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.linalg import sqrtm
import numpy as np
from math import sqrt
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import MNIST, CIFAR10
import sys
from tqdm import tqdm
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import copy
import gc
import warnings
import argparse
warnings.filterwarnings("ignore")
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser(description="Run model with specified parameters.")
parser = argparse.ArgumentParser(description="Run model with specified parameters.")
parser.add_argument("--train_size_a_0", type=int, required=True, help="Train size A_0")
parser.add_argument("--train_size_a_1", type=int, required=True, help="Train size A_1")
parser.add_argument("--train_size_b_0", type=int, required=True, help="Train size B_0")
parser.add_argument("--train_size_b_1", type=int, required=True, help="Train size B_1")
parser.add_argument("--test_size_a_0", type=int, required=True, help="Test size A_0")
parser.add_argument("--test_size_a_1", type=int, required=True, help="Test size A_1")
parser.add_argument("--test_size_b_0", type=int, required=True, help="Test size B_0")
parser.add_argument("--test_size_b_1", type=int, required=True, help="Test size B_1")
parser.add_argument("--penalty", type=float, required=True, help="Penalty parameter")

args = parser.parse_args()

resnet18 = models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Identity()
resnet18.to(device).eval()
# print(resnet18)

# penalty = 4.282
penalty = args.penalty
eps = 0.01
# width = 0.15
# gap = 0.05
# MCsample = 500
# sigma = 1e-1

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# normalize = transforms.Normalize(mean=[0.559, 0.571, 0.586],
#                                  std=[0.230, 0.226, 0.249])

preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
labels = np.load('../CIFAR-10-C/labels.npy')
selected_indices_0 = labels == 0
selected_indices_1 = labels == 1

def data_preprocess(images):
    images_0 = torch.stack([preprocess(image) for image in images[selected_indices_0]]).to(device)
    images_1 = torch.stack([preprocess(image) for image in images[selected_indices_1]]).to(device)
    with torch.no_grad():
        embedding_0 = resnet18(images_0)
        embedding_1 = resnet18(images_1)
        embedding_0 = torch.concatenate([embedding_0, torch.ones(embedding_0.size()[0],1).to(device)], dim=1)
        embedding_1 = torch.concatenate([embedding_1, torch.ones(embedding_1.size()[0],1).to(device)], dim=1)
        perm = torch.randperm(len(embedding_0))
        embedding_0 = embedding_0[perm]
        embedding_1 = embedding_1[perm]
        # print(embedding_0.size(), embedding_1.size())
    return embedding_0, embedding_1


images = np.load('../CIFAR-10-C/brightness.npy')
images_a_0_embedding, images_a_1_embedding = data_preprocess(images)
images = np.load('../CIFAR-10-C/contrast.npy')
images_b_0_embedding, images_b_1_embedding = data_preprocess(images)
# images = np.load('../CIFAR-10-C/defocus_blur.npy')
# images_c_0_embedding, images_c_1_embedding = data_preprocess(images)
# images = np.load('../CIFAR-10-C/elastic_transform.npy')
# images_d_0_embedding, images_d_1_embedding = data_preprocess(images)
# images = np.load('../CIFAR-10-C/fog.npy')
# images_e_0_embedding, images_e_1_embedding = data_preprocess(images)
# images = np.load('../CIFAR-10-C/frost.npy')
# images_f_0_embedding, images_f_1_embedding = data_preprocess(images)
# images = np.load('../CIFAR-10-C/gaussian_blur.npy')
# images_g_0_embedding, images_g_1_embedding = data_preprocess(images)
# images = np.load('../CIFAR-10-C/gaussian_noise.npy')
# images_h_0_embedding, images_h_1_embedding = data_preprocess(images)
# images = np.load('../CIFAR-10-C/glass_blur.npy')
# images_i_0_embedding, images_i_1_embedding = data_preprocess(images)
# images = np.load('../CIFAR-10-C/impulse_noise.npy')
# images_j_0_embedding, images_j_1_embedding = data_preprocess(images)
# Assuming the embeddings are loaded as numpy arrays
datasets = [
    images_a_0_embedding, 
    images_a_1_embedding, 
    images_b_0_embedding, 
    images_b_1_embedding
]

# Concatenate all datasets to calculate the mean and standard deviation
combined_data = torch.cat(datasets, dim=0)  # Shape: (20000, 513)

# Compute mean and std across the feature dimension (dim=0)
mean = combined_data.mean(dim=0, keepdim=True)
std = combined_data.std(dim=0, keepdim=True)

# Normalize each dataset using the same mean and std
normalized_datasets = [(dataset - mean) / (std + 1e-6) for dataset in datasets]

# Unpack the normalized datasets
images_a_0_embedding, images_a_1_embedding, images_b_0_embedding, images_b_1_embedding = normalized_datasets

P_x = torch.concatenate([
    images_a_0_embedding[4000:],
    images_b_0_embedding[4000:],
    # images_c_0_embedding[4000:],
    # images_d_0_embedding[4000:],
    # images_e_0_embedding[4000:],
    # images_f_0_embedding[4000:],
    # images_g_0_embedding[4000:],
    # images_h_0_embedding[4000:],
    # images_i_0_embedding[4000:],
    # images_j_0_embedding[4000:],
    images_a_1_embedding[4000:],
    images_b_1_embedding[4000:],
    # images_c_1_embedding[4000:],
    # images_d_1_embedding[4000:],
    # images_e_1_embedding[4000:],
    # images_f_1_embedding[4000:],
    # images_g_1_embedding[4000:],
    # images_h_1_embedding[4000:],
    # images_i_1_embedding[4000:],
    # images_j_1_embedding[4000:]
])
# P_y = torch.concatenate([torch.zeros(10000), torch.ones(10000)]).to(device)
P_y = torch.concatenate([torch.zeros(2000), torch.ones(2000)]).to(device)
Q_x = torch.concatenate([
    images_a_0_embedding[:4000],
    images_b_0_embedding[:4000],
    # images_c_0_embedding[:4000],
    # images_d_0_embedding[:4000],
    # images_e_0_embedding[:4000],
    # images_f_0_embedding[:4000],
    # images_g_0_embedding[:4000],
    # images_h_0_embedding[:4000],
    # images_i_0_embedding[:4000],
    # images_j_0_embedding[:4000],
    images_a_1_embedding[:4000],
    images_b_1_embedding[:4000],
    # images_c_1_embedding[:4000],
    # images_d_1_embedding[:4000],
    # images_e_1_embedding[:4000],
    # images_f_1_embedding[:4000],
    # images_g_1_embedding[:4000],
    # images_h_1_embedding[:4000],
    # images_i_1_embedding[:4000],
    # images_j_1_embedding[:4000]
])
# Q_y = torch.concatenate([torch.zeros(40000), torch.ones(40000)]).to(device)
Q_y = torch.concatenate([torch.zeros(8000), torch.ones(8000)]).to(device)
# del resnet18
allocated_memory = torch.cuda.memory_allocated()
print(f"Already allocated: {allocated_memory / 1024 ** 2} MB")
print(P_x[0].sum(), P_y.size())

class dataset:
    def __init__(self, X, y, num, score, base_loss, base_acc, noise):
        self.X = X
        self.y = y
        self.num = num
        self.score = score
        self.base_loss = base_loss
        self.base_acc = base_acc
        self.noise = noise

def sigmoid(z):
    return 1/(1 + torch.exp(-z))

def generate_data_cifar10(indices, dataset, label, noise):
    X = dataset[indices]
    y = torch.ones(X.size()[0]) * label
    y = torch.tensor(y, dtype=torch.int)
    y = y.to(device)
    y[:int(noise*X.size()[0])] = 1 - y[:int(noise*X.size()[0])]
    return X, y, X.size()[0]

def generate_train_cifar10(X, y, train_number, train_size, noise_level):
    train_dataset = []
    for i in range(train_number):
        train_X, train_y = subsample(X, y, train_size)
        train_dataset.append(dataset(train_X, train_y, train_X.size()[0], 0, 0, 0, noise_level*eps))
    return train_dataset

def generate_train_cifar10_distribution(train_ratio, test_ratio, noise_level):
    train_max = max(train_ratio[0], train_ratio[1], train_ratio[2], train_ratio[3])
    train_size = (int(train_ratio[0]*4000/train_max), int(train_ratio[1]*4000/train_max), int(train_ratio[2]*4000/train_max), int(train_ratio[3]*4000/train_max))
    test_max = max(test_ratio[0], test_ratio[1], test_ratio[2], test_ratio[3])
    test_size = (int(test_ratio[0]*1000/test_max), int(test_ratio[1]*1000/test_max), int(test_ratio[2]*1000/test_max), int(test_ratio[3]*1000/test_max))
    perm_indices_a_0 = torch.randperm(4000).to(device)
    sample_indices_a_0 = perm_indices_a_0[:train_size[0]]
    perm_indices_a_1 = torch.randperm(4000).to(device)
    sample_indices_a_1 = perm_indices_a_1[:train_size[2]]
    perm_indices_b_0 = torch.randperm(4000).to(device)
    sample_indices_b_0 = perm_indices_b_0[:train_size[1]]
    perm_indices_b_1 = torch.randperm(4000).to(device)
    sample_indices_b_1 = perm_indices_b_1[:train_size[3]]
    train_X_a_0, train_y_a_0, _ = generate_data_cifar10(sample_indices_a_0, images_a_0_embedding[:4000], 0, eps * noise_level)
    train_X_a_1, train_y_a_1, _ = generate_data_cifar10(sample_indices_a_1, images_a_1_embedding[:4000], 1, eps * noise_level)
    train_X_b_0, train_y_b_0, _ = generate_data_cifar10(sample_indices_b_0, images_b_0_embedding[:4000], 0, eps * noise_level)
    train_X_b_1, train_y_b_1, _ = generate_data_cifar10(sample_indices_b_1, images_b_1_embedding[:4000], 1, eps * noise_level)
    train_X = torch.concatenate([train_X_a_0, train_X_b_0, train_X_a_1, train_X_b_1])
    train_y = torch.concatenate([train_y_a_0, train_y_b_0, train_y_a_1, train_y_b_1])
    test_X = torch.concatenate([
        images_a_0_embedding[4000:4000+test_size[0]],
        images_b_0_embedding[4000:4000+test_size[1]],
        images_a_1_embedding[4000:4000+test_size[2]],
        images_b_1_embedding[4000:4000+test_size[3]],
    ]).to(device)
    test_y = torch.concatenate([
        torch.zeros(test_size[0]+test_size[1]), 
        torch.ones(test_size[2]+test_size[3]),
    ]).to(device)
    return train_X, train_y, test_X, test_y, (train_size[0], train_size[1], train_size[2], train_size[3])

def generate_train_cifar10_distribution_copy(train_X, train_y, train_ratio, test_ratio):
    test_a_label_ratio = test_ratio[0]/test_ratio[2]
    test_b_label_ratio = test_ratio[1]/test_ratio[3]
    train_size_a_0, train_size_b_0, train_size_a_1, train_size_b_1 = train_ratio
    target_size_a_0 = int(train_size_a_1 * test_a_label_ratio)
    target_size_b_0 = int(train_size_b_1 * test_b_label_ratio)
    if train_size_a_0 < target_size_a_0:
        num_extra = int(target_size_a_0/train_size_a_0) - 1
        # extra_indices = torch.randint(0, train_size_a_0, (num_extra,))
        indices = torch.range(0,train_size_a_0-1)
        extra_indices = torch.tensor([])
        for g in range(num_extra):
            extra_indices = torch.cat([extra_indices, indices])
        extra_indices = extra_indices.to(torch.long)
        new_train_X_a_0 = torch.cat([train_X[:train_size_a_0], train_X[:train_size_a_0][extra_indices]])
        new_train_X_a_1 = train_X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]
        new_train_y_a_0 = torch.cat([train_y[:train_size_a_0], train_y[:train_size_a_0][extra_indices]])
        new_train_y_a_1 = train_y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]
    else:
        target_size_a_1 = int(train_size_a_0 / test_a_label_ratio)
        num_extra = int(target_size_a_1/train_size_a_1) - 1
        # extra_indices = torch.randint(0, train_size_a_1, (num_extra,))
        indices = torch.range(0,train_size_a_1-1)
        extra_indices = torch.tensor([])
        for g in range(num_extra):
            extra_indices = torch.cat([extra_indices, indices])
        extra_indices = extra_indices.to(torch.long)
        new_train_X_a_1 = torch.cat([train_X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1], train_X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][extra_indices]])
        new_train_X_a_0 = train_X[:train_size_a_0]
        new_train_y_a_1 = torch.cat([train_y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1], train_y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][extra_indices]])
        new_train_y_a_0 = train_y[:train_size_a_0]

    if train_size_b_0 < target_size_b_0:
        num_extra = int(target_size_b_0/train_size_b_0) - 1
        # extra_indices = torch.randint(0, train_size_b_0, (num_extra,))
        indices = torch.range(0,train_size_b_0-1)
        extra_indices = torch.tensor([])
        for g in range(num_extra):
            extra_indices = torch.cat([extra_indices, indices])
        extra_indices = extra_indices.to(torch.long)
        new_train_X_b_0 = torch.cat([train_X[train_size_a_0:train_size_a_0+train_size_b_0], train_X[train_size_a_0:train_size_a_0+train_size_b_0][extra_indices]])
        new_train_X_b_1 = train_X[train_size_a_0+train_size_b_0+train_size_a_1:]
        new_train_y_b_0 = torch.cat([train_y[train_size_a_0:train_size_a_0+train_size_b_0], train_y[train_size_a_0:train_size_a_0+train_size_b_0][extra_indices]])
        new_train_y_b_1 = train_y[train_size_a_0+train_size_b_0+train_size_a_1:]
    else:
        target_size_b_1 = int(train_size_b_0 / test_b_label_ratio)
        num_extra = int(target_size_b_1/train_size_b_1) - 1
        # extra_indices = torch.randint(0, train_size_b_1, (num_extra,))
        indices = torch.range(0,train_size_b_1-1)
        extra_indices = torch.tensor([])
        for g in range(num_extra):
            extra_indices = torch.cat([extra_indices, indices])
        extra_indices = extra_indices.to(torch.long)
        new_train_X_b_1 = torch.cat([train_X[train_size_a_0+train_size_b_0+train_size_a_1:], train_X[train_size_a_0+train_size_b_0+train_size_a_1:][extra_indices]])
        new_train_X_b_0 = train_X[train_size_a_0:train_size_a_0+train_size_b_0]
        new_train_y_b_1 = torch.cat([train_y[train_size_a_0+train_size_b_0+train_size_a_1:], train_y[train_size_a_0+train_size_b_0+train_size_a_1:][extra_indices]])
        new_train_y_b_0 = train_y[train_size_a_0:train_size_a_0+train_size_b_0]
    curated_train_X = torch.concatenate([new_train_X_a_0, new_train_X_b_0, new_train_X_a_1, new_train_X_b_1])
    curated_train_y = torch.concatenate([new_train_y_a_0, new_train_y_b_0, new_train_y_a_1, new_train_y_b_1])
    return curated_train_X, curated_train_y

def generate_train_cifar10_distribution_delete(train_X, train_y, train_ratio, test_ratio):
    test_a_label_ratio = test_ratio[0]/test_ratio[2]
    test_b_label_ratio = test_ratio[1]/test_ratio[3]
    train_size_a_0, train_size_b_0, train_size_a_1, train_size_b_1 = train_ratio
    target_size_a_0 = int(train_size_a_1 * test_a_label_ratio)
    target_size_b_0 = int(train_size_b_1 * test_b_label_ratio)

    if train_size_a_0 > target_size_a_0:
        residual_indices = torch.randperm(train_size_a_0)[:target_size_a_0]
        new_train_X_a_0 = train_X[:train_size_a_0][residual_indices]
        new_train_X_a_1 = train_X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]
        new_train_y_a_0 = train_y[:train_size_a_0][residual_indices]
        new_train_y_a_1 = train_y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]
    else:
        target_size_a_1 = int(train_size_a_0 / test_a_label_ratio)
        residual_indices = torch.randperm(train_size_a_1)[:target_size_a_1]
        new_train_X_a_1 = train_X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][residual_indices]
        new_train_X_a_0 = train_X[:train_size_a_0]
        new_train_y_a_1 = train_y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][residual_indices]
        new_train_y_a_0 = train_y[:train_size_a_0]

    if train_size_b_0 > target_size_b_0:
        residual_indices = torch.randperm(train_size_b_0)[:target_size_b_0]
        new_train_X_b_0 = train_X[train_size_a_0:train_size_a_0+train_size_b_0][residual_indices]
        new_train_X_b_1 = train_X[train_size_a_0+train_size_b_0+train_size_a_1:]
        new_train_y_b_0 = train_y[train_size_a_0:train_size_a_0+train_size_b_0][residual_indices]
        new_train_y_b_1 = train_y[train_size_a_0+train_size_b_0+train_size_a_1:]
    else:
        target_size_b_1 = int(train_size_b_0 / test_b_label_ratio)
        residual_indices = torch.randperm(train_size_b_1)[:target_size_b_1]
        new_train_X_b_1 = train_X[train_size_a_0+train_size_b_0+train_size_a_1:][residual_indices]
        new_train_X_b_0 = train_X[train_size_a_0:train_size_a_0+train_size_b_0]
        new_train_y_b_1 = train_y[train_size_a_0+train_size_b_0+train_size_a_1:][residual_indices]
        new_train_y_b_0 = train_y[train_size_a_0:train_size_a_0+train_size_b_0]

    curated_train_X = torch.concatenate([new_train_X_a_0, new_train_X_b_0, new_train_X_a_1, new_train_X_b_1])
    curated_train_y = torch.concatenate([new_train_y_a_0, new_train_y_b_0, new_train_y_a_1, new_train_y_b_1])
    return curated_train_X, curated_train_y

def generate_train_cifar10_distribution_denoise(train_X, train_y, train_ratio, noise_level, ratio = 1):
    train_size_a_0, train_size_b_0, train_size_a_1, train_size_b_1 = train_ratio
    noise = noise_level*eps
    new_train_X_a_0 = train_X[:train_size_a_0][int(ratio*noise*train_size_a_0):]
    new_train_X_b_0 = train_X[train_size_a_0:train_size_a_0+train_size_b_0][int(ratio*noise*train_size_b_0):]
    new_train_X_a_1 = train_X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][int(ratio*noise*train_size_a_1):]
    new_train_X_b_1 = train_X[train_size_a_0+train_size_b_0+train_size_a_1:][int(ratio*noise*train_size_b_1):]
    new_train_y_a_0 = train_y[:train_size_a_0][int(ratio*noise*train_size_a_0):]
    new_train_y_b_0 = train_y[train_size_a_0:train_size_a_0+train_size_b_0][int(ratio*noise*train_size_b_0):]
    new_train_y_a_1 = train_y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][int(ratio*noise*train_size_a_1):]
    new_train_y_b_1 = train_y[train_size_a_0+train_size_b_0+train_size_a_1:][int(ratio*noise*train_size_b_1):]
    curated_train_X = torch.concatenate([new_train_X_a_0, new_train_X_b_0, new_train_X_a_1, new_train_X_b_1])
    curated_train_y = torch.concatenate([new_train_y_a_0, new_train_y_b_0, new_train_y_a_1, new_train_y_b_1])
    return curated_train_X, curated_train_y

def subsample(X, y, size):
    perm = torch.randperm(len(y))
    sample_X = X[perm[:size]]
    sample_y = y[perm[:size]]
    return sample_X, sample_y

def compute_hessian(mu, X):
    sigm = sigmoid(X @ mu.t())
    diag_sigm = (sigm * (1 - sigm)).flatten()
    res = torch.eye(X.size(1), device=device)/penalty
    res += (X.t() * diag_sigm) @ X
    return res

def compute_score(mu0, Q0, lg0, mu1, Q1, lg1, mu2, Q2, lg2):
    Q = Q1 + Q2 - Q0
    Q_t_L = torch.linalg.cholesky(Q)
    Q_t_L_inv = torch.linalg.solve_triangular(Q_t_L, torch.eye(Q_t_L.size(0), device=device), upper=False)
    Q_inv = Q_t_L_inv.T @ Q_t_L_inv
    mu = torch.matmul(Q_inv, torch.matmul(Q1, mu1) + torch.matmul(Q2, mu2) - torch.matmul(Q0, mu0))

    lg12 = 2 * torch.sum(torch.log(torch.diagonal(Q_t_L)))

    lg = lg1+lg2-lg12-lg0

    sqr = torch.matmul(mu.T, torch.matmul(Q, mu)) - torch.matmul(mu1.T, torch.matmul(Q1, mu1)) - torch.matmul(mu2.T, torch.matmul(Q2, mu2)) + torch.matmul(mu0.T, torch.matmul(Q0, mu0))

    score = 0.5 * (lg + sqr)
    # print(lg1,lg2,lg12,lg0,sqr)
    return score.item()

def compute_data_score_err(mu_test, Q_test, test_X, test_y, train_X, train_y, lg2):
    test_N = test_y.size()[0]
    M = test_X.size()[1]

    mu0 = torch.zeros((1, M))
    mu0 = mu0.to(device)
    Q0 = torch.eye(M)/penalty
    Q0 = Q0.to(device)
    lg0 = -M * torch.log(torch.tensor(penalty))
    
    train = LogisticRegression(fit_intercept = False, C = penalty, max_iter=5000).fit(train_X.cpu(), train_y.cpu())
    # print(train.score(torch.cat([Q_x[1000:2000], Q_x[5000:6000], Q_x[9000:10000], Q_x[13000:14000]]).cpu(), torch.cat([Q_y[1000:2000], Q_y[5000:6000], Q_y[9000:10000], Q_y[13000:14000]]).cpu()))
    # print(train.score(P_x.cpu(), P_y.cpu()))
    mu_train = torch.tensor(train.coef_, dtype=torch.float32, device=device)
    # mu_train_numpy = mu_train.detach().squeeze().cpu().numpy()

    Q_train = compute_hessian(mu_train, train_X)
    Q_train_L = torch.linalg.cholesky(Q_train)
    # Q_train_inverse = Q_train_L_inv.T @ Q_train_L_inv
    # Q_train_inverse = torch.inverse(Q_train)
    # Q_numpy = Q_train_inverse.detach().cpu().numpy()

    lg1 = 2 * torch.sum(torch.log(torch.diagonal(Q_train_L)))

    score = compute_score(mu0.t(), Q0, lg0, mu_train.t(), Q_train, lg1, mu_test.t(), Q_test, lg2)

    test_y = test_y.float()
    criterion = nn.BCELoss()

    base_predictive = sigmoid(torch.matmul(test_X, mu_train.t())).squeeze()
    base_predictions = (base_predictive >= 0.5).float()
    base_loss = criterion(base_predictive, test_y)
    base_acc = (base_predictions == test_y).float().mean()
    
    return score, base_loss.item(), base_acc.item()

def get_err_score(train_data, test_X, test_y, train_number):
    test = LogisticRegression(fit_intercept = False, C = penalty, max_iter=5000).fit(test_X.cpu(), test_y.cpu())
    mu_test = torch.tensor(test.coef_, dtype=torch.float32, device=device)
    Q_test = compute_hessian(mu_test, test_X)

    L = torch.linalg.cholesky(Q_test)
    lg2 = 2 * torch.sum(torch.log(torch.diagonal(L)))

    for i in range(train_number):
        train_data[i].score, train_data[i].base_loss, train_data[i].base_acc = compute_data_score_err(mu_test, Q_test, test_X, test_y, train_data[i].X, train_data[i].y, lg2)


# def random_copy(train_data, copy_num, num_candidate, train_size):
#     new_train_data = []
#     for i in range(num_candidate):
#         perm_indices = torch.randperm(train_size)
#         sample_indices = perm_indices[:copy_num]
#         new_train_X = torch.concatenate([train_data[i].X, train_data[i].X[sample_indices]])
#         new_train_y = torch.concatenate([train_data[i].y, train_data[i].y[sample_indices]])
#         new_train_data.append(dataset(new_train_X, new_train_y, new_train_X.size()[0], 0, 0, 0, 0, 0, 0, train_data[i].noise_ratio, train_data[i].label_ratio, train_data[i].bias_ratio))
#     return new_train_data

def mimic_label_copy(train_data, num_candidate, test_ratio):
    new_train_data = []
    test_a_label_ratio = test_ratio[0]/test_ratio[2]
    test_b_label_ratio = test_ratio[1]/test_ratio[3]
    for i in range(num_candidate):
        train_size_a_0, train_size_b_0, train_size_a_1, train_size_b_1 = train_data[i].bias
        target_size_a_0 = int(train_size_a_1 * test_a_label_ratio)
        target_size_b_0 = int(train_size_b_1 * test_b_label_ratio)

        if train_size_a_0 < target_size_a_0:
            num_extra = int(target_size_a_0/train_size_a_0) - 1
            # extra_indices = torch.randint(0, train_size_a_0, (num_extra,))
            indices = torch.range(0,train_size_a_0-1)
            extra_indices = torch.tensor([])
            for g in range(num_extra):
                extra_indices = torch.cat([extra_indices, indices])
            extra_indices = extra_indices.to(torch.long)
            new_train_X_a_0 = torch.cat([train_data[i].X[:train_size_a_0], train_data[i].X[:train_size_a_0][extra_indices]])
            new_train_X_a_1 = train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]
            new_train_y_a_0 = torch.cat([train_data[i].y[:train_size_a_0], train_data[i].y[:train_size_a_0][extra_indices]])
            new_train_y_a_1 = train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]
        else:
            target_size_a_1 = int(train_size_a_0 / test_a_label_ratio)
            num_extra = int(target_size_a_1/train_size_a_1) - 1
            # extra_indices = torch.randint(0, train_size_a_1, (num_extra,))
            indices = torch.range(0,train_size_a_1-1)
            extra_indices = torch.tensor([])
            for g in range(num_extra):
                extra_indices = torch.cat([extra_indices, indices])
            extra_indices = extra_indices.to(torch.long)
            new_train_X_a_1 = torch.cat([train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1], train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][extra_indices]])
            new_train_X_a_0 = train_data[i].X[:train_size_a_0]
            new_train_y_a_1 = torch.cat([train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1], train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][extra_indices]])
            new_train_y_a_0 = train_data[i].y[:train_size_a_0]

        if train_size_b_0 < target_size_b_0:
            num_extra = int(target_size_b_0/train_size_b_0) - 1
            # extra_indices = torch.randint(0, train_size_b_0, (num_extra,))
            indices = torch.range(0,train_size_b_0-1)
            extra_indices = torch.tensor([])
            for g in range(num_extra):
                extra_indices = torch.cat([extra_indices, indices])
            extra_indices = extra_indices.to(torch.long)
            new_train_X_b_0 = torch.cat([train_data[i].X[train_size_a_0:train_size_a_0+train_size_b_0], train_data[i].X[train_size_a_0:train_size_a_0+train_size_b_0][extra_indices]])
            new_train_X_b_1 = train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:]
            new_train_y_b_0 = torch.cat([train_data[i].y[train_size_a_0:train_size_a_0+train_size_b_0], train_data[i].y[train_size_a_0:train_size_a_0+train_size_b_0][extra_indices]])
            new_train_y_b_1 = train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:]
        else:
            target_size_b_1 = int(train_size_b_0 / test_b_label_ratio)
            num_extra = int(target_size_b_1/train_size_b_1) - 1
            # extra_indices = torch.randint(0, train_size_b_1, (num_extra,))
            indices = torch.range(0,train_size_b_1-1)
            extra_indices = torch.tensor([])
            for g in range(num_extra):
                extra_indices = torch.cat([extra_indices, indices])
            extra_indices = extra_indices.to(torch.long)
            new_train_X_b_1 = torch.cat([train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:], train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:][extra_indices]])
            new_train_X_b_0 = train_data[i].X[train_size_a_0:train_size_a_0+train_size_b_0]
            new_train_y_b_1 = torch.cat([train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:], train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:][extra_indices]])
            new_train_y_b_0 = train_data[i].y[train_size_a_0:train_size_a_0+train_size_b_0]

        new_train_X = torch.concatenate([new_train_X_a_0, new_train_X_b_0, new_train_X_a_1, new_train_X_b_1])
        new_train_y = torch.concatenate([new_train_y_a_0, new_train_y_b_0, new_train_y_a_1, new_train_y_b_1])
        new_train_data.append(dataset(new_train_X, new_train_y, new_train_X.size()[0], 0, 0, 0, train_data[i].noise, torch.tensor([new_train_X_a_0.size()[0], new_train_X_b_0.size()[0], new_train_X_a_1.size()[0], new_train_X_b_1.size()[0]]).to(device), [], []))
        # print(torch.tensor([new_train_X_a_0.size(), new_train_X_b_0.size(), new_train_X_a_1.size(), new_train_X_b_1.size()]).to(device))
    return new_train_data

def mimic_label_delete(train_data, num_candidate, test_ratio):
    new_train_data = []
    test_a_label_ratio = test_ratio[0]/test_ratio[2]
    test_b_label_ratio = test_ratio[1]/test_ratio[3]
    for i in range(num_candidate):
        train_size_a_0, train_size_b_0, train_size_a_1, train_size_b_1 = train_data[i].bias
        target_size_a_0 = int(train_size_a_1 * test_a_label_ratio)
        target_size_b_0 = int(train_size_b_1 * test_b_label_ratio)

        if train_size_a_0 > target_size_a_0:
            residual_indices = torch.randperm(train_size_a_0)[:target_size_a_0]
            new_train_X_a_0 = train_data[i].X[:train_size_a_0][residual_indices]
            new_train_X_a_1 = train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]
            new_train_y_a_0 = train_data[i].y[:train_size_a_0][residual_indices]
            new_train_y_a_1 = train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]
        else:
            target_size_a_1 = int(train_size_a_0 / test_a_label_ratio)
            residual_indices = torch.randperm(train_size_a_1)[:target_size_a_1]
            new_train_X_a_1 = train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][residual_indices]
            new_train_X_a_0 = train_data[i].X[:train_size_a_0]
            new_train_y_a_1 = train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][residual_indices]
            new_train_y_a_0 = train_data[i].y[:train_size_a_0]

        if train_size_b_0 > target_size_b_0:
            residual_indices = torch.randperm(train_size_b_0)[:target_size_b_0]
            new_train_X_b_0 = train_data[i].X[train_size_a_0:train_size_a_0+train_size_b_0][residual_indices]
            new_train_X_b_1 = train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:]
            new_train_y_b_0 = train_data[i].y[train_size_a_0:train_size_a_0+train_size_b_0][residual_indices]
            new_train_y_b_1 = train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:]
        else:
            target_size_b_1 = int(train_size_b_0 / test_b_label_ratio)
            residual_indices = torch.randperm(train_size_b_1)[:target_size_b_1]
            new_train_X_b_1 = train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:][residual_indices]
            new_train_X_b_0 = train_data[i].X[train_size_a_0:train_size_a_0+train_size_b_0]
            new_train_y_b_1 = train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:][residual_indices]
            new_train_y_b_0 = train_data[i].y[train_size_a_0:train_size_a_0+train_size_b_0]

        new_train_X = torch.concatenate([new_train_X_a_0, new_train_X_b_0, new_train_X_a_1, new_train_X_b_1])
        new_train_y = torch.concatenate([new_train_y_a_0, new_train_y_b_0, new_train_y_a_1, new_train_y_b_1])
        new_train_data.append(dataset(new_train_X, new_train_y, new_train_X.size()[0], 0, 0, 0, train_data[i].noise, torch.tensor([new_train_X_a_0.size()[0], new_train_X_b_0.size()[0], new_train_X_a_1.size()[0], new_train_X_b_1.size()[0]]).to(device), [], []))
        # print(torch.tensor([new_train_X_a_0.size()[0], new_train_X_b_0.size()[0], new_train_X_a_1.size()[0], new_train_X_b_1.size()[0]]).to(device))
    return new_train_data

def mimic_bias_copy(train_data, num_candidate, test_ratio):
    new_train_data = []
    test_0_bias_ratio = test_ratio[0]/test_ratio[1]
    test_1_bias_ratio = test_ratio[2]/test_ratio[3]
    for i in range(num_candidate):
        train_size_a_0, train_size_b_0, train_size_a_1, train_size_b_1 = train_data[i].bias
        target_size_a_0 = int(train_size_b_0 * test_0_bias_ratio)
        target_size_a_1 = int(train_size_b_1 * test_1_bias_ratio)

        if train_size_a_0 < target_size_a_0:
            num_extra = int(target_size_a_0/train_size_a_0) - 1
            # extra_indices = torch.randint(0, train_size_a_0, (num_extra,))
            indices = torch.range(0,train_size_a_0-1)
            extra_indices = torch.tensor([])
            for g in range(num_extra):
                extra_indices = torch.cat([extra_indices, indices])
            extra_indices = extra_indices.to(torch.long)
            new_train_X_a_0 = torch.cat([train_data[i].X[:train_size_a_0], train_data[i].X[:train_size_a_0][extra_indices]])
            new_train_X_b_0 = train_data[i].X[train_size_a_0:train_size_a_0+train_size_b_0]
            new_train_y_a_0 = torch.cat([train_data[i].y[:train_size_a_0], train_data[i].y[:train_size_a_0][extra_indices]])
            new_train_y_b_0 = train_data[i].y[train_size_a_0:train_size_a_0+train_size_b_0]
        else:
            target_size_b_0 = int(train_size_a_0 / test_0_bias_ratio)
            num_extra = int(target_size_b_0/train_size_b_0) - 1
            # extra_indices = torch.randint(0, train_size_b_0, (num_extra,))
            indices = torch.range(0,train_size_b_0-1)
            extra_indices = torch.tensor([])
            for g in range(num_extra):
                extra_indices = torch.cat([extra_indices, indices])
            extra_indices = extra_indices.to(torch.long)
            new_train_X_b_0 = torch.cat([train_data[i].X[train_size_a_0:train_size_a_0+train_size_b_0], train_data[i].X[train_size_a_0:train_size_a_0+train_size_b_0][extra_indices]])
            new_train_X_a_0 = train_data[i].X[:train_size_a_0]
            new_train_y_b_0 = torch.cat([train_data[i].y[train_size_a_0:train_size_a_0+train_size_b_0], train_data[i].y[train_size_a_0:train_size_a_0+train_size_b_0][extra_indices]])
            new_train_y_a_0 = train_data[i].y[:train_size_a_0]

        if train_size_a_1 < target_size_a_1:
            num_extra = int(target_size_a_1/train_size_a_1) - 1
            # extra_indices = torch.randint(0, train_size_a_1, (num_extra,))
            indices = torch.range(0,train_size_a_1-1)
            extra_indices = torch.tensor([])
            for g in range(num_extra):
                extra_indices = torch.cat([extra_indices, indices])
            extra_indices = extra_indices.to(torch.long)
            new_train_X_a_1 = torch.cat([train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1], train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][extra_indices]])
            new_train_X_b_1 = train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:]
            new_train_y_a_1 = torch.cat([train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1], train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][extra_indices]])
            new_train_y_b_1 = train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:]
        else:
            target_size_b_1 = int(train_size_a_1 / test_1_bias_ratio)
            num_extra = int(target_size_b_1/train_size_b_1) - 1
            # extra_indices = torch.randint(0, train_size_b_1, (num_extra,))
            indices = torch.range(0,train_size_b_1-1)
            extra_indices = torch.tensor([])
            for g in range(num_extra):
                extra_indices = torch.cat([extra_indices, indices])
            extra_indices = extra_indices.to(torch.long)
            new_train_X_b_1 = torch.cat([train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:], train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:][extra_indices]])
            new_train_X_a_1 = train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]
            new_train_y_b_1 = torch.cat([train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:], train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:][extra_indices]])
            new_train_y_a_1 = train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]

        new_train_X = torch.concatenate([new_train_X_a_0, new_train_X_b_0, new_train_X_a_1, new_train_X_b_1])
        new_train_y = torch.concatenate([new_train_y_a_0, new_train_y_b_0, new_train_y_a_1, new_train_y_b_1])
        new_train_data.append(dataset(new_train_X, new_train_y, new_train_X.size()[0], 0, 0, 0, train_data[i].noise, torch.tensor([new_train_X_a_0.size()[0], new_train_X_b_0.size()[0], new_train_X_a_1.size()[0], new_train_X_b_1.size()[0]]).to(device), [], []))
        # print(torch.tensor([new_train_X_a_0.size()[0], new_train_X_b_0.size()[0], new_train_X_a_1.size()[0], new_train_X_b_1.size()[0]]).to(device))
    return new_train_data

def mimic_bias_delete(train_data, num_candidate, test_ratio):
    new_train_data = []
    test_0_bias_ratio = test_ratio[0]/test_ratio[1]
    test_1_bias_ratio = test_ratio[2]/test_ratio[3]
    for i in range(num_candidate):
        train_size_a_0, train_size_b_0, train_size_a_1, train_size_b_1 = train_data[i].bias
        target_size_a_0 = int(train_size_b_0 * test_0_bias_ratio)
        target_size_a_1 = int(train_size_b_1 * test_1_bias_ratio)

        if train_size_a_0 > target_size_a_0:
            residual_indices = torch.randperm(train_size_a_0)[:target_size_a_0]
            new_train_X_a_0 = train_data[i].X[:train_size_a_0][residual_indices]
            new_train_X_b_0 = train_data[i].X[train_size_a_0:train_size_a_0+train_size_b_0]
            new_train_y_a_0 = train_data[i].y[:train_size_a_0][residual_indices]
            new_train_y_b_0 = train_data[i].y[train_size_a_0:train_size_a_0+train_size_b_0]
        else:
            target_size_b_0 = int(train_size_a_0 / test_0_bias_ratio)
            residual_indices = torch.randperm(train_size_b_0)[:target_size_b_0]
            new_train_X_b_0 = train_data[i].X[train_size_a_0:train_size_a_0+train_size_b_0][residual_indices]
            new_train_X_a_0 = train_data[i].X[:train_size_a_0]
            new_train_y_b_0 = train_data[i].y[train_size_a_0:train_size_a_0+train_size_b_0][residual_indices]
            new_train_y_a_0 = train_data[i].y[:train_size_a_0]

        if train_size_a_1 > target_size_a_1:
            residual_indices = torch.randperm(train_size_a_1)[:target_size_a_1]
            new_train_X_a_1 = train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][residual_indices]
            new_train_X_b_1 = train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:]
            new_train_y_a_1 = train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][residual_indices]
            new_train_y_b_1 = train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:]
        else:
            target_size_b_1 = int(train_size_a_1 / test_1_bias_ratio)
            residual_indices = torch.randperm(train_size_b_1)[:target_size_b_1]
            new_train_X_b_1 = train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:][residual_indices]
            new_train_X_a_1 = train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]
            new_train_y_b_1 = train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:][residual_indices]
            new_train_y_a_1 = train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]

        new_train_X = torch.concatenate([new_train_X_a_0, new_train_X_b_0, new_train_X_a_1, new_train_X_b_1])
        new_train_y = torch.concatenate([new_train_y_a_0, new_train_y_b_0, new_train_y_a_1, new_train_y_b_1])
        new_train_data.append(dataset(new_train_X, new_train_y, new_train_X.size()[0], 0, 0, 0, train_data[i].noise, torch.tensor([new_train_X_a_0.size()[0], new_train_X_b_0.size()[0], new_train_X_a_1.size()[0], new_train_X_b_1.size()[0]]).to(device), [], []))
        # print(torch.tensor([new_train_X_a_0.size()[0], new_train_X_b_0.size()[0], new_train_X_a_1.size()[0], new_train_X_b_1.size()[0]]).to(device))
    return new_train_data

def data_denoise(train_data, num_candidate, ratio = 0.5):
    new_train_data = []
    for i in range(num_candidate):
        train_size_a_0, train_size_b_0, train_size_a_1, train_size_b_1 = train_data[i].bias
        noise = train_data[i].noise
        new_train_X_a_0 = train_data[i].X[:train_size_a_0][int(ratio*noise*train_size_a_0):]
        new_train_X_b_0 = train_data[i].X[train_size_a_0:train_size_a_0+train_size_b_0][int(ratio*noise*train_size_b_0):]
        new_train_X_a_1 = train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][int(ratio*noise*train_size_a_1):]
        new_train_X_b_1 = train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:][int(ratio*noise*train_size_b_1):]
        new_train_y_a_0 = train_data[i].y[:train_size_a_0][int(ratio*noise*train_size_a_0):]
        new_train_y_b_0 = train_data[i].y[train_size_a_0:train_size_a_0+train_size_b_0][int(ratio*noise*train_size_b_0):]
        new_train_y_a_1 = train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][int(ratio*noise*train_size_a_1):]
        new_train_y_b_1 = train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:][int(ratio*noise*train_size_b_1):]
        new_train_X = torch.concatenate([new_train_X_a_0, new_train_X_b_0, new_train_X_a_1, new_train_X_b_1])
        new_train_y = torch.concatenate([new_train_y_a_0, new_train_y_b_0, new_train_y_a_1, new_train_y_b_1])
        new_train_data.append(dataset(new_train_X, new_train_y, new_train_X.size()[0], 0, 0, 0, train_data[i].noise*(1-ratio), torch.tensor([new_train_X_a_0.size()[0], new_train_X_b_0.size()[0], new_train_X_a_1.size()[0], new_train_X_b_1.size()[0]]).to(device), [], []))
        # print(torch.tensor([new_train_X_a_0.size()[0], new_train_X_b_0.size()[0], new_train_X_a_1.size()[0], new_train_X_b_1.size()[0]]).to(device))
    return new_train_data

T = 1
D = 3000
train_size_a_0 = args.train_size_a_0
train_size_a_1 = args.train_size_a_1
train_size_b_0 = args.train_size_b_0
train_size_b_1 = args.train_size_b_1
num_candidate = 1
test_size_a_0 = args.test_size_a_0
test_size_a_1 = args.test_size_a_1
test_size_b_0 = args.test_size_b_0
test_size_b_1 = args.test_size_b_1
noise_levels = [10]

# with open("output_distribution_copy.txt", "a") as file:
#     file.write("train size: {}, {}, {}, {}; test size: {}, {}, {}, {} \n".format(
#             train_size_a_0, train_size_a_1, train_size_b_0, train_size_b_1,
#             test_size_a_0, test_size_a_1, test_size_b_0, test_size_b_1,
#         ))
#     file.write("{} ".format(penalty))

criterion = nn.BCELoss()

test_label_ratio = (test_size_a_0 + test_size_b_0)/(test_size_a_1 + test_size_b_1)
test_bias_ratio = (test_size_a_0 + test_size_a_1)/(test_size_b_0 + test_size_b_1)
test_size_all = test_size_a_0 + test_size_a_1 + test_size_b_0 + test_size_b_1
train_size_all = train_size_a_0 + train_size_b_0 + train_size_a_1 + train_size_b_1
test_ratio = (test_size_a_0, test_size_b_0, test_size_a_1, test_size_b_1)
train_ratio = (train_size_a_0, train_size_b_0, train_size_a_1, train_size_b_1)
for noise_level in noise_levels:
    copy_score_change = []
    copy_loss_change = []
    copy_acc_change = []
    delete_score_change = []
    delete_loss_change = []
    delete_acc_change = []
    denoise_score_change = []
    denoise_loss_change = []
    denoise_acc_change = []
    for _ in range(10):
        new_train_X, new_train_y, new_test_X, new_test_y, new_train_ratio = generate_train_cifar10_distribution(train_ratio, test_ratio, noise_level)
        pre_score = 0
        copy_score = 0
        delete_score = 0
        denoise_score = 0
        pre_loss = 0
        copy_loss = 0
        delete_loss = 0
        denoise_loss = 0
        pre_acc = 0
        copy_acc = 0
        delete_acc = 0
        denoise_acc = 0
        for d in tqdm(range(D)):
            copy_train_X, copy_train_y = generate_train_cifar10_distribution_copy(new_train_X, new_train_y, new_train_ratio, test_ratio)
            delete_train_X, delete_train_y = generate_train_cifar10_distribution_delete(new_train_X, new_train_y, new_train_ratio, test_ratio)
            denoise_train_X, denoise_train_y = generate_train_cifar10_distribution_denoise(new_train_X, new_train_y, new_train_ratio, noise_level)

            # print(new_train_X.size(), new_train_y.size(), new_test_X.size(), new_test_y.size(), copy_train_data_X.size(), copy_train_data_y.size(), delete_train_data_X.size(), delete_train_data_y.size(), denoise_train_data_X.size(), denoise_train_data_y.size())
            # sys.exit()

            # sample_test_X_a_0, sample_test_y_a_0 = subsample(images_a_0_embedding[4000:], torch.zeros(1000).to(device), test_size_a_0)
            # sample_test_X_a_1, sample_test_y_a_1 = subsample(images_a_1_embedding[4000:], torch.ones(1000).to(device), test_size_a_1)
            # sample_test_X_b_0, sample_test_y_b_0 = subsample(images_b_0_embedding[4000:], torch.zeros(1000).to(device), test_size_b_0)
            # sample_test_X_b_1, sample_test_y_b_1 = subsample(images_b_1_embedding[4000:], torch.ones(1000).to(device), test_size_b_1)
            # sample_test_X = torch.concatenate([sample_test_X_a_0, sample_test_X_b_0, sample_test_X_a_1, sample_test_X_b_1])
            # sample_test_y = torch.concatenate([sample_test_y_a_0, sample_test_y_b_0, sample_test_y_a_1, sample_test_y_b_1])

            sample_test_X, sample_test_y = subsample(new_test_X, new_test_y, test_size_all)

            train_data = generate_train_cifar10(new_train_X, new_train_y, num_candidate, train_size_all, noise_level)
            copy_train_data = generate_train_cifar10(copy_train_X, copy_train_y, num_candidate, train_size_all, noise_level)
            delete_train_data = generate_train_cifar10(delete_train_X, delete_train_y, num_candidate, train_size_all, noise_level)
            denoise_train_data = generate_train_cifar10(denoise_train_X, denoise_train_y, num_candidate, train_size_all, noise_level)

            get_err_score(train_data, sample_test_X, sample_test_y, num_candidate)
            get_err_score(copy_train_data, sample_test_X, sample_test_y, num_candidate)
            get_err_score(delete_train_data, sample_test_X, sample_test_y, num_candidate)
            get_err_score(denoise_train_data, sample_test_X, sample_test_y, num_candidate)

            for i in range(num_candidate):
                pre_score += train_data[i].score
                copy_score += copy_train_data[i].score
                delete_score += delete_train_data[i].score
                denoise_score += denoise_train_data[i].score
                pre_loss += train_data[i].base_loss
                copy_loss += copy_train_data[i].base_loss
                delete_loss += delete_train_data[i].base_loss
                denoise_loss += denoise_train_data[i].base_loss
                pre_acc += train_data[i].base_acc
                copy_acc += copy_train_data[i].base_acc
                delete_acc += delete_train_data[i].base_acc
                denoise_acc += denoise_train_data[i].base_acc
        pre_score /= (T*D*num_candidate)
        copy_score /= (T*D*num_candidate)
        delete_score /= (T*D*num_candidate)
        denoise_score /= (T*D*num_candidate)
        pre_loss /= (T*D*num_candidate)
        copy_loss /= (T*D*num_candidate)
        delete_loss /= (T*D*num_candidate)
        denoise_loss /= (T*D*num_candidate)
        pre_acc /= (T*D*num_candidate)
        copy_acc /= (T*D*num_candidate)
        delete_acc /= (T*D*num_candidate)
        denoise_acc /= (T*D*num_candidate)
        copy_score_change.append(copy_score-pre_score)
        copy_loss_change.append(copy_loss-pre_loss)
        copy_acc_change.append(copy_acc-pre_acc)
        delete_score_change.append(delete_score-pre_score)
        delete_loss_change.append(delete_loss-pre_loss)
        delete_acc_change.append(delete_acc-pre_acc)
        denoise_score_change.append(denoise_score-pre_score)
        denoise_loss_change.append(denoise_loss-pre_loss)
        denoise_acc_change.append(denoise_acc-pre_acc)

    copy_score_mean = np.array(copy_score_change).mean()
    copy_score_std = np.array(copy_score_change).std()
    copy_loss_mean = np.array(copy_loss_change).mean()
    copy_loss_std = np.array(copy_loss_change).std()
    copy_acc_mean = np.array(copy_acc_change).mean()
    copy_acc_std = np.array(copy_acc_change).std()
    delete_score_mean = np.array(delete_score_change).mean()
    delete_score_std = np.array(delete_score_change).std()
    delete_loss_mean = np.array(delete_loss_change).mean()
    delete_loss_std = np.array(delete_loss_change).std()
    delete_acc_mean = np.array(delete_acc_change).mean()
    delete_acc_std = np.array(delete_acc_change).std()
    denoise_score_mean = np.array(denoise_score_change).mean()
    denoise_score_std = np.array(denoise_score_change).std()
    denoise_loss_mean = np.array(denoise_loss_change).mean()
    denoise_loss_std = np.array(denoise_loss_change).std()
    denoise_acc_mean = np.array(denoise_acc_change).mean()
    denoise_acc_std = np.array(denoise_acc_change).std()

    with open("output_distribution.txt", "a") as file:
        file.write("{} & {}\\% & Copy & {:.4f}\\pm{:.4f} & {:.4f}\\pm{:.4f} & {:.4f}\\pm{:.4f} \\\\\n".format(
            penalty, noise_level, copy_score_mean, copy_score_std, copy_loss_mean, copy_loss_std, copy_acc_mean, copy_acc_std
        ))
        file.write("{} & {}\\% & Delete & {:.4f}\\pm{:.4f} & {:.4f}\\pm{:.4f} & {:.4f}\\pm{:.4f} \\\\\n".format(
            penalty, noise_level, delete_score_mean, delete_score_std, delete_loss_mean, delete_loss_std, delete_acc_mean, delete_acc_std
        ))
        file.write("{} & {}\\% & Denoise & {:.4f}\\pm{:.4f} & {:.4f}\\pm{:.4f} & {:.4f}\\pm{:.4f} \\\\\n".format(
            penalty, noise_level, denoise_score_mean, denoise_score_std, denoise_loss_mean, denoise_loss_std, denoise_acc_mean, denoise_acc_std
        ))