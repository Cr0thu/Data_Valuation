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
    def __init__(self, X, y, num, score, base_loss, base_acc, noise, bias, bias_loss, bias_acc):
        self.X = X
        self.y = y
        self.num = num
        self.score = score
        self.base_loss = base_loss
        self.base_acc = base_acc
        self.noise = noise
        self.bias = bias
        self.bias_loss = bias_loss
        self.bias_acc = bias_acc

def sigmoid(z):
    return 1/(1 + torch.exp(-z))

def generate_data_cifar10(indices, dataset, label, noise):
    X = dataset[indices]
    y = torch.ones(X.size()[0]) * label
    y = torch.tensor(y, dtype=torch.int)
    y = y.to(device)
    y[:int(noise*X.size()[0])] = 1 - y[:int(noise*X.size()[0])]
    # print(y)
    return X, y, X.size()[0]

# def generate_dataset_cifar10(sample_indices, noise):
    # perm_indices = torch.randperm(80000).to(device)
    # sample_indices = perm_indices[:N]
    a_0 = sample_indices[(sample_indices >= 0) & (sample_indices < 4000)]
    b_0 = sample_indices[(sample_indices >= 4000) & (sample_indices < 8000)] - 4000
    c_0 = sample_indices[(sample_indices >= 8000) & (sample_indices < 12000)] - 8000
    d_0 = sample_indices[(sample_indices >= 12000) & (sample_indices < 16000)] - 12000
    e_0 = sample_indices[(sample_indices >= 16000) & (sample_indices < 20000)] - 16000
    f_0 = sample_indices[(sample_indices >= 20000) & (sample_indices < 24000)] - 20000
    g_0 = sample_indices[(sample_indices >= 24000) & (sample_indices < 28000)] - 24000
    h_0 = sample_indices[(sample_indices >= 28000) & (sample_indices < 32000)] - 28000
    i_0 = sample_indices[(sample_indices >= 32000) & (sample_indices < 36000)] - 32000
    j_0 = sample_indices[(sample_indices >= 36000) & (sample_indices < 40000)] - 36000
    a_1 = sample_indices[(sample_indices >= 40000) & (sample_indices < 44000)] - 40000
    b_1 = sample_indices[(sample_indices >= 44000) & (sample_indices < 48000)] - 44000
    c_1 = sample_indices[(sample_indices >= 48000) & (sample_indices < 52000)] - 48000
    d_1 = sample_indices[(sample_indices >= 52000) & (sample_indices < 56000)] - 52000
    e_1 = sample_indices[(sample_indices >= 56000) & (sample_indices < 60000)] - 56000
    f_1 = sample_indices[(sample_indices >= 60000) & (sample_indices < 64000)] - 60000
    g_1 = sample_indices[(sample_indices >= 64000) & (sample_indices < 68000)] - 64000
    h_1 = sample_indices[(sample_indices >= 68000) & (sample_indices < 72000)] - 68000
    i_1 = sample_indices[(sample_indices >= 72000) & (sample_indices < 76000)] - 72000
    j_1 = sample_indices[(sample_indices >= 76000) & (sample_indices < 80000)] - 76000
    a_X_0, a_y_0, a_num_0 = generate_data_cifar10(a_0, images_a_0_embedding[:4000], 0, noise)
    a_X_1, a_y_1, a_num_1 = generate_data_cifar10(a_1, images_a_1_embedding[:4000], 1, noise)
    b_X_0, b_y_0, b_num_0 = generate_data_cifar10(b_0, images_b_0_embedding[:4000], 0, noise)
    b_X_1, b_y_1, b_num_1 = generate_data_cifar10(b_1, images_b_1_embedding[:4000], 1, noise)
    c_X_0, c_y_0, c_num_0 = generate_data_cifar10(c_0, images_c_0_embedding[:4000], 0, noise)
    c_X_1, c_y_1, c_num_1 = generate_data_cifar10(c_1, images_c_1_embedding[:4000], 1, noise)
    d_X_0, d_y_0, d_num_0 = generate_data_cifar10(d_0, images_d_0_embedding[:4000], 0, noise)
    d_X_1, d_y_1, d_num_1 = generate_data_cifar10(d_1, images_d_1_embedding[:4000], 1, noise)
    e_X_0, e_y_0, e_num_0 = generate_data_cifar10(e_0, images_e_0_embedding[:4000], 0, noise)
    e_X_1, e_y_1, e_num_1 = generate_data_cifar10(e_1, images_e_1_embedding[:4000], 1, noise)
    f_X_0, f_y_0, f_num_0 = generate_data_cifar10(f_0, images_f_0_embedding[:4000], 0, noise)
    f_X_1, f_y_1, f_num_1 = generate_data_cifar10(f_1, images_f_1_embedding[:4000], 1, noise)
    g_X_0, g_y_0, g_num_0 = generate_data_cifar10(g_0, images_g_0_embedding[:4000], 0, noise)
    g_X_1, g_y_1, g_num_1 = generate_data_cifar10(g_1, images_g_1_embedding[:4000], 1, noise)
    h_X_0, h_y_0, h_num_0 = generate_data_cifar10(h_0, images_h_0_embedding[:4000], 0, noise)
    h_X_1, h_y_1, h_num_1 = generate_data_cifar10(h_1, images_h_1_embedding[:4000], 1, noise)
    i_X_0, i_y_0, i_num_0 = generate_data_cifar10(i_0, images_i_0_embedding[:4000], 0, noise)
    i_X_1, i_y_1, i_num_1 = generate_data_cifar10(i_1, images_i_1_embedding[:4000], 1, noise)
    j_X_0, j_y_0, j_num_0 = generate_data_cifar10(j_0, images_j_0_embedding[:4000], 0, noise)
    j_X_1, j_y_1, j_num_1 = generate_data_cifar10(j_1, images_j_1_embedding[:4000], 1, noise)
    X = torch.concatenate([
        a_X_0,
        b_X_0,
        c_X_0,
        d_X_0,
        e_X_0,
        f_X_0,
        g_X_0,
        h_X_0,
        i_X_0,
        j_X_0,
        a_X_1,
        b_X_1,
        c_X_1,
        d_X_1,
        e_X_1,
        f_X_1,
        g_X_1,
        h_X_1,
        i_X_1,
        j_X_1
    ], axis = 0)
    y = torch.concatenate([
        a_y_0,
        b_y_0,
        c_y_0,
        d_y_0,
        e_y_0,
        f_y_0,
        g_y_0,
        h_y_0,
        i_y_0,
        j_y_0,
        a_y_1,
        b_y_1,
        c_y_1,
        d_y_1,
        e_y_1,
        f_y_1,
        g_y_1,
        h_y_1,
        i_y_1,
        j_y_1
    ], axis = 0)
    bias = torch.tensor([
        a_num_0+a_num_1,
        b_num_0+b_num_1,
        c_num_0+c_num_1,
        d_num_0+d_num_1,
        e_num_0+e_num_1,
        f_num_0+f_num_1,
        g_num_0+g_num_1,
        h_num_0+h_num_1,
        i_num_0+i_num_1,
        j_num_0+j_num_1
    ]).to(device)
    return bias, X, y

def generate_train_cifar10(train_size_a_0, train_size_a_1, train_size_b_0, train_size_b_1, train_number, noise_level):
    train_dataset = []
    bias = torch.tensor([
        train_size_a_0, train_size_b_0,
        train_size_a_1, train_size_b_1
    ]).to(device)
    for i in range(train_number):
        perm_indices_a_0 = torch.randperm(4000).to(device)
        sample_indices_a_0 = perm_indices_a_0[:train_size_a_0]
        perm_indices_a_1 = torch.randperm(4000).to(device)
        sample_indices_a_1 = perm_indices_a_1[:train_size_a_1]
        perm_indices_b_0 = torch.randperm(4000).to(device)
        sample_indices_b_0 = perm_indices_b_0[:train_size_b_0]
        perm_indices_b_1 = torch.randperm(4000).to(device)
        sample_indices_b_1 = perm_indices_b_1[:train_size_b_1]
        train_X_a_0, train_y_a_0, _ = generate_data_cifar10(sample_indices_a_0, images_a_0_embedding[:4000], 0, eps * noise_level)
        train_X_a_1, train_y_a_1, _ = generate_data_cifar10(sample_indices_a_1, images_a_1_embedding[:4000], 1, eps * noise_level)
        train_X_b_0, train_y_b_0, _ = generate_data_cifar10(sample_indices_b_0, images_b_0_embedding[:4000], 0, eps * noise_level)
        train_X_b_1, train_y_b_1, _ = generate_data_cifar10(sample_indices_b_1, images_b_1_embedding[:4000], 1, eps * noise_level)
        train_X = torch.concatenate([train_X_a_0, train_X_b_0, train_X_a_1, train_X_b_1])
        train_y = torch.concatenate([train_y_a_0, train_y_b_0, train_y_a_1, train_y_b_1])
        train_dataset.append(dataset(train_X, train_y, train_X.size()[0], 0, 0, 0, noise_level*eps, bias, [], []))
    return train_dataset

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

def compute_data_score_err(mu_test, Q_test, test_X, test_y, train_X, train_y, lg2, bias):
    test_N = test_y.size()[0]
    M = test_X.size()[1]
    test_size_a_0, test_size_b_0, test_size_a_1, test_size_b_1 = bias

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

    base_predictive_0 = sigmoid(torch.matmul(test_X[:test_size_a_0], mu_train.t())).squeeze()
    base_predictions_0 = (base_predictive_0 >= 0.5).float()
    base_loss_0 = criterion(base_predictive_0, test_y[:test_size_a_0])
    base_acc_0 = (base_predictions_0 == test_y[:test_size_a_0]).float().mean()

    base_predictive_1 = sigmoid(torch.matmul(test_X[test_size_a_0:test_size_a_0+test_size_b_0], mu_train.t())).squeeze()
    base_predictions_1 = (base_predictive_1 >= 0.5).float()
    base_loss_1 = criterion(base_predictive_1, test_y[test_size_a_0:test_size_a_0+test_size_b_0])
    base_acc_1 = (base_predictions_1 == test_y[test_size_a_0:test_size_a_0+test_size_b_0]).float().mean()

    base_predictive_2 = sigmoid(torch.matmul(test_X[test_size_a_0+test_size_b_0:test_size_a_0+test_size_b_0+test_size_a_1], mu_train.t())).squeeze()
    base_predictions_2 = (base_predictive_2 >= 0.5).float()
    base_loss_2 = criterion(base_predictive_2, test_y[test_size_a_0+test_size_b_0:test_size_a_0+test_size_b_0+test_size_a_1])
    base_acc_2 = (base_predictions_2 == test_y[test_size_a_0+test_size_b_0:test_size_a_0+test_size_b_0+test_size_a_1]).float().mean()

    base_predictive_3 = sigmoid(torch.matmul(test_X[test_size_a_0+test_size_b_0+test_size_a_1:], mu_train.t())).squeeze()
    base_predictions_3 = (base_predictive_3 >= 0.5).float()
    base_loss_3 = criterion(base_predictive_3, test_y[test_size_a_0+test_size_b_0+test_size_a_1:])
    base_acc_3 = (base_predictions_3 == test_y[test_size_a_0+test_size_b_0+test_size_a_1:]).float().mean()
    
    return score, base_loss.item(), base_acc.item(), np.array([base_loss_0.item(), base_loss_1.item(), base_loss_2.item(), base_loss_3.item()]), np.array([base_acc_0.item(), base_acc_1.item(), base_acc_2.item(), base_acc_3.item()])

def get_err_score(train_data, test_X, test_y, train_number, test_bias):
    test = LogisticRegression(fit_intercept = False, C = penalty, max_iter=5000).fit(test_X.cpu(), test_y.cpu())
    mu_test = torch.tensor(test.coef_, dtype=torch.float32, device=device)
    Q_test = compute_hessian(mu_test, test_X)

    L = torch.linalg.cholesky(Q_test)
    lg2 = 2 * torch.sum(torch.log(torch.diagonal(L)))

    for i in range(train_number):
        train_data[i].score, train_data[i].base_loss, train_data[i].base_acc, train_data[i].bias_loss, train_data[i].bias_acc = compute_data_score_err(mu_test, Q_test, test_X, test_y, train_data[i].X, train_data[i].y, lg2, test_bias)


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
D = 1000
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

with open("output_new.txt", "a") as file:
    file.write("train size: {}, {}, {}, {}; test size: {}, {}, {}, {} \n".format(
            train_size_a_0, train_size_a_1, train_size_b_0, train_size_b_1,
            test_size_a_0, test_size_a_1, test_size_b_0, test_size_b_1,
        ))
    file.write("{} ".format(penalty))

criterion = nn.BCELoss()

test_label_ratio = (test_size_a_0 + test_size_b_0)/(test_size_a_1 + test_size_b_1)
test_bias_ratio = (test_size_a_0 + test_size_a_1)/(test_size_b_0 + test_size_b_1)
test_size = test_size_a_0 + test_size_a_1 + test_size_b_0 + test_size_b_1
test_ratio = (test_size_a_0, test_size_b_0, test_size_a_1, test_size_b_1)
for noise_level in noise_levels:
    pre_score = 0
    mimic_label_copy_score = 0
    mimic_label_delete_socre = 0
    mimic_bias_copy_score = 0
    mimic_bias_delete_score = 0
    data_denoise_score = 0
    pre_loss = 0
    mimic_label_copy_loss = 0
    mimic_label_delete_loss = 0
    mimic_bias_copy_loss = 0
    mimic_bias_delete_loss = 0
    data_denoise_loss = 0
    pre_acc = 0
    mimic_label_copy_acc = 0
    mimic_label_delete_acc = 0
    mimic_bias_copy_acc = 0
    mimic_bias_delete_acc = 0
    data_denoise_acc = 0
    pre_bias_loss = np.array([0.,0.,0.,0.])
    mimic_label_copy_bias_loss = np.array([0.,0.,0.,0.])
    mimic_label_delete_bias_loss = np.array([0.,0.,0.,0.])
    mimic_bias_copy_bias_loss = np.array([0.,0.,0.,0.])
    mimic_bias_delete_bias_loss = np.array([0.,0.,0.,0.])
    data_denoise_bias_loss = np.array([0.,0.,0.,0.])
    pre_bias_acc = np.array([0.,0.,0.,0.])
    mimic_label_copy_bias_acc = np.array([0.,0.,0.,0.])
    mimic_label_delete_bias_acc = np.array([0.,0.,0.,0.])
    mimic_bias_copy_bias_acc = np.array([0.,0.,0.,0.])
    mimic_bias_delete_bias_acc = np.array([0.,0.,0.,0.])
    data_denoise_bias_acc = np.array([0.,0.,0.,0.])
    for d in tqdm(range(D)):
        test_X = P_x
        test_y = P_y
        # penalty = 10000
    #     best_penalty = 1
    #     best_loss = 100
    #     for c in range(1, 11):
    #         loss = 0
    #         for i in range(30):
    #             sub_test_X, sub_test_y = subsample(Q_x, Q_y, 200)
    #             test = LogisticRegression(fit_intercept = False, C = c, max_iter=50000).fit(sub_test_X.cpu(), sub_test_y.cpu())
    #             # acc += test.score(test_X.cpu(), test_y.cpu())
    #             mu = torch.tensor(test.coef_, dtype=torch.float32, device=device)
    #             predictive = sigmoid(torch.matmul(test_X, mu.t())).squeeze()
    #             predictions = (predictive >= 0.5).float()
    #             loss += criterion(predictive, test_y)
    #         if loss < best_loss:
    #             penalty = c
    #             best_loss = loss
    #         print(loss)
    #     print(penalty)
    #     for t in range(30):
    #         loss_1 = 0
    #         loss_2 = 0
    #         for i in range(30):
    #             sub_test_X, sub_test_y = subsample(Q_x, Q_y, 200)
    #             test = LogisticRegression(fit_intercept = False, C = penalty+0.1, max_iter=50000).fit(sub_test_X.cpu(), sub_test_y.cpu())
    #             # acc_1 += test.score(test_X.cpu(), test_y.cpu())
    #             mu = torch.tensor(test.coef_, dtype=torch.float32, device=device)
    #             predictive = sigmoid(torch.matmul(test_X, mu.t())).squeeze()
    #             predictions = (predictive >= 0.5).float()
    #             loss_1 += criterion(predictive, test_y)
    #         for i in range(30):
    #             sub_test_X, sub_test_y = subsample(Q_x, Q_y, 200)
    #             test = LogisticRegression(fit_intercept = False, C = penalty-0.1, max_iter=50000).fit(sub_test_X.cpu(), sub_test_y.cpu())
    #             # acc_2 += test.score(test_X.cpu(), test_y.cpu())
    #             mu = torch.tensor(test.coef_, dtype=torch.float32, device=device)
    #             predictive = sigmoid(torch.matmul(test_X, mu.t())).squeeze()
    #             predictions = (predictive >= 0.5).float()
    #             loss_2 += criterion(predictive, test_y)
    #         # test = LogisticRegression(fit_intercept = False, C = penalty, max_iter=50000).fit(Q_x.cpu(), Q_y.cpu())
    #         # acc_1 = test.score(test_X.cpu(), test_y.cpu())
    #         # test = LogisticRegression(fit_intercept = False, C = penalty-0.1, max_iter=50000).fit(Q_x.cpu(), Q_y.cpu())
    #         # acc_2 = test.score(test_X.cpu(), test_y.cpu())
    #         if loss_1 < best_loss:
    #             best_loss = loss_1
    #             best_penalty = penalty+0.1
    #         if loss_2 < best_loss:
    #             best_loss = loss_2
    #             best_penalty = penalty-0.1
    #         penalty = penalty - (loss_1.item()-loss_2.item())/8
    #         print(loss_1, loss_2, penalty)
    #     # print(acc)
    #     print(best_penalty, best_loss, penalty)
    #     sys.exit()

        train_data = generate_train_cifar10(train_size_a_0, train_size_a_1, train_size_b_0, train_size_b_1, num_candidate, noise_level)
        mimic_label_copy_train_data = mimic_label_copy(train_data, num_candidate, test_ratio)
        mimic_label_delete_train_data = mimic_label_delete(train_data, num_candidate, test_ratio)
        mimic_bias_copy_train_data = mimic_bias_copy(train_data, num_candidate, test_ratio)
        mimic_bias_delete_train_data = mimic_bias_delete(train_data, num_candidate, test_ratio)
        data_denoise_train_data = data_denoise(train_data, num_candidate, ratio=1)

        # for t in range(T):
        sample_test_X_a_0, sample_test_y_a_0 = subsample(images_a_0_embedding[4000:], torch.zeros(1000).to(device), test_size_a_0)
        sample_test_X_a_1, sample_test_y_a_1 = subsample(images_a_1_embedding[4000:], torch.ones(1000).to(device), test_size_a_1)
        sample_test_X_b_0, sample_test_y_b_0 = subsample(images_b_0_embedding[4000:], torch.zeros(1000).to(device), test_size_b_0)
        sample_test_X_b_1, sample_test_y_b_1 = subsample(images_b_1_embedding[4000:], torch.ones(1000).to(device), test_size_b_1)
        sample_test_X = torch.concatenate([sample_test_X_a_0, sample_test_X_b_0, sample_test_X_a_1, sample_test_X_b_1])
        sample_test_y = torch.concatenate([sample_test_y_a_0, sample_test_y_b_0, sample_test_y_a_1, sample_test_y_b_1])

        get_err_score(train_data, sample_test_X, sample_test_y, num_candidate, test_ratio)
        get_err_score(mimic_label_copy_train_data, sample_test_X, sample_test_y, num_candidate, test_ratio)
        get_err_score(mimic_label_delete_train_data, sample_test_X, sample_test_y, num_candidate, test_ratio)
        get_err_score(mimic_bias_copy_train_data, sample_test_X, sample_test_y, num_candidate, test_ratio)
        get_err_score(mimic_bias_delete_train_data, sample_test_X, sample_test_y, num_candidate, test_ratio)
        get_err_score(data_denoise_train_data, sample_test_X, sample_test_y, num_candidate, test_ratio)

        for i in range(num_candidate):
            pre_score += train_data[i].score
            mimic_label_copy_score += mimic_label_copy_train_data[i].score
            mimic_label_delete_socre += mimic_label_delete_train_data[i].score
            mimic_bias_copy_score += mimic_bias_copy_train_data[i].score
            mimic_bias_delete_score += mimic_bias_delete_train_data[i].score
            data_denoise_score += data_denoise_train_data[i].score
            pre_loss += train_data[i].base_loss
            mimic_label_copy_loss += mimic_label_copy_train_data[i].base_loss
            mimic_label_delete_loss += mimic_label_delete_train_data[i].base_loss
            mimic_bias_copy_loss += mimic_bias_copy_train_data[i].base_loss
            mimic_bias_delete_loss += mimic_bias_delete_train_data[i].base_loss
            data_denoise_loss += data_denoise_train_data[i].base_loss
            pre_acc += train_data[i].base_acc
            mimic_label_copy_acc += mimic_label_copy_train_data[i].base_acc
            mimic_label_delete_acc += mimic_label_delete_train_data[i].base_acc
            mimic_bias_copy_acc += mimic_bias_copy_train_data[i].base_acc
            mimic_bias_delete_acc += mimic_bias_delete_train_data[i].base_acc
            data_denoise_acc += data_denoise_train_data[i].base_acc
            pre_bias_loss += train_data[i].bias_loss
            mimic_label_copy_bias_loss += mimic_label_copy_train_data[i].bias_loss
            mimic_label_delete_bias_loss += mimic_label_delete_train_data[i].bias_loss
            mimic_bias_copy_bias_loss += mimic_bias_copy_train_data[i].bias_loss
            mimic_bias_delete_bias_loss += mimic_bias_delete_train_data[i].bias_loss
            data_denoise_bias_loss += data_denoise_train_data[i].bias_loss
            pre_bias_acc += train_data[i].bias_acc
            mimic_label_copy_bias_acc += mimic_label_copy_train_data[i].bias_acc
            mimic_label_delete_bias_acc += mimic_label_delete_train_data[i].bias_acc
            mimic_bias_copy_bias_acc += mimic_bias_copy_train_data[i].bias_acc
            mimic_bias_delete_bias_acc += mimic_bias_delete_train_data[i].bias_acc
            data_denoise_bias_acc += data_denoise_train_data[i].bias_acc
    pre_score /= (T*D*num_candidate)
    mimic_label_copy_score /= (T*D*num_candidate)
    mimic_label_delete_socre /= (T*D*num_candidate)
    mimic_bias_copy_score /= (T*D*num_candidate)
    mimic_bias_delete_score /= (T*D*num_candidate)
    data_denoise_score /= (T*D*num_candidate)
    pre_loss /= (T*D*num_candidate)
    mimic_label_copy_loss /= (T*D*num_candidate)
    mimic_label_delete_loss /= (T*D*num_candidate)
    mimic_bias_copy_loss /= (T*D*num_candidate)
    mimic_bias_delete_loss /= (T*D*num_candidate)
    data_denoise_loss /= (T*D*num_candidate)
    pre_acc /= (T*D*num_candidate)
    mimic_label_copy_acc /= (T*D*num_candidate)
    mimic_label_delete_acc /= (T*D*num_candidate)
    mimic_bias_copy_acc /= (T*D*num_candidate)
    mimic_bias_delete_acc /= (T*D*num_candidate)
    data_denoise_acc /= (T*D*num_candidate)
    pre_bias_loss /= (T*D*num_candidate)
    mimic_label_copy_bias_loss /= (T*D*num_candidate)
    mimic_label_delete_bias_loss /= (T*D*num_candidate)
    mimic_bias_copy_bias_loss /= (T*D*num_candidate)
    mimic_bias_delete_bias_loss /= (T*D*num_candidate)
    data_denoise_bias_loss /= (T*D*num_candidate)
    pre_bias_acc /= (T*D*num_candidate)
    mimic_label_copy_bias_acc /= (T*D*num_candidate)
    mimic_label_delete_bias_acc /= (T*D*num_candidate)
    mimic_bias_copy_bias_acc /= (T*D*num_candidate)
    mimic_bias_delete_bias_acc /= (T*D*num_candidate)
    data_denoise_bias_acc /= (T*D*num_candidate)
    # print("original score: ", '%.4f'%pre_score, ", mimic label copy: ", '%.4f'%(mimic_label_copy_score - pre_score), ", mimic label delete: ", '%.4f'%(mimic_label_delete_socre - pre_score), ", mimic bias copy: ", '%.4f'%(mimic_bias_copy_score - pre_score), ", mimic bias delete: ", '%.4f'%(mimic_bias_delete_score - pre_score), ", data denoise: ", '%.4f'%(data_denoise_score - pre_score))
    # print("original loss: ", '%.4f'%pre_loss, ", mimic label copy: ", '%.4f'%(mimic_label_copy_loss - pre_loss), ", mimic label delete: ", '%.4f'%(mimic_label_delete_loss - pre_loss), ", mimic bias copy: ", '%.4f'%(mimic_bias_copy_loss - pre_loss), ", mimic bias delete: ", '%.4f'%(mimic_bias_delete_loss - pre_loss), ", data denoise: ", '%.4f'%(data_denoise_loss - pre_loss))
    # print("original acc: ", '%.4f'%pre_acc, ", mimic label copy: ", '%.4f'%(mimic_label_copy_acc - pre_acc), ", mimic label delete: ", '%.4f'%(mimic_label_delete_acc - pre_acc), ", mimic bias copy: ", '%.4f'%(mimic_bias_copy_acc - pre_acc), ", mimic bias delete: ", '%.4f'%(mimic_bias_delete_acc - pre_acc), ", data denoise: ", '%.4f'%(data_denoise_acc - pre_acc))
    # print("&", "{}\\%".format(noise_level),"&", "PMI", "&", '%.4f'%pre_score, "&", '%.4f'%(mimic_label_copy_score - pre_score), "&", '%.4f'%(mimic_label_delete_socre - pre_score), "&", '%.4f'%(mimic_bias_copy_score - pre_score), "&", '%.4f'%(mimic_bias_delete_score - pre_score), "&", '%.4f'%(data_denoise_score - pre_score), "\\\\")
    # print("&", "&", "Loss", "&", '%.4f'%pre_loss, "&", '%.4f'%(mimic_label_copy_loss - pre_loss), "&", '%.4f'%(mimic_label_delete_loss - pre_loss), "&", '%.4f'%(mimic_bias_copy_loss - pre_loss), "&", '%.4f'%(mimic_bias_delete_loss - pre_loss), "&", '%.4f'%(data_denoise_loss - pre_loss), "\\\\")
    # print("&", "&", "Acc", "&", '%.4f'%pre_acc, "&", '%.4f'%(mimic_label_copy_acc - pre_acc), "&", '%.4f'%(mimic_label_delete_acc - pre_acc), "&", '%.4f'%(mimic_bias_copy_acc - pre_acc), "&", '%.4f'%(mimic_bias_delete_acc - pre_acc), "&", '%.4f'%(data_denoise_acc - pre_acc), "\\\\")
    # Writing to a file
    with open("output_new.txt", "a") as file:
        file.write("& {}\\% & PMI & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\\n".format(
            noise_level, pre_score, mimic_label_copy_score - pre_score, mimic_label_delete_socre - pre_score,
            mimic_bias_copy_score - pre_score, mimic_bias_delete_score - pre_score, data_denoise_score - pre_score
        ))
        file.write("& & Loss & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\\n".format(
            pre_loss, mimic_label_copy_loss - pre_loss, mimic_label_delete_loss - pre_loss,
            mimic_bias_copy_loss - pre_loss, mimic_bias_delete_loss - pre_loss, data_denoise_loss - pre_loss
        ))
        file.write("& & Acc & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\\n".format(
            pre_acc, mimic_label_copy_acc - pre_acc, mimic_label_delete_acc - pre_acc,
            mimic_bias_copy_acc - pre_acc, mimic_bias_delete_acc - pre_acc, data_denoise_acc - pre_acc
        ))
    # print("original loss for cluster a0: ", '%.4f'%pre_bias_loss[0], ", mimic label copy: ", '%.4f'%(mimic_label_copy_bias_loss[0] - pre_bias_loss[0]), ", mimic label delete: ", '%.4f'%(mimic_label_delete_bias_loss[0] - pre_bias_loss[0]), ", mimic bias copy: ", '%.4f'%(mimic_bias_copy_bias_loss[0] - pre_bias_loss[0]), ", mimic bias delete: ", '%.4f'%(mimic_bias_delete_bias_loss[0] - pre_bias_loss[0]), ", data denoise: ", '%.4f'%(data_denoise_bias_loss[0] - pre_bias_loss[0]))
    # print("original loss for cluster b0: ", '%.4f'%pre_bias_loss[1], ", mimic label copy: ", '%.4f'%(mimic_label_copy_bias_loss[1] - pre_bias_loss[1]), ", mimic label delete: ", '%.4f'%(mimic_label_delete_bias_loss[1] - pre_bias_loss[1]), ", mimic bias copy: ", '%.4f'%(mimic_bias_copy_bias_loss[1] - pre_bias_loss[1]), ", mimic bias delete: ", '%.4f'%(mimic_bias_delete_bias_loss[1] - pre_bias_loss[1]), ", data denoise: ", '%.4f'%(data_denoise_bias_loss[1] - pre_bias_loss[1]))
    # print("original loss for cluster a1: ", '%.4f'%pre_bias_loss[2], ", mimic label copy: ", '%.4f'%(mimic_label_copy_bias_loss[2] - pre_bias_loss[2]), ", mimic label delete: ", '%.4f'%(mimic_label_delete_bias_loss[2] - pre_bias_loss[2]), ", mimic bias copy: ", '%.4f'%(mimic_bias_copy_bias_loss[2] - pre_bias_loss[2]), ", mimic bias delete: ", '%.4f'%(mimic_bias_delete_bias_loss[2] - pre_bias_loss[2]), ", data denoise: ", '%.4f'%(data_denoise_bias_loss[2] - pre_bias_loss[2]))
    # print("original loss for cluster b1: ", '%.4f'%pre_bias_loss[3], ", mimic label copy: ", '%.4f'%(mimic_label_copy_bias_loss[3] - pre_bias_loss[3]), ", mimic label delete: ", '%.4f'%(mimic_label_delete_bias_loss[3] - pre_bias_loss[3]), ", mimic bias copy: ", '%.4f'%(mimic_bias_copy_bias_loss[3] - pre_bias_loss[3]), ", mimic bias delete: ", '%.4f'%(mimic_bias_delete_bias_loss[3] - pre_bias_loss[3]), ", data denoise: ", '%.4f'%(data_denoise_bias_loss[3] - pre_bias_loss[3]))
    # print("original acc for cluster a0: ", '%.4f'%pre_bias_acc[0], ", mimic label copy: ", '%.4f'%(mimic_label_copy_bias_acc[0] - pre_bias_acc[0]), ", mimic label delete: ", '%.4f'%(mimic_label_delete_bias_acc[0] - pre_bias_acc[0]), ", mimic bias copy: ", '%.4f'%(mimic_bias_copy_bias_acc[0] - pre_bias_acc[0]), ", mimic bias delete: ", '%.4f'%(mimic_bias_delete_bias_acc[0] - pre_bias_acc[0]), ", data denoise: ", '%.4f'%(data_denoise_bias_acc[0] - pre_bias_acc[0]))
    # print("original acc for cluster b0: ", '%.4f'%pre_bias_acc[1], ", mimic label copy: ", '%.4f'%(mimic_label_copy_bias_acc[1] - pre_bias_acc[1]), ", mimic label delete: ", '%.4f'%(mimic_label_delete_bias_acc[1] - pre_bias_acc[1]), ", mimic bias copy: ", '%.4f'%(mimic_bias_copy_bias_acc[1] - pre_bias_acc[1]), ", mimic bias delete: ", '%.4f'%(mimic_bias_delete_bias_acc[1] - pre_bias_acc[1]), ", data denoise: ", '%.4f'%(data_denoise_bias_acc[1] - pre_bias_acc[1]))
    # print("original acc for cluster a1: ", '%.4f'%pre_bias_acc[2], ", mimic label copy: ", '%.4f'%(mimic_label_copy_bias_acc[2] - pre_bias_acc[2]), ", mimic label delete: ", '%.4f'%(mimic_label_delete_bias_acc[2] - pre_bias_acc[2]), ", mimic bias copy: ", '%.4f'%(mimic_bias_copy_bias_acc[2] - pre_bias_acc[2]), ", mimic bias delete: ", '%.4f'%(mimic_bias_delete_bias_acc[2] - pre_bias_acc[2]), ", data denoise: ", '%.4f'%(data_denoise_bias_acc[2] - pre_bias_acc[2]))
    # print("original acc for cluster b1: ", '%.4f'%pre_bias_acc[3], ", mimic label copy: ", '%.4f'%(mimic_label_copy_bias_acc[3] - pre_bias_acc[3]), ", mimic label delete: ", '%.4f'%(mimic_label_delete_bias_acc[3] - pre_bias_acc[3]), ", mimic bias copy: ", '%.4f'%(mimic_bias_copy_bias_acc[3] - pre_bias_acc[3]), ", mimic bias delete: ", '%.4f'%(mimic_bias_delete_bias_acc[3] - pre_bias_acc[3]), ", data denoise: ", '%.4f'%(data_denoise_bias_acc[3] - pre_bias_acc[3]))
            # for i in range(num_candidate):
            #     print("original score: ", train_data[i].score, ", change of score: ", new_train_data[i].score - train_data[i].score)
            #     print("original base loss: ", train_data[i].base_loss, ", change of base loss: ", new_train_data[i].base_loss - train_data[i].base_loss)
            #     # print("original post loss: ", train_data[i].post_loss, ", change of post loss: ", new_train_data[i].post_loss - train_data[i].post_loss)
            #     # print("original smooth loss: ", train_data[i].smooth, ", change of smooth loss: ", new_train_data[i].smooth - train_data[i].smooth)5
            #     print("original base acc: ", train_data[i].base_acc, ", change of base acc: ", new_train_data[i].base_acc - train_data[i].base_acc)
            #     # print("original post acc: ", train_data[i].post_acc, ", change of post acc: ", new_train_data[i].post_acc - train_data[i].post_acc)

