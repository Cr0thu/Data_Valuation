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
import signal
import sys
import fcntl
import os

def signal_handler(signum, frame):
    print("\nReceived signal to terminate. Cleaning up...")
    # Release CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class TorchLogisticRegression:
    """
    PyTorch implementation of Logistic Regression to replace sklearn's LogisticRegression
    Supports GPU acceleration and maintains the same interface as sklearn
    """
    def __init__(self, fit_intercept=True, C=1.0, max_iter=1000, tol=1e-6, lr=1.0):
        self.fit_intercept = fit_intercept
        self.C = C  # Regularization strength (inverse of lambda)
        # Reduce max_iter for large C values to prevent excessive computation
        self.max_iter = min(max_iter, max(100, int(1000/max(1, C/10))))
        self.tol = tol
        self.lr = lr
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_features_in_ = None
        
    def _add_intercept(self, X):
        """Add intercept term to features if fit_intercept=True"""
        if self.fit_intercept:
            intercept = torch.ones(X.shape[0], 1, device=X.device)
            return torch.cat([X, intercept], dim=1)
        return X
    
    def _sigmoid(self, z):
        """Sigmoid activation function with numerical stability"""
        return torch.sigmoid(torch.clamp(z, -250, 250))
    
    def _compute_loss(self, X, y, weights):
        """Compute logistic loss with L2 regularization"""
        z = torch.matmul(X, weights)
        sigmoid_z = self._sigmoid(z)
        
        # Binary cross-entropy loss
        epsilon = 1e-15  # For numerical stability
        sigmoid_z = torch.clamp(sigmoid_z, epsilon, 1 - epsilon)
        bce_loss = -torch.mean(y * torch.log(sigmoid_z) + (1 - y) * torch.log(1 - sigmoid_z))
        
        # L2 regularization (excluding intercept term if present)
        if self.fit_intercept:
            l2_reg = (1 / (2 * self.C)) * torch.sum(weights[:-1] ** 2)
        else:
            l2_reg = (1 / (2 * self.C)) * torch.sum(weights ** 2)
        
        return bce_loss + l2_reg
    
    def _compute_gradient(self, X, y, weights):
        """Compute gradient for logistic regression with L2 regularization"""
        z = torch.matmul(X, weights)
        sigmoid_z = self._sigmoid(z)
        
        # Gradient of binary cross-entropy
        gradient = torch.matmul(X.t(), sigmoid_z - y) / X.shape[0]
        
        # Add L2 regularization gradient (excluding intercept)
        if self.fit_intercept:
            reg_gradient = weights / self.C
            reg_gradient[-1] = 0  # Don't regularize intercept
        else:
            reg_gradient = weights / self.C
        
        return gradient + reg_gradient
    
    def fit(self, X, y):
        """
        Fit the logistic regression model
        
        Parameters:
        X: features (numpy array or torch tensor)
        y: target labels (numpy array or torch tensor)
        
        Returns:
        self
        """
        # Convert to torch tensors and move to device
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32, device=device)
        
        X = X.to(device)
        y = y.to(device).float()
        
        # Store input info
        self.n_features_in_ = X.shape[1]
        self.classes_ = torch.unique(y)
        
        # Add intercept if needed
        X_with_intercept = self._add_intercept(X)
        
        # Initialize weights
        n_features = X_with_intercept.shape[1]
        weights = torch.zeros(n_features, 1, device=device, requires_grad=False)
        
        # Optimized Newton-Raphson with adaptive method selection
        prev_loss = float('inf')
        use_newton = True
        gradient_step_size = 1.0
        
        # For large C values, use L-BFGS-like approach
        if self.C > 10:
            use_newton = False
            gradient_step_size = min(1.0, 10.0 / self.C)
        
        for iteration in range(self.max_iter):
            # Compute predictions
            z = torch.matmul(X_with_intercept, weights)
            p = self._sigmoid(z)
            
            # Compute gradient
            gradient = torch.matmul(X_with_intercept.t(), p - y.unsqueeze(1)) / X_with_intercept.shape[0]
            
            # Add L2 regularization to gradient
            if self.fit_intercept:
                reg_gradient = weights / self.C
                reg_gradient[-1] = 0  # Don't regularize intercept
            else:
                reg_gradient = weights / self.C
            gradient += reg_gradient
            
            if use_newton and iteration < 100:  # Only use Newton for first 100 iterations
                # Compute Hessian (approximate)
                W = p * (1 - p)
                W = torch.clamp(W, min=1e-8)  # Avoid numerical issues
                hessian = torch.matmul(X_with_intercept.t() * W.t(), X_with_intercept) / X_with_intercept.shape[0]
                
                # Add L2 regularization to Hessian
                if self.fit_intercept:
                    reg_hessian = torch.eye(hessian.shape[0], device=device) / self.C
                    reg_hessian[-1, -1] = 0  # Don't regularize intercept
                else:
                    reg_hessian = torch.eye(hessian.shape[0], device=device) / self.C
                hessian += reg_hessian
                
                # Use Cholesky decomposition instead of direct inverse for stability
                try:
                    L = torch.linalg.cholesky(hessian)
                    step = torch.cholesky_solve(gradient, L)
                    weights = weights - self.lr * step
                except:
                    # Fallback to gradient descent
                    use_newton = False
                    weights = weights - gradient_step_size * gradient
            else:
                # Gradient descent with adaptive step size
                step_size = gradient_step_size / (1 + iteration * 0.001)  # Decay step size
                weights = weights - step_size * gradient
            
            # Early stopping based on gradient norm (faster than loss computation)
            grad_norm = torch.norm(gradient)
            if grad_norm < self.tol * 10:  # Gradient is small enough
                break
                
            # Check convergence less frequently to save computation
            if iteration % 20 == 0:
                current_loss = self._compute_loss(X_with_intercept, y.unsqueeze(1), weights)
                if abs(prev_loss - current_loss) < self.tol:
                    break
                if current_loss > prev_loss * 1.01:  # If loss increased significantly, reduce step size
                    gradient_step_size *= 0.8
                prev_loss = current_loss
        
        # Store coefficients
        if self.fit_intercept:
            self.coef_ = weights[:-1].t()  # Shape: (1, n_features)
            self.intercept_ = weights[-1].item()
        else:
            self.coef_ = weights.t()  # Shape: (1, n_features)
            self.intercept_ = 0.0
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        X: features (numpy array or torch tensor)
        
        Returns:
        probabilities: torch tensor of shape (n_samples, 2)
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Convert to torch tensor and move to device
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=device)
        
        X = X.to(device)
        
        # Add intercept if needed
        X_with_intercept = self._add_intercept(X)
        
        # Compute predictions
        if self.fit_intercept:
            weights = torch.cat([self.coef_.t(), torch.tensor([[self.intercept_]], device=device)], dim=1)
        else:
            weights = self.coef_.t()
        
        z = torch.matmul(X_with_intercept, weights)
        prob_1 = self._sigmoid(z).squeeze()
        prob_0 = 1 - prob_1
        
        return torch.stack([prob_0, prob_1], dim=1)
    
    def predict(self, X):
        """
        Predict class labels
        
        Parameters:
        X: features (numpy array or torch tensor)
        
        Returns:
        predictions: torch tensor of predicted class labels
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).float()
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels
        
        Parameters:
        X: features (numpy array or torch tensor)
        y: target labels (numpy array or torch tensor)
        
        Returns:
        accuracy: float
        """
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32, device=device)
        
        y = y.to(device)
        predictions = self.predict(X)
        accuracy = (predictions == y).float().mean()
        return accuracy.item()
    
    def decision_function(self, X):
        """
        Predict confidence scores for samples
        
        Parameters:
        X: features (numpy array or torch tensor)
        
        Returns:
        scores: torch tensor of shape (n_samples,)
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=device)
        
        X = X.to(device)
        X_with_intercept = self._add_intercept(X)
        
        if self.fit_intercept:
            weights = torch.cat([self.coef_.t(), torch.tensor([[self.intercept_]], device=device)], dim=1)
        else:
            weights = self.coef_.t()
        
        return torch.matmul(X_with_intercept, weights).squeeze()
    
    @staticmethod
    def fit_multiple_models(datasets, fit_intercept=True, C=1.0, max_iter=1000, tol=1e-6, lr=1.0):
        """
        Fit multiple logistic regression models in parallel on GPU
        
        Parameters:
        datasets: list of (X, y) tuples
        
        Returns:
        list of fitted TorchLogisticRegression models
        """
        models = []
        for X, y in datasets:
            model = TorchLogisticRegression(
                fit_intercept=fit_intercept, 
                C=C, 
                max_iter=max_iter, 
                tol=tol, 
                lr=lr
            )
            model.fit(X, y)
            models.append(model)
        return models

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
parser.add_argument("--gpu_id", type=int, required=True, help="GPU device ID to use")
parser.add_argument("--run_id", type=int, required=True, help="Run ID (1-10)")
parser.add_argument("--noise_level", type=float, default=0, help="Noise level for data generation")

args = parser.parse_args()

# Set device based on argument
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output file specific to this run
output_file = f"output_copy_{args.run_id}_train_{args.train_size_a_0}_{args.train_size_a_1}_{args.train_size_b_0}_{args.train_size_b_1}_test_{args.test_size_a_0}_{args.test_size_a_1}_{args.test_size_b_0}_{args.test_size_b_1}.txt"

resnet50 = models.resnet50(pretrained=True)
resnet50.fc = torch.nn.Identity()
resnet50.to(device).eval()
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
        embedding_0 = resnet50(images_0)
        embedding_1 = resnet50(images_1)
        # Create ones tensor once and reuse
        ones_0 = torch.ones(embedding_0.size()[0], 1, device=device)
        ones_1 = torch.ones(embedding_1.size()[0], 1, device=device)
        embedding_0 = torch.cat([embedding_0, ones_0], dim=1)
        embedding_1 = torch.cat([embedding_1, ones_1], dim=1)
        perm = torch.randperm(len(embedding_0), device=device)
        embedding_0 = embedding_0[perm]
        embedding_1 = embedding_1[perm]
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
    y = torch.ones(X.size()[0], device=device) * label
    y = y.to(torch.int)
    y[:int(noise*X.size()[0])] = 1 - y[:int(noise*X.size()[0])]
    # print(y)
    return X, y, X.size()[0]

def generate_train_cifar10(train_size_a_0, train_size_a_1, train_size_b_0, train_size_b_1, train_number, noise_level):
    train_dataset = []
    bias = torch.tensor([
        train_size_a_0, train_size_b_0,
        train_size_a_1, train_size_b_1
    ], device=device)
    for i in range(train_number):
        perm_indices_a_0 = torch.randperm(4000, device=device)
        sample_indices_a_0 = perm_indices_a_0[:train_size_a_0]
        perm_indices_a_1 = torch.randperm(4000, device=device)
        sample_indices_a_1 = perm_indices_a_1[:train_size_a_1]
        perm_indices_b_0 = torch.randperm(4000, device=device)
        sample_indices_b_0 = perm_indices_b_0[:train_size_b_0]
        perm_indices_b_1 = torch.randperm(4000, device=device)
        sample_indices_b_1 = perm_indices_b_1[:train_size_b_1]
        train_X_a_0, train_y_a_0, _ = generate_data_cifar10(sample_indices_a_0, images_a_0_embedding[:4000], 0, eps * noise_level)
        train_X_a_1, train_y_a_1, _ = generate_data_cifar10(sample_indices_a_1, images_a_1_embedding[:4000], 1, eps * noise_level)
        train_X_b_0, train_y_b_0, _ = generate_data_cifar10(sample_indices_b_0, images_b_0_embedding[:4000], 0, eps * noise_level)
        train_X_b_1, train_y_b_1, _ = generate_data_cifar10(sample_indices_b_1, images_b_1_embedding[:4000], 1, eps * noise_level)
        train_X = torch.cat([train_X_a_0, train_X_b_0, train_X_a_1, train_X_b_1], dim=0)
        train_y = torch.cat([train_y_a_0, train_y_b_0, train_y_a_1, train_y_b_1], dim=0)
        train_dataset.append(dataset(train_X, train_y, train_X.size()[0], 0, 0, 0, noise_level*eps, bias, [], []))
    return train_dataset

def subsample(X, y, size):
    perm = torch.randperm(len(y), device=device)
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

def compute_data_score_err(mu_test, Q_test, test_X, test_y, train_X, train_y, lg2, bias, detailed_loss_acc = False):
    test_N = test_y.size()[0]
    M = test_X.size()[1]
    test_size_a_0, test_size_b_0, test_size_a_1, test_size_b_1 = bias

    mu0 = torch.zeros((1, M), device=device)
    Q0 = torch.eye(M, device=device)/penalty
    lg0 = -M * torch.log(torch.tensor(penalty, device=device))
    
    train = TorchLogisticRegression(fit_intercept=False, C=penalty, max_iter=5000).fit(train_X, train_y)
    # print(train.score(test_X.cpu(), test_y.cpu()))
    # print(train.score(P_x, P_y))
    mu_train = train.coef_.to(device)
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

    if detailed_loss_acc:
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

        detailed_loss = [base_loss_0.item(), base_loss_1.item(), base_loss_2.item(), base_loss_3.item()]
        detailed_acc = [base_acc_0.item(), base_acc_1.item(), base_acc_2.item(), base_acc_3.item()]
    else:
        detailed_loss = []
        detailed_acc = []
    
    return score, base_loss.item(), base_acc.item(), torch.tensor(detailed_loss, device=device), torch.tensor(detailed_acc, device=device)

def get_err_score(train_data, test_X, test_y, train_number, test_bias):
    test = TorchLogisticRegression(fit_intercept=False, C=penalty, max_iter=5000).fit(test_X, test_y)
    mu_test = test.coef_.to(device)
    Q_test = compute_hessian(mu_test, test_X)

    L = torch.linalg.cholesky(Q_test)
    lg2 = 2 * torch.sum(torch.log(torch.diagonal(L)))

    for i in range(train_number):
        train_data[i].score, train_data[i].base_loss, train_data[i].base_acc, train_data[i].bias_loss, train_data[i].bias_acc = compute_data_score_err(mu_test, Q_test, test_X, test_y, train_data[i].X, train_data[i].y, lg2, test_bias)

def mimic_label_copy(train_data, num_candidate, test_ratio):
    new_train_data = []
    test_a_label_ratio = test_ratio[0]/test_ratio[2]
    test_b_label_ratio = test_ratio[1]/test_ratio[3]
    for i in range(num_candidate):
        train_size_a_0, train_size_b_0, train_size_a_1, train_size_b_1 = train_data[i].bias
        target_size_a_0 = int(train_size_a_1 * test_a_label_ratio)
        target_size_b_0 = int(train_size_b_1 * test_b_label_ratio)

        # Pre-allocate tensors to avoid repeated concatenation
        if train_size_a_0 < target_size_a_0:
            num_extra = int(target_size_a_0/train_size_a_0) - 1
            indices = torch.range(0,train_size_a_0-1, device=device)
            extra_indices = torch.tensor([], device=device)
            for g in range(num_extra):
                extra_indices = torch.cat([extra_indices, indices])
            extra_indices = extra_indices.to(torch.long)
            
            # Use in-place operations where possible
            new_train_X_a_0 = torch.cat([train_data[i].X[:train_size_a_0], train_data[i].X[:train_size_a_0][extra_indices]])
            new_train_X_a_1 = train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]
            new_train_y_a_0 = torch.cat([train_data[i].y[:train_size_a_0], train_data[i].y[:train_size_a_0][extra_indices]])
            new_train_y_a_1 = train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]
        else:
            target_size_a_1 = int(train_size_a_0 / test_a_label_ratio)
            num_extra = int(target_size_a_1/train_size_a_1) - 1
            indices = torch.range(0,train_size_a_1-1, device=device)
            extra_indices = torch.tensor([], device=device)
            for g in range(num_extra):
                extra_indices = torch.cat([extra_indices, indices])
            extra_indices = extra_indices.to(torch.long)
            
            new_train_X_a_1 = torch.cat([train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1], train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][extra_indices]])
            new_train_X_a_0 = train_data[i].X[:train_size_a_0]
            new_train_y_a_1 = torch.cat([train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1], train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][extra_indices]])
            new_train_y_a_0 = train_data[i].y[:train_size_a_0]

        if train_size_b_0 < target_size_b_0:
            num_extra = int(target_size_b_0/train_size_b_0) - 1
            indices = torch.range(0,train_size_b_0-1, device=device)
            extra_indices = torch.tensor([], device=device)
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
            indices = torch.range(0,train_size_b_1-1, device=device)
            extra_indices = torch.tensor([], device=device)
            for g in range(num_extra):
                extra_indices = torch.cat([extra_indices, indices])
            extra_indices = extra_indices.to(torch.long)
            
            new_train_X_b_1 = torch.cat([train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:], train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:][extra_indices]])
            new_train_X_b_0 = train_data[i].X[train_size_a_0:train_size_a_0+train_size_b_0]
            new_train_y_b_1 = torch.cat([train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:], train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:][extra_indices]])
            new_train_y_b_0 = train_data[i].y[train_size_a_0:train_size_a_0+train_size_b_0]

        # Use torch.cat instead of torch.concatenate for better performance
        new_train_X = torch.cat([new_train_X_a_0, new_train_X_b_0, new_train_X_a_1, new_train_X_b_1], dim=0)
        new_train_y = torch.cat([new_train_y_a_0, new_train_y_b_0, new_train_y_a_1, new_train_y_b_1], dim=0)
        
        # Create bias tensor directly on device
        bias_tensor = torch.tensor([new_train_X_a_0.size()[0], new_train_X_b_0.size()[0], new_train_X_a_1.size()[0], new_train_X_b_1.size()[0]], device=device)
        new_train_data.append(dataset(new_train_X, new_train_y, new_train_X.size()[0], 0, 0, 0, train_data[i].noise, bias_tensor, [], []))
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
            residual_indices = torch.randperm(train_size_a_0, device=device)[:target_size_a_0]
            new_train_X_a_0 = train_data[i].X[:train_size_a_0][residual_indices]
            new_train_X_a_1 = train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]
            new_train_y_a_0 = train_data[i].y[:train_size_a_0][residual_indices]
            new_train_y_a_1 = train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]
        else:
            target_size_a_1 = int(train_size_a_0 / test_a_label_ratio)
            residual_indices = torch.randperm(train_size_a_1, device=device)[:target_size_a_1]
            new_train_X_a_1 = train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][residual_indices]
            new_train_X_a_0 = train_data[i].X[:train_size_a_0]
            new_train_y_a_1 = train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][residual_indices]
            new_train_y_a_0 = train_data[i].y[:train_size_a_0]

        if train_size_b_0 > target_size_b_0:
            residual_indices = torch.randperm(train_size_b_0, device=device)[:target_size_b_0]
            new_train_X_b_0 = train_data[i].X[train_size_a_0:train_size_a_0+train_size_b_0][residual_indices]
            new_train_X_b_1 = train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:]
            new_train_y_b_0 = train_data[i].y[train_size_a_0:train_size_a_0+train_size_b_0][residual_indices]
            new_train_y_b_1 = train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:]
        else:
            target_size_b_1 = int(train_size_b_0 / test_b_label_ratio)
            residual_indices = torch.randperm(train_size_b_1, device=device)[:target_size_b_1]
            new_train_X_b_1 = train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:][residual_indices]
            new_train_X_b_0 = train_data[i].X[train_size_a_0:train_size_a_0+train_size_b_0]
            new_train_y_b_1 = train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:][residual_indices]
            new_train_y_b_0 = train_data[i].y[train_size_a_0:train_size_a_0+train_size_b_0]

        new_train_X = torch.concatenate([new_train_X_a_0, new_train_X_b_0, new_train_X_a_1, new_train_X_b_1])
        new_train_y = torch.concatenate([new_train_y_a_0, new_train_y_b_0, new_train_y_a_1, new_train_y_b_1])
        new_train_data.append(dataset(new_train_X, new_train_y, new_train_X.size()[0], 0, 0, 0, train_data[i].noise, torch.tensor([new_train_X_a_0.size()[0], new_train_X_b_0.size()[0], new_train_X_a_1.size()[0], new_train_X_b_1.size()[0]], device=device), [], []))
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
            indices = torch.range(0,train_size_a_0-1, device=device)
            extra_indices = torch.tensor([], device=device)
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
            indices = torch.range(0,train_size_b_0-1, device=device)
            extra_indices = torch.tensor([], device=device)
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
            indices = torch.range(0,train_size_a_1-1, device=device)
            extra_indices = torch.tensor([], device=device)
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
            indices = torch.range(0,train_size_b_1-1, device=device)
            extra_indices = torch.tensor([], device=device)
            for g in range(num_extra):
                extra_indices = torch.cat([extra_indices, indices])
            extra_indices = extra_indices.to(torch.long)
            new_train_X_b_1 = torch.cat([train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:], train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:][extra_indices]])
            new_train_X_a_1 = train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]
            new_train_y_b_1 = torch.cat([train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:], train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:][extra_indices]])
            new_train_y_a_1 = train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]

        new_train_X = torch.concatenate([new_train_X_a_0, new_train_X_b_0, new_train_X_a_1, new_train_X_b_1])
        new_train_y = torch.concatenate([new_train_y_a_0, new_train_y_b_0, new_train_y_a_1, new_train_y_b_1])
        new_train_data.append(dataset(new_train_X, new_train_y, new_train_X.size()[0], 0, 0, 0, train_data[i].noise, torch.tensor([new_train_X_a_0.size()[0], new_train_X_b_0.size()[0], new_train_X_a_1.size()[0], new_train_X_b_1.size()[0]], device=device), [], []))
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
            residual_indices = torch.randperm(train_size_a_0, device=device)[:target_size_a_0]
            new_train_X_a_0 = train_data[i].X[:train_size_a_0][residual_indices]
            new_train_X_b_0 = train_data[i].X[train_size_a_0:train_size_a_0+train_size_b_0]
            new_train_y_a_0 = train_data[i].y[:train_size_a_0][residual_indices]
            new_train_y_b_0 = train_data[i].y[train_size_a_0:train_size_a_0+train_size_b_0]
        else:
            target_size_b_0 = int(train_size_a_0 / test_0_bias_ratio)
            residual_indices = torch.randperm(train_size_b_0, device=device)[:target_size_b_0]
            new_train_X_b_0 = train_data[i].X[train_size_a_0:train_size_a_0+train_size_b_0][residual_indices]
            new_train_X_a_0 = train_data[i].X[:train_size_a_0]
            new_train_y_b_0 = train_data[i].y[train_size_a_0:train_size_a_0+train_size_b_0][residual_indices]
            new_train_y_a_0 = train_data[i].y[:train_size_a_0]

        if train_size_a_1 > target_size_a_1:
            residual_indices = torch.randperm(train_size_a_1, device=device)[:target_size_a_1]
            new_train_X_a_1 = train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][residual_indices]
            new_train_X_b_1 = train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:]
            new_train_y_a_1 = train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1][residual_indices]
            new_train_y_b_1 = train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:]
        else:
            target_size_b_1 = int(train_size_a_1 / test_1_bias_ratio)
            residual_indices = torch.randperm(train_size_b_1, device=device)[:target_size_b_1]
            new_train_X_b_1 = train_data[i].X[train_size_a_0+train_size_b_0+train_size_a_1:][residual_indices]
            new_train_X_a_1 = train_data[i].X[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]
            new_train_y_b_1 = train_data[i].y[train_size_a_0+train_size_b_0+train_size_a_1:][residual_indices]
            new_train_y_a_1 = train_data[i].y[train_size_a_0+train_size_b_0:train_size_a_0+train_size_b_0+train_size_a_1]

        new_train_X = torch.concatenate([new_train_X_a_0, new_train_X_b_0, new_train_X_a_1, new_train_X_b_1])
        new_train_y = torch.concatenate([new_train_y_a_0, new_train_y_b_0, new_train_y_a_1, new_train_y_b_1])
        new_train_data.append(dataset(new_train_X, new_train_y, new_train_X.size()[0], 0, 0, 0, train_data[i].noise, torch.tensor([new_train_X_a_0.size()[0], new_train_X_b_0.size()[0], new_train_X_a_1.size()[0], new_train_X_b_1.size()[0]], device=device), [], []))
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
        new_train_data.append(dataset(new_train_X, new_train_y, new_train_X.size()[0], 0, 0, 0, train_data[i].noise*(1-ratio), torch.tensor([new_train_X_a_0.size()[0], new_train_X_b_0.size()[0], new_train_X_a_1.size()[0], new_train_X_b_1.size()[0]], device=device), [], []))
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

# with open("output_copy.txt", "a") as file:
#     file.write("train size: {}, {}, {}, {}; test size: {}, {}, {}, {} \n".format(
#             train_size_a_0, train_size_a_1, train_size_b_0, train_size_b_1,
#             test_size_a_0, test_size_a_1, test_size_b_0, test_size_b_1,
#         ))

criterion = nn.BCELoss()

test_label_ratio = (test_size_a_0 + test_size_b_0)/(test_size_a_1 + test_size_b_1)
test_bias_ratio = (test_size_a_0 + test_size_a_1)/(test_size_b_0 + test_size_b_1)
test_size = test_size_a_0 + test_size_a_1 + test_size_b_0 + test_size_b_1
test_ratio = (test_size_a_0, test_size_b_0, test_size_a_1, test_size_b_1)

noise_level = args.noise_level
pre_score = []
mimic_label_copy_score = []
pre_loss = []
mimic_label_copy_loss = []
pre_acc = []
mimic_label_copy_acc = []

print(f"Running on GPU id: {args.gpu_id}, run_id: {args.run_id}")

# Enable memory efficient attention if available
if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(True)

for d in tqdm(range(D), desc=f"GPU {args.gpu_id}, run_id {args.run_id}"):
    # Clear cache periodically to prevent memory buildup
    if d % 100 == 0 and d > 0:
        torch.cuda.empty_cache()
        gc.collect()
    
    test_X = P_x
    test_y = P_y

    train_data = generate_train_cifar10(train_size_a_0, train_size_a_1, train_size_b_0, train_size_b_1, num_candidate, noise_level)
    mimic_label_copy_train_data = mimic_label_copy(train_data, num_candidate, test_ratio)

    sample_test_X_a_0, sample_test_y_a_0 = subsample(images_a_0_embedding[4000:], torch.zeros(1000, device=device), test_size_a_0)
    sample_test_X_a_1, sample_test_y_a_1 = subsample(images_a_1_embedding[4000:], torch.ones(1000, device=device), test_size_a_1)
    sample_test_X_b_0, sample_test_y_b_0 = subsample(images_b_0_embedding[4000:], torch.zeros(1000, device=device), test_size_b_0)
    sample_test_X_b_1, sample_test_y_b_1 = subsample(images_b_1_embedding[4000:], torch.ones(1000, device=device), test_size_b_1)
    sample_test_X = torch.cat([sample_test_X_a_0, sample_test_X_b_0, sample_test_X_a_1, sample_test_X_b_1], dim=0)
    sample_test_y = torch.cat([sample_test_y_a_0, sample_test_y_b_0, sample_test_y_a_1, sample_test_y_b_1], dim=0)

    get_err_score(train_data, sample_test_X, sample_test_y, num_candidate, test_ratio)
    get_err_score(mimic_label_copy_train_data, sample_test_X, sample_test_y, num_candidate, test_ratio)
    
    # Output the accuracy of trained logistic regression models
    if d == 0:
        print(f"Iteration {d}:")
        for i in range(num_candidate):
            print(f"  Original model accuracy: {train_data[i].base_acc:.4f}")
            print(f"  Mimic label copy model accuracy: {mimic_label_copy_train_data[i].base_acc:.4f}")
            print(f"  Accuracy difference: {mimic_label_copy_train_data[i].base_acc - train_data[i].base_acc:.4f}")
        print("---")

    for i in range(num_candidate):
        pre_score.append(train_data[i].score)
        mimic_label_copy_score.append(mimic_label_copy_train_data[i].score)
        pre_loss.append(train_data[i].base_loss)
        mimic_label_copy_loss.append(mimic_label_copy_train_data[i].base_loss)
        pre_acc.append(train_data[i].base_acc)
        mimic_label_copy_acc.append(mimic_label_copy_train_data[i].base_acc)

# Calculate statistics
pre_score_tensor = torch.tensor(pre_score, device=device)
mimic_label_copy_score_tensor = torch.tensor(mimic_label_copy_score, device=device)
pre_loss_tensor = torch.tensor(pre_loss, device=device)
mimic_label_copy_loss_tensor = torch.tensor(mimic_label_copy_loss, device=device)
pre_acc_tensor = torch.tensor(pre_acc, device=device)
mimic_label_copy_acc_tensor = torch.tensor(mimic_label_copy_acc, device=device)

mimic_label_copy_score_mean = (mimic_label_copy_score_tensor - pre_score_tensor).mean().item()
mimic_label_copy_score_std = (mimic_label_copy_score_tensor - pre_score_tensor).std().item()
mimic_label_copy_loss_mean = (mimic_label_copy_loss_tensor - pre_loss_tensor).mean().item()
mimic_label_copy_loss_std = (mimic_label_copy_loss_tensor - pre_loss_tensor).std().item()
mimic_label_copy_acc_mean = (mimic_label_copy_acc_tensor - pre_acc_tensor).mean().item()
mimic_label_copy_acc_std = (mimic_label_copy_acc_tensor - pre_acc_tensor).std().item()

print(f"\n=== SUMMARY ===")
print(f"Average original model accuracy: {pre_acc_tensor.mean().item():.4f} ± {pre_acc_tensor.std().item():.4f}")
print(f"Average mimic label copy model accuracy: {mimic_label_copy_acc_tensor.mean().item():.4f} ± {mimic_label_copy_acc_tensor.std().item():.4f}")
print(f"Accuracy change (mean ± std): {mimic_label_copy_acc_mean:.4f} ± {mimic_label_copy_acc_std:.4f}")
print(f"Score change (mean ± std): {mimic_label_copy_score_mean:.4f} ± {mimic_label_copy_score_std:.4f}")
print(f"Loss change (mean ± std): {mimic_label_copy_loss_mean:.4f} ± {mimic_label_copy_loss_std:.4f}")
print("=" * 50)

# 使用文件锁确保并行写入安全
with open(output_file, "a") as file:
    try:
        # 获取文件锁
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        file.write("{} & Copy & {:.4f}\\pm{:.4f} & {:.4f}\\pm{:.4f} & {:.4f}\\pm{:.4f}\n".format(
            penalty, mimic_label_copy_score_mean, mimic_label_copy_score_std, 
            mimic_label_copy_loss_mean, mimic_label_copy_loss_std, 
            mimic_label_copy_acc_mean, mimic_label_copy_acc_std
        ))
        file.flush()  # 确保数据立即写入
        print(f"Results written to {output_file}")
    finally:
        # 释放文件锁
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)

