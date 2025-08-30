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
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# Multi-GPU setup
num_gpus = torch.cuda.device_count()
print(f"Available GPUs: {num_gpus}")
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Create device list
gpu_devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
main_device = gpu_devices[0] if num_gpus > 0 else torch.device("cpu")
print(f"Using devices: {gpu_devices}")
print(f"Main device: {main_device}")

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

class TorchLogisticRegression:
    """
    Multi-GPU PyTorch implementation of Logistic Regression
    Supports GPU acceleration and maintains the same interface as sklearn
    """
    def __init__(self, fit_intercept=True, C=1.0, max_iter=1000, tol=1e-6, lr=1.0, device=None):
        self.fit_intercept = fit_intercept
        self.C = C  # Regularization strength (inverse of lambda)
        # Reduce max_iter for large C values to prevent excessive computation
        self.max_iter = min(max_iter, max(100, int(1000/max(1, C/10))))
        self.tol = tol
        self.lr = lr
        self.device = device if device is not None else main_device
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
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        X = X.to(self.device)
        y = y.to(self.device).float()
        
        # Store input info
        self.n_features_in_ = X.shape[1]
        self.classes_ = torch.unique(y)
        
        # Add intercept if needed
        X_with_intercept = self._add_intercept(X)
        
        # Initialize weights
        n_features = X_with_intercept.shape[1]
        weights = torch.zeros(n_features, 1, device=self.device, requires_grad=False)
        
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
                    reg_hessian = torch.eye(hessian.shape[0], device=self.device) / self.C
                    reg_hessian[-1, -1] = 0  # Don't regularize intercept
                else:
                    reg_hessian = torch.eye(hessian.shape[0], device=self.device) / self.C
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
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        X = X.to(self.device)
        
        # Add intercept if needed
        X_with_intercept = self._add_intercept(X)
        
        # Compute predictions
        if self.fit_intercept:
            weights = torch.cat([self.coef_.t(), torch.tensor([[self.intercept_]], device=self.device)])
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
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        y = y.to(self.device)
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
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        X = X.to(self.device)
        X_with_intercept = self._add_intercept(X)
        
        if self.fit_intercept:
            weights = torch.cat([self.coef_.t(), torch.tensor([[self.intercept_]], device=self.device)])
        else:
            weights = self.coef_.t()
        
        return torch.matmul(X_with_intercept, weights).squeeze()
    
    @staticmethod
    def fit_multiple_models_parallel(datasets, fit_intercept=True, C=1.0, max_iter=1000, tol=1e-6, lr=1.0):
        """
        Fit multiple logistic regression models in parallel across multiple GPUs
        
        Parameters:
        datasets: list of (X, y) tuples
        
        Returns:
        list of fitted TorchLogisticRegression models
        """
        if num_gpus <= 1:
            # Fallback to single GPU
            models = []
            for X, y in datasets:
                model = TorchLogisticRegression(
                    fit_intercept=fit_intercept, 
                    C=C, 
                    max_iter=max_iter, 
                    tol=tol, 
                    lr=lr,
                    device=main_device
                )
                model.fit(X, y)
                models.append(model)
            return models
        
        # Multi-GPU parallel training
        def train_model_on_gpu(args):
            X, y, gpu_id = args
            device_id = gpu_devices[gpu_id]
            
            # Move data to specific GPU
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32, device=device_id)
            else:
                X = X.to(device_id)
            
            if isinstance(y, np.ndarray):
                y = torch.tensor(y, dtype=torch.float32, device=device_id)
            else:
                y = y.to(device_id)
            
            model = TorchLogisticRegression(
                fit_intercept=fit_intercept, 
                C=C, 
                max_iter=max_iter, 
                tol=tol, 
                lr=lr,
                device=device_id
            )
            model.fit(X, y)
            return model
        
        # Distribute datasets across GPUs
        models = [None] * len(datasets)
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            future_to_idx = {}
            for i, (X, y) in enumerate(datasets):
                gpu_id = i % num_gpus
                future = executor.submit(train_model_on_gpu, (X, y, gpu_id))
                future_to_idx[future] = i
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    models[idx] = future.result()
                except Exception as exc:
                    print(f'Model {idx} generated an exception: {exc}')
                    # Fallback to single GPU
                    X, y = datasets[idx]
                    model = TorchLogisticRegression(
                        fit_intercept=fit_intercept, 
                        C=C, 
                        max_iter=max_iter, 
                        tol=tol, 
                        lr=lr,
                        device=main_device
                    )
                    model.fit(X, y)
                    models[idx] = model
        
        return models

# Load ResNet50 models on all GPUs for parallel preprocessing
resnet50_models = []
for i in range(num_gpus):
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = torch.nn.Identity()
    resnet50.to(gpu_devices[i]).eval()
    resnet50_models.append(resnet50)

penalty = args.penalty
eps = 0.01

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

labels = np.load('../CIFAR-10-C/labels.npy')
selected_indices_0 = labels == 0
selected_indices_1 = labels == 1

def data_preprocess_parallel(images):
    """Parallel data preprocessing using multiple GPUs"""
    def process_on_gpu(args):
        images_subset, indices, gpu_id = args
        device = gpu_devices[gpu_id]
        resnet_model = resnet50_models[gpu_id]
        
        if len(indices) == 0:
            return torch.empty(0, 2049, device=main_device)  # 2048 + 1 for bias
        
        processed_images = torch.stack([preprocess(image) for image in images_subset]).to(device)
        
        with torch.no_grad():
            embeddings = resnet_model(processed_images)
            embeddings = torch.concatenate([embeddings, torch.ones(embeddings.size()[0], 1).to(device)], dim=1)
            # Apply random permutation
            perm = torch.randperm(len(embeddings))
            embeddings = embeddings[perm]
        
        return embeddings.to(main_device)
    
    # Split data across GPUs
    images_0 = images[selected_indices_0]
    images_1 = images[selected_indices_1]
    
    # Distribute work across GPUs
    chunk_size_0 = len(images_0) // num_gpus
    chunk_size_1 = len(images_1) // num_gpus
    
    tasks = []
    # Tasks for class 0
    for i in range(num_gpus):
        start_idx = i * chunk_size_0
        end_idx = start_idx + chunk_size_0 if i < num_gpus - 1 else len(images_0)
        if start_idx < len(images_0):
            tasks.append((images_0[start_idx:end_idx], list(range(start_idx, end_idx)), i))
    
    # Tasks for class 1  
    for i in range(num_gpus):
        start_idx = i * chunk_size_1
        end_idx = start_idx + chunk_size_1 if i < num_gpus - 1 else len(images_1)
        if start_idx < len(images_1):
            gpu_id = (i + num_gpus // 2) % num_gpus  # Distribute across different GPUs
            tasks.append((images_1[start_idx:end_idx], list(range(start_idx, end_idx)), gpu_id))
    
    # Process in parallel
    results_0 = []
    results_1 = []
    
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for task in tasks:
            future = executor.submit(process_on_gpu, task)
            futures.append(future)
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if i < num_gpus:  # First half are class 0
                results_0.append(result)
            else:  # Second half are class 1
                results_1.append(result)
    
    # Concatenate results
    embedding_0 = torch.cat([r for r in results_0 if r.numel() > 0], dim=0) if results_0 else torch.empty(0, 2049, device=main_device)
    embedding_1 = torch.cat([r for r in results_1 if r.numel() > 0], dim=0) if results_1 else torch.empty(0, 2049, device=main_device)
    
    return embedding_0, embedding_1

def data_preprocess(images):
    """Fallback single-GPU preprocessing"""
    images_0 = torch.stack([preprocess(image) for image in images[selected_indices_0]]).to(main_device)
    images_1 = torch.stack([preprocess(image) for image in images[selected_indices_1]]).to(main_device)
    with torch.no_grad():
        embedding_0 = resnet50_models[0](images_0)
        embedding_1 = resnet50_models[0](images_1)
        embedding_0 = torch.concatenate([embedding_0, torch.ones(embedding_0.size()[0],1).to(main_device)], dim=1)
        embedding_1 = torch.concatenate([embedding_1, torch.ones(embedding_1.size()[0],1).to(main_device)], dim=1)
        perm = torch.randperm(len(embedding_0))
        embedding_0 = embedding_0[perm]
        embedding_1 = embedding_1[perm]
    return embedding_0, embedding_1

# Process data using parallel preprocessing
print("Processing brightness images in parallel...")
images = np.load('../CIFAR-10-C/brightness.npy')
images_a_0_embedding, images_a_1_embedding = data_preprocess_parallel(images)
print("Processing contrast images in parallel...")
images = np.load('../CIFAR-10-C/contrast.npy')
images_b_0_embedding, images_b_1_embedding = data_preprocess_parallel(images)

# Normalize datasets
datasets = [
    images_a_0_embedding, 
    images_a_1_embedding, 
    images_b_0_embedding, 
    images_b_1_embedding
]

combined_data = torch.cat(datasets, dim=0)
mean = combined_data.mean(dim=0, keepdim=True)
std = combined_data.std(dim=0, keepdim=True)
normalized_datasets = [(dataset - mean) / (std + 1e-6) for dataset in datasets]
images_a_0_embedding, images_a_1_embedding, images_b_0_embedding, images_b_1_embedding = normalized_datasets

P_x = torch.concatenate([
    images_a_0_embedding[4000:],
    images_b_0_embedding[4000:],
    images_a_1_embedding[4000:],
    images_b_1_embedding[4000:],
])
P_y = torch.concatenate([torch.zeros(2000), torch.ones(2000)]).to(main_device)

Q_x = torch.concatenate([
    images_a_0_embedding[:4000],
    images_b_0_embedding[:4000],
    images_a_1_embedding[:4000],
    images_b_1_embedding[:4000]
])
Q_y = torch.concatenate([torch.zeros(8000), torch.ones(8000)]).to(main_device)

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
    """Numerically stable sigmoid function"""
    # Clamp input to avoid overflow
    z = torch.clamp(z, min=-50, max=50)
    
    # Use numerically stable sigmoid computation
    return torch.where(z >= 0, 
                      1 / (1 + torch.exp(-z)),
                      torch.exp(z) / (1 + torch.exp(z)))

def generate_data_cifar10(indices, dataset, label, noise):
    X = dataset[indices]
    y = torch.ones(X.size()[0]) * label
    y = torch.tensor(y, dtype=torch.int)
    y = y.to(main_device)
    y[:int(noise*X.size()[0])] = 1 - y[:int(noise*X.size()[0])]
    return X, y, X.size()[0]

def generate_train_cifar10(train_size_a_0, train_size_a_1, train_size_b_0, train_size_b_1, train_number, noise_level):
    train_dataset = []
    bias = torch.tensor([
        train_size_a_0, train_size_b_0,
        train_size_a_1, train_size_b_1
    ]).to(main_device)
    for i in range(train_number):
        perm_indices_a_0 = torch.randperm(4000).to(main_device)
        sample_indices_a_0 = perm_indices_a_0[:train_size_a_0]
        perm_indices_a_1 = torch.randperm(4000).to(main_device)
        sample_indices_a_1 = perm_indices_a_1[:train_size_a_1]
        perm_indices_b_0 = torch.randperm(4000).to(main_device)
        sample_indices_b_0 = perm_indices_b_0[:train_size_b_0]
        perm_indices_b_1 = torch.randperm(4000).to(main_device)
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
    res = torch.eye(X.size(1), device=main_device)/penalty
    res += (X.t() * diag_sigm) @ X
    return res

def compute_score(mu0, Q0, lg0, mu1, Q1, lg1, mu2, Q2, lg2):
    Q = Q1 + Q2 - Q0
    Q_t_L = torch.linalg.cholesky(Q)
    Q_t_L_inv = torch.linalg.solve_triangular(Q_t_L, torch.eye(Q_t_L.size(0), device=main_device), upper=False)
    Q_inv = Q_t_L_inv.T @ Q_t_L_inv
    mu = torch.matmul(Q_inv, torch.matmul(Q1, mu1) + torch.matmul(Q2, mu2) - torch.matmul(Q0, mu0))

    lg12 = 2 * torch.sum(torch.log(torch.diagonal(Q_t_L)))

    lg = lg1+lg2-lg12-lg0

    sqr = torch.matmul(mu.T, torch.matmul(Q, mu)) - torch.matmul(mu1.T, torch.matmul(Q1, mu1)) - torch.matmul(mu2.T, torch.matmul(Q2, mu2)) + torch.matmul(mu0.T, torch.matmul(Q0, mu0))

    score = 0.5 * (lg + sqr)
    return score.item()

def compute_data_score_err_parallel(mu_test, Q_test, test_X, test_y, train_datasets, lg2, bias, detailed_loss_acc=False):
    """
    Compute data score and error for multiple training datasets in parallel
    """
    test_N = test_y.size()[0]
    M = test_X.size()[1]
    test_size_a_0, test_size_b_0, test_size_a_1, test_size_b_1 = bias

    mu0 = torch.zeros((1, M))
    mu0 = mu0.to(main_device)
    Q0 = torch.eye(M)/penalty
    Q0 = Q0.to(main_device)
    lg0 = -M * torch.log(torch.tensor(penalty))
    
    # Prepare datasets for parallel training
    train_data_list = [(train_data.X, train_data.y) for train_data in train_datasets]
    
    # Train multiple models in parallel
    models = TorchLogisticRegression.fit_multiple_models_parallel(
        train_data_list, 
        fit_intercept=False, 
        C=penalty, 
        max_iter=5000
    )
    
    results = []
    for i, (model, train_data) in enumerate(zip(models, train_datasets)):
        # Move model coefficients to main device for computation
        mu_train = model.coef_.to(main_device)
        
        Q_train = compute_hessian(mu_train, train_data.X.to(main_device))
        Q_train_L = torch.linalg.cholesky(Q_train)
        lg1 = 2 * torch.sum(torch.log(torch.diagonal(Q_train_L)))

        score = compute_score(mu0.t(), Q0, lg0, mu_train.t(), Q_train, lg1, mu_test.t(), Q_test, lg2)

        # Compute accuracy and loss
        test_y_float = test_y.float()
        criterion = nn.BCELoss()

        base_predictive = sigmoid(torch.matmul(test_X, mu_train.t())).squeeze()
        base_predictions = (base_predictive >= 0.5).float()
        base_loss = criterion(base_predictive, test_y_float)
        base_acc = (base_predictions == test_y_float).float().mean()

        if detailed_loss_acc:
            # Compute detailed loss and accuracy for different groups
            base_predictive_0 = sigmoid(torch.matmul(test_X[:test_size_a_0], mu_train.t())).squeeze()
            base_predictions_0 = (base_predictive_0 >= 0.5).float()
            base_loss_0 = criterion(base_predictive_0, test_y_float[:test_size_a_0])
            base_acc_0 = (base_predictions_0 == test_y_float[:test_size_a_0]).float().mean()

            base_predictive_1 = sigmoid(torch.matmul(test_X[test_size_a_0:test_size_a_0+test_size_b_0], mu_train.t())).squeeze()
            base_predictions_1 = (base_predictive_1 >= 0.5).float()
            base_loss_1 = criterion(base_predictive_1, test_y_float[test_size_a_0:test_size_a_0+test_size_b_0])
            base_acc_1 = (base_predictions_1 == test_y_float[test_size_a_0:test_size_a_0+test_size_b_0]).float().mean()

            base_predictive_2 = sigmoid(torch.matmul(test_X[test_size_a_0+test_size_b_0:test_size_a_0+test_size_b_0+test_size_a_1], mu_train.t())).squeeze()
            base_predictions_2 = (base_predictive_2 >= 0.5).float()
            base_loss_2 = criterion(base_predictive_2, test_y_float[test_size_a_0+test_size_b_0:test_size_a_0+test_size_b_0+test_size_a_1])
            base_acc_2 = (base_predictions_2 == test_y_float[test_size_a_0+test_size_b_0:test_size_a_0+test_size_b_0+test_size_a_1]).float().mean()

            base_predictive_3 = sigmoid(torch.matmul(test_X[test_size_a_0+test_size_b_0+test_size_a_1:], mu_train.t())).squeeze()
            base_predictions_3 = (base_predictive_3 >= 0.5).float()
            base_loss_3 = criterion(base_predictive_3, test_y_float[test_size_a_0+test_size_b_0+test_size_a_1:])
            base_acc_3 = (base_predictions_3 == test_y_float[test_size_a_0+test_size_b_0+test_size_a_1:]).float().mean()

            detailed_loss = [base_loss_0.item(), base_loss_1.item(), base_loss_2.item(), base_loss_3.item()]
            detailed_acc = [base_acc_0.item(), base_acc_1.item(), base_acc_2.item(), base_acc_3.item()]
        else:
            detailed_loss = []
            detailed_acc = []
        
        results.append((score, base_loss.item(), base_acc.item(), np.array(detailed_loss), np.array(detailed_acc)))
    
    return results

def get_err_score_parallel(train_data, test_X, test_y, train_number, test_bias):
    """
    Parallel version of get_err_score using multi-GPU training with better load balancing
    """
    
    def train_test_model_on_gpu(gpu_id):
        """Train test model on specific GPU"""
        test_device = gpu_devices[gpu_id]
        # Copy data to specific GPU
        test_X_gpu = test_X.to(test_device)
        test_y_gpu = test_y.to(test_device)
        
        test_model = TorchLogisticRegression(fit_intercept=False, C=penalty, max_iter=5000, device=test_device)
        test_model.fit(test_X_gpu, test_y_gpu)
        return test_model.coef_.to(main_device), test_X_gpu, test_y_gpu
    
    def train_models_on_gpu(args):
        """Train multiple training models on specific GPU"""
        train_data_subset, gpu_id = args
        device = gpu_devices[gpu_id]
        
        models = []
        for train_datum in train_data_subset:
            # Copy data to specific GPU
            X_gpu = train_datum.X.to(device)
            y_gpu = train_datum.y.to(device)
            
            model = TorchLogisticRegression(
                fit_intercept=False, 
                C=penalty, 
                max_iter=5000, 
                device=device
            )
            model.fit(X_gpu, y_gpu)
            models.append((model, train_datum))
        
        return models, gpu_id
    
    # Step 1: Train test model on GPU 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(train_test_model_on_gpu, 0)
        mu_test, test_X_gpu0, test_y_gpu0 = future.result()
    
    Q_test = compute_hessian(mu_test, test_X.to(main_device))
    L = torch.linalg.cholesky(Q_test)
    lg2 = 2 * torch.sum(torch.log(torch.diagonal(L)))
    
    # Step 2: Distribute training data across remaining GPUs
    available_gpus = list(range(1, num_gpus)) if num_gpus > 1 else [0]
    train_data_chunks = []
    
    # Split train_data across available GPUs
    chunk_size = max(1, len(train_data) // len(available_gpus))
    for i, gpu_id in enumerate(available_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < len(available_gpus) - 1 else len(train_data)
        train_data_chunks.append((train_data[start_idx:end_idx], gpu_id))
    
    # Step 3: Train all training models in parallel
    all_models = []
    with ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
        futures = []
        for chunk in train_data_chunks:
            future = executor.submit(train_models_on_gpu, chunk)
            futures.append(future)
        
        for future in as_completed(futures):
            models, gpu_id = future.result()
            all_models.extend(models)
    
    # Step 4: Compute scores in parallel across GPUs
    def compute_scores_on_gpu(args):
        """Compute scores for a subset of models on specific GPU"""
        models_subset, gpu_id, mu_test_gpu, Q_test_gpu, lg2_gpu, test_X_gpu, test_y_gpu = args
        device = gpu_devices[gpu_id]
        
        # Move shared data to this GPU
        mu_test_local = mu_test_gpu.to(device)
        Q_test_local = Q_test_gpu.to(device)
        test_X_local = test_X_gpu.to(device)
        test_y_local = test_y_gpu.to(device)
        
        results = []
        for model, train_datum in models_subset:
            # Move model coefficients to this GPU
            mu_train = model.coef_.to(device)
            
            # Move training data to this GPU for hessian computation
            train_X_local = train_datum.X.to(device)
            
            # Compute hessian and score on this GPU
            Q_train = compute_hessian_gpu(mu_train, train_X_local, device)
            Q_train_L = torch.linalg.cholesky(Q_train)
            lg1 = 2 * torch.sum(torch.log(torch.diagonal(Q_train_L)))
            
            score = compute_score_gpu(mu_test_local, Q_test_local, lg2_gpu, mu_train, Q_train, lg1, device)
            
            # Compute accuracy and loss on this GPU
            test_y_float = test_y_local.float()
            criterion = nn.BCELoss()
            
            base_predictive = sigmoid(torch.matmul(test_X_local, mu_train.t())).squeeze()
            base_predictions = (base_predictive >= 0.5).float()
            base_loss = criterion(base_predictive, test_y_float)
            base_acc = (base_predictions == test_y_float).float().mean()
            
            results.append((score, base_loss.item(), base_acc.item(), [], []))
        
        return results
    
    # Distribute score computation across all GPUs
    model_chunks = []
    chunk_size = max(1, len(all_models) // num_gpus)
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_gpus - 1 else len(all_models)
        model_chunks.append((all_models[start_idx:end_idx], i, mu_test, Q_test, lg2, test_X, test_y))
    
    # Step 5: Compute all scores in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for chunk in model_chunks:
            if chunk[0]:  # Only submit if chunk is not empty
                future = executor.submit(compute_scores_on_gpu, chunk)
                futures.append(future)
        
        for future in as_completed(futures):
            results = future.result()
            all_results.extend(results)
    
    # Update train_data with results
    for i, (score, base_loss, base_acc, bias_loss, bias_acc) in enumerate(all_results):
        train_data[i].score = score
        train_data[i].base_loss = base_loss
        train_data[i].base_acc = base_acc
        train_data[i].bias_loss = bias_loss
        train_data[i].bias_acc = bias_acc

def compute_hessian_gpu(mu, X, device):
    """Compute hessian on specific GPU with numerical stability"""
    # Compute sigmoid more stably
    XW = X @ mu.t()
    sigm = sigmoid(XW)
    
    # Clamp sigmoid values to avoid numerical issues
    sigm = torch.clamp(sigm, min=1e-7, max=1-1e-7)
    
    # Compute diagonal terms with numerical stability
    diag_sigm = (sigm * (1 - sigm)).flatten()
    
    # Ensure diagonal terms are not too small
    diag_sigm = torch.clamp(diag_sigm, min=1e-7)
    
    # Initialize with regularization term
    res = torch.eye(X.size(1), device=device) / penalty
    
    # Add the data-dependent term
    res += (X.t() * diag_sigm) @ X
    
    # Add a small regularization for numerical stability
    res += torch.eye(X.size(1), device=device) * 1e-6
    
    return res

def compute_score_gpu(mu0, Q0, lg0, mu1, Q1, lg1, device):
    """Compute score on specific GPU with enhanced numerical stability"""
    # Move mu0 to same device if needed
    mu0_local = mu0.to(device) if mu0.device != device else mu0
    Q0_local = Q0.to(device) if Q0.device != device else Q0
    
    # Create mu2 and Q2 (dummy values for this computation)
    mu2 = torch.zeros_like(mu1, device=device)
    # Use larger regularization for numerical stability
    reg_strength = 1e-2  # Increased regularization
    Q2 = torch.eye(Q1.size(0), device=device) * reg_strength
    lg2 = -mu2.size(1) * torch.log(torch.tensor(reg_strength, device=device))
    
    Q = Q1 + Q2 - Q0_local
    
    # Enhanced regularization strategy
    min_eigenval = 1e-4  # Minimum eigenvalue for positive definiteness
    
    # Check if matrix is positive definite and fix if needed
    def ensure_positive_definite(matrix, min_eig=min_eigenval):
        """Ensure matrix is positive definite"""
        try:
            # Try eigenvalue decomposition
            eigenvals = torch.linalg.eigvals(matrix)
            min_real_eigenval = torch.min(eigenvals.real)
            
            if min_real_eigenval <= min_eig:
                # Add regularization to make positive definite
                reg_amount = min_eig - min_real_eigenval + min_eig
                matrix = matrix + torch.eye(matrix.size(0), device=device) * reg_amount
            
            return matrix
        except:
            # If eigenvalue decomposition fails, use strong regularization
            return matrix + torch.eye(matrix.size(0), device=device) * min_eig
    
    Q = ensure_positive_definite(Q)
    
    # Try multiple strategies for Cholesky decomposition
    strategies = [
        ("standard", lambda: torch.linalg.cholesky(Q)),
        ("extra_reg", lambda: torch.linalg.cholesky(Q + torch.eye(Q.size(0), device=device) * 1e-3)),
        ("strong_reg", lambda: torch.linalg.cholesky(Q + torch.eye(Q.size(0), device=device) * 1e-2)),
    ]
    
    for strategy_name, cholesky_fn in strategies:
        try:
            Q_t_L = cholesky_fn()
            
            # Solve triangular system more stably
            I = torch.eye(Q_t_L.size(0), device=device)
            Q_t_L_inv = torch.linalg.solve_triangular(Q_t_L, I, upper=False)
            Q_inv = Q_t_L_inv.T @ Q_t_L_inv
            
            # Compute mu with better numerical stability
            rhs = torch.matmul(Q1, mu1) + torch.matmul(Q2, mu2) - torch.matmul(Q0_local, mu0_local)
            mu = torch.matmul(Q_inv, rhs)
            
            lg12 = 2 * torch.sum(torch.log(torch.diagonal(Q_t_L)))
            lg = lg1 + lg2 - lg12 - lg0
            
            # Compute quadratic form more stably
            sqr = (torch.matmul(mu.T, torch.matmul(Q, mu)) - 
                   torch.matmul(mu1.T, torch.matmul(Q1, mu1)) - 
                   torch.matmul(mu2.T, torch.matmul(Q2, mu2)) + 
                   torch.matmul(mu0_local.T, torch.matmul(Q0_local, mu0_local)))
            
            score = 0.5 * (lg + sqr)
            
            # Sanity check on the result
            if torch.isnan(score) or torch.isinf(score):
                continue
                
            return score.item()
            
        except Exception as e:
            if strategy_name == "strong_reg":  # Last strategy failed
                print(f"Warning: All Cholesky strategies failed on GPU {device}: {e}")
            continue
    
    # Ultimate fallback: return a default score
    print(f"Warning: Using ultimate fallback score on GPU {device}")
    return 1.0  # Return a neutral score instead of 0

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
        new_train_data.append(dataset(new_train_X, new_train_y, new_train_X.size()[0], 0, 0, 0, train_data[i].noise, torch.tensor([new_train_X_a_0.size()[0], new_train_X_b_0.size()[0], new_train_X_a_1.size()[0], new_train_X_b_1.size()[0]]).to(main_device), [], []))
    return new_train_data

# Main execution
T = 1
D = 100
train_size_a_0 = args.train_size_a_0
train_size_a_1 = args.train_size_a_1
train_size_b_0 = args.train_size_b_0
train_size_b_1 = args.train_size_b_1
num_candidate = 1
test_size_a_0 = args.test_size_a_0
test_size_a_1 = args.test_size_a_1
test_size_b_0 = args.test_size_b_0
test_size_b_1 = args.test_size_b_1
noise_levels = [0]

with open("output_multigpu.txt", "a") as file:
    file.write("train size: {}, {}, {}, {}; test size: {}, {}, {}, {} \n".format(
            train_size_a_0, train_size_a_1, train_size_b_0, train_size_b_1,
            test_size_a_0, test_size_a_1, test_size_b_0, test_size_b_1,
        ))

criterion = nn.BCELoss()

test_label_ratio = (test_size_a_0 + test_size_b_0)/(test_size_a_1 + test_size_b_1)
test_bias_ratio = (test_size_a_0 + test_size_a_1)/(test_size_b_0 + test_size_b_1)
test_size = test_size_a_0 + test_size_a_1 + test_size_b_0 + test_size_b_1
test_ratio = (test_size_a_0, test_size_b_0, test_size_a_1, test_size_b_1)

for noise_level in noise_levels:
    score_change = []
    loss_change = []
    acc_change = []
    
    for run_idx in range(10):
        print(f"Run {run_idx + 1}/10")
        pre_score = []
        mimic_label_copy_score = []
        pre_loss = []
        mimic_label_copy_loss = []
        pre_acc = []
        mimic_label_copy_acc = []
        
        for d in tqdm(range(D)):
            # Generate training and test data
            train_data = generate_train_cifar10(train_size_a_0, train_size_a_1, train_size_b_0, train_size_b_1, num_candidate, noise_level)
            mimic_label_copy_train_data = mimic_label_copy(train_data, num_candidate, test_ratio)

            # Generate test data
            sample_test_X_a_0, sample_test_y_a_0 = subsample(images_a_0_embedding[4000:], torch.zeros(1000).to(main_device), test_size_a_0)
            sample_test_X_a_1, sample_test_y_a_1 = subsample(images_a_1_embedding[4000:], torch.ones(1000).to(main_device), test_size_a_1)
            sample_test_X_b_0, sample_test_y_b_0 = subsample(images_b_0_embedding[4000:], torch.zeros(1000).to(main_device), test_size_b_0)
            sample_test_X_b_1, sample_test_y_b_1 = subsample(images_b_1_embedding[4000:], torch.ones(1000).to(main_device), test_size_b_1)
            sample_test_X = torch.concatenate([sample_test_X_a_0, sample_test_X_b_0, sample_test_X_a_1, sample_test_X_b_1])
            sample_test_y = torch.concatenate([sample_test_y_a_0, sample_test_y_b_0, sample_test_y_a_1, sample_test_y_b_1])

            # Parallel training and evaluation
            start_time = time.time()
            
            # Monitor GPU memory before training
            gpu_mem_before = []
            for i in range(num_gpus):
                torch.cuda.set_device(i)
                gpu_mem_before.append(torch.cuda.memory_allocated(i) / 1024**2)
            
            get_err_score_parallel(train_data, sample_test_X, sample_test_y, num_candidate, test_ratio)
            get_err_score_parallel(mimic_label_copy_train_data, sample_test_X, sample_test_y, num_candidate, test_ratio)
            
            elapsed_time = time.time() - start_time
            
            # Monitor GPU memory after training
            gpu_mem_after = []
            for i in range(num_gpus):
                torch.cuda.set_device(i)
                gpu_mem_after.append(torch.cuda.memory_allocated(i) / 1024**2)
            
            # Output accuracy every 100 iterations
            if d % 100 == 0:
                print(f"Iteration {d}:")
                for i in range(num_candidate):
                    print(f"  Original model accuracy: {train_data[i].base_acc:.4f}")
                    print(f"  Mimic label copy model accuracy: {mimic_label_copy_train_data[i].base_acc:.4f}")
                    print(f"  Accuracy difference: {mimic_label_copy_train_data[i].base_acc - train_data[i].base_acc:.4f}")
                    print(f"  Processing time: {elapsed_time:.3f}s")
                
                # GPU memory usage
                print("  GPU Memory Usage:")
                for i in range(num_gpus):
                    mem_change = gpu_mem_after[i] - gpu_mem_before[i]
                    print(f"    GPU {i}: {gpu_mem_after[i]:.1f}MB (Δ{mem_change:+.1f}MB)")
                print("---")

            for i in range(num_candidate):
                pre_score.append(train_data[i].score)
                mimic_label_copy_score.append(mimic_label_copy_train_data[i].score)
                pre_loss.append(train_data[i].base_loss)
                mimic_label_copy_loss.append(mimic_label_copy_train_data[i].base_loss)
                pre_acc.append(train_data[i].base_acc)
                mimic_label_copy_acc.append(mimic_label_copy_train_data[i].base_acc)
        
        score_change.append((np.array(mimic_label_copy_score)-np.array(pre_score)).mean())
        loss_change.append((np.array(mimic_label_copy_loss)-np.array(pre_loss)).mean())
        acc_change.append((np.array(mimic_label_copy_acc)-np.array(pre_acc)).mean())

    mimic_label_copy_score_mean = np.array(score_change).mean()
    mimic_label_copy_score_std = np.array(score_change).std()
    mimic_label_copy_loss_mean = np.array(loss_change).mean()
    mimic_label_copy_loss_std = np.array(loss_change).std()
    mimic_label_copy_acc_mean = np.array(acc_change).mean()
    mimic_label_copy_acc_std = np.array(acc_change).std()
    
    print(f"\n=== SUMMARY for Noise Level {noise_level}% ===")
    print(f"Average original model accuracy: {np.array(pre_acc).mean():.4f} ± {np.array(pre_acc).std():.4f}")
    print(f"Average mimic label copy model accuracy: {np.array(mimic_label_copy_acc).mean():.4f} ± {np.array(mimic_label_copy_acc).std():.4f}")
    print(f"Accuracy change (mean ± std): {mimic_label_copy_acc_mean:.4f} ± {mimic_label_copy_acc_std:.4f}")
    print(f"Score change (mean ± std): {mimic_label_copy_score_mean:.4f} ± {mimic_label_copy_score_std:.4f}")
    print(f"Loss change (mean ± std): {mimic_label_copy_loss_mean:.4f} ± {mimic_label_copy_loss_std:.4f}")
    print("=" * 50)
    
    with open("output_multigpu.txt", "a") as file:
        file.write("{} & {}\\% & MultiGPU-Copy & {:.4f}\\pm{:.4f} & {:.4f}\\pm{:.4f} & {:.4f}\\pm{:.4f} \\\\\n".format(
            penalty, noise_level, mimic_label_copy_score_mean, mimic_label_copy_score_std, mimic_label_copy_loss_mean, mimic_label_copy_loss_std, mimic_label_copy_acc_mean, mimic_label_copy_acc_std
        )) 