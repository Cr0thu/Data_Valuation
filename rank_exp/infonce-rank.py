import os
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from torchvision.datasets import CIFAR10
import torchvision.models as models
from scipy.stats import kendalltau
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# base_path = "/scratch/gpfs/xq5452/test/pmi/PMI/experiment-4.1/colorized-MNIST-master"
# train_path = os.path.join(base_path, "training")
# test_path = os.path.join(base_path, "testing")

resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet18.fc = torch.nn.Identity()
resnet18.to(device).eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# Load the original CIFAR-10 dataset
cifar10_dataset = CIFAR10(root='../data', train=True, download=True)
images, labels = cifar10_dataset.data, np.array(cifar10_dataset.targets)

# Select indices for labels 0 and 1
selected_indices_0 = labels == 0
selected_indices_1 = labels == 1

def data_preprocess(images):
    images_0 = torch.stack([preprocess(Image.fromarray(image)) for image in images[selected_indices_0]]).to(device)
    images_1 = torch.stack([preprocess(Image.fromarray(image)) for image in images[selected_indices_1]]).to(device)
    with torch.no_grad():
        embedding_0 = resnet18(images_0)
        embedding_1 = resnet18(images_1)
        # embedding_0 = torch.concatenate([embedding_0, torch.ones(embedding_0.size()[0],1).to(device)], dim=1)
        # embedding_1 = torch.concatenate([embedding_1, torch.ones(embedding_1.size()[0],1).to(device)], dim=1)
        # perm = torch.randperm(len(embedding_0))
        # embedding_0 = embedding_0[perm]
        # embedding_1 = embedding_1[perm]
    return embedding_0, embedding_1

# Preprocess images for labels 0 and 1
images_a_0_embedding, images_a_1_embedding = data_preprocess(images)

# Since we are only using labels 0 and 1, we don't need to load another set of images
datasets = [
    images_a_0_embedding, 
    images_a_1_embedding
]

# Concatenate all datasets to calculate the mean and standard deviation
combined_data = torch.cat(datasets, dim=0)  # Shape: (20000, 512)

# Compute mean and std across the feature dimension (dim=0)
mean = combined_data.mean(dim=0, keepdim=True)
std = combined_data.std(dim=0, keepdim=True)

# Normalize each dataset using the same mean and std
normalized_datasets = [(dataset - mean) / (std + 1e-6) for dataset in datasets]

# Unpack the normalized datasets and append a 1 to each data point
images_a_0_embedding, images_a_1_embedding = [
    torch.cat([dataset, torch.ones(dataset.size(0), 1, device=device)], dim=1) 
    for dataset in normalized_datasets
]

train_size = 100
test_size = 100

# normalize = transforms.Normalize(mean=[0.5], std=[0.5])
# preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     normalize
# ])

rho_values = [0.2621, 0.2750, 0.2894, 0.3062, 0.3271, 0.3552, 0.3950, 0.4514, 0.4877, 0.5000]
T_ = 1000
K = 20
temperature = 0.07  # Temperature parameter for InfoNCE
feature_dim = 1024   # Feature dimension
batch_size = 64      # Batch size
epochs = 100         # Number of training epochs
lr = 0.001          # Learning rate

p1 = 0.2
p2 = 0.8


class FeatureEncoder(nn.Module):
    """Feature encoder that maps input data to feature space"""
    def __init__(self, input_dim, feature_dim):
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        # Feature normalization, important for InfoNCE
        features = nn.functional.normalize(features, dim=1)
        return features


# def load_colored_mnist_data(path, label, color, preprocess):
#     data = []
#     targets = []
#     label_path = os.path.join(path, str(label), color)
#     for img_name in os.listdir(label_path):
#         img_path = os.path.join(label_path, img_name)
#         img = Image.open(img_path).convert('RGB')
#         img = preprocess(img)
#         data.append(img.view(-1).numpy())  # flatten
#         targets.append(label)
#     return data, targets


# def load_data_for_r_D_r_T(r_D, r_T):
#     train_data_0, train_targets_0 = load_colored_mnist_data(train_path, 0, "blue", preprocess)
#     train_data_1, train_targets_1 = load_colored_mnist_data(train_path, 1, "blue", preprocess)
#     test_data_0,  test_targets_0  = load_colored_mnist_data(test_path,  0, "blue", preprocess)
#     test_data_1,  test_targets_1  = load_colored_mnist_data(test_path,  1, "blue", preprocess)

#     n_train_0 = random.sample(range(len(train_data_0)), int(train_size * r_D))
#     n_train_1 = random.sample(range(len(train_data_1)), int(train_size * (1 - r_D)))
#     n_test_0  = random.sample(range(len(test_data_0)),  int(test_size  * r_T))
#     n_test_1  = random.sample(range(len(test_data_1)),  int(test_size  * (1 - r_T)))

#     train_data = [train_data_0[i] for i in n_train_0] + [train_data_1[i] for i in n_train_1]
#     train_targets = [train_targets_0[i] for i in n_train_0] + [train_targets_1[i] for i in n_train_1]
#     test_data = [test_data_0[i] for i in n_test_0] + [test_data_1[i] for i in n_test_1]
#     test_targets = [test_targets_0[i] for i in n_test_0] + [test_targets_1[i] for i in n_test_1]

#     return (np.array(train_data), np.array(train_targets),
#             np.array(test_data),  np.array(test_targets))

def load_data_for_r_D_r_T(r_D, r_T):
        n_train_0 = random.sample(range(len(images_a_0_embedding[:4000])), int(train_size * r_D))
        n_train_1 = random.sample(range(len(images_a_1_embedding[:4000])), int(train_size - train_size * r_D))
        n_test_0 = random.sample(range(len(images_a_0_embedding[4000:])), int(test_size * r_T))
        n_test_1 = random.sample(range(len(images_a_1_embedding[4000:])), int(test_size - test_size * r_T))
        
        # Concatenate data and targets first
        train_data = torch.cat([
            images_a_0_embedding[:4000][n_train_0],
            images_a_1_embedding[:4000][n_train_1]
        ])
        train_targets = torch.cat([
            torch.zeros(len(n_train_0), device=device),
            torch.ones(len(n_train_1), device=device)
        ])
        
        # Create random permutation indices
        train_perm = torch.randperm(len(train_data))
        # Shuffle both data and targets using same permutation
        train_data = train_data[train_perm]
        train_targets = train_targets[train_perm]

        # Same for test data
        test_data = torch.cat([
            images_a_0_embedding[4000:][n_test_0],
            images_a_1_embedding[4000:][n_test_1]
        ])
        test_targets = torch.cat([
            torch.zeros(len(n_test_0), device=device),
            torch.ones(len(n_test_1), device=device)
        ])
        
        # Shuffle test data
        test_perm = torch.randperm(len(test_data))
        test_data = test_data[test_perm]
        test_targets = test_targets[test_perm]

        # Convert to numpy arrays for sklearn
        train_data = train_data.cpu().numpy()
        train_targets = train_targets.cpu().numpy()
        test_data = test_data.cpu().numpy()
        test_targets = test_targets.cpu().numpy()

        return train_data, train_targets, test_data, test_targets

def generate_r_D_r_T(rho):
    prob = random.random()
    if prob < rho:
        r_D, r_T = p1, p1
    elif prob < 0.5:
        r_D, r_T = p1, p2
    elif prob < 0.5 + rho:
        r_D, r_T = p2, p2
    else:
        r_D, r_T = p2, p1
    return r_D, r_T


def infonce_loss(q, k, temperature=0.07):
    """
    Calculate InfoNCE loss
    q: query vectors, shape (batch_size, feature_dim)
    k: key vectors, shape (batch_size, feature_dim)
    temperature: temperature coefficient
    """
    # Compute similarity matrix (batch_size x batch_size)
    sim_matrix = torch.matmul(q, k.transpose(0, 1)) / temperature
    
    # Positive samples are on the diagonal
    labels = torch.arange(q.shape[0]).to(sim_matrix.device)
    
    # Calculate InfoNCE loss (cross entropy form)
    loss = nn.CrossEntropyLoss()(sim_matrix, labels)
    
    return loss


def estimate_mutual_information_infonce(train_data, train_targets, test_data, test_targets, 
                                        epochs=50, batch_size=32, temperature=0.07):
    """
    Estimate mutual information between train and test sets using InfoNCE method
    """
    input_dim = train_data.shape[1]
    
    # Create two feature encoders for train and test sets
    train_encoder = FeatureEncoder(input_dim, feature_dim).to(device)
    test_encoder = FeatureEncoder(input_dim, feature_dim).to(device)
    
    # Optimizer
    optimizer = optim.Adam(list(train_encoder.parameters()) + 
                          list(test_encoder.parameters()), 
                          lr=lr)
    
    # Convert to PyTorch tensors
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    
    # Create data loaders
    train_indices = list(range(len(train_data)))
    test_indices = list(range(len(test_data)))
    
    # Train the model
    train_encoder.train()
    test_encoder.train()
    
    for epoch in range(epochs):
        # Shuffle indices
        random.shuffle(train_indices)
        random.shuffle(test_indices)
        
        total_loss = 0.0
        num_batches = 0
        
        # Process by batch
        for i in range(0, min(len(train_indices), len(test_indices)), batch_size):
            if i + batch_size > min(len(train_indices), len(test_indices)):
                continue
                
            # Get current batch indices
            batch_train_indices = train_indices[i:i+batch_size]
            batch_test_indices = test_indices[i:i+batch_size]
            
            # Extract batch data
            batch_train_data = train_data_tensor[batch_train_indices]
            batch_test_data = test_data_tensor[batch_test_indices]
            
            # Get feature representations
            train_features = train_encoder(batch_train_data)
            test_features = test_encoder(batch_test_data)
            
            # Calculate InfoNCE loss (bidirectional)
            loss_train_test = infonce_loss(train_features, test_features, temperature)
            loss_test_train = infonce_loss(test_features, train_features, temperature)
            loss = (loss_train_test + loss_test_train) / 2
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    # Evaluation phase
    train_encoder.eval()
    test_encoder.eval()
    
    with torch.no_grad():
        # Get all features
        train_features = train_encoder(train_data_tensor)
        test_features = test_encoder(test_data_tensor)
        
        # Calculate InfoNCE bound
        sim_matrix = torch.matmul(train_features, test_features.transpose(0, 1)) / temperature
        
        # Extract diagonal elements (positive sample similarities)
        pos_samples = torch.diag(sim_matrix)
        
        # Calculate mutual information estimate using InfoNCE bound
        n_samples = pos_samples.size(0)
        mi_estimate = torch.mean(pos_samples) - torch.log(torch.tensor(n_samples, dtype=torch.float32, device=device))
        
    return mi_estimate.cpu().numpy()


def main_experiment_infonce():
    kendall_tau_list = []
    rho_avg_scores = {rho: [] for rho in rho_values}

    for k in range(K):
        print(f"Running experiment repetition {k + 1}/{K}")
        scores = []
        rhos = []

        for rho in rho_values:
            print(f"  rho = {rho}")
            rho_scores = []

            for _ in tqdm(range(T_), desc=f"Processing rho={rho}"):
                r_D, r_T = generate_r_D_r_T(rho)

                train_data, train_targets, test_data, test_targets = load_data_for_r_D_r_T(r_D, r_T)
                train_data = train_data.astype(np.float32)
                test_data = test_data.astype(np.float32)

                # Use InfoNCE to estimate mutual information
                mi_estimate = estimate_mutual_information_infonce(
                    train_data, train_targets, 
                    test_data, test_targets,
                    epochs=epochs, 
                    batch_size=batch_size, 
                    temperature=temperature
                )
                
                rho_scores.append(mi_estimate)

            avg_score = np.mean(rho_scores)
            scores.append(avg_score)
            rhos.append(rho)
            rho_avg_scores[rho].append(avg_score)
            print(f"    Average InfoNCE MI estimate for rho = {rho}: {avg_score:.4f}")

        kendall_corr, _ = kendalltau(rhos, scores)
        kendall_tau_list.append(kendall_corr)
        print(f"Kendall's Tau for repetition {k+1}/{K}: {kendall_corr:.4f}\n")

    avg_kendall_tau = np.mean(kendall_tau_list)
    final_avg_scores = {rho: np.mean(rho_avg_scores[rho]) for rho in rho_values}

    print("\nFinal Results:")
    print(f"Average Kendall's Tau over {K} repetitions: {avg_kendall_tau:.4f}")
    print("Average InfoNCE MI estimates for each rho:")
    for rho, avg_score in final_avg_scores.items():
        print(f"  rho = {rho}: {avg_score:.4f}")


if __name__ == "__main__":
    main_experiment_infonce()