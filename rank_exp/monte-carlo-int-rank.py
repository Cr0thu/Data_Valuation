import os
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import kendalltau
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import torchvision.models as models


os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


train_size = 50
test_size = 30

resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet18.fc = torch.nn.Identity()
resnet18.to(device).eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
labels = np.load('../CIFAR-10-C/labels.npy')
selected_indices_0 = labels == 0
selected_indices_1 = labels == 1

rho_values = [0.2621, 0.2750, 0.2894, 0.3062, 0.3271, 0.3552, 0.3950, 0.4514, 0.4877, 0.5000]
T_ = 100
K = 1
penalty = 1
alpha = 1.0  

p1 = 0.2
p2 = 0.8

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

def load_data_for_r_D_r_T(r_D, r_T):
    n_train_0 = random.sample(range(len(images_a_0_embedding[:4000])), int(train_size * r_D))
    n_train_1 = random.sample(range(len(images_a_1_embedding[:4000])), int(train_size * (1-r_D)))
    n_test_0 = random.sample(range(len(images_a_0_embedding[4000:])), int(test_size * r_T))
    n_test_1 = random.sample(range(len(images_a_1_embedding[4000:])), int(test_size * (1-r_T)))
    
    train_data = torch.cat([
        images_a_0_embedding[:4000][n_train_0],
        images_a_1_embedding[:4000][n_train_1]
    ])
    train_targets = torch.cat([
        torch.zeros(len(n_train_0), device=device),
        torch.ones(len(n_train_1), device=device)
    ])
    test_data = torch.cat([
        images_a_0_embedding[4000:][n_test_0],
        images_a_1_embedding[4000:][n_test_1]
    ])
    test_targets = torch.cat([
        torch.zeros(len(n_test_0), device=device),
        torch.ones(len(n_test_1), device=device)
    ])

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


def fit_logistic_regression(train_data, train_targets, penalty=1.0):

    C_ = 1.0 / penalty
    clf = LogisticRegression(fit_intercept=False, C=C_, max_iter=5000)
    clf.fit(train_data, train_targets)
    w = clf.coef_[0]  # (d,)
    return w

def compute_hessian(w, X, y, penalty=1.0, alpha=1.0):

    N, d = X.shape
    w_t = torch.tensor(w, dtype=torch.float32, device=device)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)


    z = X_t.mv(w_t)  # (N,)
    p = torch.sigmoid(z)  # (N,)

    # diag(p(1-p))
    w_vec = p * (1.0 - p)  # (N,)

    X_weighted = X_t * w_vec.unsqueeze(1)  # (N,d)
    Hess_data = X_t.t().mm(X_weighted)     # (d,d)

    Hess_prior = penalty * torch.eye(d, device=device)

    Hess = Hess_data + Hess_prior
    return Hess.cpu().numpy()  


def sample_from_posterior_laplace(w_map, Hess, n_samples=100):
    d = len(w_map)
    
    # Compute Cholesky decomposition of the Hessian
    L = np.linalg.cholesky(Hess + 1e-10*np.eye(d))
    
    # Generate standard normal samples
    z = np.random.randn(n_samples, d)
    
    # Solve the system using triangular solve
    samples = w_map + np.linalg.solve(L.T, np.linalg.solve(L, z.T)).T
    return samples


def sample_from_prior(d, n_samples=100, alpha=1.0):
    # Directly sample from standard normal and scale by alpha
    samples = alpha * np.random.randn(n_samples, d)
    return samples

def log_likelihood_of_dataset(w, data, targets):

    logits = data @ w  # (N,)

    logits = np.clip(logits, -20, 20)
    probs = 1.0/(1.0 + np.exp(-logits))

    eps = 1e-12
    ll = 0.0
    for i, yi in enumerate(targets):
        if yi == 1:
            ll += np.log(probs[i] + eps)
        else:
            ll += np.log(1.0 - probs[i] + eps)
    return ll

def monte_carlo_estimate_likelihood(data, targets, list_of_w):

    log_likelihoods = []
    for w in list_of_w:
        ll = log_likelihood_of_dataset(w, data, targets)
        log_likelihoods.append(ll)

    log_likelihoods = np.array(log_likelihoods)
    max_ll = np.max(log_likelihoods)
    shifted_sum = np.sum(np.exp(log_likelihoods - max_ll))
    mean_likelihood = np.exp(max_ll)* (shifted_sum / len(list_of_w))
    return mean_likelihood

def compute_score_laplace(train_data, train_targets, test_data, test_targets,
                          penalty=1.0, alpha=1.0, n_samples=100):

    # 1) w_map
    w_map = fit_logistic_regression(train_data, train_targets, penalty=penalty)

    # 2) Hessian
    H = compute_hessian(w_map, train_data, train_targets, penalty=penalty, alpha=alpha)

    posterior_samples = sample_from_posterior_laplace(w_map, H, n_samples=n_samples)

    d = len(w_map)
    prior_samples = sample_from_prior(d, n_samples=n_samples, alpha=alpha)

    pT_given_D = monte_carlo_estimate_likelihood(test_data, test_targets, posterior_samples)
    pT = monte_carlo_estimate_likelihood(test_data, test_targets, prior_samples)

    pT_given_D = max(pT_given_D, 1e-15)
    pT = max(pT, 1e-15)

    return np.log(pT_given_D) - np.log(pT)

def main_experiment():
    kendall_tau_list = []
    rho_avg_scores = {rho: [] for rho in rho_values}

    alpha_ = 1.0
    penalty_ = 1
    n_samples_ = 1000

    for k in range(K):
        print(f"Running experiment repetition {k + 1}/{K}")
        scores = []
        rhos = []

        for rho in rho_values:
            print(f"  rho = {rho}")
            rho_scores = []

            # for _ in tqdm(range(T_), desc="Repeating for rho"):
            for _ in tqdm(range(T_), desc="Repeating for rho", ncols=100):

                r_D, r_T = generate_r_D_r_T(rho)

                train_data, train_targets, test_data, test_targets = load_data_for_r_D_r_T(r_D, r_T)
                train_data = train_data.astype(np.float32)
                test_data = test_data.astype(np.float32)

                score = compute_score_laplace(train_data, train_targets, 
                                              test_data, test_targets,
                                              penalty=penalty_,
                                              alpha=alpha_,
                                              n_samples=n_samples_)
                rho_scores.append(score)

            avg_score = np.mean(rho_scores)
            scores.append(avg_score)
            rhos.append(rho)
            rho_avg_scores[rho].append(avg_score)
            print(f"    Average PMI for rho = {rho}: {avg_score:.4f}")

        kendall_corr, _ = kendalltau(rhos, scores)
        kendall_tau_list.append(kendall_corr)
        print(f"Kendall's Tau for repetition {k+1}/{K}: {kendall_corr:.4f}\n")

    avg_kendall_tau = np.mean(kendall_tau_list)
    final_avg_scores = {rho: np.mean(rho_avg_scores[rho]) for rho in rho_values}

    print("\nFinal Results:")
    print(f"Average Kendall's Tau over {K} repetitions: {avg_kendall_tau:.4f}")
    print("Average scores for each rho:")
    for rho, avg_score in final_avg_scores.items():
        print(f"  rho = {rho}: {avg_score:.4f}")


if __name__ == "__main__":
    main_experiment()
