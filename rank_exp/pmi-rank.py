import os
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from sklearn.decomposition import PCA
import sys
from lmi import lmi
from torchvision.datasets import CIFAR10
import time
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
np.seterr(divide='ignore', invalid='ignore')

def main():
    parser = argparse.ArgumentParser(description='Run experiments with different parameters.')
    parser.add_argument('--train_size', type=int, required=True, help='Size of the training set', default=50)
    parser.add_argument('--test_size', type=int, required=True, help='Size of the test set', default=50)
    parser.add_argument('--penalty', type=int, required=True, help='Penalty parameter', default=10000)
    parser.add_argument('--T', type=int, required=True, help='Number of iterations T', default=2000)
    parser.add_argument('--K', type=int, required=True, help='Number of repetitions K', default=10)
    parser.add_argument('--N_dims', type=int, required=True, help='Number of dimensions N_dims', default=128)
    parser.add_argument('--lmi_only', action='store_true', help='Only run LMI calculation')
    parser.add_argument('--pmi_only', action='store_true', help='Only run PMI calculation')
    parser.add_argument('--fix', action='store_true', help='Fixed probability value')
    parser.add_argument('--rand', action='store_true', help='Random probability value')

    args = parser.parse_args()
    print(f"Running experiment with train_size={args.train_size}, test_size={args.test_size}, penalty={args.penalty}, T={args.T}, K={args.K}, N_dims={args.N_dims}, lmi_only={args.lmi_only}, fix={args.fix}, rand={args.rand}")

    # Use the parsed arguments
    train_size = args.train_size
    test_size = args.test_size
    penalty = args.penalty
    T = args.T
    K = args.K
    N_dims = args.N_dims
    lmi_only = args.lmi_only
    pmi_only = args.pmi_only
    fix = args.fix
    rand = args.rand

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

    # images_a_0_embedding, images_a_1_embedding = normalized_datasets

    # print(images_a_0_embedding[0])
    # print(images_a_1_embedding[0])
    # sys.exit()

    def sigmoid(z):
        return 1/(1 + torch.exp(-z))


    def generate_r_D_r_T(rho, rand = rand, fix = fix, p1_small_rand = 0.4, p2_small_rand = 0.4, p1_large_rand = 0.6, p2_large_rand = 0.6):
        if rand:
            p1_small = random.choice([0.1, 0.2, 0.3, 0.4])
            p2_small = random.choice([0.1, 0.2, 0.3, 0.4])
            p1_large = random.choice([0.6, 0.7, 0.8, 0.9])
            p2_large = random.choice([0.6, 0.7, 0.8, 0.9])
        elif fix:
            p1_small = 0.4
            p2_small = 0.4
            p1_large = 0.6
            p2_large = 0.6
        else:
            p1_small = p1_small_rand
            p2_small = p2_small_rand
            p1_large = p1_large_rand
            p2_large = p2_large_rand

        prob = random.random()

        if prob < rho:  
            r_D = p1_small
            r_T = p2_small
        elif prob < 0.5: 
            r_D = p1_large
            r_T = p2_small
        elif prob < 0.5 + rho: 
            r_D = p1_large
            r_T = p2_large
        else:  
            r_D = p1_small
            r_T = p2_large
        
        return r_D, r_T


    def compute_hessian(mu, X):
        sigm = sigmoid(X @ mu.t())
        diag_sigm = (sigm * (1 - sigm)).flatten()
        res = torch.eye(X.size(1), device=device) / penalty
        res = res + (X.t() * diag_sigm) @ X
        return res

    def compute_score(mu0, Q0, lg0, mu1, Q1, lg1, mu2, Q2, lg2, rho):
        Q = Q1 + Q2 - Q0
        epsilon = 0
        # lg12 = - torch.log(torch.linalg.det(Q))
        Q_t_L = torch.linalg.cholesky(Q + epsilon * torch.eye(Q.size(0), device=Q.device))
        Q_t_L_inv = torch.linalg.solve_triangular(Q_t_L, torch.eye(Q_t_L.size(0), device=device), upper=False)
        Q_inv = Q_t_L_inv.T @ Q_t_L_inv
        # Q_inv = torch.inverse(Q)
        mu = torch.matmul(Q_inv, torch.matmul(Q1, mu1) + torch.matmul(Q2, mu2) - torch.matmul(Q0, mu0))

        lg12 = 2 * torch.sum(torch.log(torch.diagonal(Q_t_L)))

        lg = lg1+lg2-lg12-lg0

        sqr = torch.matmul(mu.T, torch.matmul(Q, mu)) - torch.matmul(mu1.T, torch.matmul(Q1, mu1)) - torch.matmul(mu2.T, torch.matmul(Q2, mu2)) + torch.matmul(mu0.T, torch.matmul(Q0, mu0))
        #sqr = sqr1 - torch.matmul(mu1.T, torch.matmul(Q1, mu1)) - torch.matmul(mu2.T, torch.matmul(Q2, mu2)) + torch.matmul(mu0.T, torch.matmul(Q0, mu0))
        score = 0.5 * (lg + sqr)
        # print("rho:", rho)
        # print("score:", score)
        # print("lg:", lg)
        # print("sqr:", sqr)
        # print("lg1,lg2,lg12,lg0:", lg1,lg2,lg12,lg0)
        # print("sqr1", torch.matmul(mu.T, torch.matmul(Q, mu)))
        # print("sqr2", torch.matmul(mu1.T, torch.matmul(Q1, mu1)))
        # print("sqr3", torch.matmul(mu2.T, torch.matmul(Q2, mu2)))
        # print("sqr4", torch.matmul(mu0.T, torch.matmul(Q0, mu0)))
        # print(lg1,lg2,lg12,lg0,sqr)
        return score.item()


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

    rho_values = [0.3418, 0.3785, 0.4054, 0.4269, 0.4439, 0.4604, 0.4734, 0.4841, 0.4963, 0.5000]

    if not lmi_only:
        spearman_tau_list = []
        rho_avg_scores = {rho: [] for rho in rho_values}
    
    if not pmi_only:
        # kendall_tau_lmi_default_list = []
        spearman_tau_lmi_aemine_list = []
        # kendall_tau_lmi_aeinfonce_list = []
        # lmi_avg_scores_default = {rho: [] for rho in rho_values}
        lmi_avg_scores_aemine = {rho: [] for rho in rho_values}
        # lmi_avg_scores_aeinfonce = {rho: [] for rho in rho_values}
    # Create a log file
    current_time = time.strftime("%Y%m%d_%H%M%S")
    log_file = open(f'experiment_log_train{train_size}_test{test_size}_T{T}_K{K}_penalty{penalty}_fix{args.fix}_rand{args.rand}_{current_time}.txt', 'a')

    def log_message(*args, **kwargs):
        # Convert all arguments to strings
        message = ' '.join(str(arg) for arg in args)
        
        # Print to console
        print(message, **kwargs)
        
        # Print to file
        print(message, file=log_file, **kwargs)
        
        # Ensure the log is written immediately
        log_file.flush()

    for k in range(K):
        log_message(f"Running experiment repetition {k + 1}/{K}")
        if not lmi_only:
            scores = []

        if not pmi_only:
            # lmi_scores_default = []
            lmi_scores_aemine = []
            # lmi_scores_aeinfonce = []
        rhos = []

        p1_small = random.choice([0.1, 0.2, 0.3, 0.4])
        p2_small = random.choice([0.1, 0.2, 0.3, 0.4])
        p1_large = random.choice([0.6, 0.7, 0.8, 0.9])
        p2_large = random.choice([0.6, 0.7, 0.8, 0.9])

        for rho in rho_values:
            log_message(f"Running experiment with rho = {rho}")
            if not lmi_only:
                rho_scores = []

            # Initialize lists to store all T groups of data for this rho iteration
            all_train_data = []
            all_test_data = []

            # Start timing for score calculation
            start_time_score = time.time()

            for _ in tqdm(range(T), desc=f"Repeating for rho = {rho}", ncols=100):
                r_D, r_T = generate_r_D_r_T(rho, p1_small, p2_small, p1_large, p2_large) 

                train_data, train_targets, test_data, test_targets = load_data_for_r_D_r_T(r_D, r_T)

                if not pmi_only:
                    # Concatenate targets to data
                    train_data_with_target = np.concatenate((train_data, train_targets[:, None]), axis=1)
                    test_data_with_target = np.concatenate((test_data, test_targets[:, None]), axis=1)

                    # Append each group of data to the list
                    all_train_data.append(train_data_with_target.flatten())
                    all_test_data.append(test_data_with_target.flatten())

                if not lmi_only:
                    model_train = LogisticRegression(fit_intercept=False, C=penalty, max_iter=5000)
                    model_train.fit(train_data, train_targets)
                    train_X = torch.tensor(train_data, dtype=torch.float32, device=device)
                    mu_train = torch.tensor(model_train.coef_, dtype=torch.float32, device=device)
                    Q_train = compute_hessian(mu_train, train_X)
                    Q_train_L = torch.linalg.cholesky(Q_train)
                    # Q_train_L_inv = torch.linalg.solve_triangular(Q_train_L, torch.eye(Q_train_L.size(0), device=device), upper=False)
                    lg1 = 2 * torch.sum(torch.log(torch.diagonal(Q_train_L)))
                    
                    model_test = LogisticRegression(fit_intercept=False, C=penalty, max_iter=5000)
                    model_test.fit(test_data, test_targets)
                    test_X = torch.tensor(test_data, dtype=torch.float32, device=device)
                    mu_test = torch.tensor(model_test.coef_, dtype=torch.float32, device=device)
                    Q_test = compute_hessian(mu_test, test_X)
                    Q_test_L = torch.linalg.cholesky(Q_test)
                    lg2 = 2 * torch.sum(torch.log(torch.diagonal(Q_test_L)))
                    mu0 = np.zeros_like(mu_train.cpu())
                    mu0 = torch.tensor(mu0, dtype=torch.float32, device=device)
                    Q0 = np.eye(train_data.shape[1]) / penalty
                    Q0 = torch.tensor(Q0, dtype=torch.float32, device=device)
                    lg0 = - train_data.shape[1] * torch.log(torch.tensor(penalty))
                    
                    score = compute_score(mu0.t(), Q0, lg0, mu_train.t(), Q_train, lg1, mu_test.t(), Q_test, lg2, rho)

                    rho_scores.append(score)

            # End timing for score calculation
            end_time_score = time.time()
            score_time = end_time_score - start_time_score
            log_message(f"Time taken for score calculation over {T} iterations: {score_time:.4f} seconds")

            if not pmi_only:
                # Concatenate all T groups of data into a large array for this rho iteration
                final_train_data = np.array(all_train_data)
                final_test_data = np.array(all_test_data)

                # Output the shape of the final concatenated data for this rho iteration
                print(f"Final train data shape for rho = {rho}:", final_train_data.shape)
                print(f"Final test data shape for rho = {rho}:", final_test_data.shape)

                # Start timing for lmi_scores_aemine calculation
                start_time_lmi = time.time()

                # lmi_score_default = np.nanmean(lmi.lmi(final_train_data, final_test_data)[0])
                lmi_score_aemine = np.nanmean(lmi.lmi(final_train_data, final_test_data, regularizer='models.AEMINE', N_dims=N_dims, epochs=500, quiet=True)[0])
                # lmi_score_aeinfonce = np.nanmean(lmi.lmi(final_train_data, final_test_data, regularizer='models.AEInfoNCE')[0])

                # End timing for lmi_scores_aemine calculation
                end_time_lmi = time.time()
                lmi_time = end_time_lmi - start_time_lmi
                log_message(f"Time taken for LMI score (AEMINE) calculation: {lmi_time:.4f} seconds")
                # sys.exit()

            rhos.append(rho)
            if not lmi_only:
                avg_score = np.mean(rho_scores)
                var_score = np.var(rho_scores)
                scores.append(avg_score)
                rho_avg_scores[rho].append(avg_score)
                log_message(f"Average score for rho = {rho}: {avg_score:.4f} (var = {var_score:.4f})")
            
            if not pmi_only:
                # lmi_scores_default.append(lmi_score_default)
                lmi_scores_aemine.append(lmi_score_aemine)
                # lmi_scores_aeinfonce.append(lmi_score_aeinfonce)
                # lmi_avg_scores_default[rho].append(lmi_score_default)
                lmi_avg_scores_aemine[rho].append(lmi_score_aemine)
                # lmi_avg_scores_aeinfonce[rho].append(lmi_score_aeinfonce)

                # log_message(f"LMI score (default) for rho = {rho}: {lmi_score_default:.4f}")
                log_message(f"LMI score (AEMINE) for rho = {rho}: {lmi_score_aemine:.4f}")
                # log_message(f"LMI score (AEInfoNCE) for rho = {rho}: {lmi_score_aeinfonce:.4f}")

        if not lmi_only:
            spearman_corr, _ = spearmanr(rhos, scores)
            spearman_tau_list.append(spearman_corr)
            log_message(f"Spearman's Rho of average scores for repetition {k + 1}/{K}: {spearman_corr:.4f}")

        if not pmi_only:
            spearman_corr_lmi_aemine, _ = spearmanr(rhos, lmi_scores_aemine)
            spearman_tau_lmi_aemine_list.append(spearman_corr_lmi_aemine)
            log_message(f"Spearman's Rho of LMI scores (AEMINE) for repetition {k + 1}/{K}: {spearman_corr_lmi_aemine:.4f}")

    log_message("\nFinal Results:")
    if not lmi_only:
        avg_spearman_tau = np.mean(spearman_tau_list)
        var_spearman_tau = np.var(spearman_tau_list)
        final_avg_scores = {rho: np.mean(rho_avg_scores[rho]) for rho in rho_values}
        final_var_scores = {rho: np.var(rho_avg_scores[rho]) for rho in rho_values}
        log_message(f"Average Spearman's Rho of average scores over {K} repetitions: {avg_spearman_tau:.4f} (var = {var_spearman_tau:.4f})")

    if not pmi_only:
        avg_spearman_tau_lmi_aemine = np.mean(spearman_tau_lmi_aemine_list)
        var_spearman_tau_lmi_aemine = np.var(spearman_tau_lmi_aemine_list)
        final_avg_scores_lmi_aemine = {rho: np.mean(lmi_avg_scores_aemine[rho]) for rho in rho_values}
        final_var_scores_lmi_aemine = {rho: np.var(lmi_avg_scores_aemine[rho]) for rho in rho_values}
        log_message(f"Average Spearman's Rho of LMI scores (AEMINE) over {K} repetitions: {avg_spearman_tau_lmi_aemine:.4f} (var = {var_spearman_tau_lmi_aemine:.4f})")
    log_message("Average scores and variances for each rho:")
    for rho in rho_values:
        if not lmi_only:
            log_message(f"  rho = {rho}: avg = {final_avg_scores[rho]:.4f}, var = {final_var_scores[rho]:.4f}")
        
        if not pmi_only:
            log_message(f"  rho = {rho}: avg_lmi_aemine = {final_avg_scores_lmi_aemine[rho]:.4f}, var_lmi_aemine = {final_var_scores_lmi_aemine[rho]:.4f}")

    # Close the log file at the end
    log_file.close()

if __name__ == "__main__":
    main()