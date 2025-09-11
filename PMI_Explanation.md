# PMI (Pointwise Mutual Information) Values Explanation

## What are PMI Values?

PMI (Pointwise Mutual Information) values measure the mutual information between training and test datasets in the context of logistic regression models. In this code, PMI is calculated using a Bayesian framework that compares the information content between different datasets.

## How PMI is Calculated

The PMI calculation in this code follows these steps:

1. **Model Training**: Logistic regression models are trained on different datasets
2. **Hessian Computation**: The Hessian matrix is computed for each model
3. **Score Calculation**: The PMI score is calculated using the formula:

```python
def compute_score(mu0, Q0, lg0, mu1, Q1, lg1, mu2, Q2, lg2):
    Q = Q1 + Q2 - Q0
    Q_t_L = torch.linalg.cholesky(Q)
    Q_t_L_inv = torch.linalg.solve_triangular(Q_t_L, torch.eye(Q_t_L.size(0), device=device), upper=False)
    Q_inv = Q_t_L_inv.T @ Q_t_L_inv
    mu = torch.matmul(Q_inv, torch.matmul(Q1, mu1) + torch.matmul(Q2, mu2) - torch.matmul(Q0, mu0))

    lg12 = 2 * torch.sum(torch.log(torch.diagonal(Q_t_L)))
    lg = lg1 + lg2 - lg12 - lg0

    sqr = torch.matmul(mu.T, torch.matmul(Q, mu)) - torch.matmul(mu1.T, torch.matmul(Q1, mu1)) - torch.matmul(mu2.T, torch.matmul(Q2, mu2)) + torch.matmul(mu0.T, torch.matmul(Q0, mu0))

    score = 0.5 * (lg + sqr)
    return score.item()
```

## What the PMI Values Mean

- **Higher PMI values** indicate stronger mutual information between datasets
- **Lower PMI values** indicate weaker mutual information
- **Positive PMI differences** suggest that the modified dataset (e.g., mimic label copy) has stronger mutual information with the test set
- **Negative PMI differences** suggest that the original dataset has stronger mutual information

## Output Format

The modified code now outputs PMI values in several places:

1. **Initial Iteration**: Shows PMI scores for the first iteration
2. **Periodic Updates**: Every 100 iterations, shows current PMI statistics
3. **Final Summary**: Comprehensive PMI statistics across all iterations
4. **File Output**: PMI values are saved to output files for analysis

## Example Output

```
Iteration 0:
  Original model accuracy: 0.8234
  Original model PMI score: -1.234567
  Mimic label copy model accuracy: 0.8456
  Mimic label copy model PMI score: -1.123456
  Accuracy difference: 0.0222
  PMI score difference: 0.111111

Iteration 100: PMI scores - Original: -1.245678 ± 0.012345, Mimic: -1.134567 ± 0.011234, Difference: 0.111111

=== SUMMARY ===
Average original model accuracy: 0.8234 ± 0.0123
Average original model PMI score: -1.234567 ± 0.012345
Average mimic label copy model accuracy: 0.8456 ± 0.0112
Average mimic label copy model PMI score: -1.123456 ± 0.011234
Accuracy change (mean ± std): 0.0222 ± 0.0011
PMI score change (mean ± std): 0.111111 ± 0.001111
Loss change (mean ± std): -0.0234 ± 0.0023
```

## Interpretation

- The PMI values help understand how well the training data aligns with the test data distribution
- Changes in PMI values when applying data modifications (like mimic label copy) show how those modifications affect the mutual information
- This can be useful for understanding data bias and model generalization 