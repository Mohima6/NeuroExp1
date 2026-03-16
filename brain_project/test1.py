import numpy as np
from scipy.stats import wishart
import scipy.linalg as la
import pandas as pd  # for CSV 
# Parameters
n_subjects = 977
n_regions = 65
n_networks = 7
np.random.seed(42)          # for reproducibility
#  each region to one of the 7 functional networks (sizes chosen to sum to 65)
network_sizes = [10, 8, 12, 9, 11, 7, 8]
region_to_network = np.repeat(np.arange(n_networks), network_sizes)
# correlation matrix with block structure
base_corr = np.eye(n_regions)
for net in range(n_networks):
    mask = region_to_network == net
    # Within‑network correlations ~ 0.5 + small random variation
    block = 0.5 + 0.1 * np.random.randn(np.sum(mask), np.sum(mask))
    # Make symmetric and set diagonal to 1
    block = (block + block.T) / 2
    np.fill_diagonal(block, 1.0)
    base_corr[np.ix_(mask, mask)] = block
# Between‑network correlations (all others) set to ~0.1
for i in range(n_regions):
    for j in range(i+1, n_regions):
        if region_to_network[i] != region_to_network[j]:
            base_corr[i, j] = 0.1 + 0.05 * np.random.randn()
            base_corr[j, i] = base_corr[i, j]
# positive definiteness (small ridge if needed)
eigvals = np.linalg.eigvalsh(base_corr)
min_eig = np.min(eigvals)
if min_eig < 1e-6:
    base_corr += (1e-6 - min_eig) * np.eye(n_regions)
# PMAT scores
# PMAT_A_CR: number correct (0–24). Approx normal with mean 15, std 4.
pmat_cr = np.random.normal(loc=15, scale=4, size=n_subjects)
pmat_cr = np.clip(pmat_cr, 0, 24)            # clip to plausible range
# PMAT_A_SI: skipped items (0–24). Negatively correlated with correct.
pmat_si = 20 - 0.6 * pmat_cr + np.random.normal(0, 2, size=n_subjects)
pmat_si = np.clip(pmat_si, 0, 24)
# Center scores for effect modulation
pmat_cr_centered = pmat_cr - np.mean(pmat_cr)
# subject‑specific correlation matrices
all_corr = []
for i in range(n_subjects):
    # Start from base correlation
    subj_corr = base_corr.copy()
    # systematic effect: edges within network 0 and network 1
    # become stronger with higher PMAT_A_CR.
    net0 = region_to_network == 0
    net1 = region_to_network == 1
    effect = 0.01 * pmat_cr_centered[i]          # small effect size
    subj_corr[np.ix_(net0, net0)] += effect
    subj_corr[np.ix_(net1, net1)] += effect
    #  symmetry and diagonal = 1
    subj_corr = (subj_corr + subj_corr.T) / 2
    np.fill_diagonal(subj_corr, 1.0)
    #  extreme values to avoid numerical issues
    subj_corr = np.clip(subj_corr, -1, 1)
    # positive definiteness
    eigvals = np.linalg.eigvalsh(subj_corr)
    if np.min(eigvals) < 1e-6:
        subj_corr += (1e-6 - np.min(eigvals)) * np.eye(n_regions)
    # random noise using Wishart distribution for realistic variability
    # current correlation; scale matrix for a covariance
    # Wishart requires a positive definite scale. We'll use subj_corr * degrees_of_freedom.
    dof = n_regions + 10          # degrees of freedom > n_regions - 1
    scale = subj_corr * dof
    # Ensure scale is positive definite
    eigvals_scale = np.linalg.eigvalsh(scale)
    if np.min(eigvals_scale) < 1e-6:
        scale += (1e-6 - np.min(eigvals_scale)) * np.eye(n_regions)
    #  covariance matrix from Wishart
    cov = wishart.rvs(df=dof, scale=scale / dof, random_state=i)
    # Convert covariance to correlation
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    all_corr.append(corr)
# 3D array
all_corr = np.array(all_corr)          # shape (977, 65, 65)
# Save to CSV files
#  upper triangle (excluding diagonal) of each correlation matrix
# yields a 2D array of shape (977, n_edges)
i_upper = np.triu_indices(n_regions, k=1)   # indices of upper triangle without diagonal
n_edges = len(i_upper[0])
flattened_corr = np.array([corr[i_upper] for corr in all_corr])
# column names for the edges (e.g., "r0_r1", "r0_r2", ...)
col_names = [f"r{i}_r{j}" for i, j in zip(i_upper[0], i_upper[1])]
# Save correlations as CSV
df_corr = pd.DataFrame(flattened_corr, columns=col_names)
df_corr.to_csv('synthetic_correlations.csv', index=False)
# Save PMAT scores as CSV
df_pmat = pd.DataFrame({
    'PMAT_A_CR': pmat_cr,
    'PMAT_A_SI': pmat_si
})
df_pmat.to_csv('synthetic_pmat.csv', index=False)


print("Synthetic dataset saved as CSV.")
print(f"Correlation matrices shape: {all_corr.shape}")
print(f"Flattened correlations shape: {flattened_corr.shape} (saved as synthetic_correlations.csv)")
print(f"PMAT_A_CR range: [{pmat_cr.min():.2f}, {pmat_cr.max():.2f}]")
print(f"PMAT_A_SI range: [{pmat_si.min():.2f}, {pmat_si.max():.2f}]")
print("PMAT scores saved as synthetic_pmat.csv")
