import numpy as np
from scipy.stats import wishart
import scipy.linalg as la
import pandas as pd  
n_subjects = 977
n_regions = 65
n_networks = 7
np.random.seed(42)          
network_sizes = [10, 8, 12, 9, 11, 7, 8]
region_to_network = np.repeat(np.arange(n_networks), network_sizes)
# correlation matrix
base_corr = np.eye(n_regions)
for net in range(n_networks):
    mask = region_to_network == net
    block = 0.5 + 0.1 * np.random.randn(np.sum(mask), np.sum(mask))
    block = (block + block.T) / 2
    np.fill_diagonal(block, 1.0)
    base_corr[np.ix_(mask, mask)] = block
for i in range(n_regions):
    for j in range(i+1, n_regions):
        if region_to_network[i] != region_to_network[j]:
            base_corr[i, j] = 0.1 + 0.05 * np.random.randn()
            base_corr[j, i] = base_corr[i, j]
eigvals = np.linalg.eigvalsh(base_corr)
min_eig = np.min(eigvals)
if min_eig < 1e-6:
    base_corr += (1e-6 - min_eig) * np.eye(n_regions)
# PMAT scores
pmat_cr = np.random.normal(loc=15, scale=4, size=n_subjects)
pmat_cr = np.clip(pmat_cr, 0, 24)            
pmat_si = 20 - 0.6 * pmat_cr + np.random.normal(0, 2, size=n_subjects)
pmat_si = np.clip(pmat_si, 0, 24)
pmat_cr_centered = pmat_cr - np.mean(pmat_cr)
all_corr = []
for i in range(n_subjects):
    subj_corr = base_corr.copy()
    net0 = region_to_network == 0
    net1 = region_to_network == 1
    effect = 0.01 * pmat_cr_centered[i]          
    subj_corr[np.ix_(net0, net0)] += effect
    subj_corr[np.ix_(net1, net1)] += effect
    subj_corr = (subj_corr + subj_corr.T) / 2
    np.fill_diagonal(subj_corr, 1.0)
    subj_corr = np.clip(subj_corr, -1, 1)
    eigvals = np.linalg.eigvalsh(subj_corr)
    if np.min(eigvals) < 1e-6:
        subj_corr += (1e-6 - np.min(eigvals)) * np.eye(n_regions)
    # Wishart requires a positive definite scale. We'll use subj_corr * degrees_of_freedom.
    dof = n_regions + 10          
    scale = subj_corr * dof
    eigvals_scale = np.linalg.eigvalsh(scale)
    if np.min(eigvals_scale) < 1e-6:
        scale += (1e-6 - np.min(eigvals_scale)) * np.eye(n_regions)
    cov = wishart.rvs(df=dof, scale=scale / dof, random_state=i)
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    all_corr.append(corr)
all_corr = np.array(all_corr)
i_upper = np.triu_indices(n_regions, k=1)   
n_edges = len(i_upper[0])
flattened_corr = np.array([corr[i_upper] for corr in all_corr])
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
