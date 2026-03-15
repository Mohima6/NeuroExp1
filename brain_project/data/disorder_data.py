
import os
import numpy as np
import pandas as pd

# =============================================================================
# Parameters
# =============================================================================
N_REGIONS = 65
N_EDGES = N_REGIONS * (N_REGIONS - 1) // 2  # 2080
N_SUBJECTS_PER_GROUP = 200
OUTPUT_FILE = "synthetic_disorder_data.csv"

# Yeo 7-network parcellation (for injecting realistic group differences)
NETWORK_SIZES = [10, 8, 12, 9, 11, 7, 8]  # total 65
region_to_network = np.repeat(np.arange(7), NETWORK_SIZES)

# =============================================================================
# Helper function to generate one group
# =============================================================================
def generate_group_correlations(n_subjects, base_corr, effect_network, effect_strength=0.0,
                                random_seed=None, reg=1e-4):
    """
    Generate n_subjects correlation matrices with a common base structure,
    plus a group-specific effect in a given network (adds to edges where at least
    one region belongs to effect_network). A small regularization term ensures
    positive definiteness.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    n_regions = base_corr.shape[0]
    all_mats = []
    for _ in range(n_subjects):
        # Start from base correlation + small random noise
        noise = np.random.randn(n_regions, n_regions) * 0.05
        noise = (noise + noise.T) / 2
        np.fill_diagonal(noise, 0)
        corr = base_corr + noise

        # Inject group effect
        if effect_network is not None and effect_strength != 0:
            for i in range(n_regions):
                for j in range(i+1, n_regions):
                    if region_to_network[i] == effect_network or region_to_network[j] == effect_network:
                        corr[i, j] += effect_strength
                        corr[j, i] = corr[i, j]

        # Regularize to ensure positive definiteness
        np.fill_diagonal(corr, 1.0 + reg)
        # Convert back to correlation matrix
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)

        # Clip to [-1,1] and enforce symmetry
        corr = np.clip(corr, -1, 1)
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)

        all_mats.append(corr)

    return np.array(all_mats)

# =============================================================================
# Build base correlation matrix (common to all groups)
# =============================================================================
np.random.seed(42)
base = np.random.randn(N_REGIONS, N_REGIONS) * 0.3
base = (base + base.T) / 2
# Make positive definite
base = base + np.eye(N_REGIONS) * 0.8
# Convert to correlation
d = np.sqrt(np.diag(base))
base = base / np.outer(d, d)
base = np.clip(base, -1, 1)
np.fill_diagonal(base, 1)

# =============================================================================
# Generate each group
# =============================================================================
groups = {}
# Healthy: no effect
groups[0] = generate_group_correlations(N_SUBJECTS_PER_GROUP, base,
                                         effect_network=None, effect_strength=0,
                                         random_seed=100)
# Alzheimer's: decrease default mode (network 1)
groups[1] = generate_group_correlations(N_SUBJECTS_PER_GROUP, base,
                                         effect_network=1, effect_strength=-0.15,
                                         random_seed=200)
# Autism: increase salience/ventral attention (network 4)
groups[2] = generate_group_correlations(N_SUBJECTS_PER_GROUP, base,
                                         effect_network=4, effect_strength=0.2,
                                         random_seed=300)
# Parkinson's: decrease somatomotor (network 5)
groups[3] = generate_group_correlations(N_SUBJECTS_PER_GROUP, base,
                                         effect_network=5, effect_strength=-0.1,
                                         random_seed=400)

# Concatenate all matrices
X_mats = np.concatenate([groups[0], groups[1], groups[2], groups[3]], axis=0)
y = np.concatenate([
    np.full(N_SUBJECTS_PER_GROUP, 0),
    np.full(N_SUBJECTS_PER_GROUP, 1),
    np.full(N_SUBJECTS_PER_GROUP, 2),
    np.full(N_SUBJECTS_PER_GROUP, 3)
])
subject_ids = np.arange(len(y))

# =============================================================================
# Flatten upper triangle to 2080 features
# =============================================================================
i_upper = np.triu_indices(N_REGIONS, k=1)
X_flat = np.array([mat[i_upper] for mat in X_mats])

# =============================================================================
# Create DataFrame and save
# =============================================================================
# Column names: subject_id, diagnosis, feat_0, feat_1, ..., feat_2079
columns = ['subject_id', 'diagnosis'] + [f'feat_{i}' for i in range(N_EDGES)]
data = np.column_stack([subject_ids, y, X_flat])
df = pd.DataFrame(data, columns=columns)
df['subject_id'] = df['subject_id'].astype(int)
df['diagnosis'] = df['diagnosis'].astype(int)

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSynthetic data saved to {OUTPUT_FILE}")
print(f"Shape: {df.shape} (subjects × features)")
print("Class distribution:")
print(df['diagnosis'].value_counts().sort_index())