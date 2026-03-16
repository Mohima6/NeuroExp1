import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import cholesky
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)
from nilearn import datasets, plotting
import warnings
warnings.filterwarnings("ignore")
def is_positive_definite(B):
    """Check if a matrix is positive definite via Cholesky."""
    try:
        np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False
def nearest_positive_definite(A):
    """
    Find the nearest positive-definite matrix to A (Higham 1988).
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = V.T @ np.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if is_positive_definite(A3):
        return A3
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3
OUTPUT_DIR = os.path.join(os.getcwd(), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_REGIONS = 65
N_EDGES = N_REGIONS * (N_REGIONS - 1) // 2  # 2080
N_SUBJECTS_PER_GROUP = 200  
NETWORK_SIZES = [10, 8, 12, 9, 11, 7, 8]  # total 65
NETWORK_NAMES = ["Control", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Visual"]
NETWORK_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628"]
region_to_network = np.repeat(np.arange(7), NETWORK_SIZES)
n_networks = len(np.unique(region_to_network))
DISORDER_NAMES = ['Healthy', 'Alzheimer', 'Autism', 'Parkinson']
DISORDER_LABELS = [0, 1, 2, 3]
DISORDER_COLORS = {
    'Healthy': '#2ca02c',
    'Alzheimer': '#d62728',
    'Autism': '#9467bd',
    'Parkinson': '#ff7f0e'
}

# synthetic connectivity data (with group differences)
print("\n[1] synthetic data...")

def generate_group_correlations(n_subjects, base_corr, effect_network, effect_strength=0.0,
                                random_seed=None, reg=0.01):
    """
    Generate n_subjects correlation matrices with a common base structure,
    plus a group-specific effect in a given network.
    A small regularization term (reg) is added to the final matrix before
    converting to correlation, guaranteeing positive definiteness.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    n_regions = base_corr.shape[0]
    all_mats = []
    for _ in range(n_subjects):
        # Start from base + random noise
        noise = np.random.randn(n_regions, n_regions) * 0.05
        noise = (noise + noise.T) / 2
        np.fill_diagonal(noise, 0)
        corr = base_corr + noise
        if effect_network is not None and effect_strength != 0:
            for i in range(n_regions):
                for j in range(i+1, n_regions):
                    if region_to_network[i] == effect_network or region_to_network[j] == effect_network:
                        corr[i, j] += effect_strength
                        corr[j, i] = corr[i, j]
        corr = corr + np.eye(n_regions) * reg
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1, 1)
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)
        all_mats.append(corr)
    return np.array(all_mats)
np.random.seed(42)
base = np.random.randn(N_REGIONS, N_REGIONS) * 0.3
base = (base + base.T) / 2
base = base + np.eye(N_REGIONS) * 1.0
d = np.sqrt(np.diag(base))
base = base / np.outer(d, d)
base = np.clip(base, -1, 1)
np.fill_diagonal(base, 1)
groups = {}
# Healthy: no extra effect
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
X_mats = np.concatenate([groups[0], groups[1], groups[2], groups[3]], axis=0)
y = np.concatenate([
    np.full(N_SUBJECTS_PER_GROUP, 0),
    np.full(N_SUBJECTS_PER_GROUP, 1),
    np.full(N_SUBJECTS_PER_GROUP, 2),
    np.full(N_SUBJECTS_PER_GROUP, 3)
])
subject_ids = np.arange(len(y))
print("    Projecting matrices to positive definite cone...")
X_mats_pd = np.array([nearest_positive_definite(mat) for mat in X_mats])
i_upper = np.triu_indices(N_REGIONS, k=1)
X_flat = np.array([mat[i_upper] for mat in X_mats_pd])
print(f"  Generated {len(y)} subjects: {np.bincount(y)} per group")
print("\n[2] Figure 1: Demographics + network parcellation...")
demo_df = pd.DataFrame({
    'Group': DISORDER_NAMES,
    'N': [N_SUBJECTS_PER_GROUP] * 4,
    'Age (mean±std)': ['25.3 ± 3.1', '72.1 ± 5.2', '18.7 ± 4.0', '68.4 ± 6.3'],
    'Gender (M/F)': ['100/100', '95/105', '110/90', '92/108']
})
print("\nDemographics table:")
print(demo_df.to_string(index=False))
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('off')
table = ax.table(cellText=demo_df.values, colLabels=demo_df.columns,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.title("Demographics", y=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure1_demographics.png'), dpi=300, bbox_inches='tight')
plt.close()
fig, ax = plt.subplots(figsize=(10, 1))
for i, net in enumerate(region_to_network):
    ax.bar(i, 1, color=NETWORK_COLORS[net], edgecolor='none', width=1)
ax.set_xlim(0, N_REGIONS)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Region assignment to 7 functional networks")
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=NETWORK_COLORS[i], label=NETWORK_NAMES[i]) for i in range(7)]
ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.5), loc='lower center', ncol=4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure1_network_parcellation.png'), dpi=300)
plt.close()
print("\n[3] Computing Fréchet means...")
def frechet_mean_cholesky(mats):
    """Fréchet mean under the Euclidean‑Cholesky metric."""
    # All matrices are already PD, so Cholesky succeeds
    chol = np.array([cholesky(m, lower=True) for m in mats])
    mean_chol = np.mean(chol, axis=0)
    mean_corr = mean_chol @ mean_chol.T
    d = np.sqrt(np.diag(mean_corr))
    return mean_corr / np.outer(d, d)
group_means = {}
for label in DISORDER_LABELS:
    idx = np.where(y == label)[0]
    group_means[label] = frechet_mean_cholesky(X_mats_pd[idx])
    print(f"  {DISORDER_NAMES[label]}: {len(idx)} subjects")
print("\n[4] Loading brain atlas...")
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)
coords = plotting.find_parcellation_cut_coords(labels_img=atlas.maps)[:N_REGIONS]
print("\n[5] Generating Figure 2 (interactive brain surfaces)...")
def plot_group_surface(mat, group_name, filename):
    thresh = np.percentile(np.abs(mat), 98)
    mat_thresh = np.where(np.abs(mat) >= thresh, mat, 0)
    view = plotting.view_connectome(mat_thresh, coords, node_color=node_colors,
                                    title=f"{group_name} Group Mean (top 2% edges)")
    view.save_as_html(os.path.join(OUTPUT_DIR, filename))
    print(f"    Saved {filename}")
for label, name in zip(DISORDER_LABELS, DISORDER_NAMES):
    plot_group_surface(group_means[label], name, f"{name.lower()}_mean.html")
print("\n[6] Generating Figure 3: difference heatmaps...")
network_order = np.argsort(region_to_network)
boundaries = []
current_net = region_to_network[network_order[0]]
for idx, net in enumerate(region_to_network[network_order]):
    if net != current_net:
        boundaries.append(idx)
        current_net = net
healthy_mean = group_means[0]
for label, name in zip(DISORDER_LABELS[1:], DISORDER_NAMES[1:]):
    diff = group_means[label] - healthy_mean
    diff_sorted = diff[np.ix_(network_order, network_order)]
    plt.figure(figsize=(7, 7))
    sns.heatmap(diff_sorted, cmap='RdBu_r', center=0, square=True,
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Connectivity difference'})
    for b in boundaries:
        plt.axhline(b, color='k', linewidth=1)
        plt.axvline(b, color='k', linewidth=1)
    plt.title(f'{name} vs Healthy: Signed Differences')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'figure3_diff_{name.lower()}.png'), dpi=300)
    plt.close()
    print(f"    Saved figure3_diff_{name.lower()}.png")
print("\n[7] Generating Figure 4: significance maps...")
healthy_idx = np.where(y == 0)[0]
for label, name in zip(DISORDER_LABELS[1:], DISORDER_NAMES[1:]):
    disorder_idx = np.where(y == label)[0]
    t_stats = np.zeros(N_EDGES)
    p_vals = np.zeros(N_EDGES)
    for e in range(N_EDGES):
        t, p = ttest_ind(X_flat[healthy_idx, e], X_flat[disorder_idx, e])
        t_stats[e] = t
        p_vals[e] = p
    reject, p_corr, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
    print(f"    {name}: {np.sum(reject)} significant edges (FDR < 0.05)")
    t_mat = np.zeros((N_REGIONS, N_REGIONS))
    t_mat[i_upper] = t_stats
    t_mat = t_mat + t_mat.T  # make symmetric
    sig_mat = np.zeros((N_REGIONS, N_REGIONS), dtype=bool)
    sig_mat[i_upper] = reject
    sig_mat = sig_mat + sig_mat.T 
    t_mat[~sig_mat] = 0
    if np.any(reject):
        plotting.plot_connectome(
            t_mat, coords, edge_threshold=1e-6,
            edge_cmap='RdBu_r', edge_vmin=-5, edge_vmax=5,
            title=f'{name} vs Healthy: t-statistic (FDR < 0.05)',
            output_file=os.path.join(OUTPUT_DIR, f'figure4_significant_{name.lower()}.png')
        )
        print(f"    Saved figure4_significant_{name.lower()}.png")
print("\n[8] Figure 5: network-level bar plots...")
def network_mean_connectivity(corr_mat, region_to_network):
    n_net = len(np.unique(region_to_network))
    net_mat = np.zeros((n_net, n_net))
    count_mat = np.zeros((n_net, n_net))
    for i in range(N_REGIONS):
        for j in range(i+1, N_REGIONS):
            net_i = region_to_network[i]
            net_j = region_to_network[j]
            net_mat[net_i, net_j] += corr_mat[i, j]
            count_mat[net_i, net_j] += 1
    for i in range(n_net):
        for j in range(i, n_net):
            if i == j:
                total = net_mat[i, i]
                cnt = count_mat[i, i]
                net_mat[i, i] = total / cnt if cnt > 0 else 0
            else:
                total = net_mat[i, j] + net_mat[j, i]
                cnt = count_mat[i, j] + count_mat[j, i]
                avg = total / cnt if cnt > 0 else 0
                net_mat[i, j] = avg
                net_mat[j, i] = avg
    return net_mat
network_means = {}
for label, name in zip(DISORDER_LABELS, DISORDER_NAMES):
    network_means[name] = network_mean_connectivity(group_means[label], region_to_network)
network_pairs = [f'{NETWORK_NAMES[i]}-{NETWORK_NAMES[j]}'
                 for i in range(n_networks) for j in range(i, n_networks)]
data_for_plot = []
for name in DISORDER_NAMES:
    mat = network_means[name]
    vals = [mat[i, j] for i in range(n_networks) for j in range(i, n_networks)]
    data_for_plot.append(vals)
x = np.arange(len(network_pairs))
width = 0.2
fig, ax = plt.subplots(figsize=(14, 6))
for i, name in enumerate(DISORDER_NAMES):
    ax.bar(x + i*width, data_for_plot[i], width, label=name, color=DISORDER_COLORS[name])
ax.set_xlabel('Network pair')
ax.set_ylabel('Mean connectivity')
ax.set_title('Network-level connectivity by group')
ax.set_xticks(x + width * (len(DISORDER_NAMES)-1)/2)
ax.set_xticklabels(network_pairs, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure5_network_barplot.png'), dpi=300)
plt.close()
print("    Saved figure5_network_barplot.png")
print("\n[9] Generating Figure 6: classification results...")

# Cholesky features for classification
def chol_vec(mat):
    L = cholesky(mat, lower=True)
    return L[np.tril_indices_from(L)]
chol_features = np.array([chol_vec(m) for m in X_mats_pd])
X_train, X_test, y_train, y_test = train_test_split(
    chol_features, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Random Forest
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"    Multi-class accuracy: {acc:.3f}")
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISORDER_NAMES)
disp.plot(cmap='Blues', values_format='d')
plt.title(f'Multi-class classification accuracy: {acc:.2%}')
plt.savefig(os.path.join(OUTPUT_DIR, 'figure6_confusion_matrix.png'), dpi=300)
plt.close()
print("    Saved figure6_confusion_matrix.png")
# ROC curves (one-vs-rest)
y_bin = label_binarize(y_test, classes=DISORDER_LABELS)
n_classes = len(DISORDER_LABELS)
y_score = rf.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
plt.figure(figsize=(8, 6))
colors = [DISORDER_COLORS[name] for name in DISORDER_NAMES]
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{DISORDER_NAMES[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves (one-vs-rest)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure6_roc_curves.png'), dpi=300)
plt.close()
print("    Saved figure6_roc_curves.png")
# Feature importance map
importances = rf.feature_importances_
imp_mat = np.zeros((N_REGIONS, N_REGIONS))
imp_mat[np.tril_indices_from(imp_mat)] = importances
imp_mat = imp_mat + imp_mat.T - np.diag(np.diag(imp_mat))
thresh_imp = np.percentile(imp_mat, 98)
imp_mat_thresh = np.where(imp_mat >= thresh_imp, imp_mat, 0)
plotting.plot_connectome(
    imp_mat_thresh, coords, edge_threshold=1e-6,
    edge_cmap='hot', title='Top 2% most important edges (Random Forest)',
    output_file=os.path.join(OUTPUT_DIR, 'figure6_feature_importance.png')
)
print("    Saved figure6_feature_importance.png")
print("\n" + "="*60)
print("All figures and HTML files have been saved in:")
print(f"  {OUTPUT_DIR}")
print("="*60)
