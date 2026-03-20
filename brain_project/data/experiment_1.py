import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy.linalg import cholesky
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import networkx as nx
from nilearn import datasets, plotting
warnings.filterwarnings("ignore")
#Load data (HCP sample, n=977 after exclusions)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
corr_path = os.path.join(parent_dir, "synthetic_correlations.csv")
pmat_path = os.path.join(parent_dir, "synthetic_pmat.csv")
df_corr = pd.read_csv(corr_path)
df_pmat = pd.read_csv(pmat_path)
X_flat = df_corr.values.astype(np.float64)
y_cr = df_pmat["PMAT_A_CR"].values.astype(np.float64)   
y_si = df_pmat["PMAT_A_SI"].values.astype(np.float64)   
n_subjects, n_edges = X_flat.shape
n_regions = 65 
i_upper = np.triu_indices(n_regions, k=1)
all_corr = np.zeros((n_subjects, n_regions, n_regions))
for i in range(n_subjects):
    mat = np.zeros((n_regions, n_regions))
    mat[i_upper] = X_flat[i]
    mat = mat + mat.T
    np.fill_diagonal(mat, 1)
    all_corr[i] = mat
# Functional network parcellation (7 networks, Schaefer style)
network_sizes = [10, 8, 12, 9, 11, 7, 8]
network_names = ["Control", "Default", "DorsAttn", "Limbic",
                 "SalVentAttn", "SomMot", "Visual"]
network_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
                  "#ff7f00", "#ffff33", "#a65628"]
region_to_network = np.repeat(np.arange(7), network_sizes)
# Top / bottom 10% groups based on PMAT_A_CR
sorted_idx = np.argsort(y_cr)
n_top = int(0.1 * n_subjects)
n_bottom = int(0.1 * n_subjects)
top_idx = sorted_idx[-n_top:]
bottom_idx = sorted_idx[:n_bottom]
# Fréchet means under Euclidean‑Cholesky metric
def frechet_mean_cholesky(mats):
    chol = np.array([cholesky(m, lower=True) for m in mats])
    mean_chol = np.mean(chol, axis=0)
    mean_corr = mean_chol @ mean_chol.T
    d = np.sqrt(np.diag(mean_corr))
    return mean_corr / np.outer(d, d)
mean_top = frechet_mean_cholesky(all_corr[top_idx])
mean_bottom = frechet_mean_cholesky(all_corr[bottom_idx])
# Difference matrix sorted by functional network
diff_mat = np.abs(mean_top - mean_bottom)
network_order = np.argsort(region_to_network)
diff_sorted = diff_mat[np.ix_(network_order, network_order)]
plt.figure(figsize=(7, 7))
sns.heatmap(diff_sorted, cmap="Reds", square=True,
            xticklabels=False, yticklabels=False,
            cbar_kws={"label": "Absolute difference"})
boundaries = np.cumsum(network_sizes)
for b in boundaries:
    plt.axhline(b, color="blue", linewidth=2)
    plt.axvline(b, color="blue", linewidth=2)
plt.title("Connectivity Difference (Top vs Bottom PMAT)")
plt.savefig("diff_matrix.png", dpi=300)
plt.show()
#  Brain surface 
#  Uses fsaverage surface; realistic cortical surface with connectome
print("Loading brain atlas and surfaces")
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)
coords = plotting.find_parcellation_cut_coords(labels_img=atlas.maps)[:n_regions]
def plot_brain_surface(mat, title, filename, threshold_percent=98):
    """Create interactive HTML brain surface with thresholded connectome."""
    # Threshold matrix to keep only strongest connections (top 2%)
    thresh_val = np.percentile(np.abs(mat), threshold_percent)
    mat_thresh = np.where(np.abs(mat) >= thresh_val, mat, 0)
    view = plotting.view_connectome(mat_thresh, coords, title=title)
    view.save_as_html(filename)
    print(f"Saved: {filename}")
# HTML shows real cortical surface
plot_brain_surface(mean_top, "Top 10% PMAT Connectivity", "top_connectome.html")
plot_brain_surface(mean_bottom, "Bottom 10% PMAT Connectivity", "bottom_connectome.html")
#save static glass brain PNGs
plotting.plot_connectome(mean_top, coords, edge_threshold="98%",
                         title="Top 10% PMAT", output_file="top_glass.png")
plotting.plot_connectome(mean_bottom, coords, edge_threshold="98%",
                         title="Bottom 10% PMAT", output_file="bottom_glass.png")
print("Static glass brain PNGs saved.")
# Predictive modeling – 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y_cr, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#  Connectome‑Based Predictive Modeling
def cpm_predict(X_train, y_train, X_test, k=100):
    corrs = np.array([np.corrcoef(X_train[:, i], y_train)[0, 1]
                      for i in range(X_train.shape[1])])
    pos = np.argsort(corrs)[-k:]
    neg = np.argsort(corrs)[:k]
    pos_train = X_train[:, pos].sum(axis=1)
    neg_train = X_train[:, neg].sum(axis=1)
    lr = LinearRegression().fit(np.column_stack([pos_train, neg_train]), y_train)
    pos_test = X_test[:, pos].sum(axis=1)
    neg_test = X_test[:, neg].sum(axis=1)
    return lr.predict(np.column_stack([pos_test, neg_test]))
cpm_pred = cpm_predict(X_train, y_train, X_test)
# LASSO (with 5‑fold CV)
lasso = LassoCV(cv=5).fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
# Cholesky + Kernel Ridge Regression (proposed) 
def chol_vec(mat):
    L = cholesky(mat, lower=True)
    return L[np.tril_indices_from(L)]
train_idx, test_idx = train_test_split(np.arange(n_subjects),
                                       test_size=0.2, random_state=42)
X_train_chol = np.array([chol_vec(all_corr[i]) for i in train_idx])
X_test_chol = np.array([chol_vec(all_corr[i]) for i in test_idx])
scaler_chol = StandardScaler()
X_train_chol = scaler_chol.fit_transform(X_train_chol)
X_test_chol = scaler_chol.transform(X_test_chol)
krr = KernelRidge(kernel="rbf")
krr.fit(X_train_chol, y_cr[train_idx])
krr_pred = krr.predict(X_test_chol)
# Results (MSE on test set)
print("\n" + "="*50)
print("Model performance on PMAT_A_CR (MSE)")
print("="*50)
print(f"CPM                    : {mean_squared_error(y_test, cpm_pred):.4f}")
print(f"LASSO                  : {mean_squared_error(y_test, lasso_pred):.4f}")
print(f"Cholesky + KRR         : {mean_squared_error(y_cr[test_idx], krr_pred):.4f}")
print("="*50)
# save results to text file
with open("prediction_results.txt", "w") as f:
    f.write("Model performance on PMAT_A_CR (MSE)\n")
    f.write(f"CPM                    : {mean_squared_error(y_test, cpm_pred):.4f}\n")
    f.write(f"LASSO                  : {mean_squared_error(y_test, lasso_pred):.4f}\n")
    f.write(f"Cholesky + KRR         : {mean_squared_error(y_cr[test_idx], krr_pred):.4f}\n")
