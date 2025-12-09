"""
Experiment for limitation mentioned in 3.1:
"the barycentric projection R​(x) is constrained to the convex hull of the target support"
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier

from common import create_results_folder
from uncertainties import uncertaintyOT, mahalanobis_dist, MSP


def get_synthetic_data(n_samples: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a synthetic dataset with topological structures designed to fail specific uncertainty metrics.
    
    Structure: out-distr blob -> in-distr circle -> in-distr circle -> out-distr circle
    """
    # 1. ID: Two Concentric Circles
    X_id, Y_id = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    
    # 2. OOD Type 1: Center Blob (Inside the Inner Circle)
    X_ood_center = np.random.normal(0, 0.1, size=(n_samples, 2))
    
    # 3. OOD Type 2: Far Outer Ring
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    r = np.random.normal(1.5, 0.1, n_samples)
    X_ood_outer = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    X_ood = np.vstack([X_ood_center, X_ood_outer])
    return X_id, Y_id, X_ood


def visualize_results(X_id: np.ndarray, X_ood: np.ndarray, X_eval: np.ndarray, 
                      metrics: dict[str, np.ndarray]) -> None:
    """
    Visualizes GT distribution and the resulting Uncertainty Maps in a 2x2 grid.
    """
    create_results_folder()

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    
    ax_gt = axs[0, 0]
    ax_gt.scatter(X_id[:, 0], X_id[:, 1], s=15, color='blue', alpha=0.5, label='In-Distribution')
    ax_gt.scatter(X_ood[:, 0], X_ood[:, 1], s=15, color='red', alpha=0.5, label='Out-of-Distribution')
    ax_gt.set_title("Ground Truth Distributions", fontsize=14)
    ax_gt.legend(loc='upper right')
    ax_gt.set_xticks([])
    ax_gt.set_yticks([])
    
    plot_locations = [(0, 1), (1, 0), (1, 1)]
    
    for (name, scores), loc in zip(metrics.items(), plot_locations):
        ax = axs[loc]
        sc = ax.scatter(X_eval[:, 0], X_eval[:, 1], c=scores, cmap='viridis', s=15, alpha=0.8)
        ax.set_title(f"{name}\nUncertainty Map", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig('results/hole.png')


def main() -> None:
    X_id, Y_train, X_ood = get_synthetic_data(n_samples=2000)
    
    clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=42)
    clf.fit(X_id, Y_train)
    
    X_eval = np.vstack([X_id, X_ood])
    
    probs_train = clf.predict_proba(X_id)
    probs_eval = clf.predict_proba(X_eval)
    
    msp_eval = MSP(probs_eval)
    maha_eval = mahalanobis_dist(X_id, Y_train, X_eval)
    
    msp_train = MSP(probs_train)
    maha_train = mahalanobis_dist(X_id, Y_train, X_id)
    
    scores_train = np.stack([msp_train, maha_train], axis=1)
    scores_eval = np.stack([msp_eval, maha_eval], axis=1)
    ot_eval = uncertaintyOT(scores_train, scores_eval)
    
    metrics_to_plot = {
        "1 - MSP": msp_eval,
        "Mahalanobis": maha_eval,
        "VecUQ-OT": ot_eval
    }
    visualize_results(X_id, X_ood, X_eval, metrics_to_plot)


if __name__ == "__main__":
    main()
