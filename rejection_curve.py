"""
Plots rejection curve for classification task on FashionMNIST Dataset. 
"""

import torch
import torchvision
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from common import create_results_folder
from uncertainties import (
    uncertaintyOT,
    mahalanobis_dist, 
    MSP
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128, bias=False)
        self.fc2 = nn.Linear(128, 10, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


def train_test_loaders(seed: int = 42, batch: int = 128) -> list[DataLoader, DataLoader]:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    full_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, 
                                                     download=True, transform=transform)
    
    total_size = len(full_dataset)
    train_size = int(0.5 * total_size)
    test_size = total_size - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    
    return train_loader, test_loader


def train_single_model(model_seed: int, train_loader: DataLoader, epochs: int = 3) -> nn.Module:
    torch.manual_seed(model_seed)
    model = Classifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    train_loader_shuffled = DataLoader(train_loader.dataset, 
                                       batch_size=train_loader.batch_size, shuffle=True)
    for _ in trange(epochs, desc='Epoch', leave=False, position=1):
        for images, labels in train_loader_shuffled:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def extract_logits_ensemble(models: list[nn.Module], 
                            loader: DataLoader) -> np.ndarray:
    """
    Runs all models in the ensemble on the loader.
    Returns: (n_models, n_samples, n_features)
    """
    all_logits = []
    
    for model in models:
        model.eval()
        model_logits = []
        for images, _ in loader:
            images = images.to(DEVICE)
            out: Tensor = model(images)
            model_logits.append(out.cpu())
        all_logits.append(np.concatenate(model_logits))
    return np.stack(all_logits)


def compute_rejection_curve(uncertainties: np.ndarray,
                            correct_mask: np.ndarray) -> list[np.ndarray, np.ndarray]:
    """
    Returns accuracy (%) per rejection rate (%) to build rejection curve.
    """
    if uncertainties.ndim > 1:
        uncertainties = uncertainties.mean(axis=1) if uncertainties.shape[1] > 1 else uncertainties.flatten()

    sorted_indices = np.argsort(uncertainties) # low -> high (more uncertain)
    sorted_correct = correct_mask[sorted_indices]
    n_samples = len(uncertainties)
    rejection_rates = np.arange(0, 1.00 + 0.01, 0.05)
    accuracies = []
    
    for rate in rejection_rates:
        keep_ratio = 1.0 - rate
        cutoff_index = int(n_samples * keep_ratio)
        
        if cutoff_index == 0:
            accuracies.append(1.0 if not accuracies else accuracies[-1])
            continue

        selected_samples = sorted_correct[:cutoff_index]        
        acc = np.mean(selected_samples)
        accuracies.append(acc)
    return np.array(accuracies) * 100, rejection_rates * 100


def plot_rejection_curves(results_dict: dict[str, list[np.ndarray, np.ndarray]]) -> None:
    """
    results_dict: { "Method Name": (accuracies, rates) }
    """
    plt.figure(figsize=(10, 6))
    
    for method_name, (acc, rates) in results_dict.items():
        plt.plot(rates, acc, label=method_name, linewidth=2)
    
    plt.title('Rejection Curves on FashionMNIST')
    plt.xlabel('Rejected data part, %')
    plt.ylabel('Accuracy, %')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rates)
    plt.savefig('results/rejection_curve.png')


def main():
    create_results_folder()
    
    train_loader, test_loader= train_test_loaders(seed=42)
    
    Y_train = np.concatenate([y for _, y in train_loader])
    Y_test  = np.concatenate([y for _, y in test_loader])

    model_seeds = [1, 2, 3, 4, 5]
    models = []
    for seed in tqdm(model_seeds, desc='Seed', leave=False, position=0):
        m = train_single_model(seed, train_loader)
        models.append(m)

    X_train_logits = extract_logits_ensemble(models, train_loader) # shape: (n_models, n_samples, n_classes)
    X_test_logits = extract_logits_ensemble(models, test_loader)
    
    mean_test_logits = np.mean(X_test_logits, axis=0)
    predictions = np.argmax(mean_test_logits, axis=1)
    correct_mask = (predictions == Y_test)

    uncertainty_results = {}
    
    unc_msp_test = MSP(mean_test_logits)
    uncertainty_results["1 - MSP"] = unc_msp_test
    
    unc_maha_test = mahalanobis_dist(X_train_logits, Y_train, X_test_logits)
    uncertainty_results["Mahalanobis"] = unc_maha_test
    
    unc_msp_train  = MSP(np.mean(X_train_logits, axis=0))
    unc_maha_train = mahalanobis_dist(X_train_logits, Y_train, X_train_logits)    
    features_train = np.stack([unc_msp_train, unc_maha_train], axis=1)
    features_test  = np.stack([unc_msp_test, unc_maha_test], axis=1)
    unc_ot_test = uncertaintyOT(features_train, features_test)
    uncertainty_results["VecUQ-OT"] = unc_ot_test
    
    plot_data = {}
    for name, unc_values in uncertainty_results.items():
        acc, rates = compute_rejection_curve(unc_values, correct_mask)
        plot_data[name] = (acc, rates)
    plot_rejection_curves(plot_data)


if __name__ == "__main__":
    main()
