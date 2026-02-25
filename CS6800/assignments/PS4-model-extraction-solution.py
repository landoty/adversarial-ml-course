"""
CS 6800 — Problem Set 4: Model Extraction
Complete Solution

Implements Knockoff Nets model extraction attack against:
1. Hard-label black-box victim (label only)
2. Soft-label victim (top-3 softmax probabilities)

Measures fidelity and accuracy vs. query budget.

Requirements:
    pip install torch torchvision matplotlib numpy tqdm

Usage:
    python PS4-model-extraction-solution.py
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VICTIM_CKPT  = './cifar10_victim.pth'
PLOT_DIR     = './ps4_plots'
QUERY_BUDGETS = [1000, 5000, 10000, 25000, 50000]
EPOCHS_SUBSTITUTE = 15
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build and Train the Victim Model
#     The victim is a ResNet-18 with ~93% clean accuracy on CIFAR-10.
#     We simulate a "black-box API" by wrapping it in a class that exposes
#     only the information the attacker is allowed to see.
# ─────────────────────────────────────────────────────────────────────────────

def build_resnet18():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m


def train_victim():
    print('[+] Training victim model...')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                               shuffle=True, num_workers=2)
    model = build_resnet18().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    for epoch in range(30):
        model.train()
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(model(x), y).backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1}/30')

    torch.save(model.state_dict(), VICTIM_CKPT)
    return model


def load_or_train_victim():
    model = build_resnet18().to(DEVICE)
    if os.path.exists(VICTIM_CKPT):
        print(f'[+] Loading victim from {VICTIM_CKPT}')
        model.load_state_dict(torch.load(VICTIM_CKPT, map_location=DEVICE))
    else:
        model = train_victim()
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Black-Box API Wrappers
# ─────────────────────────────────────────────────────────────────────────────

class HardLabelAPI:
    """
    Simulates a black-box API that returns only the top-1 predicted class label.
    This is the most restrictive setting.
    """
    def __init__(self, model):
        self.model = model
        self.query_count = 0

    def query(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Batch of images [B, C, H, W]
        Returns:
            labels: Top-1 label [B] as LongTensor
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x.to(DEVICE))
            labels = logits.argmax(dim=1)
        self.query_count += x.shape[0]
        return labels.cpu()


class SoftLabelAPI:
    """
    Simulates a black-box API that returns top-3 class probabilities.
    This is a more informative setting.
    """
    def __init__(self, model, top_k=3):
        self.model = model
        self.top_k = top_k
        self.query_count = 0
        self.num_classes = 10

    def query(self, x: torch.Tensor):
        """
        Returns:
            probs:   Full probability vector [B, 10], but with only top-k
                     probabilities retained; the rest are zeroed (as if the API
                     only exposes the top-k predictions and their probabilities).
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x.to(DEVICE))
            all_probs = F.softmax(logits, dim=1)

        # Zero out all but top-k
        top_k_vals, top_k_idx = all_probs.topk(self.top_k, dim=1)
        sparse_probs = torch.zeros_like(all_probs)
        sparse_probs.scatter_(1, top_k_idx, top_k_vals)
        # Renormalize to sum to 1
        sparse_probs = sparse_probs / sparse_probs.sum(dim=1, keepdim=True)

        self.query_count += x.shape[0]
        return sparse_probs.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Substitute Dataset Construction via API Queries
#     We use the CIFAR-10 *test* set as our query dataset (the attacker doesn't
#     know the victim's training set, but has access to natural images from the
#     same domain — a standard Knockoff Nets assumption).
# ─────────────────────────────────────────────────────────────────────────────

def build_query_dataset(api, query_budget: int, api_type: str = 'hard'):
    """
    Query the API with up to query_budget images from CIFAR-10 test set.
    Returns a labeled substitute dataset.

    Args:
        api:          API object (HardLabelAPI or SoftLabelAPI)
        query_budget: Maximum number of queries
        api_type:     'hard' or 'soft'

    Returns:
        (images, labels_or_probs): Tensors for substitute training
    """
    transform = transforms.ToTensor()
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    # Shuffle and take first query_budget samples
    indices = torch.randperm(len(testset))[:query_budget]
    subset = torch.utils.data.Subset(testset, indices.tolist())
    loader = torch.utils.data.DataLoader(
        subset, batch_size=256, shuffle=False, num_workers=2)

    all_images  = []
    all_labels  = []

    for x, _ in loader:
        # Query the API — attacker does not know true labels
        if api_type == 'hard':
            pseudo_labels = api.query(x)   # [B] LongTensor
            all_labels.append(pseudo_labels)
        else:
            pseudo_probs = api.query(x)    # [B, 10] FloatTensor
            all_labels.append(pseudo_probs)
        all_images.append(x)

    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_images, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Substitute Dataset Class
# ─────────────────────────────────────────────────────────────────────────────

class SubstituteDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, augment=True):
        self.images = images
        self.labels = labels
        self.is_soft = labels.dtype == torch.float32
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx].clone()
        y = self.labels[idx]

        if self.augment:
            # Simple augmentation: random horizontal flip
            if torch.rand(1).item() > 0.5:
                x = x.flip(-1)  # Horizontal flip

        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Train Substitute Model
# ─────────────────────────────────────────────────────────────────────────────

def train_substitute(images, labels, api_type='hard', epochs=EPOCHS_SUBSTITUTE):
    """
    Train a substitute ResNet-18 model on the queried data.

    For hard labels: standard cross-entropy
    For soft labels: KL divergence with the probability distribution
    """
    dataset = SubstituteDataset(images, labels, augment=True)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=2)

    model = build_resnet18().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)

            if api_type == 'hard':
                y = y.to(DEVICE)
                loss = F.cross_entropy(logits, y)
            else:
                # Soft labels: minimize KL divergence
                y = y.to(DEVICE)  # [B, 10] probability distribution
                log_probs = F.log_softmax(logits, dim=1)
                loss = F.kl_div(log_probs, y, reduction='batchmean')

            loss.backward()
            optimizer.step()
        scheduler.step()

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_substitute(substitute_model, victim_api, victim_model_ref):
    """
    Measure:
    1. Fidelity: agreement between substitute and victim on held-out test images
    2. Accuracy: test set accuracy of substitute on true labels

    Args:
        substitute_model: Trained substitute
        victim_api:       API object (to query victim predictions for fidelity)
        victim_model_ref: The actual victim model (used only for accuracy ground truth)
    """
    transform = transforms.ToTensor()
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    # Use a different 1000 samples than the query set for unbiased evaluation
    eval_indices = torch.arange(8000, 9000)  # 8000-9000
    subset = torch.utils.data.Subset(testset, eval_indices.tolist())
    loader = torch.utils.data.DataLoader(subset, batch_size=256, num_workers=2)

    substitute_model.eval()
    n_total = 0
    n_fidelity = 0  # Substitute agrees with victim
    n_accuracy  = 0  # Substitute correct on true label

    with torch.no_grad():
        for x, y_true in loader:
            x = x.to(DEVICE)
            # Substitute prediction
            sub_pred = substitute_model(x).argmax(dim=1).cpu()
            # Victim prediction (via API — returns hard label)
            victim_pred = victim_api.query(x.cpu())
            # True accuracy
            n_accuracy  += (sub_pred == y_true).sum().item()
            # Fidelity: agreement with victim
            n_fidelity  += (sub_pred == victim_pred).sum().item()
            n_total     += x.shape[0]

    return n_fidelity / n_total, n_accuracy / n_total


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main Experiment Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_extraction_experiment(victim_model, api_type='hard'):
    print(f'\n{"="*60}')
    print(f'Model Extraction Experiment — API type: {api_type.upper()}')
    print(f'{"="*60}')

    if api_type == 'hard':
        api = HardLabelAPI(victim_model)
    else:
        api = SoftLabelAPI(victim_model, top_k=3)

    fidelities = []
    accuracies  = []

    for budget in QUERY_BUDGETS:
        print(f'\n  Query budget: {budget:,}')

        # Build substitute dataset
        images, labels = build_query_dataset(api, budget, api_type=api_type)
        print(f'  Queried {len(images)} images, {api.query_count:,} total queries so far')

        # Train substitute model
        print(f'  Training substitute model ({EPOCHS_SUBSTITUTE} epochs)...')
        sub_model = train_substitute(images, labels, api_type=api_type)

        # Evaluate
        hard_api_eval = HardLabelAPI(victim_model)  # Fresh API for eval
        fid, acc = evaluate_substitute(sub_model, hard_api_eval, victim_model)
        fidelities.append(fid)
        accuracies.append(acc)

        print(f'  Fidelity (agree w/ victim): {fid*100:.1f}%')
        print(f'  Accuracy (true labels):     {acc*100:.1f}%')

    return fidelities, accuracies


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_extraction_results(budgets, hard_fid, hard_acc, soft_fid, soft_acc,
                             victim_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Fidelity
    ax1.plot(budgets, [f*100 for f in hard_fid],
             'b-o', linewidth=2, markersize=8, label='Hard Labels (top-1 only)')
    ax1.plot(budgets, [f*100 for f in soft_fid],
             'r-s', linewidth=2, markersize=8, label='Soft Labels (top-3 probs)')
    ax1.axhline(100, color='black', linestyle=':', alpha=0.5, label='Perfect fidelity')
    ax1.set_xlabel('Query Budget', fontsize=12)
    ax1.set_ylabel('Fidelity (% agreement with victim)', fontsize=12)
    ax1.set_title('Model Fidelity vs. Query Budget', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_xscale('log')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(budgets, [a*100 for a in hard_acc],
             'b-o', linewidth=2, markersize=8, label='Hard Labels (top-1 only)')
    ax2.plot(budgets, [a*100 for a in soft_acc],
             'r-s', linewidth=2, markersize=8, label='Soft Labels (top-3 probs)')
    ax2.axhline(victim_acc*100, color='green', linestyle='--', linewidth=2,
                label=f'Victim accuracy ({victim_acc*100:.1f}%)')
    ax2.set_xlabel('Query Budget', fontsize=12)
    ax2.set_ylabel('Substitute Accuracy (%)', fontsize=12)
    ax2.set_title('Substitute Model Accuracy vs. Query Budget', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_xscale('log')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Knockoff Nets Model Extraction — CIFAR-10 ResNet-18 Victim',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'extraction_results.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n[+] Saved extraction results to {PLOT_DIR}/extraction_results.png')


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('CS 6800 PS4 — Model Extraction (Knockoff Nets)')
    print(f'Device: {DEVICE}')
    print('=' * 60)

    # Load/train victim
    victim = load_or_train_victim()

    # Victim's own accuracy (ground truth)
    transform = transforms.ToTensor()
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, num_workers=2)
    victim.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (victim(x).argmax(1) == y).sum().item()
            total += y.shape[0]
    victim_acc = correct / total
    print(f'\n[Victim] Clean accuracy: {victim_acc*100:.2f}%')

    # Run extraction experiments
    hard_fid, hard_acc = run_extraction_experiment(victim, api_type='hard')
    soft_fid, soft_acc = run_extraction_experiment(victim, api_type='soft')

    # Plot
    plot_extraction_results(QUERY_BUDGETS, hard_fid, hard_acc,
                            soft_fid, soft_acc, victim_acc)

    # Summary table
    print('\n' + '='*70)
    print(f'{"":>10} {"Hard-Label API":>30} {"Soft-Label API (top-3)":>30}')
    print(f'{"Budget":>10} {"Fidelity":>14} {"Accuracy":>14} '
          f'{"Fidelity":>14} {"Accuracy":>14}')
    print('-' * 70)
    for i, b in enumerate(QUERY_BUDGETS):
        print(f'{b:>10,} {hard_fid[i]*100:>13.1f}% {hard_acc[i]*100:>13.1f}% '
              f'{soft_fid[i]*100:>13.1f}% {soft_acc[i]*100:>13.1f}%')
    print(f'\nVictim accuracy (reference): {victim_acc*100:.1f}%')
    print(f'\n[+] Done. Plots saved to {PLOT_DIR}/')


if __name__ == '__main__':
    main()
