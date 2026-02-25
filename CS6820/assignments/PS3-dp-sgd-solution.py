"""
CS 6820 — Problem Set 3: Differential Privacy with DP-SGD
Complete Solution

Implements:
1. Baseline LeNet-5 training on MNIST (non-private)
2. DP-SGD training with Opacus at ε ∈ {0.5, 1, 3, 10}
3. Clipping norm sweep at fixed ε=3
4. Privacy-utility curve
5. Membership inference attack (loss-threshold) against private vs. non-private

Requirements:
    pip install torch torchvision opacus matplotlib numpy scipy tqdm

Usage:
    python PS3-dp-sgd-solution.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    HAS_OPACUS = True
except ImportError:
    print('[!] Opacus not installed. Install with: pip install opacus')
    print('[!] DP-SGD experiments will be skipped.')
    HAS_OPACUS = False

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS       = 20
BATCH_SIZE   = 256           # Larger batches benefit DP (better gradient signal)
TARGET_DELTA = 1e-5
TARGET_EPS   = [0.5, 1.0, 3.0, 10.0]
CLIP_NORMS   = [0.1, 0.5, 1.0, 5.0]
FIXED_EPS    = 3.0           # Fixed epsilon for clipping norm sweep
PLOT_DIR     = './ps3_plots'
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LeNet-5 for MNIST (Opacus-compatible)
#     Opacus requires GroupNorm instead of BatchNorm.
#     We use a standard LeNet-5 without batch norm.
# ─────────────────────────────────────────────────────────────────────────────

class LeNet5(nn.Module):
    """
    LeNet-5 adapted for MNIST (28×28 grayscale).
    Uses no BatchNorm layers for Opacus compatibility.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.pool  = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 6, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 16, 5, 5]
        x = x.view(x.shape[0], -1)            # [B, 400]
        x = F.relu(self.fc1(x))               # [B, 120]
        x = F.relu(self.fc2(x))               # [B, 84]
        return self.fc3(x)                    # [B, 10]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Data
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders():
    transform = transforms.ToTensor()
    trainset  = torchvision.datasets.MNIST('./data', train=True,
                                            download=True, transform=transform)
    testset   = torchvision.datasets.MNIST('./data', train=False,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader  = torch.utils.data.DataLoader(
        testset,  batch_size=512, shuffle=False, num_workers=2)
    return trainloader, testloader


def get_train_subset(n=5000):
    """Return a small training subset for membership inference evaluation."""
    transform = transforms.ToTensor()
    trainset  = torchvision.datasets.MNIST('./data', train=True,
                                            download=True, transform=transform)
    indices   = torch.arange(n)
    return torch.utils.data.Subset(trainset, indices.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Baseline Training (Non-Private)
# ─────────────────────────────────────────────────────────────────────────────

def train_baseline(trainloader, testloader):
    """Train a standard LeNet-5 without any DP."""
    print('\n' + '='*55)
    print('Training: Non-private baseline')
    print('='*55)

    model = LeNet5().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        scheduler.step()

    acc = evaluate(model, testloader)
    print(f'Baseline accuracy: {acc*100:.2f}%')
    return model, acc


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DP-SGD Training with Opacus
# ─────────────────────────────────────────────────────────────────────────────

def train_dp_sgd(target_epsilon: float, max_grad_norm: float = 1.0,
                 trainloader=None, testloader=None):
    """
    Train LeNet-5 on MNIST with DP-SGD using Opacus.

    Opacus automatically:
    1. Computes per-sample gradients (via grad hooks)
    2. Clips each per-sample gradient to max_grad_norm
    3. Adds calibrated Gaussian noise to the clipped aggregate
    4. Tracks the privacy budget via RDP accountant

    Args:
        target_epsilon:  Target ε at TARGET_DELTA
        max_grad_norm:   Clipping norm C
    """
    if not HAS_OPACUS:
        return None, 0.0, None

    print(f'\n  Training DP-SGD: target ε={target_epsilon}, C={max_grad_norm}')

    model = LeNet5().to(DEVICE)
    # Opacus validation: ensure model is compatible
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        model = ModuleValidator.fix(model)

    optimizer = optim.SGD(model.parameters(), lr=0.05,
                          momentum=0, weight_decay=0)
    # Note: Opacus is not compatible with momentum in the standard implementation
    # (momentum introduces implicit composition of gradients across steps).
    # Use SGD without momentum, or use the GradSampleModule workaround.

    privacy_engine = PrivacyEngine(accountant='rdp')

    model, optimizer, dp_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        epochs=EPOCHS,
        target_epsilon=target_epsilon,
        target_delta=TARGET_DELTA,
        max_grad_norm=max_grad_norm,
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for x, y in dp_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            eps = privacy_engine.get_epsilon(TARGET_DELTA)
            acc = evaluate(model, testloader)
            print(f'    Epoch {epoch+1}/{EPOCHS}  ε={eps:.2f}  acc={acc*100:.1f}%')

    final_eps = privacy_engine.get_epsilon(TARGET_DELTA)
    final_acc = evaluate(model, testloader)
    print(f'  Final: ε={final_eps:.3f}, acc={final_acc*100:.2f}%')

    return model, final_acc, privacy_engine


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.shape[0]
    return correct / total


def compute_per_sample_loss(model, dataset, batch_size=256):
    """
    Compute per-sample training loss for membership inference.
    Lower loss → more likely to be a training member.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(x), y)
            losses.extend(loss.cpu().tolist())
    return np.array(losses)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Membership Inference Attack (Loss Threshold)
#
#     The simplest MI attack: predict "member" if per-sample loss < τ.
#     At the optimal threshold, compute TPR (correctly identify members)
#     and FPR (incorrectly flag non-members as members).
#     MI advantage = |TPR - FPR|
# ─────────────────────────────────────────────────────────────────────────────

def membership_inference_attack(model, n_members=2000, n_nonmembers=2000):
    """
    Run a loss-threshold membership inference attack.

    Members:    First n_members samples from the training set (in-training)
    Non-members: Test set samples (not in training)

    Returns:
        mi_advantage: |TPR - FPR| at the optimal threshold
        auc_roc:      Area under the ROC curve
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    transform = transforms.ToTensor()

    # Member losses (training set)
    train_full = torchvision.datasets.MNIST('./data', train=True,
                                             download=True, transform=transform)
    member_set = torch.utils.data.Subset(train_full, range(n_members))
    member_losses = compute_per_sample_loss(model, member_set)

    # Non-member losses (test set)
    test_full = torchvision.datasets.MNIST('./data', train=False,
                                            download=True, transform=transform)
    nonmember_set = torch.utils.data.Subset(test_full, range(n_nonmembers))
    nonmember_losses = compute_per_sample_loss(model, nonmember_set)

    # Labels: 1 = member, 0 = non-member
    y_true  = np.concatenate([np.ones(n_members), np.zeros(n_nonmembers)])
    # Score: negative loss (high = more likely member)
    y_score = np.concatenate([-member_losses, -nonmember_losses])

    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    # MI advantage = max |TPR - FPR| over all thresholds
    mi_advantage = float(np.max(tpr - fpr))

    return mi_advantage, auc, fpr, tpr


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_privacy_utility(epsilon_vals, accuracies, baseline_acc):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epsilon_vals, [a*100 for a in accuracies],
            'b-o', linewidth=2, markersize=9, label='DP-SGD accuracy')
    ax.axhline(baseline_acc*100, color='green', linestyle='--', linewidth=2,
               label=f'Non-private baseline ({baseline_acc*100:.1f}%)')
    ax.set_xlabel('Privacy Budget ε', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Privacy-Utility Tradeoff on MNIST\n(LeNet-5, DP-SGD)', fontsize=12)
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.set_xticks(epsilon_vals)
    ax.set_xticklabels([str(e) for e in epsilon_vals])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'privacy_utility_curve.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[+] Saved privacy-utility curve')


def plot_clip_norm_sweep(clip_norms, accuracies, eps):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(clip_norms, [a*100 for a in accuracies],
            'r-s', linewidth=2, markersize=9)
    ax.set_xlabel('Clipping Norm C', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'Effect of Clipping Norm on Accuracy\n'
                 f'(DP-SGD ε={eps}, LeNet-5, MNIST)', fontsize=12)
    ax.set_xscale('log')
    ax.set_xticks(clip_norms)
    ax.set_xticklabels([str(c) for c in clip_norms])
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'clip_norm_sweep.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[+] Saved clipping norm sweep')


def plot_mi_roc(fprs_fprs, tprs_tprs, labels, aucs):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['#e74c3c', '#2980b9', '#27ae60', '#f39c12']
    for i, (fpr, tpr, label, auc, color) in enumerate(
            zip(fprs_fprs, tprs_tprs, labels, aucs, colors)):
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{label} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Membership Inference ROC Curves\nLoss-Threshold Attack on MNIST',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'mi_roc_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[+] Saved MI ROC curves')


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('CS 6820 PS3 — Differential Privacy with DP-SGD')
    print(f'Device: {DEVICE}')
    print('=' * 60)

    trainloader, testloader = get_loaders()

    # ── 1. Baseline ──
    baseline_model, baseline_acc = train_baseline(trainloader, testloader)

    # ── 2. DP-SGD for each epsilon ──
    dp_models   = {}
    dp_accs     = []

    if HAS_OPACUS:
        print('\n' + '='*55)
        print('DP-SGD Training: Epsilon Sweep')
        print('='*55)
        for eps in TARGET_EPS:
            model, acc, engine = train_dp_sgd(
                eps, max_grad_norm=1.0,
                trainloader=trainloader, testloader=testloader)
            dp_models[eps] = model
            dp_accs.append(acc)
    else:
        dp_accs = [0.0] * len(TARGET_EPS)

    # ── 3. Clipping Norm Sweep at ε=3 ──
    clip_accs = []
    if HAS_OPACUS:
        print('\n' + '='*55)
        print(f'Clipping Norm Sweep at ε={FIXED_EPS}')
        print('='*55)
        for C in CLIP_NORMS:
            _, acc, _ = train_dp_sgd(
                FIXED_EPS, max_grad_norm=C,
                trainloader=trainloader, testloader=testloader)
            clip_accs.append(acc)
            print(f'  C={C}  acc={acc*100:.1f}%')
    else:
        clip_accs = [0.0] * len(CLIP_NORMS)

    # ── 4. Membership Inference ──
    print('\n' + '='*55)
    print('Membership Inference Attack')
    print('='*55)

    all_fprs, all_tprs, all_labels, all_aucs = [], [], [], []

    # Non-private model
    mi_adv_nat, auc_nat, fpr_nat, tpr_nat = membership_inference_attack(baseline_model)
    all_fprs.append(fpr_nat); all_tprs.append(tpr_nat)
    all_labels.append('Non-private'); all_aucs.append(auc_nat)
    print(f'Non-private model:  MI advantage={mi_adv_nat:.3f}, AUC={auc_nat:.3f}')

    # DP models
    mi_results = {}
    if HAS_OPACUS:
        for eps in [1.0, 3.0, 10.0]:
            if eps in dp_models and dp_models[eps] is not None:
                mi_adv, auc, fpr, tpr = membership_inference_attack(dp_models[eps])
                mi_results[eps] = (mi_adv, auc)
                all_fprs.append(fpr); all_tprs.append(tpr)
                all_labels.append(f'DP ε={eps}'); all_aucs.append(auc)
                print(f'DP ε={eps}:  MI advantage={mi_adv:.3f}, AUC={auc:.3f}')

    # ── 5. Plots ──
    plot_privacy_utility(TARGET_EPS, dp_accs, baseline_acc)
    plot_clip_norm_sweep(CLIP_NORMS, clip_accs, FIXED_EPS)
    if len(all_fprs) > 1:
        plot_mi_roc(all_fprs, all_tprs, all_labels, all_aucs)

    # ── 6. Summary Table ──
    print('\n' + '='*60)
    print('RESULTS SUMMARY')
    print('='*60)
    print(f'{"Method":<20} {"Test Acc":>12} {"MI Advantage":>15}')
    print('-' * 50)
    print(f'{"Non-private":<20} {baseline_acc*100:>11.2f}% {mi_adv_nat:>14.3f}')
    for i, eps in enumerate(TARGET_EPS):
        adv_str = f'{mi_results[eps][0]:.3f}' if eps in mi_results else 'N/A'
        print(f'{"DP ε="+str(eps):<20} {dp_accs[i]*100:>11.2f}% {adv_str:>15}')

    print(f'\n[+] Plots saved to {PLOT_DIR}/')
    print('[+] Done.')


if __name__ == '__main__':
    main()
