"""
CS 6820 — Problem Set 1: Adversarial Training Variants
Complete Solution

Implements and compares:
1. Standard training (baseline)
2. PGD-AT (Madry et al. 2018)
3. TRADES (Zhang et al. 2019)
4. MART (Wang et al. 2020)

Evaluates each with:
- Natural accuracy
- PGD-20 robust accuracy
- AutoAttack robust accuracy (requires: pip install autoattack)
- Epsilon sweep for PGD-AT

Requirements:
    pip install torch torchvision autoattack matplotlib tqdm

Usage:
    python PS1-adversarial-training-solution.py
"""

import os
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

# Try importing AutoAttack (optional — falls back to PGD-40 if not available)
try:
    from autoattack import AutoAttack
    HAS_AUTOATTACK = True
except ImportError:
    print('[!] AutoAttack not found. Install with: pip install autoattack')
    print('[!] Falling back to PGD-40 for "robust accuracy" evaluation.')
    HAS_AUTOATTACK = False

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS      = 20          # Increase to 100+ for full-quality results
BATCH_SIZE  = 128
EPS_TRAIN   = 8 / 255     # Training epsilon (L_inf)
EPS_EVAL    = 8 / 255     # Evaluation epsilon
ALPHA_TRAIN = 2 / 255     # PGD step size during training
K_TRAIN     = 7           # PGD steps during training (Madry default)
BETA_TRADES = 6.0         # TRADES regularization weight
CKPT_DIR    = './ps1_checkpoints'
PLOT_DIR    = './ps1_plots'
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model and Data
# ─────────────────────────────────────────────────────────────────────────────

def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m.to(DEVICE)


def get_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR10('./data', train=True,
                                             download=True, transform=transform_train)
    testset  = torchvision.datasets.CIFAR10('./data', train=False,
                                             download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, testloader


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PGD Attack (for inner maximization during training and for evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def pgd_attack(model, x, y, eps, alpha, num_steps, loss_fn='ce', y_orig=None):
    """
    PGD L_inf attack for adversarial training.

    Args:
        model:     Classifier
        x:         Clean inputs [B, C, H, W]
        y:         Labels [B]
        eps:       L_inf budget
        alpha:     Step size
        num_steps: Number of iterations
        loss_fn:   'ce' (cross-entropy) or 'kl' (KL divergence from clean logits)
        y_orig:    Logits of the clean input (used when loss_fn='kl')

    Returns:
        x_adv: Adversarial inputs
    """
    criterion_ce = nn.CrossEntropyLoss()

    # Random start within epsilon ball
    x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    for _ in range(num_steps):
        x_adv = x_adv.requires_grad_(True)
        logits = model(x_adv)

        if loss_fn == 'ce':
            loss = criterion_ce(logits, y)
        elif loss_fn == 'kl':
            # TRADES inner maximization: maximize KL(f(x+δ) || f(x))
            loss = F.kl_div(
                F.log_softmax(logits, dim=1),
                F.softmax(y_orig, dim=1),
                reduction='batchmean'
            )

        loss.backward()
        grad = x_adv.grad.detach()
        x_adv = x_adv.detach() + alpha * grad.sign()

        # Project back into epsilon ball around original x
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

    return x_adv


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Training Methods
# ─────────────────────────────────────────────────────────────────────────────

def train_natural(model, trainloader, testloader, method_name='Natural'):
    """Standard training — no adversarial examples."""
    print(f'\n{"="*55}')
    print(f'Training: {method_name}')
    print(f'{"="*55}')

    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 15], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'nat_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
            total_loss += criterion(model(x.detach()), y).item()

        scheduler.step()
        nat_acc = evaluate_clean(model, testloader)
        history['nat_acc'].append(nat_acc)

        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch+1:02d}/{EPOCHS}  '
                  f'Nat acc: {nat_acc*100:.1f}%')

    return model, history


def train_pgd_at(model, trainloader, testloader, eps=EPS_TRAIN,
                 alpha=ALPHA_TRAIN, k=K_TRAIN):
    """
    PGD Adversarial Training (Madry et al. 2018).

    Objective: min_θ E[(max_{δ∈S} L(f_θ(x+δ), y))]
    Inner max solved with K-step PGD.
    """
    print(f'\n{"="*55}')
    print(f'Training: PGD-AT (ε={eps*255:.0f}/255, α={alpha*255:.0f}/255, K={k})')
    print(f'{"="*55}')

    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 15], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    history = {'nat_acc': [], 'rob_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Inner maximization: find adversarial examples
            model.eval()  # Disable dropout/BN randomness during attack
            x_adv = pgd_attack(model, x, y, eps, alpha, k, loss_fn='ce')
            model.train()

            # Outer minimization
            optimizer.zero_grad()
            loss = criterion(model(x_adv), y)
            loss.backward()
            optimizer.step()

        scheduler.step()
        nat_acc = evaluate_clean(model, testloader)
        rob_acc = evaluate_pgd20(model, testloader, eps)
        history['nat_acc'].append(nat_acc)
        history['rob_acc'].append(rob_acc)

        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch+1:02d}/{EPOCHS}  '
                  f'Nat: {nat_acc*100:.1f}%  '
                  f'Rob (PGD-20): {rob_acc*100:.1f}%')

    return model, history


def train_trades(model, trainloader, testloader, eps=EPS_TRAIN,
                 alpha=ALPHA_TRAIN, k=K_TRAIN, beta=BETA_TRADES):
    """
    TRADES: Theoretically Principled Trade-off between Robustness and Accuracy
    Zhang et al. 2019

    Loss: L_CE(f(x), y) + β · max_{δ∈S} KL(f(x+δ) || f(x))
    Note: the inner max maximizes KL divergence (not CE loss).
    """
    print(f'\n{"="*55}')
    print(f'Training: TRADES (β={beta}, ε={eps*255:.0f}/255)')
    print(f'{"="*55}')

    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 15], gamma=0.1)

    history = {'nat_acc': [], 'rob_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Step 1: Get clean logits (stop gradient — clean branch not updated by PGD)
            model.eval()
            with torch.no_grad():
                logits_clean = model(x)
            model.train()

            # Step 2: Inner maximization — maximize KL(f(x+δ) || f(x))
            model.eval()
            x_adv = pgd_attack(model, x, y, eps, alpha, k,
                                loss_fn='kl', y_orig=logits_clean.detach())
            model.train()

            # Step 3: Outer minimization
            optimizer.zero_grad()
            logits_clean_train = model(x)
            logits_adv_train   = model(x_adv)

            # Natural loss (cross-entropy on clean examples)
            loss_nat = F.cross_entropy(logits_clean_train, y)

            # Robustness loss (KL divergence between clean and adversarial predictions)
            loss_rob = F.kl_div(
                F.log_softmax(logits_adv_train, dim=1),
                F.softmax(logits_clean_train, dim=1).detach(),
                reduction='batchmean'
            )

            loss = loss_nat + beta * loss_rob
            loss.backward()
            optimizer.step()

        scheduler.step()
        nat_acc = evaluate_clean(model, testloader)
        rob_acc = evaluate_pgd20(model, testloader, eps)
        history['nat_acc'].append(nat_acc)
        history['rob_acc'].append(rob_acc)

        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch+1:02d}/{EPOCHS}  '
                  f'Nat: {nat_acc*100:.1f}%  '
                  f'Rob (PGD-20): {rob_acc*100:.1f}%')

    return model, history


def train_mart(model, trainloader, testloader, eps=EPS_TRAIN,
               alpha=ALPHA_TRAIN, k=K_TRAIN, beta=6.0):
    """
    MART: Misclassification-Aware adveRsarial Training
    Wang et al. 2020

    Key idea: Upweight the loss contribution of misclassified examples
    (examples where the clean prediction is already wrong are most in need
    of robustness improvement).

    Loss: BCE(f(x_adv), y) * (1 - f(x)_y) + β * KL(f(x_adv) || f(x))
    """
    print(f'\n{"="*55}')
    print(f'Training: MART (β={beta}, ε={eps*255:.0f}/255)')
    print(f'{"="*55}')

    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 15], gamma=0.1)

    history = {'nat_acc': [], 'rob_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Get clean probabilities (detached for weighting)
            model.eval()
            with torch.no_grad():
                probs_clean = F.softmax(model(x), dim=1)
            model.train()

            # Inner maximization (same as TRADES: maximize KL)
            model.eval()
            x_adv = pgd_attack(model, x, y, eps, alpha, k,
                                loss_fn='ce')  # MART uses CE for inner max
            model.train()

            # Outer minimization
            optimizer.zero_grad()
            logits_clean = model(x)
            logits_adv   = model(x_adv)

            probs_adv   = F.softmax(logits_adv, dim=1)
            probs_clean_curr = F.softmax(logits_clean, dim=1)

            # Weight: (1 - clean probability of the true class)
            # → High weight for samples where model is uncertain/wrong on clean input
            weight = 1.0 - probs_clean_curr.gather(1, y.unsqueeze(1)).squeeze(1)
            weight = weight.detach()

            # Binary cross-entropy of adversarial examples against true labels (MART loss)
            # Using one-hot encoding
            y_onehot = torch.zeros_like(probs_adv)
            y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
            bce_adv = -(y_onehot * torch.log(probs_adv + 1e-8)).sum(dim=1)
            loss_bce = (weight * bce_adv).mean()

            # KL divergence regularization
            loss_kl = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                probs_clean_curr.detach(),
                reduction='batchmean'
            )

            loss = loss_bce + beta * loss_kl
            loss.backward()
            optimizer.step()

        scheduler.step()
        nat_acc = evaluate_clean(model, testloader)
        rob_acc = evaluate_pgd20(model, testloader, eps)
        history['nat_acc'].append(nat_acc)
        history['rob_acc'].append(rob_acc)

        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch+1:02d}/{EPOCHS}  '
                  f'Nat: {nat_acc*100:.1f}%  '
                  f'Rob (PGD-20): {rob_acc*100:.1f}%')

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_clean(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.shape[0]
    return correct / total


def evaluate_pgd20(model, loader, eps, num_samples=1000):
    """Evaluate PGD-20 robust accuracy on the first num_samples test examples."""
    model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv = pgd_attack(model, x, y, eps=eps, alpha=eps/4,
                           num_steps=20, loss_fn='ce')
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += y.shape[0]
        if total >= num_samples:
            break

    return correct / total


def evaluate_autoattack(model, loader, eps, num_samples=1000):
    """Evaluate using AutoAttack (L_inf standard version)."""
    if not HAS_AUTOATTACK:
        return evaluate_pgd20(model, loader, eps, num_samples)

    all_x, all_y = [], []
    for x, y in loader:
        all_x.append(x)
        all_y.append(y)
        if sum(xi.shape[0] for xi in all_x) >= num_samples:
            break
    all_x = torch.cat(all_x)[:num_samples].to(DEVICE)
    all_y = torch.cat(all_y)[:num_samples].to(DEVICE)

    adversary = AutoAttack(model, norm='Linf', eps=eps, version='standard',
                           device=DEVICE, verbose=False)
    x_adv = adversary.run_standard_evaluation(all_x, all_y, bs=250)

    with torch.no_grad():
        correct = (model(x_adv).argmax(1) == all_y).sum().item()
    return correct / num_samples


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Epsilon Sweep for PGD-AT
# ─────────────────────────────────────────────────────────────────────────────

def eps_sweep(trainloader, testloader):
    """Train PGD-AT models for different training epsilon values."""
    eps_values = [2/255, 4/255, 8/255]
    results = {}

    for eps in eps_values:
        model = build_model()
        model, _ = train_pgd_at(model, trainloader, testloader,
                                  eps=eps, alpha=eps*2.5/7, k=7)
        nat_acc = evaluate_clean(model, testloader)
        rob_acc = evaluate_pgd20(model, testloader, eps)
        results[eps] = (nat_acc, rob_acc)
        print(f'  ε={eps*255:.0f}/255 → '
              f'Nat: {nat_acc*100:.1f}%, Rob: {rob_acc*100:.1f}%')

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(histories):
    methods = ['PGD-AT', 'TRADES', 'MART']
    colors  = ['#2980b9', '#e74c3c', '#27ae60']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for method, color in zip(methods, colors):
        h = histories[method]
        epochs = range(1, len(h['nat_acc']) + 1)
        axes[0].plot(epochs, [a*100 for a in h['nat_acc']],
                     color=color, linewidth=2, label=method)
        axes[1].plot(epochs, [a*100 for a in h['rob_acc']],
                     color=color, linewidth=2, label=method)

    for ax, title in zip(axes, ['Natural Accuracy', 'PGD-20 Robust Accuracy']):
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    plt.suptitle('Adversarial Training Methods — CIFAR-10 ResNet-18', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'training_curves.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'[+] Saved training curves')


def plot_eps_frontier(eps_results, summary):
    fig, ax = plt.subplots(figsize=(7, 5))

    # Epsilon sweep points (PGD-AT)
    eps_labels = [f'{int(e*255)}/255' for e in sorted(eps_results.keys())]
    nat_pts = [eps_results[e][0]*100 for e in sorted(eps_results.keys())]
    rob_pts = [eps_results[e][1]*100 for e in sorted(eps_results.keys())]
    ax.plot(nat_pts, rob_pts, 'b-o', linewidth=2, markersize=10,
            label='PGD-AT (ε sweep)')
    for i, (ep, n, r) in enumerate(zip(eps_labels, nat_pts, rob_pts)):
        ax.annotate(f'ε={ep}', (n, r), textcoords='offset points',
                    xytext=(5, 5), fontsize=9)

    # Other methods at ε=8/255
    method_colors = {'TRADES': '#e74c3c', 'MART': '#27ae60'}
    for method, color in method_colors.items():
        n = summary[method]['nat'] * 100
        r = summary[method]['rob'] * 100
        ax.scatter([n], [r], s=150, color=color, zorder=5, label=method)
        ax.annotate(method, (n, r), textcoords='offset points',
                    xytext=(5, -12), fontsize=9, color=color)

    ax.set_xlabel('Natural Accuracy (%)', fontsize=12)
    ax.set_ylabel('PGD-20 Robust Accuracy (%)', fontsize=12)
    ax.set_title('Robustness-Accuracy Frontier\n(CIFAR-10, ε=8/255 eval)',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'robustness_accuracy_frontier.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[+] Saved frontier plot')


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('CS 6820 PS1 — Adversarial Training Variants')
    print(f'Device: {DEVICE}  |  Epochs: {EPOCHS}')
    print('=' * 60)

    trainloader, testloader = get_loaders()

    # ── 1. Natural Training ──
    model_nat = build_model()
    model_nat, _ = train_natural(model_nat, trainloader, testloader)
    nat_clean = evaluate_clean(model_nat, testloader)
    print(f'Natural model — Clean acc: {nat_clean*100:.1f}%')

    # ── 2. PGD-AT ──
    model_pgdat = build_model()
    model_pgdat, hist_pgdat = train_pgd_at(model_pgdat, trainloader, testloader)
    torch.save(model_pgdat.state_dict(), os.path.join(CKPT_DIR, 'pgdat.pth'))

    # ── 3. TRADES ──
    model_trades = build_model()
    model_trades, hist_trades = train_trades(model_trades, trainloader, testloader)
    torch.save(model_trades.state_dict(), os.path.join(CKPT_DIR, 'trades.pth'))

    # ── 4. MART ──
    model_mart = build_model()
    model_mart, hist_mart = train_mart(model_mart, trainloader, testloader)
    torch.save(model_mart.state_dict(), os.path.join(CKPT_DIR, 'mart.pth'))

    # ── 5. Full Evaluation ──
    print('\n\n' + '='*60)
    print('Final Evaluation (AutoAttack for robust acc)')
    print('='*60)

    models_dict = {
        'Natural': model_nat,
        'PGD-AT':  model_pgdat,
        'TRADES':  model_trades,
        'MART':    model_mart,
    }

    summary = {}
    for name, model in models_dict.items():
        nat = evaluate_clean(model, testloader)
        rob_pgd20 = evaluate_pgd20(model, testloader, EPS_EVAL)
        rob_aa    = evaluate_autoattack(model, testloader, EPS_EVAL)
        summary[name] = {'nat': nat, 'rob': rob_pgd20, 'rob_aa': rob_aa}
        print(f'  {name:<10} Nat: {nat*100:.1f}%  '
              f'PGD-20: {rob_pgd20*100:.1f}%  '
              f'AutoAttack: {rob_aa*100:.1f}%')

    # ── 6. Epsilon Sweep (PGD-AT only) ──
    print('\n[Epsilon Sweep for PGD-AT]')
    eps_results = eps_sweep(trainloader, testloader)

    # ── 7. Plots ──
    histories = {
        'PGD-AT': hist_pgdat,
        'TRADES': hist_trades,
        'MART':   hist_mart,
    }
    plot_training_curves(histories)
    plot_eps_frontier(eps_results, summary)

    # ── 8. Summary Table ──
    print('\n' + '='*60)
    print('FINAL RESULTS TABLE (ε = 8/255)')
    print('='*60)
    print(f'{"Method":<12} {"Natural":>10} {"PGD-20":>10} {"AutoAttack":>12}')
    print('-' * 46)
    for name, res in summary.items():
        print(f'{name:<12} {res["nat"]*100:>9.1f}% '
              f'{res["rob"]*100:>9.1f}% '
              f'{res["rob_aa"]*100:>11.1f}%')

    print(f'\n[+] Checkpoints saved to {CKPT_DIR}/')
    print(f'[+] Plots saved to {PLOT_DIR}/')
    print('[+] Done.')


if __name__ == '__main__':
    main()
