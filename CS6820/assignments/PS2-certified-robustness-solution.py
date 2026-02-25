"""
CS 6820 — Problem Set 2: Certified Robustness
Complete Solution

Implements:
1. Randomized smoothing certifier (Cohen et al. 2019)
   - Train ResNet with Gaussian noise augmentation (σ = 0.25, 0.5, 1.0)
   - Monte Carlo certification with Clopper-Pearson confidence interval
   - Certified accuracy curves at radii [0.0, 0.25, 0.5, 0.75, 1.0]
2. IBP (Interval Bound Propagation) for a 3-layer MLP on MNIST
3. Comparison plots: certified vs. empirical robustness

Requirements:
    pip install torch torchvision matplotlib scipy numpy tqdm

Usage:
    python PS2-certified-robustness-solution.py
"""

import os
import math
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
from scipy import stats
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS    = 20
SIGMAS    = [0.25, 0.5, 1.0]        # Noise levels for smoothing
RADII     = [0.0, 0.25, 0.5, 0.75, 1.0]   # Certification radii
N_CERT    = 1000                    # MC samples for certification
N_CERT_TIGHT = 100000               # MC samples for tight certification
ALPHA_CP  = 0.001                   # Confidence level for Clopper-Pearson
N_EVAL    = 500                     # Number of test points to certify
IBP_EPS   = 0.1                     # IBP certification epsilon (L_inf)
PLOT_DIR  = './ps2_plots'
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Models
# ─────────────────────────────────────────────────────────────────────────────

class SmallResNet(nn.Module):
    """Small ResNet for MNIST (adapted for 1-channel 28×28 input)."""
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        base = models.resnet18(weights=None)
        # Adapt first conv for grayscale
        base.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(base.fc.in_features, 10)
        self.base = base

    def forward(self, x):
        return self.base(x)


class SimpleMLP(nn.Module):
    """3-layer MLP for MNIST — used for IBP certification (small network)."""
    def __init__(self, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Data Loaders
# ─────────────────────────────────────────────────────────────────────────────

def get_mnist_loaders():
    transform = transforms.ToTensor()
    trainset  = torchvision.datasets.MNIST('./data', train=True,
                                            download=True, transform=transform)
    testset   = torchvision.datasets.MNIST('./data', train=False,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                               shuffle=True, num_workers=2)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=256,
                                               shuffle=False, num_workers=2)
    return trainloader, testloader


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Training with Gaussian Noise Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def train_smoothed_model(sigma, epochs=EPOCHS):
    """
    Train a model with Gaussian noise augmentation at level σ.

    During training, each input x is perturbed with N(0, σ²I) noise.
    This trains the base classifier f used by the smoothed classifier g.
    """
    print(f'\n{"="*50}')
    print(f'Training smoothed model: σ = {sigma}')
    print(f'{"="*50}')

    trainloader, testloader = get_mnist_loaders()
    model = SmallResNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Add Gaussian noise during training
            noise = torch.randn_like(x) * sigma
            x_noisy = torch.clamp(x + noise, 0.0, 1.0)

            optimizer.zero_grad()
            criterion(model(x_noisy), y).backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            # Clean accuracy (no noise)
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in testloader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    correct += (model(xb).argmax(1) == yb).sum().item()
                    total   += yb.shape[0]
            print(f'  Epoch {epoch+1}/{epochs}  Clean acc: {correct/total*100:.1f}%')

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Clopper-Pearson Confidence Interval
# ─────────────────────────────────────────────────────────────────────────────

def clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """
    Compute the lower bound of the (1-alpha) Clopper-Pearson confidence interval
    for a Binomial proportion p, given k successes out of n trials.

    The lower bound satisfies:
        P(Binomial(n, p_low) >= k) = alpha/2

    Using the relationship to the Beta distribution:
        p_low = Beta(alpha/2; k, n-k+1)

    Args:
        k:     Number of successes
        n:     Number of trials
        alpha: Significance level (e.g., 0.001 → 99.9% CI lower bound)

    Returns:
        p_low: Lower bound on the true proportion
    """
    if k == 0:
        return 0.0
    return stats.beta.ppf(alpha / 2, k, n - k + 1)


def clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    """Upper bound of Clopper-Pearson CI."""
    if k == n:
        return 1.0
    return stats.beta.ppf(1 - alpha / 2, k + 1, n - k)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Randomized Smoothing Certifier (Cohen et al. 2019)
# ─────────────────────────────────────────────────────────────────────────────

def smooth_predict(model, x: torch.Tensor, sigma: float,
                   n: int = N_CERT, batch_size: int = 500) -> torch.Tensor:
    """
    Compute the smoothed classifier's output probabilities via Monte Carlo.

    g(x) = argmax_c P(f(x + ε) = c)  where ε ~ N(0, σ²I)

    Returns vote counts per class: shape [10]
    """
    model.eval()
    x = x.to(DEVICE)
    counts = torch.zeros(10, dtype=torch.long)

    remaining = n
    while remaining > 0:
        batch_n = min(batch_size, remaining)
        # Sample noise and add to input
        noise = torch.randn(batch_n, *x.shape).to(DEVICE) * sigma
        x_noisy = torch.clamp(x.unsqueeze(0) + noise, 0.0, 1.0)
        with torch.no_grad():
            preds = model(x_noisy).argmax(dim=1)  # [batch_n]
        for c in range(10):
            counts[c] += (preds == c).sum().item()
        remaining -= batch_n

    return counts


def certify_sample(model, x: torch.Tensor, sigma: float,
                   n: int = N_CERT, alpha: float = ALPHA_CP):
    """
    Certify a single sample x using randomized smoothing.

    Returns:
        prediction:     The smoothed classifier's prediction class (or -1 for ABSTAIN)
        certified_radius: Certified L2 radius (0.0 if ABSTAIN)

    Algorithm (Cohen et al. 2019):
    1. Get vote counts from n_0 samples (for selection — we skip this optimization
       and use n samples directly for simplicity)
    2. Let c_A = class with most votes, n_A = vote count
    3. Compute lower Clopper-Pearson bound p_A_low = CP_lower(n_A, n, α)
    4. If p_A_low > 0.5: certify at radius R = σ · Φ⁻¹(p_A_low)
       Else: ABSTAIN
    """
    counts = smooth_predict(model, x, sigma, n=n)
    c_A = counts.argmax().item()          # Predicted class
    n_A = counts[c_A].item()             # Vote count for top class

    # Lower confidence bound on the probability of the top class
    p_A_lower = clopper_pearson_lower(n_A, n, alpha)

    if p_A_lower > 0.5:
        # Certified radius: R = σ · Φ⁻¹(p_A_lower)
        # Φ⁻¹ is the quantile function of the standard normal
        radius = sigma * stats.norm.ppf(p_A_lower)
        return c_A, radius
    else:
        return -1, 0.0  # ABSTAIN


def compute_certified_accuracy(model, testloader, sigma, radii,
                                n_eval=N_EVAL, n_cert=N_CERT):
    """
    Compute certified accuracy at each radius in radii.

    For each test sample:
    - If certified at radius r AND prediction is correct: counts as certified correct at r
    - If ABSTAIN or incorrect: does not count

    Returns dict: radius → certified_accuracy
    """
    print(f'  Certifying {n_eval} samples with σ={sigma}, n={n_cert}...')
    model.eval()

    certified_correct = {r: 0 for r in radii}
    n_abstain = 0
    total = 0

    for x_batch, y_batch in tqdm(testloader, desc='  Batches', leave=False):
        for i in range(len(x_batch)):
            if total >= n_eval:
                break
            x = x_batch[i]
            y = y_batch[i].item()

            pred, radius = certify_sample(model, x, sigma, n=n_cert)

            if pred == -1:
                n_abstain += 1
            elif pred == y:
                # Certified correct — counts for all radii ≤ certified radius
                for r in radii:
                    if radius >= r:
                        certified_correct[r] += 1

            total += 1
        if total >= n_eval:
            break

    cert_acc = {r: certified_correct[r] / n_eval for r in radii}
    print(f'  Abstain rate: {n_abstain/n_eval*100:.1f}%')
    return cert_acc


def compute_pgd50_accuracy(model, testloader, eps_l2, n_eval=N_EVAL):
    """
    Compute empirical robust accuracy against PGD-50 L2 attack.
    Used to compare against certified accuracy.
    """
    model.eval()
    correct = 0
    total = 0

    for x, y in testloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        # PGD-L2 attack
        alpha_l2 = eps_l2 / 10
        x_adv = x + torch.randn_like(x) * 0.01
        x_adv = torch.clamp(x_adv, 0, 1).detach()

        for _ in range(50):
            x_adv = x_adv.requires_grad_(True)
            loss = nn.CrossEntropyLoss()(model(x_adv), y)
            loss.backward()
            grad = x_adv.grad.detach()
            # Normalize gradient to unit L2 norm
            grad_norm = grad.view(grad.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1)
            grad_unit = grad / (grad_norm + 1e-8)
            x_adv = x_adv.detach() + alpha_l2 * grad_unit
            # Project to L2 ball
            delta = x_adv - x
            delta_norm = delta.view(delta.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1)
            factor = torch.clamp(eps_l2 / (delta_norm + 1e-8), max=1.0)
            x_adv = torch.clamp(x + delta * factor, 0, 1).detach()

        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += y.shape[0]
        if total >= n_eval:
            break

    return correct / n_eval


# ─────────────────────────────────────────────────────────────────────────────
# 6.  IBP (Interval Bound Propagation) for MLP
# ─────────────────────────────────────────────────────────────────────────────

def ibp_forward(model: SimpleMLP, x: torch.Tensor,
                eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Propagate an L_inf epsilon ball through the MLP using interval arithmetic.

    For each layer with weights W and bias b:
    - Input interval: [x_lower, x_upper] = [x - eps, x + eps] (clipped to [0,1])
    - After linear layer W: mid = W @ x_mid; rad = |W| @ x_rad
    - After ReLU: lower = max(lower, 0); upper = max(upper, 0)

    Returns:
        logits_lower: [B, 10] lower bound on output logits
        logits_upper: [B, 10] upper bound on output logits
    """
    x_flat = x.view(x.shape[0], -1)

    # Initial interval (clamped to [0, 1])
    l = torch.clamp(x_flat - eps, 0.0, 1.0)
    u = torch.clamp(x_flat + eps, 0.0, 1.0)

    def ibp_linear(l, u, weight, bias):
        """Propagate through a linear layer."""
        w_pos = torch.clamp(weight, min=0)
        w_neg = torch.clamp(weight, max=0)
        # Lower bound: use negative weights with upper, positive weights with lower
        l_out = l @ w_pos.T + u @ w_neg.T + bias
        # Upper bound: use positive weights with upper, negative weights with lower
        u_out = u @ w_pos.T + l @ w_neg.T + bias
        return l_out, u_out

    def ibp_relu(l, u):
        """Propagate through ReLU."""
        return torch.clamp(l, min=0), torch.clamp(u, min=0)

    # Layer 1
    l, u = ibp_linear(l, u, model.fc1.weight.detach(), model.fc1.bias.detach())
    l, u = ibp_relu(l, u)

    # Layer 2
    l, u = ibp_linear(l, u, model.fc2.weight.detach(), model.fc2.bias.detach())
    l, u = ibp_relu(l, u)

    # Layer 3 (output — no ReLU)
    l, u = ibp_linear(l, u, model.fc3.weight.detach(), model.fc3.bias.detach())

    return l, u


def ibp_certified_accuracy(model: SimpleMLP, testloader, eps: float,
                            n_eval: int = N_EVAL) -> float:
    """
    Compute IBP certified accuracy at L_inf radius eps.

    A sample (x, y) is certified robust if the IBP lower bound on the
    true class logit > upper bound on all other class logits:
        logits_lower[y] > max_{c ≠ y} logits_upper[c]

    This means no perturbation δ with ||δ||_inf ≤ eps can change the prediction.
    """
    model.eval()
    certified = 0
    total = 0

    for x, y in tqdm(testloader, desc='  IBP certifying', leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        l, u = ibp_forward(model, x, eps)

        # For each sample: check if true class lower bound > all other upper bounds
        B = x.shape[0]
        for i in range(B):
            if total >= n_eval:
                break
            c = y[i].item()
            true_lower = l[i, c].item()
            # Worst-case max over other classes
            other_upper = u[i].clone()
            other_upper[c] = -1e9
            max_other_upper = other_upper.max().item()
            if true_lower > max_other_upper:
                certified += 1
            total += 1
        if total >= n_eval:
            break

    return certified / n_eval


def train_ibp_model(eps: float = IBP_EPS, epochs: int = EPOCHS):
    """
    Train an MLP with IBP regularization for certified robustness.

    IBP training loss: (1-κ) · CE(f(x), y) + κ · CE(f_lower_worst(x, eps), y)
    where f_lower_worst is the IBP lower bound on the correct class.
    """
    print(f'\n{"="*50}')
    print(f'Training IBP MLP: ε={eps} (L_inf)')
    print(f'{"="*50}')

    trainloader, testloader = get_mnist_loaders()
    model = SimpleMLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    kappa = 0.5  # Balance between natural and IBP loss

    for epoch in range(epochs):
        model.train()
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            # Natural loss
            logits = model(x)
            loss_nat = F.cross_entropy(logits, y)

            # IBP loss: penalize when verified worst-case output is wrong
            l_out, u_out = ibp_forward(model, x, eps)
            # Worst-case: lower bound on true class, upper bound on others
            B = x.shape[0]
            y_onehot = torch.zeros_like(l_out)
            y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
            # For each sample, the worst-case logit for the true class is the lower bound
            # The worst-case logit for each other class is the upper bound
            worst_case_logits = u_out.clone()
            worst_case_logits[range(B), y] = l_out[range(B), y]
            loss_ibp = F.cross_entropy(worst_case_logits, y)

            loss = (1 - kappa) * loss_nat + kappa * loss_ibp
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in testloader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    correct += (model(xb).argmax(1) == yb).sum().item()
                    total   += yb.shape[0]
            print(f'  Epoch {epoch+1}/{epochs}  '
                  f'Clean acc: {correct/total*100:.1f}%')

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_certified_accuracy(all_cert_accs, all_emp_accs, sigmas, radii):
    """
    Plot certified accuracy curves for each σ, with empirical PGD accuracy overlaid.
    """
    colors  = ['#2980b9', '#e74c3c', '#27ae60']
    styles  = ['-o', '-s', '-^']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, (sigma, ax) in enumerate(zip(sigmas, axes)):
        cert_vals = [all_cert_accs[sigma][r] * 100 for r in radii]
        emp_vals  = all_emp_accs[sigma]  # List of (radius, acc) tuples

        ax.plot(radii, cert_vals,
                color=colors[i], linestyle='-', marker='o',
                linewidth=2, markersize=8, label='Certified accuracy')

        # Plot empirical PGD accuracy at corresponding radii
        emp_radii = [e[0] for e in emp_vals]
        emp_accs_pct = [e[1]*100 for e in emp_vals]
        ax.plot(emp_radii, emp_accs_pct,
                color=colors[i], linestyle='--', marker='s',
                linewidth=2, markersize=8, alpha=0.7, label='PGD-50 empirical')

        ax.fill_between(radii, cert_vals, emp_accs_pct[:len(radii)],
                        alpha=0.1, color=colors[i],
                        label='Certification gap')

        ax.set_title(f'σ = {sigma}', fontsize=13)
        ax.set_xlabel('Certified Radius (L₂)', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.02, max(radii) + 0.05)

    plt.suptitle('Randomized Smoothing: Certified vs. Empirical Robustness\n'
                 'MNIST ResNet (n=1000 MC samples)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'certified_accuracy_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[+] Saved certified accuracy curves')


def plot_sigma_comparison(all_cert_accs, sigmas, radii):
    """Compare all σ values on one plot."""
    colors = ['#2980b9', '#e74c3c', '#27ae60']
    markers = ['o', 's', '^']

    fig, ax = plt.subplots(figsize=(8, 5))
    for sigma, color, marker in zip(sigmas, colors, markers):
        cert_vals = [all_cert_accs[sigma][r]*100 for r in radii]
        ax.plot(radii, cert_vals, color=color, marker=marker,
                linewidth=2, markersize=9, label=f'σ = {sigma}')

    ax.set_xlabel('Certified Radius (L₂)', fontsize=12)
    ax.set_ylabel('Certified Accuracy (%)', fontsize=12)
    ax.set_title('Effect of σ on Certified Accuracy\n(MNIST, n=1000 MC samples)',
                 fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'sigma_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[+] Saved σ comparison plot')


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('CS 6820 PS2 — Certified Robustness')
    print(f'Device: {DEVICE}')
    print('=' * 60)

    _, testloader = get_mnist_loaders()

    all_cert_accs = {}   # sigma → {radius → cert_acc}
    all_emp_accs  = {}   # sigma → list of (radius, pgd50_acc)

    # ── Part 1: Randomized Smoothing ──
    for sigma in SIGMAS:
        model = train_smoothed_model(sigma)

        # Certified accuracy at each radius
        cert_acc = compute_certified_accuracy(
            model, testloader, sigma, RADII, n_eval=N_EVAL, n_cert=N_CERT)
        all_cert_accs[sigma] = cert_acc

        # Empirical PGD-50 robustness at matching radii
        emp_accs = []
        for r in RADII[1:]:  # Skip r=0.0 (always 100% for PGD-0)
            pgd_acc = compute_pgd50_accuracy(model, testloader, r, n_eval=N_EVAL)
            emp_accs.append((r, pgd_acc))
            print(f'  σ={sigma}  r={r:.2f}  '
                  f'Certified: {cert_acc[r]*100:.1f}%  '
                  f'PGD-50: {pgd_acc*100:.1f}%')
        all_emp_accs[sigma] = emp_accs

    # ── Part 2: IBP MLP ──
    print('\n\n' + '='*60)
    print('IBP Certification (3-layer MLP on MNIST)')
    print('='*60)
    ibp_model = train_ibp_model(eps=IBP_EPS)

    ibp_cert_acc = ibp_certified_accuracy(ibp_model, testloader, IBP_EPS,
                                           n_eval=N_EVAL)
    # For comparison: PGD-50 at same epsilon (converted to L2: eps * sqrt(d))
    eps_l2_equiv = IBP_EPS * math.sqrt(784)  # L_inf → L2 rough conversion
    ibp_emp_acc  = compute_pgd50_accuracy(ibp_model, testloader,
                                           eps_l2=0.5, n_eval=N_EVAL)

    print(f'\nIBP results at ε={IBP_EPS} L_inf:')
    print(f'  IBP certified accuracy: {ibp_cert_acc*100:.1f}%')
    print(f'  PGD-50 empirical (L2=0.5): {ibp_emp_acc*100:.1f}%')

    # ── Part 3: Plots ──
    plot_certified_accuracy(all_cert_accs, all_emp_accs, SIGMAS, RADII)
    plot_sigma_comparison(all_cert_accs, SIGMAS, RADII)

    # ── Final Summary Table ──
    print('\n' + '='*60)
    print('CERTIFIED ACCURACY TABLE')
    print('='*60)
    print(f'{"Method":<18}' + ''.join(f'{r:>8.2f}' for r in RADII))
    print('-' * (18 + 8*len(RADII)))
    for sigma in SIGMAS:
        row = f'RS σ={sigma:<6}'
        row += ''.join(f'{all_cert_accs[sigma][r]*100:>7.1f}%' for r in RADII)
        print(row)
    print(f'IBP (ε=0.1)     ' + f'{ibp_cert_acc*100:>7.1f}%' + ' (certified at ε=0.1 L_inf)')

    print(f'\n[+] Plots saved to {PLOT_DIR}/')
    print('[+] Done.')


if __name__ == '__main__':
    main()
