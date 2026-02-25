"""
CS 6810 — Programming Assignment 2: Black-Box Attack Suite
Complete Solution

Implements:
1. MI-FGSM (Momentum Iterative FGSM) — transfer-based black-box attack
2. NES (Natural Evolution Strategies) — score-based black-box attack
3. HopSkipJump — decision-based black-box attack
4. 4×4 Transferability matrix across ResNet-18, ViT-B/16, MobileNetV2, DenseNet-121
5. Query budget curves for NES and HopSkipJump

Requirements:
    pip install torch torchvision timm matplotlib numpy tqdm

Note: ViT requires the timm library. If unavailable, ViT is replaced with EfficientNet-B0.

Usage:
    python PA2-black-box-attacks-solution.py
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

try:
    import timm
    HAS_TIMM = True
except ImportError:
    print('[!] timm not found. ViT will be replaced with EfficientNet-B0.')
    HAS_TIMM = False

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_EVAL    = 500           # Samples for attack evaluation
BATCH     = 50
EPS       = 8 / 255       # L_inf budget
PLOT_DIR  = './pa2_plots'
CKPT_DIR  = './pa2_checkpoints'
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model Zoo — four architectures for transferability study
# ─────────────────────────────────────────────────────────────────────────────

def build_model(name: str, pretrained_imagenet: bool = False):
    """
    Build one of four architectures. For CIFAR-10 evaluation we either:
    (a) use a model trained on CIFAR-10 (preferred), or
    (b) adapt an ImageNet model to CIFAR-10 by replacing the final layer.

    For the transferability study we use CIFAR-10-trained models.
    """
    if name == 'resnet18':
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 10)
    elif name == 'mobilenetv2':
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 10)
    elif name == 'densenet121':
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, 10)
    elif name == 'vit':
        if HAS_TIMM:
            # Small ViT suitable for CIFAR-10
            m = timm.create_model('vit_small_patch16_224', pretrained=False,
                                  num_classes=10, img_size=32)
        else:
            # Fallback: EfficientNet-B0
            m = models.efficientnet_b0(weights=None)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, 10)
            name = 'efficientnet_b0'
    else:
        raise ValueError(f'Unknown model: {name}')
    return m.to(DEVICE), name


def train_model_cifar10(model, name: str, epochs: int = 25):
    """Train a model on CIFAR-10 and save checkpoint."""
    ckpt = os.path.join(CKPT_DIR, f'{name}_cifar10.pth')
    if os.path.exists(ckpt):
        print(f'  Loading {name} from checkpoint')
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        return model

    print(f'  Training {name} on CIFAR-10 ({epochs} epochs)...')
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10('./data', train=True,
                                             download=True, transform=transform)
    loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                 momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f'    Epoch {epoch+1}/{epochs}')

    torch.save(model.state_dict(), ckpt)
    model.eval()
    return model


def load_models():
    """Load all four models (train if needed)."""
    model_names = ['resnet18', 'vit', 'mobilenetv2', 'densenet121']
    zoo = {}
    for name in model_names:
        print(f'\n[+] Preparing {name}')
        m, actual_name = build_model(name)
        m = train_model_cifar10(m, actual_name)
        zoo[actual_name] = m
    return zoo


def get_test_subset():
    """Return first N_EVAL CIFAR-10 test images as tensors."""
    testset = torchvision.datasets.CIFAR10(
        './data', train=False, download=True, transform=transforms.ToTensor())
    subset = torch.utils.data.Subset(testset, range(N_EVAL))
    loader = torch.utils.data.DataLoader(subset, batch_size=N_EVAL)
    x, y = next(iter(loader))
    return x.to(DEVICE), y.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MI-FGSM (Momentum Iterative FGSM)
#     Dong et al. 2018 "Boosting Adversarial Attacks with Momentum"
#
#     Update rule:
#       g_{t+1} = μ · g_t + ∇_x L / ||∇_x L||_1
#       x_{t+1} = Clip_{x, ε}(x_t + α · sign(g_{t+1}))
# ─────────────────────────────────────────────────────────────────────────────

def mi_fgsm(model, x, y, eps=EPS, alpha=None, num_steps=10, mu=1.0):
    """
    Momentum Iterative FGSM (MI-FGSM).

    The momentum term g accumulates a velocity vector that smooths the
    attack direction across steps, helping escape local optima and improving
    transfer to black-box models.

    Args:
        model:     Surrogate model (white-box access)
        x:         Clean inputs [B, C, H, W]
        y:         True labels [B]
        eps:       L_inf budget
        alpha:     Step size (default: eps/num_steps)
        num_steps: Number of steps
        mu:        Decay factor for momentum (typically 1.0)

    Returns:
        x_adv: Adversarial examples optimized on model, intended to transfer
    """
    if alpha is None:
        alpha = eps / num_steps

    criterion = nn.CrossEntropyLoss()
    x_adv = x.clone().detach()
    g     = torch.zeros_like(x)  # Momentum accumulator

    for _ in range(num_steps):
        x_adv = x_adv.requires_grad_(True)
        loss = criterion(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad.detach()

        # Normalize gradient by its L1 norm (per sample)
        l1_norm = grad.view(grad.shape[0], -1).abs().sum(dim=1).view(-1, 1, 1, 1)
        grad_normalized = grad / (l1_norm + 1e-8)

        # Update momentum
        g = mu * g + grad_normalized

        # Update adversarial example
        x_adv = x_adv.detach() + alpha * g.sign()

        # Project back into epsilon ball
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

    return x_adv


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NES Attack (Natural Evolution Strategies)
#     Ilyas et al. 2018 "Black-Box Adversarial Attacks with Limited Queries"
#
#     Gradient estimate: ∇L ≈ (1/nσ) Σ_i δ_i · L(x + σδ_i)
#     where δ_i ~ N(0, I) are antithetic pairs.
# ─────────────────────────────────────────────────────────────────────────────

def nes_attack(model_fn, x, y, eps=EPS, alpha=None, num_steps=50,
               n_samples=50, sigma=0.001, targeted=False):
    """
    NES (Natural Evolution Strategies) black-box attack.

    Uses score-based gradient estimation: query the model at random
    perturbations of the current x, use the response to estimate ∇L.

    Args:
        model_fn:  Function x → logits (called for each query)
        x:         Clean input [1, C, H, W] (single sample for clarity)
        y:         True label [1]
        eps:       L_inf budget
        alpha:     Step size
        num_steps: Number of NES steps
        n_samples: Number of antithetic pairs per step
        sigma:     Smoothing radius for gradient estimation

    Returns:
        x_adv:       Adversarial example
        query_count: Number of model queries used
    """
    if alpha is None:
        alpha = eps / num_steps * 2

    x_adv = x.clone().detach()
    query_count = 0

    for step in range(num_steps):
        # Antithetic gradient estimation
        grad_est = torch.zeros_like(x_adv)

        for _ in range(n_samples):
            # Sample random direction
            delta = torch.randn_like(x_adv)

            # Query at x + σδ and x - σδ (antithetic pairs)
            x_plus  = torch.clamp(x_adv + sigma * delta, 0.0, 1.0)
            x_minus = torch.clamp(x_adv - sigma * delta, 0.0, 1.0)

            with torch.no_grad():
                logits_plus  = model_fn(x_plus)
                logits_minus = model_fn(x_minus)

            # Loss (cross-entropy, maximize)
            loss_plus  = F.cross_entropy(logits_plus,  y).item()
            loss_minus = F.cross_entropy(logits_minus, y).item()

            # Gradient estimate via finite differences along δ direction
            grad_est += (loss_plus - loss_minus) / (2 * sigma) * delta
            query_count += 2

        grad_est /= n_samples

        # Gradient sign update
        x_adv = x_adv + alpha * grad_est.sign()

        # Project back to epsilon ball
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

    return x_adv, query_count


# ─────────────────────────────────────────────────────────────────────────────
# 4.  HopSkipJump Attack (Decision-Based)
#     Chen et al. 2020 — operates with hard-label outputs only
#
#     Algorithm:
#     1. Find initial adversarial example (random noise search)
#     2. Binary search on the line between clean and adversarial to
#        find the decision boundary
#     3. Estimate the gradient of the boundary via Monte Carlo
#     4. Take a step along estimated gradient, then project back to boundary
# ─────────────────────────────────────────────────────────────────────────────

def hop_skip_jump(predict_label_fn, x, y, eps=EPS, num_steps=20,
                  n_boundary_queries=100, step_size=0.1):
    """
    Simplified HopSkipJump attack (decision-based).

    Only requires hard labels from the model. Finds the nearest decision
    boundary point using gradient estimation from binary queries.

    Args:
        predict_label_fn: x → predicted class (integer), uses hard labels only
        x:                Clean input [1, C, H, W]
        y:                True label (integer)
        eps:              L_inf constraint
        num_steps:        Number of attack steps
        n_boundary_queries: Queries for boundary gradient estimation

    Returns:
        x_adv:       Adversarial example (or x if no adversarial found)
        query_count: Total queries used
    """
    query_count = 0

    def is_adversarial(x_t):
        nonlocal query_count
        pred = predict_label_fn(x_t)
        query_count += 1
        return int(pred) != int(y)

    # Step 1: Find an initial adversarial example via random search
    x_adv = None
    for _ in range(1000):
        candidate = torch.rand_like(x)  # Random image
        if is_adversarial(candidate):
            x_adv = candidate.clone()
            break

    if x_adv is None:
        # Fallback: use maximally perturbed version
        x_adv = torch.clamp(x + eps * torch.sign(torch.randn_like(x)), 0, 1)

    # Step 2: Binary search to the decision boundary
    def binary_search_boundary(x_clean, x_adv, n_steps=25):
        lo, hi = 0.0, 1.0
        for _ in range(n_steps):
            mid = (lo + hi) / 2
            x_mid = (1 - mid) * x_clean + mid * x_adv
            if is_adversarial(x_mid):
                hi = mid
            else:
                lo = mid
        return (1 - hi) * x_clean + hi * x_adv

    x_adv = binary_search_boundary(x, x_adv)

    for step in range(num_steps):
        # Step 3: Estimate gradient at the boundary via random perturbations
        grad_est = torch.zeros_like(x)

        for _ in range(n_boundary_queries):
            rv = torch.randn_like(x)
            rv /= rv.view(-1).norm() + 1e-8
            # Perturb along this direction and check which side of boundary we land on
            x_perturbed = torch.clamp(x_adv + step_size * rv, 0, 1)
            if is_adversarial(x_perturbed):
                grad_est += rv  # This direction stays adversarial → gradient points there
            else:
                grad_est -= rv  # This direction leaves adversarial region

        grad_est /= n_boundary_queries

        # Step 4: Move toward clean image along gradient, then project back to boundary
        x_candidate = x_adv - step_size * grad_est  # Move toward clean image
        x_candidate = torch.clamp(x_candidate, 0, 1)

        if not is_adversarial(x_candidate):
            # Binary search back to boundary
            x_adv = binary_search_boundary(x, x_adv)
        else:
            x_adv = binary_search_boundary(x, x_candidate)

        # Enforce L_inf constraint
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, 0, 1)

    return x_adv, query_count


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Transferability Matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_transferability_matrix(zoo, x_test, y_test):
    """
    Compute attack success rate for all surrogate→target model pairs.
    Attack: MI-FGSM (10 steps, μ=1.0) on surrogate; evaluate on target.

    Returns:
        matrix: |M|×|M| numpy array of transfer rates
        model_names: list of model names (rows = surrogate, cols = target)
    """
    model_names = list(zoo.keys())
    n = len(model_names)
    matrix = np.zeros((n, n))

    for i, surrogate_name in enumerate(model_names):
        surrogate = zoo[surrogate_name]
        surrogate.eval()

        print(f'\n  Surrogate: {surrogate_name}')

        # Generate adversarial examples using surrogate
        all_adv = []
        for b_start in range(0, N_EVAL, BATCH):
            xb = x_test[b_start:b_start+BATCH]
            yb = y_test[b_start:b_start+BATCH]
            x_adv = mi_fgsm(surrogate, xb, yb)
            all_adv.append(x_adv.detach())
        x_adv_all = torch.cat(all_adv)

        for j, target_name in enumerate(model_names):
            target = zoo[target_name]
            target.eval()

            # Evaluate transfer rate
            with torch.no_grad():
                preds = []
                for b_start in range(0, N_EVAL, BATCH):
                    xb = x_adv_all[b_start:b_start+BATCH]
                    pred = target(xb).argmax(1)
                    preds.append(pred)
                preds = torch.cat(preds)

            # Attack success rate = fraction where adversarial is misclassified
            asr = (preds != y_test).float().mean().item()
            matrix[i, j] = asr
            print(f'    → {target_name}: ASR = {asr*100:.1f}%')

    return matrix, model_names


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Query Budget Analysis for NES and HopSkipJump
# ─────────────────────────────────────────────────────────────────────────────

def nes_query_budget_experiment(target_model, x_test, y_test, n_samples=20):
    """Run NES with increasing step counts; measure success rate vs. query count."""
    step_counts = [5, 10, 20, 50, 100]
    results = []

    for steps in step_counts:
        success = 0
        total_queries = 0

        for i in range(n_samples):
            x_single = x_test[i].unsqueeze(0)
            y_single = y_test[i].unsqueeze(0)

            def model_fn(xb):
                return target_model(xb)

            x_adv, qc = nes_attack(model_fn, x_single, y_single,
                                    num_steps=steps, n_samples=30)
            total_queries += qc

            with torch.no_grad():
                pred = target_model(x_adv).argmax(1)
            if pred.item() != y_single.item():
                success += 1

        asr = success / n_samples
        avg_q = total_queries / n_samples
        results.append((steps, avg_q, asr))
        print(f'    NES steps={steps}: avg_queries={avg_q:.0f}, ASR={asr*100:.1f}%')

    return results


def hsj_distortion_experiment(target_model, x_test, y_test, n_samples=10):
    """Measure L2 distortion of HSJ adversarial examples vs. query count."""
    step_counts = [5, 10, 20]
    results = []

    for steps in step_counts:
        distortions = []
        query_counts = []

        for i in range(n_samples):
            x_single = x_test[i].unsqueeze(0)
            y_true = y_test[i].item()

            def pred_fn(xb):
                with torch.no_grad():
                    return target_model(xb).argmax(1).item()

            x_adv, qc = hop_skip_jump(pred_fn, x_single, y_true,
                                       num_steps=steps, n_boundary_queries=50)
            # L2 distortion
            dist = (x_adv - x_single).view(-1).norm().item()
            distortions.append(dist)
            query_counts.append(qc)

        avg_dist  = np.mean(distortions)
        avg_query = np.mean(query_counts)
        results.append((steps, avg_query, avg_dist))
        print(f'    HSJ steps={steps}: avg_queries={avg_query:.0f}, '
              f'avg_L2={avg_dist:.3f}')

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_transfer_matrix(matrix, model_names):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Attack Success Rate')

    n = len(model_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(model_names, fontsize=10)
    ax.set_xlabel('Target Model', fontsize=12)
    ax.set_ylabel('Surrogate Model', fontsize=12)
    ax.set_title('MI-FGSM Transferability Matrix\n(CIFAR-10, ε=8/255)',
                 fontsize=12)

    for i in range(n):
        for j in range(n):
            color = 'white' if matrix[i, j] > 0.6 else 'black'
            ax.text(j, i, f'{matrix[i,j]*100:.0f}%',
                    ha='center', va='center', fontsize=10, color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'transferability_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[+] Saved transferability matrix')


def plot_query_budget(nes_results, hsj_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # NES: ASR vs. query count
    nes_q = [r[1] for r in nes_results]
    nes_asr = [r[2]*100 for r in nes_results]
    ax1.plot(nes_q, nes_asr, 'b-o', linewidth=2, markersize=9)
    ax1.set_xlabel('Avg. Query Count', fontsize=12)
    ax1.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax1.set_title('NES: Success Rate vs. Query Budget', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # HSJ: L2 distortion vs. query count
    hsj_q = [r[1] for r in hsj_results]
    hsj_d = [r[2] for r in hsj_results]
    ax2.plot(hsj_q, hsj_d, 'r-s', linewidth=2, markersize=9)
    ax2.set_xlabel('Avg. Query Count', fontsize=12)
    ax2.set_ylabel('Avg. L₂ Distortion', fontsize=12)
    ax2.set_title('HopSkipJump: Distortion vs. Query Budget', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Black-Box Attack Query Efficiency (CIFAR-10)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'query_budget_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[+] Saved query budget curves')


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('CS 6810 PA2 — Black-Box Attack Suite')
    print(f'Device: {DEVICE}')
    print('=' * 60)

    # Load all models
    print('\n[1] Loading/training model zoo...')
    zoo = load_models()
    model_names = list(zoo.keys())

    # Load test data
    print('\n[2] Loading test data...')
    x_test, y_test = get_test_subset()
    print(f'    Loaded {len(x_test)} test samples')

    # Transferability matrix
    print('\n[3] Computing transferability matrix...')
    transfer_matrix, _ = compute_transferability_matrix(zoo, x_test, y_test)
    plot_transfer_matrix(transfer_matrix, model_names)

    # Use ResNet-18 as the target for query budget experiments
    target = zoo[model_names[0]]
    target.eval()

    # NES query budget
    print('\n[4] NES query budget experiment (target=ResNet-18)...')
    nes_results = nes_query_budget_experiment(target, x_test, y_test, n_samples=20)

    # HopSkipJump distortion
    print('\n[5] HopSkipJump distortion experiment (target=ResNet-18)...')
    hsj_results = hsj_distortion_experiment(target, x_test, y_test, n_samples=10)

    plot_query_budget(nes_results, hsj_results)

    # Summary
    print('\n' + '='*60)
    print('TRANSFERABILITY MATRIX (rows=surrogate, cols=target)')
    print('='*60)
    header = f'{"Surrogate→":>15}' + ''.join(f'{n:>16}' for n in model_names)
    print(header)
    print('-' * (15 + 16*len(model_names)))
    for i, sname in enumerate(model_names):
        row = f'{sname:>15}'
        for j in range(len(model_names)):
            row += f'{transfer_matrix[i,j]*100:>15.1f}%'
        print(row)

    print(f'\n[+] All plots saved to {PLOT_DIR}/')
    print('[+] Done.')


if __name__ == '__main__':
    main()
