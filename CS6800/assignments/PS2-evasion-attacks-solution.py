"""
CS 6800 — Problem Set 2: Evasion Attacks from Scratch
Complete Solution

Implements FGSM, BIM, and PGD from scratch in PyTorch.
Evaluates on CIFAR-10 under L_inf and L2 norms.
Generates epsilon-accuracy curves and step-count analysis.

Requirements:
    pip install torch torchvision matplotlib numpy tqdm

Usage:
    python PS2-evasion-attacks-solution.py

The script will:
1. Load a pretrained CIFAR-10 ResNet-18 (trains one if no checkpoint found)
2. Run FGSM, BIM, PGD under L_inf for eps in [2, 4, 8, 16]/255
3. Run FGSM, BIM, PGD under L2 for eps in [0.25, 0.5, 1.0, 2.0]
4. Save plots to ./ps2_plots/
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
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
NUM_EVAL_SAMPLES = 1000   # Use first 1000 test samples for speed
BATCH_SIZE = 100
CHECKPOINT_PATH = './cifar10_resnet18.pth'
PLOT_DIR = './ps2_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model: ResNet-18 for CIFAR-10
#     (10-class head; torchvision pretrained is ImageNet-1k, so we retrain)
# ─────────────────────────────────────────────────────────────────────────────

def build_resnet18_cifar():
    """Return a ResNet-18 with 10-class output head for CIFAR-10."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def train_model(model, trainloader, epochs=20):
    """Quick training loop to get a clean-accuracy baseline."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch+1}/{epochs}  loss={running_loss/len(trainloader):.3f}')
    return model


def load_or_train_model():
    """Load checkpoint if available, otherwise train from scratch."""
    model = build_resnet18_cifar()
    if os.path.exists(CHECKPOINT_PATH):
        print(f'[+] Loading checkpoint from {CHECKPOINT_PATH}')
        state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        print('[+] No checkpoint found. Training ResNet-18 on CIFAR-10 for 20 epochs...')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True,
                                                transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                  shuffle=True, num_workers=2)
        model = train_model(model, trainloader, epochs=20)
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f'[+] Saved checkpoint to {CHECKPOINT_PATH}')
    model.to(DEVICE)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def get_test_loader():
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    # Take only the first NUM_EVAL_SAMPLES
    subset = torch.utils.data.Subset(testset, range(NUM_EVAL_SAMPLES))
    loader = torch.utils.data.DataLoader(subset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)
    return loader


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Utility: Clipping / Projection
# ─────────────────────────────────────────────────────────────────────────────

def clip_linf(x_adv, x_orig, eps):
    """Project x_adv back into the L_inf ball of radius eps around x_orig,
    and clamp to valid pixel range [0, 1]."""
    delta = x_adv - x_orig
    delta = torch.clamp(delta, -eps, eps)
    return torch.clamp(x_orig + delta, 0.0, 1.0)


def clip_l2(x_adv, x_orig, eps):
    """Project x_adv back onto the L2 ball of radius eps around x_orig,
    then clamp to [0, 1]."""
    delta = x_adv - x_orig
    # Compute L2 norm per sample: shape [B]
    norms = delta.view(delta.shape[0], -1).norm(dim=1)
    # Scale down perturbations that exceed radius
    factor = torch.clamp(eps / (norms + 1e-10), max=1.0)
    factor = factor.view(-1, 1, 1, 1)
    delta = delta * factor
    return torch.clamp(x_orig + delta, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FGSM  (Fast Gradient Sign Method)
#     Goodfellow et al. 2015
#     x' = x + eps * sign(∇_x L(f(x), y))
# ─────────────────────────────────────────────────────────────────────────────

def fgsm(model, x, y, eps, norm='linf'):
    """
    One-step FGSM attack.

    Args:
        model: PyTorch classifier (eval mode, no grad for params)
        x:     Clean input batch [B, C, H, W] in [0, 1]
        y:     True labels [B]
        eps:   Perturbation budget
        norm:  'linf' or 'l2'

    Returns:
        x_adv: Adversarial examples [B, C, H, W]
    """
    criterion = nn.CrossEntropyLoss()
    x_adv = x.clone().detach().requires_grad_(True)

    logits = model(x_adv)
    loss = criterion(logits, y)
    loss.backward()

    grad = x_adv.grad.detach()

    if norm == 'linf':
        # Sign of gradient: maximally increases loss under L_inf constraint
        x_adv = x + eps * grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:  # l2
        # Normalize gradient to unit L2 norm, then step by eps
        grad_norm = grad.view(grad.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1)
        grad_unit = grad / (grad_norm + 1e-10)
        x_adv = x + eps * grad_unit
        x_adv = clip_l2(x_adv, x, eps)

    return x_adv.detach()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  BIM  (Basic Iterative Method)
#     Kurakin et al. 2017
#     x_{t+1} = Clip_{x, eps}(x_t + alpha * sign(∇_x L(f(x_t), y)))
# ─────────────────────────────────────────────────────────────────────────────

def bim(model, x, y, eps, alpha=None, num_steps=10, norm='linf'):
    """
    Basic Iterative Method (BIM) attack.

    Args:
        model:     PyTorch classifier
        x:         Clean inputs [B, C, H, W]
        y:         True labels [B]
        eps:       Maximum perturbation budget
        alpha:     Step size per iteration (default: eps/4)
        num_steps: Number of iterations
        norm:      'linf' or 'l2'

    Returns:
        x_adv: Adversarial examples
    """
    if alpha is None:
        alpha = eps / 4.0  # Conservative step size

    criterion = nn.CrossEntropyLoss()
    x_adv = x.clone().detach()

    for _ in range(num_steps):
        x_adv = x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = criterion(logits, y)
        loss.backward()
        grad = x_adv.grad.detach()

        if norm == 'linf':
            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = clip_linf(x_adv, x, eps)
        else:  # l2
            grad_norm = grad.view(grad.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1)
            grad_unit = grad / (grad_norm + 1e-10)
            x_adv = x_adv.detach() + alpha * grad_unit
            x_adv = clip_l2(x_adv, x, eps)

    return x_adv.detach()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PGD  (Projected Gradient Descent)
#     Madry et al. 2018
#     Same as BIM but with random initialization inside the epsilon ball.
#     Multiple restarts for stronger evaluation.
# ─────────────────────────────────────────────────────────────────────────────

def pgd(model, x, y, eps, alpha=None, num_steps=40, num_restarts=10, norm='linf'):
    """
    PGD attack with multiple random restarts.

    Key difference from BIM: starts from a random point in the epsilon ball
    rather than from x, and runs multiple restarts keeping the strongest result.

    Args:
        model:        PyTorch classifier
        x:            Clean inputs [B, C, H, W]
        y:            True labels [B]
        eps:          Maximum perturbation budget
        alpha:        Step size (default: 2.5 * eps / num_steps, Madry rule of thumb)
        num_steps:    PGD iterations per restart
        num_restarts: Number of random restarts (use 1 for training, 10 for evaluation)
        norm:         'linf' or 'l2'

    Returns:
        x_best: Best (highest-loss) adversarial examples across all restarts
    """
    if alpha is None:
        alpha = 2.5 * eps / num_steps  # Madry et al. rule of thumb

    criterion = nn.CrossEntropyLoss(reduction='none')  # Per-sample loss

    x_best = x.clone().detach()
    loss_best = torch.zeros(x.shape[0]).to(DEVICE)

    for restart in range(num_restarts):
        # Random initialization within epsilon ball
        if norm == 'linf':
            x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
        else:  # l2
            noise = torch.randn_like(x)
            noise_norm = noise.view(noise.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1)
            random_eps = torch.empty(x.shape[0], 1, 1, 1).uniform_(0, eps).to(DEVICE)
            x_adv = x + random_eps * noise / (noise_norm + 1e-10)
            x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

        for step in range(num_steps):
            x_adv = x_adv.requires_grad_(True)
            logits = model(x_adv)
            # Use per-sample loss for tracking best
            loss = criterion(logits, y)
            loss.mean().backward()
            grad = x_adv.grad.detach()

            if norm == 'linf':
                x_adv = x_adv.detach() + alpha * grad.sign()
                x_adv = clip_linf(x_adv, x, eps)
            else:  # l2
                grad_norm = grad.view(grad.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1)
                grad_unit = grad / (grad_norm + 1e-10)
                x_adv = x_adv.detach() + alpha * grad_unit
                x_adv = clip_l2(x_adv, x, eps)

        # Evaluate final loss for this restart; keep best per sample
        with torch.no_grad():
            logits_final = model(x_adv)
            loss_final = criterion(logits_final, y)

        improved = loss_final > loss_best
        x_best[improved] = x_adv[improved].detach()
        loss_best[improved] = loss_final[improved].detach()

    return x_best


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Evaluation: Robust Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_robust_accuracy(model, loader, attack_fn, attack_kwargs):
    """
    Evaluate robust accuracy: fraction of samples that remain correctly
    classified after the attack.

    Args:
        model:        Classifier in eval mode
        loader:       DataLoader (test set subset)
        attack_fn:    Function(model, x, y, **kwargs) → x_adv
        attack_kwargs: Keyword arguments passed to attack_fn

    Returns:
        (clean_acc, robust_acc): Tuple of floats in [0, 1]
    """
    model.eval()
    total = 0
    clean_correct = 0
    robust_correct = 0

    for x, y in tqdm(loader, desc='  Evaluating', leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Clean accuracy
        with torch.no_grad():
            clean_pred = model(x).argmax(dim=1)
        clean_correct += (clean_pred == y).sum().item()

        # Adversarial accuracy
        x_adv = attack_fn(model, x, y, **attack_kwargs)
        with torch.no_grad():
            adv_pred = model(x_adv).argmax(dim=1)
        robust_correct += (adv_pred == y).sum().item()

        total += x.shape[0]

    return clean_correct / total, robust_correct / total


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Sweep: epsilon vs. robust accuracy
# ─────────────────────────────────────────────────────────────────────────────

def run_linf_sweep(model, loader):
    """Run all three attacks across L_inf epsilon values."""
    eps_values = [2/255, 4/255, 8/255, 16/255]

    results = {
        'FGSM':  {'clean': [], 'robust': []},
        'BIM':   {'clean': [], 'robust': []},
        'PGD':   {'clean': [], 'robust': []},
    }

    for eps in eps_values:
        print(f'\n  L_inf eps = {eps*255:.0f}/255')

        _, fgsm_acc = evaluate_robust_accuracy(
            model, loader, fgsm, {'eps': eps, 'norm': 'linf'})
        print(f'    FGSM robust acc: {fgsm_acc*100:.1f}%')

        _, bim_acc = evaluate_robust_accuracy(
            model, loader, bim,
            {'eps': eps, 'alpha': eps/4, 'num_steps': 10, 'norm': 'linf'})
        print(f'    BIM  robust acc: {bim_acc*100:.1f}%')

        _, pgd_acc = evaluate_robust_accuracy(
            model, loader, pgd,
            {'eps': eps, 'num_steps': 40, 'num_restarts': 10, 'norm': 'linf'})
        print(f'    PGD  robust acc: {pgd_acc*100:.1f}%')

        results['FGSM']['robust'].append(fgsm_acc)
        results['BIM']['robust'].append(bim_acc)
        results['PGD']['robust'].append(pgd_acc)

    return eps_values, results


def run_l2_sweep(model, loader):
    """Run all three attacks across L2 epsilon values."""
    eps_values = [0.25, 0.5, 1.0, 2.0]

    results = {
        'FGSM': {'robust': []},
        'BIM':  {'robust': []},
        'PGD':  {'robust': []},
    }

    for eps in eps_values:
        print(f'\n  L2 eps = {eps}')

        _, fgsm_acc = evaluate_robust_accuracy(
            model, loader, fgsm, {'eps': eps, 'norm': 'l2'})
        print(f'    FGSM robust acc: {fgsm_acc*100:.1f}%')

        # For L2 BIM, alpha = eps/4
        _, bim_acc = evaluate_robust_accuracy(
            model, loader, bim,
            {'eps': eps, 'alpha': eps/4, 'num_steps': 10, 'norm': 'l2'})
        print(f'    BIM  robust acc: {bim_acc*100:.1f}%')

        _, pgd_acc = evaluate_robust_accuracy(
            model, loader, pgd,
            {'eps': eps, 'num_steps': 40, 'num_restarts': 5, 'norm': 'l2'})
        print(f'    PGD  robust acc: {pgd_acc*100:.1f}%')

        results['FGSM']['robust'].append(fgsm_acc)
        results['BIM']['robust'].append(bim_acc)
        results['PGD']['robust'].append(pgd_acc)

    return eps_values, results


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Analysis: Attack success vs. PGD step count
# ─────────────────────────────────────────────────────────────────────────────

def pgd_step_analysis(model, loader, eps=8/255):
    """
    Measure robust accuracy as a function of number of PGD steps
    to show convergence behavior.
    """
    step_counts = [1, 5, 10, 20, 40, 100]
    robust_accs = []

    for steps in step_counts:
        _, acc = evaluate_robust_accuracy(
            model, loader, pgd,
            {'eps': eps, 'num_steps': steps, 'num_restarts': 3, 'norm': 'linf'})
        robust_accs.append(acc)
        print(f'    PGD-{steps:3d}  robust acc: {acc*100:.1f}%')

    return step_counts, robust_accs


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Visualization: Adversarial Example Images
# ─────────────────────────────────────────────────────────────────────────────

def visualize_examples(model, loader, eps=8/255, n=10):
    """Show original vs adversarial examples with predictions."""
    model.eval()
    x, y = next(iter(loader))
    x, y = x[:n].to(DEVICE), y[:n].to(DEVICE)

    x_adv = pgd(model, x, y, eps=eps, num_steps=40, num_restarts=1, norm='linf')

    with torch.no_grad():
        clean_preds = model(x).argmax(dim=1).cpu().numpy()
        adv_preds   = model(x_adv).argmax(dim=1).cpu().numpy()

    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        # Original
        img = x[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'True: {CIFAR10_CLASSES[y[i].item()]}\n'
                              f'Pred: {CIFAR10_CLASSES[clean_preds[i]]}',
                             fontsize=7)
        axes[0, i].axis('off')

        # Adversarial
        adv_img = x_adv[i].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(np.clip(adv_img, 0, 1))
        color = 'red' if adv_preds[i] != y[i].item() else 'green'
        axes[1, i].set_title(f'Adv pred:\n{CIFAR10_CLASSES[adv_preds[i]]}',
                              fontsize=7, color=color)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=9)
    axes[1, 0].set_ylabel(f'PGD-40 ε=8/255', fontsize=9)
    plt.suptitle('Adversarial Examples (red title = successful attack)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'adversarial_examples.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[+] Saved adversarial examples to {PLOT_DIR}/adversarial_examples.png')


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_linf_curves(eps_values, results):
    """Plot epsilon vs. robust accuracy for L_inf attacks."""
    eps_labels = [f'{int(e*255)}/255' for e in eps_values]
    colors = {'FGSM': '#e74c3c', 'BIM': '#f39c12', 'PGD': '#2980b9'}
    markers = {'FGSM': 'o', 'BIM': 's', 'PGD': '^'}

    fig, ax = plt.subplots(figsize=(7, 5))
    for attack_name in ['FGSM', 'BIM', 'PGD']:
        accs = [a * 100 for a in results[attack_name]['robust']]
        ax.plot(eps_labels, accs,
                color=colors[attack_name],
                marker=markers[attack_name],
                linewidth=2, markersize=8,
                label=attack_name)

    ax.set_xlabel('Perturbation Budget ε (L∞)', fontsize=12)
    ax.set_ylabel('Robust Accuracy (%)', fontsize=12)
    ax.set_title('L∞ Robust Accuracy vs. Perturbation Budget\n'
                 '(ResNet-18, CIFAR-10, n=1000)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'linf_accuracy_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[+] Saved L∞ curves to {PLOT_DIR}/linf_accuracy_curves.png')


def plot_l2_curves(eps_values, results):
    """Plot epsilon vs. robust accuracy for L2 attacks."""
    colors = {'FGSM': '#e74c3c', 'BIM': '#f39c12', 'PGD': '#2980b9'}
    markers = {'FGSM': 'o', 'BIM': 's', 'PGD': '^'}

    fig, ax = plt.subplots(figsize=(7, 5))
    for attack_name in ['FGSM', 'BIM', 'PGD']:
        accs = [a * 100 for a in results[attack_name]['robust']]
        ax.plot(eps_values, accs,
                color=colors[attack_name],
                marker=markers[attack_name],
                linewidth=2, markersize=8,
                label=attack_name)

    ax.set_xlabel('Perturbation Budget ε (L₂)', fontsize=12)
    ax.set_ylabel('Robust Accuracy (%)', fontsize=12)
    ax.set_title('L₂ Robust Accuracy vs. Perturbation Budget\n'
                 '(ResNet-18, CIFAR-10, n=1000)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'l2_accuracy_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[+] Saved L₂ curves to {PLOT_DIR}/l2_accuracy_curves.png')


def plot_pgd_steps(step_counts, robust_accs):
    """Plot robust accuracy vs. number of PGD steps."""
    fig, ax = plt.subplots(figsize=(7, 5))
    accs_pct = [a * 100 for a in robust_accs]
    ax.semilogx(step_counts, accs_pct,
                color='#2980b9', marker='^', linewidth=2, markersize=8)
    ax.set_xlabel('Number of PGD Steps', fontsize=12)
    ax.set_ylabel('Robust Accuracy (%)', fontsize=12)
    ax.set_title('PGD Convergence: Robust Accuracy vs. Step Count\n'
                 '(ε = 8/255 L∞, 3 restarts, ResNet-18 CIFAR-10)', fontsize=12)
    ax.set_xticks(step_counts)
    ax.set_xticklabels([str(s) for s in step_counts])
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'pgd_step_convergence.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[+] Saved PGD step convergence to {PLOT_DIR}/pgd_step_convergence.png')


# ─────────────────────────────────────────────────────────────────────────────
# 12.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('CS 6800 PS2 — Evasion Attacks from Scratch')
    print(f'Device: {DEVICE}')
    print('=' * 60)

    # Load model
    model = load_or_train_model()

    # Load test data
    loader = get_test_loader()

    # Sanity check: clean accuracy
    print('\n[1] Evaluating clean accuracy...')
    clean_total, clean_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            clean_correct += (preds == y).sum().item()
            clean_total += x.shape[0]
    clean_acc = clean_correct / clean_total
    print(f'    Clean accuracy: {clean_acc*100:.2f}%')

    # L_inf sweep
    print('\n[2] L_inf sweep: FGSM, BIM, PGD...')
    linf_eps, linf_results = run_linf_sweep(model, loader)
    plot_linf_curves(linf_eps, linf_results)

    # L2 sweep
    print('\n[3] L2 sweep: FGSM, BIM, PGD...')
    l2_eps, l2_results = run_l2_sweep(model, loader)
    plot_l2_curves(l2_eps, l2_results)

    # PGD step analysis
    print('\n[4] PGD step-count analysis (eps=8/255 L_inf)...')
    step_counts, step_accs = pgd_step_analysis(model, loader, eps=8/255)
    plot_pgd_steps(step_counts, step_accs)

    # Visualize examples
    print('\n[5] Generating adversarial example visualization...')
    visualize_examples(model, loader, eps=8/255)

    # Summary table
    print('\n' + '=' * 60)
    print('SUMMARY TABLE: L∞ Robust Accuracy (%)')
    print('=' * 60)
    header = f'{"eps":>8}' + ''.join(f'{"FGSM":>10}{"BIM":>10}{"PGD":>10}')
    print(header)
    print('-' * 38)
    for i, eps in enumerate(linf_eps):
        row = f'{eps*255:>5.0f}/255'
        row += f'{linf_results["FGSM"]["robust"][i]*100:>10.1f}'
        row += f'{linf_results["BIM"]["robust"][i]*100:>10.1f}'
        row += f'{linf_results["PGD"]["robust"][i]*100:>10.1f}'
        print(row)

    print(f'\n[+] All plots saved to {PLOT_DIR}/')
    print('[+] Done.')


if __name__ == '__main__':
    main()
