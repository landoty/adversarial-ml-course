"""
CS 6810 — Programming Assignment 1: White-Box Attacks
Solution Implementation

Implements:
  - PGD-40 (Projected Gradient Descent, L-infinity) as reference baseline
  - C&W L2 attack (binary search on c, tanh change of variables, Adam optimizer)
  - C&W L-inf attack (Lagrangian slack variable method)
  - Evaluation on CIFAR-10 with pretrained ResNet-18
  - Attack success rate, average perturbation, timing comparison
  - Convergence plots (loss vs. optimization step for C&W L2)

Usage:
  python PA1-white-box-attacks-solution.py

Requirements:
  pip install torch torchvision matplotlib numpy tqdm
"""

import os
import time
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster environments
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Global Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Attack hyperparameters
EPS_LINF = 8 / 255          # L-infinity budget (8/255 is the standard CIFAR-10 budget)
EPS_L2   = 0.5              # L-2 budget for C&W L2 evaluation
ALPHA    = 2 / 255          # PGD step size
PGD_ITER = 40               # PGD iterations
CW_LR    = 1e-2             # C&W inner optimizer learning rate
CW_ITER  = 1000             # C&W inner iterations per binary search step
CW_BSEARCH_STEPS = 9        # C&W binary search steps
CW_KAPPA = 0.0              # C&W confidence margin (0 = minimum distortion)
CW_INIT_C = 1e-3            # C&W initial trade-off constant

N_EVAL   = 200              # Number of test images to evaluate
BATCH    = 50               # Evaluation batch size (for model loading)
SEED     = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model and Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def get_cifar10_test_loader(n_samples=N_EVAL, batch_size=BATCH):
    """
    Load CIFAR-10 test set. Returns a DataLoader with at most n_samples images.
    Pixel values are in [0, 1] (no normalization — we normalize inside the model wrapper).
    """
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    # Subset to first n_samples for efficiency
    indices = list(range(n_samples))
    subset = torch.utils.data.Subset(testset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
    return loader


class NormalizedResNet18(nn.Module):
    """
    ResNet-18 with CIFAR-10 normalization baked in.
    Expects inputs in [0, 1]; normalizes internally using CIFAR-10 statistics.
    This is crucial: attacks operate in [0,1] space but the model needs normalized inputs.
    """
    CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR10_STD  = [0.2023, 0.1994, 0.2010]

    def __init__(self):
        super().__init__()
        self.normalize = transforms.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD)

        # Load a pretrained torchvision ResNet-18 adapted for CIFAR-10
        # In a real course setting, students would load a checkpoint. Here we
        # modify the first conv to handle 32x32 inputs (standard CIFAR adaptation).
        base = models.resnet18(weights=None)
        # Replace first conv: kernel 7x7 stride 2 → 3x3 stride 1 (for 32x32 inputs)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the MaxPool that halves spatial dim (would reduce 32→16→4 after two downsamples)
        base.maxpool = nn.Identity()
        # Adjust output classes to 10
        base.fc = nn.Linear(512, 10)
        self.backbone = base

    def forward(self, x):
        # x is expected in [0,1]; normalize per CIFAR-10 statistics
        x_norm = self.normalize(x)
        return self.backbone(x_norm)


def load_model(checkpoint_path=None):
    """
    Load the model. If a checkpoint is provided, load its weights.
    Otherwise, use randomly initialized weights (for demonstration only —
    students must provide a pretrained checkpoint for meaningful results).
    """
    model = NormalizedResNet18().to(DEVICE)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("[WARNING] No checkpoint found. Using random weights.")
        print("          For real evaluation, provide a pretrained ResNet-18 checkpoint.")
        print("          You can train one with: python train_cifar10.py")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PGD Attack (Baseline)
# ─────────────────────────────────────────────────────────────────────────────

class PGDAttack:
    """
    PGD (Projected Gradient Descent) L-infinity attack.

    Reference: Madry et al. (2018), "Towards Deep Learning Models Resistant
    to Adversarial Attacks."

    Algorithm:
      x_0 = x + Uniform(-epsilon, epsilon)  (random start)
      For t in 1..T:
        grad = ∇_x L(f(x_t), y)
        x_{t+1} = Clip_{x,ε}(x_t + alpha * sign(grad))
      Return x_T
    """

    def __init__(self, model, epsilon=EPS_LINF, alpha=ALPHA, n_iter=PGD_ITER,
                 random_start=True, loss_fn=None):
        self.model   = model
        self.epsilon = epsilon
        self.alpha   = alpha
        self.n_iter  = n_iter
        self.random_start = random_start
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    def perturb(self, x, y):
        """
        Run PGD on a batch of inputs.

        Parameters:
          x : Tensor [B, C, H, W] in [0, 1]
          y : Tensor [B] with true class labels

        Returns:
          x_adv : Tensor [B, C, H, W] adversarial examples in [0, 1]
        """
        x_adv = x.clone().detach()

        if self.random_start:
            # Uniform random initialization within the epsilon ball
            delta = torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv + delta, 0, 1).detach()

        for _ in range(self.n_iter):
            x_adv.requires_grad_(True)

            logits = self.model(x_adv)
            loss   = self.loss_fn(logits, y)

            # Compute gradient of loss w.r.t. x_adv
            grad = torch.autograd.grad(loss, x_adv)[0]

            # FGSM step: move in the sign of the gradient
            x_adv = x_adv.detach() + self.alpha * grad.sign()

            # Project onto L-infinity ball around the original x
            delta  = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv  = torch.clamp(x + delta, 0, 1).detach()

        return x_adv


# ─────────────────────────────────────────────────────────────────────────────
# 3.  C&W L2 Attack
# ─────────────────────────────────────────────────────────────────────────────

def tanh_space(x):
    """Map x ∈ [0,1] to w ∈ ℝ via arctanh(2x - 1). Clip for numerical stability."""
    return torch.atanh(torch.clamp(2.0 * x - 1.0, -1 + 1e-6, 1 - 1e-6))

def from_tanh_space(w):
    """Map w ∈ ℝ back to x ∈ (0,1) via (tanh(w) + 1) / 2."""
    return (torch.tanh(w) + 1.0) / 2.0

def cw_hinge_loss(logits, target, kappa=0.0):
    """
    C&W hinge loss for a targeted attack.

    g(x') = max( max_{i != t} logits[i] - logits[t], -kappa )

    This is positive when the attack has NOT succeeded and -kappa when it
    has succeeded with margin kappa.

    Parameters:
      logits : Tensor [B, K]
      target : Tensor [B] with target class indices (for targeted attack)
               or True labels (for untargeted, negate interpretation)
      kappa  : float, confidence margin (0 = stop at boundary)

    Returns:
      loss : Tensor [B], per-example hinge loss
    """
    B, K = logits.shape
    # One-hot mask for target class
    target_one_hot = F.one_hot(target, K).float()

    # logits_target: target class logit for each example
    logits_target = (logits * target_one_hot).sum(dim=1)

    # max over non-target logits: set target logit to -inf before max
    logits_other = logits - 1e9 * target_one_hot
    logits_best_other = logits_other.max(dim=1).values

    # Hinge loss: > 0 means attack failed (other class still winning)
    hinge = logits_best_other - logits_target

    return torch.clamp(hinge, min=-kappa)


class CWL2Attack:
    """
    Carlini & Wagner L2 adversarial attack.

    Reference: Carlini & Wagner (2017), "Towards Evaluating the Robustness
    of Neural Networks." IEEE S&P.

    Formulation:
      min_w  || (tanh(w)+1)/2 - x ||_2^2 + c * g( (tanh(w)+1)/2 )

    where:
      w is the unconstrained optimization variable
      g(x') = max( max_{i≠t} logits(x')[i] - logits(x')[t], -kappa )

    Binary search over c to find the minimum-distortion adversarial example.
    """

    def __init__(self, model, c_init=CW_INIT_C, kappa=CW_KAPPA,
                 n_binary_search=CW_BSEARCH_STEPS, n_iter=CW_ITER,
                 lr=CW_LR, targeted=False):
        self.model            = model
        self.c_init           = c_init
        self.kappa            = kappa
        self.n_binary_search  = n_binary_search
        self.n_iter           = n_iter
        self.lr               = lr
        self.targeted         = targeted  # True: attack to make model predict target

    def perturb(self, x, y, record_convergence=False):
        """
        Run C&W L2 on a SINGLE example (not a batch) for clarity.

        Parameters:
          x  : Tensor [C, H, W] in [0, 1], original image
          y  : int or Tensor scalar, true label (untargeted) or target label (targeted)
          record_convergence : bool, if True record loss at each step

        Returns:
          best_adv    : Tensor [C, H, W], best adversarial example found
          best_dist   : float, L2 distortion of best adversarial example
          convergence : list of (step, total_loss, distortion, attack_loss) if record_convergence
        """
        x = x.unsqueeze(0).to(DEVICE)          # [1, C, H, W]
        y_tensor = torch.tensor([y], device=DEVICE)

        # Binary search bounds on c
        c_low  = 0.0
        c_high = 1e10   # stand-in for +infinity

        best_adv  = x.clone()
        best_dist = float('inf')
        convergence_log = []

        # Initialize w from x using tanh inverse
        w_init = tanh_space(x).detach()

        for bs_step in range(self.n_binary_search):
            # Current trade-off constant
            c = (c_low + c_high) / 2.0 if c_high < 1e9 else self.c_init * (2 ** bs_step)

            # Fresh optimization variable for each binary search step
            w = w_init.clone().requires_grad_(True)

            # Adam optimizer for w
            optimizer = torch.optim.Adam([w], lr=self.lr)

            attack_succeeded_this_round = False

            for step in range(self.n_iter):
                # Map w back to image space
                x_adv = from_tanh_space(w)      # [1, C, H, W], in (0, 1)

                # Distortion term: L2 squared
                delta = x_adv - x
                distortion = (delta ** 2).sum()

                # Attack loss term: C&W hinge loss
                logits = self.model(x_adv)
                attack_loss = cw_hinge_loss(logits, y_tensor, kappa=self.kappa)

                # Total C&W objective
                total_loss = distortion + c * attack_loss.sum()

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Record convergence if requested (detach to avoid memory leak)
                if record_convergence and bs_step == self.n_binary_search - 1:
                    convergence_log.append((
                        step,
                        total_loss.item(),
                        delta.norm(p=2).item(),
                        attack_loss.item()
                    ))

                # Check if adversarial and update best
                with torch.no_grad():
                    pred = self.model(x_adv).argmax(dim=1).item()
                    dist = delta.norm(p=2).item()

                    if self.targeted:
                        is_adv = (pred == y)
                    else:
                        is_adv = (pred != y)

                    if is_adv and dist < best_dist:
                        best_adv  = x_adv.detach().clone()
                        best_dist = dist
                        attack_succeeded_this_round = True

            # Binary search update
            if attack_succeeded_this_round:
                c_high = c   # Attack succeeded; try smaller c (less emphasis on attack)
            else:
                c_low = c    # Attack failed; try larger c (more emphasis on attack)

        return best_adv.squeeze(0), best_dist, convergence_log


# ─────────────────────────────────────────────────────────────────────────────
# 4.  C&W L-inf Attack (Lagrangian with slack variables)
# ─────────────────────────────────────────────────────────────────────────────

class CWLinfAttack:
    """
    C&W L-infinity attack using the Lagrangian method with per-coordinate slack.

    The L-inf version of C&W is less commonly used than PGD for L-inf evaluation,
    but is included here for completeness per C&W (2017) Section V.

    The approach: maintain a current L-inf bound tau; minimize the C&W attack
    loss plus a penalty for violating |delta_i| <= tau; update tau to the
    minimum satisfied value.

    Simpler (and common in practice) approach shown here: use PGD-style C&W
    with the margin loss substituted for cross-entropy. This gives C&W-Linf
    in the spirit of the paper without the full slack-variable procedure.
    """

    def __init__(self, model, epsilon=EPS_LINF, kappa=CW_KAPPA,
                 n_iter=CW_ITER, lr=CW_LR, alpha=ALPHA):
        self.model   = model
        self.epsilon = epsilon
        self.kappa   = kappa
        self.n_iter  = n_iter
        self.lr      = lr
        self.alpha   = alpha

    def perturb(self, x, y):
        """
        C&W L-inf: PGD with C&W margin loss instead of cross-entropy.
        This avoids softmax saturation in the gradient, making it more
        effective against gradient-masked defenses.

        Parameters:
          x : Tensor [B, C, H, W] in [0, 1]
          y : Tensor [B] with true labels (untargeted attack)

        Returns:
          x_adv : Tensor [B, C, H, W], adversarial examples
        """
        x_adv = x.clone().detach()

        # Random start within L-inf ball
        delta = torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv + delta, 0, 1).detach()

        for _ in range(self.n_iter):
            x_adv.requires_grad_(True)
            logits = self.model(x_adv)

            # Untargeted C&W margin loss:
            # g_untargeted = max(logits[y] - max_{i≠y} logits[i], -kappa)
            # We want to INCREASE this loss (untargeted attack decreases it)
            B, K = logits.shape
            y_onehot = F.one_hot(y, K).float()
            logits_true  = (logits * y_onehot).sum(dim=1)
            logits_other = (logits - 1e9 * y_onehot).max(dim=1).values
            margin_loss   = torch.clamp(logits_true - logits_other, min=-self.kappa)

            # Maximize margin loss (untargeted) = minimize negation
            loss = -margin_loss.mean()

            grad = torch.autograd.grad(loss, x_adv)[0]

            # PGD step
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            delta  = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv  = torch.clamp(x + delta, 0, 1).detach()

        return x_adv


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Evaluation Loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_attack(model, loader, attack, attack_name, device=DEVICE):
    """
    Evaluate an attack on the given data loader.

    Returns a dict with:
      clean_acc      : float, clean accuracy on evaluated examples
      adv_acc        : float, accuracy on adversarial examples (lower = better attack)
      attack_sr      : float, attack success rate = 1 - adv_acc (on correctly classified)
      avg_linf_dist  : float, average L-inf distortion
      avg_l2_dist    : float, average L-2 distortion
      elapsed_sec    : float, total wall-clock time for the attack
    """
    model.eval()
    total = 0
    clean_correct = 0
    adv_correct   = 0
    linf_dists    = []
    l2_dists      = []
    attack_successes_on_clean_correct = 0
    clean_correct_count = 0

    t_start = time.time()
    for x_batch, y_batch in tqdm(loader, desc=f"{attack_name}", leave=False):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Clean accuracy
        with torch.no_grad():
            clean_logits = model(x_batch)
            clean_preds  = clean_logits.argmax(dim=1)
            correct_mask = (clean_preds == y_batch)

        clean_correct += correct_mask.sum().item()
        total += x_batch.shape[0]

        # Run attack
        x_adv = attack.perturb(x_batch, y_batch)

        # Adversarial accuracy
        with torch.no_grad():
            adv_logits = model(x_adv)
            adv_preds  = adv_logits.argmax(dim=1)

        adv_correct += (adv_preds == y_batch).sum().item()

        # Count attack successes only on originally correctly classified examples
        clean_correct_count += correct_mask.sum().item()
        attack_successes_on_clean_correct += (
            correct_mask & (adv_preds != y_batch)
        ).sum().item()

        # Distortion metrics
        delta = (x_adv - x_batch).view(x_batch.shape[0], -1)
        linf_dists.extend(delta.abs().max(dim=1).values.cpu().tolist())
        l2_dists.extend(delta.norm(p=2, dim=1).cpu().tolist())

    elapsed = time.time() - t_start

    clean_acc = clean_correct / total
    adv_acc   = adv_correct / total
    attack_sr = (attack_successes_on_clean_correct / clean_correct_count
                 if clean_correct_count > 0 else 0.0)

    return {
        'attack_name'   : attack_name,
        'clean_acc'     : clean_acc,
        'adv_acc'       : adv_acc,
        'attack_sr'     : attack_sr,
        'avg_linf_dist' : np.mean(linf_dists),
        'avg_l2_dist'   : np.mean(l2_dists),
        'max_linf_dist' : np.max(linf_dists),
        'elapsed_sec'   : elapsed,
        'n_examples'    : total,
    }


def evaluate_cw_l2(model, loader, cw_attack, device=DEVICE, n_examples=None):
    """
    Evaluate C&W L2 example by example (since C&W works per-example, not batched).

    Returns a dict with the same keys as evaluate_attack plus:
      convergence_data : convergence log from the last example
    """
    model.eval()
    total = 0
    clean_correct = 0
    adv_correct   = 0
    l2_dists      = []
    attack_successes = 0
    clean_correct_count = 0
    convergence_data = None

    t_start = time.time()
    example_idx = 0

    for x_batch, y_batch in tqdm(loader, desc="C&W L2", leave=False):
        for i in range(x_batch.shape[0]):
            if n_examples is not None and example_idx >= n_examples:
                break

            x_i = x_batch[i].to(device)
            y_i = y_batch[i].item()

            # Clean prediction
            with torch.no_grad():
                clean_pred = model(x_i.unsqueeze(0)).argmax().item()

            clean_correct += int(clean_pred == y_i)
            total += 1

            # Record convergence for the last example
            is_last = (n_examples is not None and example_idx == n_examples - 1)
            x_adv, dist, conv_log = cw_attack.perturb(
                x_i, y_i, record_convergence=is_last
            )
            if is_last:
                convergence_data = conv_log

            # Adversarial prediction
            with torch.no_grad():
                adv_pred = model(x_adv.unsqueeze(0)).argmax().item()

            adv_correct += int(adv_pred == y_i)
            l2_dists.append(dist if dist < float('inf') else 0.0)

            # Attack success (on clean-correct examples)
            if clean_pred == y_i:
                clean_correct_count += 1
                if adv_pred != y_i:
                    attack_successes += 1

            example_idx += 1

        if n_examples is not None and example_idx >= n_examples:
            break

    elapsed = time.time() - t_start
    clean_acc = clean_correct / total
    adv_acc   = adv_correct / total
    attack_sr = attack_successes / clean_correct_count if clean_correct_count > 0 else 0.0

    return {
        'attack_name'    : 'C&W L2',
        'clean_acc'      : clean_acc,
        'adv_acc'        : adv_acc,
        'attack_sr'      : attack_sr,
        'avg_l2_dist'    : np.mean(l2_dists),
        'avg_linf_dist'  : 0.0,  # Not the primary metric for C&W L2
        'max_linf_dist'  : 0.0,
        'elapsed_sec'    : elapsed,
        'n_examples'     : total,
        'convergence'    : convergence_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Visualization: Convergence Plot for C&W L2
# ─────────────────────────────────────────────────────────────────────────────

def plot_cw_convergence(convergence_data, save_path='cw_l2_convergence.png'):
    """
    Plot the C&W L2 convergence curve: total loss, distortion, and attack loss
    vs. optimization step number.

    Parameters:
      convergence_data : list of (step, total_loss, distortion, attack_loss)
      save_path : file to save the figure
    """
    if not convergence_data:
        print("No convergence data to plot.")
        return

    steps       = [d[0] for d in convergence_data]
    total_loss  = [d[1] for d in convergence_data]
    distortions = [d[2] for d in convergence_data]
    attack_loss = [d[3] for d in convergence_data]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps, total_loss, color='steelblue', lw=1.5)
    axes[0].set_xlabel('Optimization Step')
    axes[0].set_ylabel('Total C&W Loss')
    axes[0].set_title('C&W L2: Total Loss Convergence')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, distortions, color='darkorange', lw=1.5)
    axes[1].set_xlabel('Optimization Step')
    axes[1].set_ylabel('L2 Distortion || δ ||₂')
    axes[1].set_title('C&W L2: Distortion (Minimized)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, attack_loss, color='crimson', lw=1.5)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Attack boundary')
    axes[2].axhline(y=-CW_KAPPA, color='green', linestyle='--', alpha=0.5,
                    label=f'κ = {CW_KAPPA}')
    axes[2].set_xlabel('Optimization Step')
    axes[2].set_ylabel('C&W Hinge Loss g(x\')')
    axes[2].set_title('C&W L2: Attack Hinge Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved to {save_path}")


def plot_attack_comparison(results, save_path='attack_comparison.png'):
    """
    Bar chart comparing attack success rate, average L2 distortion, and timing.
    """
    attacks       = [r['attack_name'] for r in results]
    success_rates = [r['attack_sr'] * 100 for r in results]
    l2_dists      = [r['avg_l2_dist'] for r in results]
    elapsed_secs  = [r['elapsed_sec'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['steelblue', 'darkorange', 'crimson', 'forestgreen'][:len(attacks)]

    # Attack success rate
    axes[0].bar(attacks, success_rates, color=colors)
    axes[0].set_ylabel('Attack Success Rate (%)')
    axes[0].set_title('Attack Success Rate\n(on originally correct examples)')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)
    axes[0].grid(True, axis='y', alpha=0.3)

    # Average L2 distortion
    axes[1].bar(attacks, l2_dists, color=colors)
    axes[1].set_ylabel('Average L2 Distortion')
    axes[1].set_title('Average L2 Perturbation Magnitude')
    for i, v in enumerate(l2_dists):
        axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
    axes[1].grid(True, axis='y', alpha=0.3)

    # Timing
    axes[2].bar(attacks, elapsed_secs, color=colors)
    axes[2].set_ylabel('Wall-Clock Time (sec)')
    axes[2].set_title(f'Computation Time\n(N={results[0]["n_examples"]} examples)')
    for i, v in enumerate(elapsed_secs):
        axes[2].text(i, v + 0.5, f'{v:.1f}s', ha='center', fontsize=9)
    axes[2].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {save_path}")


def display_adversarial_examples(model, x_batch, y_batch, attacks_dict,
                                 class_names, n_show=5,
                                 save_path='adversarial_examples.png'):
    """
    Visualize original and adversarial examples side by side for each attack.

    Parameters:
      model       : the classifier
      x_batch     : Tensor [B, C, H, W] of clean images
      y_batch     : Tensor [B] of true labels
      attacks_dict: dict mapping attack name -> attack object
      class_names : list of class name strings
      n_show      : number of examples to display
      save_path   : file to save the figure
    """
    model.eval()
    n_attacks = len(attacks_dict)
    n_cols = n_attacks + 1  # +1 for original
    fig, axes = plt.subplots(n_show, n_cols, figsize=(3 * n_cols, 3 * n_show))

    x_show = x_batch[:n_show]
    y_show = y_batch[:n_show]

    for row in range(n_show):
        x_i = x_show[row]
        y_i = y_show[row].item()

        # Original image
        ax = axes[row, 0]
        img = x_i.cpu().permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        with torch.no_grad():
            pred = model(x_i.unsqueeze(0).to(DEVICE)).argmax().item()
        ax.set_title(f'Original\nTrue: {class_names[y_i]}\nPred: {class_names[pred]}',
                     fontsize=7)
        ax.axis('off')

        # Adversarial examples for each attack
        for col, (name, attack) in enumerate(attacks_dict.items(), start=1):
            x_adv_batch = attack.perturb(x_show.to(DEVICE), y_show.to(DEVICE))
            x_adv_i = x_adv_batch[row]
            ax = axes[row, col]
            adv_img = x_adv_i.cpu().permute(1, 2, 0).numpy()
            # Show difference amplified for visibility
            diff = np.abs(adv_img - img) * 10
            ax.imshow(np.clip(adv_img, 0, 1))
            with torch.no_grad():
                adv_pred = model(x_adv_i.unsqueeze(0).to(DEVICE)).argmax().item()
            linf_d = np.abs(adv_img - img).max()
            l2_d   = np.linalg.norm(adv_img - img)
            ax.set_title(f'{name}\nPred: {class_names[adv_pred]}\n'
                         f'L∞={linf_d:.3f} L2={l2_d:.3f}', fontsize=7)
            ax.axis('off')

    plt.suptitle('Adversarial Examples: Original vs. Attacked', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Adversarial example visualization saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results):
    """Pretty-print a table of attack results."""
    header = (f"{'Attack':<20} {'Clean Acc':>10} {'Adv Acc':>10} "
              f"{'Succ Rate':>10} {'Avg L2':>10} {'Avg Linf':>10} {'Time (s)':>10}")
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(f"{r['attack_name']:<20} "
              f"{r['clean_acc']*100:>9.1f}% "
              f"{r['adv_acc']*100:>9.1f}% "
              f"{r['attack_sr']*100:>9.1f}% "
              f"{r['avg_l2_dist']:>10.4f} "
              f"{r['avg_linf_dist']:>10.4f} "
              f"{r['elapsed_sec']:>10.1f}")
    print("=" * len(header) + "\n")


def main():
    """Main evaluation pipeline."""
    print(f"Device: {DEVICE}")
    print(f"Evaluating on {N_EVAL} CIFAR-10 test images.\n")

    CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    # ── Load model ──────────────────────────────────────────────────────────
    model = load_model(checkpoint_path='resnet18_cifar10.pth')

    # ── Load data ────────────────────────────────────────────────────────────
    loader = get_cifar10_test_loader(n_samples=N_EVAL, batch_size=50)

    # ── Quick clean accuracy check ───────────────────────────────────────────
    model.eval()
    correct = 0
    total_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb.to(DEVICE)).argmax(dim=1)
            correct += (preds == yb.to(DEVICE)).sum().item()
            total_count += xb.shape[0]
    print(f"Clean accuracy: {correct/total_count*100:.1f}% ({correct}/{total_count})\n")

    # ── Instantiate attacks ──────────────────────────────────────────────────
    pgd40 = PGDAttack(model, epsilon=EPS_LINF, alpha=ALPHA, n_iter=40,
                      random_start=True)

    cw_linf = CWLinfAttack(model, epsilon=EPS_LINF, kappa=CW_KAPPA,
                            n_iter=100, lr=CW_LR, alpha=ALPHA)

    cw_l2 = CWL2Attack(model, c_init=CW_INIT_C, kappa=CW_KAPPA,
                        n_binary_search=CW_BSEARCH_STEPS, n_iter=CW_ITER,
                        lr=CW_LR, targeted=False)

    # ── Evaluate batch attacks (PGD, C&W Linf) ──────────────────────────────
    results = []

    print("Evaluating PGD-40 (L-inf)...")
    res_pgd = evaluate_attack(model, loader, pgd40, "PGD-40 (Linf)")
    results.append(res_pgd)

    print("Evaluating C&W L-inf...")
    res_cw_linf = evaluate_attack(model, loader, cw_linf, "C&W Linf")
    results.append(res_cw_linf)

    # ── Evaluate C&W L2 (per-example, with convergence logging) ─────────────
    # Evaluate on a smaller subset due to the high computational cost of C&W
    N_CW = min(50, N_EVAL)  # Reduce for tractability in a course setting
    print(f"Evaluating C&W L2 on {N_CW} examples (binary search + Adam)...")
    loader_small = get_cifar10_test_loader(n_samples=N_CW, batch_size=1)
    res_cw_l2 = evaluate_cw_l2(model, loader_small, cw_l2,
                                 n_examples=N_CW)
    results.append(res_cw_l2)

    # ── Print results table ──────────────────────────────────────────────────
    print_results_table(results)

    # ── Generate convergence plot ─────────────────────────────────────────────
    if res_cw_l2.get('convergence'):
        plot_cw_convergence(res_cw_l2['convergence'], save_path='cw_l2_convergence.png')

    # ── Generate comparison bar chart ─────────────────────────────────────────
    plot_attack_comparison(results, save_path='attack_comparison.png')

    # ── Visualize adversarial examples ───────────────────────────────────────
    sample_batch, sample_labels = next(iter(loader))
    display_adversarial_examples(
        model,
        sample_batch[:5],
        sample_labels[:5],
        attacks_dict={
            'PGD-40': pgd40,
            'C&W Linf': cw_linf,
        },
        class_names=CIFAR10_CLASSES,
        n_show=5,
        save_path='adversarial_examples.png'
    )

    print("\nDone. Generated files:")
    print("  cw_l2_convergence.png  — C&W L2 convergence curves")
    print("  attack_comparison.png  — attack comparison bar chart")
    print("  adversarial_examples.png — example adversarial images")


if __name__ == '__main__':
    main()
