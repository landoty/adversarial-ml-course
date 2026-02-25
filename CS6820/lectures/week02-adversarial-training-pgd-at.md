# CS6820 — Week 02: Adversarial Training I
## PGD-AT and the Min-Max Formulation

**Prerequisites:** Week 01 (defense landscape), working knowledge of SGD and backpropagation, familiarity with PGD attacks.

**Learning Objectives:**
- Derive the min-max formulation of adversarial training from first principles
- Understand why solving the inner maximization with PGD is a reasonable approximation
- Recognize the robust overfitting phenomenon and know how to mitigate it
- Implement a complete PGD-AT training loop in PyTorch
- Understand the tradeoffs between fast and standard adversarial training

---

## 1. The Min-Max Formulation

### 1.1 Standard Empirical Risk Minimization

Recall that standard supervised learning minimizes the empirical risk:

$$\min_\theta \hat{R}(\theta) = \frac{1}{n} \sum_{i=1}^n L(f_\theta(x_i), y_i)$$

where $L$ is the loss function (typically cross-entropy), $f_\theta$ is the neural network parameterized by $\theta$, and $(x_i, y_i)_{i=1}^n$ is the training dataset. We minimize this using stochastic gradient descent on mini-batches.

The problem: standard ERM does not account for adversarial perturbations. A model that achieves low empirical risk on the training distribution can still be fooled by small perturbations of the inputs.

### 1.2 The Adversarial Risk Objective

We want the model to perform well not just on the training distribution, but on all perturbed inputs within the threat model. The **adversarial risk** is:

$$R_{\text{adv}}(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \max_{\delta \in \mathcal{S}} L(f_\theta(x + \delta), y) \right]$$

where $\mathcal{S}$ is the set of allowed perturbations. For the L-infinity threat model: $\mathcal{S} = \{\delta : \|\delta\|_\infty \leq \epsilon\}$.

The inner maximization $\max_{\delta \in \mathcal{S}} L(f_\theta(x + \delta), y)$ computes the **worst-case loss** for input $x$: the loss incurred when the adversary applies the optimal perturbation to fool the model.

**Adversarial training** minimizes the adversarial risk:

$$\min_\theta \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \max_{\delta \in \mathcal{S}} L(f_\theta(x + \delta), y) \right]$$

This is the **min-max formulation** of adversarial training. The outer minimization adjusts model parameters to minimize worst-case loss; the inner maximization finds the worst-case perturbation for the current parameters.

### 1.3 Why This is a Saddle-Point Problem

The min-max objective defines a **saddle point problem**. We are looking for parameters $\theta^*$ and perturbations $\delta^*(x)$ such that:

$$(\theta^*, \delta^*(x)) \text{ is a saddle point: } \theta^* \text{ minimizes, } \delta^*(x) \text{ maximizes}$$

Formally, a saddle point $(\theta^*, \delta^*)$ satisfies:

$$\Phi(\theta^*, \delta) \leq \Phi(\theta^*, \delta^*) \leq \Phi(\theta, \delta^*)$$

for all $\theta$ and $\delta$, where $\Phi(\theta, \delta) = L(f_\theta(x + \delta), y)$.

**Why it's hard:** Saddle point problems are generally harder than pure minimization or maximization problems. In our case:

1. The inner problem (maximization over $\delta$) is non-convex in $\delta$ because $L(f_\theta(\cdot), y)$ is generally a non-convex function of the input.
2. The outer problem (minimization over $\theta$) is non-convex in $\theta$ because neural networks are non-convex.
3. The two problems are coupled: the optimal $\delta^*$ depends on $\theta$, and the optimal $\theta^*$ depends on the distribution over $\delta$.

This means:
- We cannot guarantee convergence to a global saddle point.
- The order of optimization matters: alternating between gradient descent on $\theta$ and gradient ascent on $\delta$ may not converge.
- Approximations to the inner maximization affect the quality of the outer minimization.

Despite these difficulties, empirical adversarial training works well in practice, and there are some theoretical arguments for why certain approximations are valid (discussed below).

### 1.4 Danskin's Theorem and Gradient Computation

A key theoretical result underlying adversarial training is **Danskin's Theorem**:

**Theorem (Danskin):** Let $\Phi(\theta, \delta) = L(f_\theta(x + \delta), y)$ and $\hat{\delta}(\theta) = \arg\max_{\delta \in \mathcal{S}} \Phi(\theta, \delta)$. Then:

$$\nabla_\theta \max_{\delta \in \mathcal{S}} \Phi(\theta, \delta) = \nabla_\theta \Phi(\theta, \hat{\delta}(\theta))$$

That is, the gradient of the max with respect to $\theta$ equals the gradient of the objective evaluated at the maximizing $\delta$.

**Implication for adversarial training:** If we can find the exact maximizer $\hat{\delta}(\theta)$ (the true adversarial example for the current $\theta$), then we can compute the correct gradient for the outer minimization by differentiating the loss at $\hat{\delta}(\theta)$ with respect to $\theta$.

**The practical approximation:** We cannot find the exact $\hat{\delta}(\theta)$ (this requires solving a non-convex maximization problem exactly). Instead, we approximate it with $K$ steps of PGD. Danskin's theorem justifies that using a better approximation of $\hat{\delta}$ gives a better gradient estimate for the outer minimization.

---

## 2. PGD-AT: Approximating the Inner Maximization

### 2.1 Projected Gradient Descent for the Inner Problem

**PGD-AT** (Madry et al. 2018) approximates the inner maximization using Projected Gradient Descent (PGD):

$$\delta^{(t+1)} = \Pi_{\mathcal{S}}\left(\delta^{(t)} + \eta \cdot \text{sign}\left(\nabla_\delta L(f_\theta(x + \delta^{(t)}), y)\right)\right)$$

where:
- $\Pi_\mathcal{S}$ is the projection onto the constraint set $\mathcal{S} = \{\delta : \|\delta\|_\infty \leq \epsilon\}$
- $\eta$ is the inner step size (typically $\eta = 2/255$ for $\epsilon = 8/255$)
- $\text{sign}(\cdot)$ is the element-wise sign function (this is FGSM-style PGD for L-infinity)

For the L-infinity constraint, the projection $\Pi_\mathcal{S}(\delta) = \text{clip}(\delta, -\epsilon, \epsilon)$ clips each element.

**Initialization:** Madry et al. recommend random initialization: start with $\delta^{(0)} \sim \text{Uniform}(-\epsilon, \epsilon)^d$ (each element independently drawn from $[-\epsilon, \epsilon]$). This avoids the adversarial examples being concentrated near the same local optima across training steps.

Additionally, the perturbed input $x + \delta$ must remain in the valid input range $[0, 1]^d$, so:

$$\delta^{(t+1)} = \text{clip}(x + \Pi_\mathcal{S}(\delta^{(t)} + \eta \cdot \text{sign}(\nabla_\delta L)), 0, 1) - x$$

### 2.2 The Number of PGD Steps During Training

**Why not use more steps?** More PGD steps give a better approximation of the inner maximization, but at the cost of $K$ times more forward-backward passes per training update.

| PGD steps $K$ | Computational cost | Quality of inner max | Practical use |
|---------------|-------------------|---------------------|---------------|
| 1 (FGSM) | 1x | Very poor | Fast AT only |
| 7 | 7x | Reasonable | Standard for CIFAR-10 |
| 20 | 20x | Good | Evaluation standard |
| 100+ | 100x+ | Near-exact | Evaluation only |

**The 7-step convention:** Madry et al. used 7 PGD steps for training on CIFAR-10 and 40 steps for MNIST. The 7-step choice is a practical compromise: it provides a reasonable approximation of the inner maximum without making training 20x slower than standard training. The training-time approximation (7 steps) is weaker than the evaluation-time attacker (typically 20-50 steps), which means the model must generalize across adversarial examples found by different numbers of steps.

**Step size for training:** The step size $\eta = 2/255$ is standard for $\epsilon = 8/255$ on CIFAR-10. This gives approximately $\epsilon / \eta = 4$ steps to traverse from one side of the constraint ball to the other. The random initialization ensures that starting from different points in the ball explores different local optima.

### 2.3 The Epsilon Selection: 8/255 as the Standard

The threat model $\epsilon = 8/255$ on CIFAR-10 became the de-facto standard for adversarial robustness benchmarking. Why?

**Human perception threshold:** 8 out of 255 pixel values corresponds to approximately 3.1% of the pixel range. For natural images, perturbations of this magnitude are generally not noticeable to human observers (though this varies by image and perturbation structure). This provides a reasonable correspondence between the technical threat model and the actual security concern (fooling classifiers without visually altering images).

**Balancing robustness and accuracy:** At very small $\epsilon$, adversarial training provides little benefit (the threat is small). At very large $\epsilon$, robust accuracy degrades severely (the threat model allows almost arbitrary input modification). $\epsilon = 8/255$ is in the "interesting" regime where there is meaningful tension between clean and robust accuracy.

**MNIST:** The standard for MNIST is $\epsilon = 0.3$ (L-infinity), which is more lenient relative to the pixel range but reflects the simpler (grayscale, lower-resolution) nature of the dataset.

---

## 3. Training Dynamics

### 3.1 How Clean and Robust Accuracy Evolve

During PGD-AT training, the training curves have a characteristic shape:

**Early training (epochs 1-50):**
- Clean accuracy rises rapidly (similar to standard training).
- Robust accuracy starts near 0% and rises slowly.
- The model is learning basic features that are somewhat robust.

**Middle training (epochs 50-150):**
- Clean accuracy continues rising but at a slower rate.
- Robust accuracy rises more rapidly as the model learns robust features.
- The gap between clean and robust accuracy is large but shrinking.

**Late training (epochs 150-200):**
- Clean accuracy may plateau or slightly decline.
- Robust accuracy on the *training set* continues improving.
- Robust accuracy on the *test set* may begin to decline — this is **robust overfitting**.

**Typical final numbers for ResNet-18 on CIFAR-10 (ε = 8/255):**
- Standard training: 94% clean accuracy, ~0% robust accuracy under PGD-20
- PGD-AT (7-step): ~84% clean accuracy, ~50-56% robust accuracy under PGD-20
- PGD-AT (7-step) evaluated under AutoAttack: ~48-52% robust accuracy

### 3.2 The Robust Overfitting Phenomenon

Rice, Wong, and Kolter (2020) identified that adversarially trained models suffer from **robust overfitting**: the test robust accuracy peaks early in training and then declines, even as the training robust accuracy continues improving.

**Empirical observation:** Training a ResNet-18 with PGD-AT on CIFAR-10 for 200 epochs:

| Epoch | Train clean acc | Train robust acc | Test clean acc | Test robust acc |
|-------|----------------|-----------------|----------------|----------------|
| 50 | 95% | 70% | 85% | 54% |
| 100 | 98% | 78% | 85% | 55% |
| 150 | 99% | 85% | 83% | 52% |
| 200 | 99% | 90% | 82% | 48% |

Notice that test robust accuracy peaks around epoch 100 and then *decreases* while train robust accuracy keeps increasing. This is robust overfitting: the model is memorizing the specific adversarial examples generated during training rather than learning genuinely robust features.

**Why does robust overfitting happen?**

Several hypotheses have been proposed:

1. **Memorization of adversarial examples:** During training, PGD generates adversarial examples for each batch. Over many epochs, the model sees the same data points many times and learns to correctly classify the adversarial examples generated from them. But the adversarial examples at epoch 200 are crafted specifically by the epoch-200 model, which is different from what the epoch-100 model would generate. The test-time attacker uses the final model to generate attacks, so the model's "memory" of earlier training adversarial examples doesn't help.

2. **Capacity issues:** The model doesn't have enough capacity to generalize robustly. Larger models tend to show less robust overfitting.

3. **Distribution shift:** The adversarial examples generated during training don't cover the full distribution of adversarial examples that a test-time attacker can find.

**Mitigation: Early stopping checkpointing**

The standard mitigation is to save a checkpoint at the epoch of peak test robust accuracy (determined by periodic PGD-20 evaluation on a validation set). This requires:
1. Setting aside a validation split (or using the test set for checkpointing, which is technically leaking test information but commonly done in the community).
2. Evaluating robust accuracy on the validation set every 10-20 epochs using PGD-20.
3. Saving the checkpoint with the highest validation robust accuracy.

Rice et al. (2020) showed that early stopping alone recovers about 2-4% robust accuracy compared to using the final checkpoint.

### 3.3 The Learning Rate Schedule

Standard learning rate schedules for PGD-AT on CIFAR-10:

**Piecewise constant schedule (Madry et al.):**
- Initial LR: 0.1
- Multiply by 0.1 at epoch 100 and 150
- Total: 200 epochs

**Cyclic learning rate (Smith 2017, used by Fast-AT):**
- Linearly ramp from 0 to 0.2 over 50% of training
- Linearly ramp back to 0 over the remaining 50%
- Works well with larger batch sizes

**Cosine annealing:**
- $\text{LR}(t) = \text{LR}_{\min} + \frac{1}{2}(\text{LR}_{\max} - \text{LR}_{\min})(1 + \cos(\pi t / T))$
- Smooth decay from initial LR to near-zero
- Used in many recent works including TRADES and MART

---

## 4. Practical Considerations for Training

### 4.1 Batch Size and Data Augmentation

**Batch size:** Standard batch size is 128. Larger batch sizes are sometimes used for stability. Note that in adversarial training, the effective computation per update is $K \times$ the computation in standard training (where $K$ is the number of PGD steps), so memory is often the limiting factor.

**Data augmentation:** Standard augmentation (random horizontal flip, random crop with 4-pixel padding) is always applied before adversarial perturbation. The order is:
1. Apply random augmentation to clean image: $x' = \text{aug}(x)$
2. Generate adversarial perturbation from $x'$: $\delta^* = \text{PGD}(f_\theta, x', y)$
3. Train on adversarial example: update $\theta$ using $L(f_\theta(x' + \delta^*), y)$

**Note:** The perturbation is generated from the augmented image, not the original. This means the adversarial constraint $\|\delta\|_\infty \leq \epsilon$ applies in augmented-image space.

### 4.2 Weight Decay and Regularization

Weight decay (L2 regularization on parameters) is important for adversarial training: $\lambda = 5 \times 10^{-4}$ is standard for CIFAR-10. Dropout is generally not used in adversarially trained models (it can interact poorly with PGD-AT).

### 4.3 Synthetic Data Augmentation (Rebuffi et al., 2021)

Rebuffi et al. (2021) showed that adversarial training can be significantly improved by augmenting the training data with synthetic images generated by diffusion models or other generative models. The intuition:

- Adversarial training is "data-hungry": it needs to learn robust features from adversarial examples.
- More diverse training data provides more robust features.
- Synthetic data generated from a diffusion model trained on CIFAR-10 (without the true labels) can be labeled by the original classifier and added to the training set.

Results: Using 500K synthetic images (50x the original CIFAR-10 size) improves robust accuracy from ~56% to ~63% at $\epsilon = 8/255$, closing a significant portion of the accuracy-robustness gap. This is now a common technique in state-of-the-art adversarial training methods.

---

## 5. PyTorch Implementation: Complete PGD-AT Training Loop

```python
"""
Complete PGD-AT Training Loop for CIFAR-10
==========================================
Implements Madry et al. (2018) adversarial training with:
- PGD inner maximization (K=7 steps, random initialization)
- Standard ResNet-18 architecture
- Piecewise constant learning rate schedule
- Robust overfitting mitigation via early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import copy

# ─── Device Setup ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Hyperparameters ──────────────────────────────────────────────────────────
EPSILON = 8 / 255       # L-inf threat model bound
ALPHA = 2 / 255         # PGD step size (η)
PGD_STEPS_TRAIN = 7     # Inner maximization steps during training
PGD_STEPS_EVAL = 20     # PGD steps for evaluation
BATCH_SIZE = 128
EPOCHS = 200
LR = 0.1
WEIGHT_DECAY = 5e-4
LR_MILESTONES = [100, 150]  # Epochs at which to multiply LR by 0.1


# ─── PGD Attack ────────────────────────────────────────────────────────────────
def pgd_attack(model, x, y, epsilon, alpha, num_steps, random_start=True):
    """
    Projected Gradient Descent attack (Madry et al. 2018).

    The attack solves approximately:
        max_{||δ||∞ ≤ ε} L(f_θ(x + δ), y)
    using K steps of sign gradient ascent with projections.

    Args:
        model:       The neural network f_θ
        x:           Clean input batch (B, C, H, W), values in [0, 1]
        y:           True labels (B,)
        epsilon:     L-inf constraint radius ε
        alpha:       Step size η (per-step perturbation magnitude)
        num_steps:   Number of PGD steps K
        random_start: Whether to initialize δ randomly in [-ε, ε]^d

    Returns:
        x_adv:       Adversarial examples (B, C, H, W)
    """
    model.eval()  # Freeze BN statistics during attack generation

    # Initialize perturbation
    if random_start:
        # Uniform random initialization in the L-inf ball
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(x)

    # Ensure the perturbed input is valid (in [0,1])
    delta = torch.clamp(x + delta, 0, 1) - x
    delta.requires_grad_(True)

    for step in range(num_steps):
        # Forward pass through the model
        output = model(x + delta)
        loss = nn.CrossEntropyLoss()(output, y)

        # Compute gradient of loss w.r.t. delta
        loss.backward()

        with torch.no_grad():
            # Gradient ascent step (maximize loss)
            # sign(∇_δ L) gives the steepest ascent direction in L-inf geometry
            grad_sign = delta.grad.sign()
            delta.data = delta.data + alpha * grad_sign

            # Project back onto L-inf ball: clip each element to [-ε, ε]
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)

            # Ensure x + δ ∈ [0, 1] (valid input range)
            delta.data = torch.clamp(x + delta.data, 0, 1) - x

        # Zero gradients for next step
        delta.grad.zero_()

    model.train()  # Restore training mode (affects BN, dropout)
    return (x + delta).detach()


# ─── Evaluation Under PGD Attack ───────────────────────────────────────────────
def evaluate_robustness(model, loader, epsilon, alpha, num_steps, device):
    """
    Evaluate clean accuracy and robust accuracy (under PGD attack) on a dataset.

    Returns:
        clean_acc:  Clean accuracy (no perturbation)
        robust_acc: Robust accuracy (against PGD-num_steps attack)
    """
    model.eval()
    total = 0
    clean_correct = 0
    robust_correct = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)
        total += batch_size

        # Clean accuracy
        with torch.no_grad():
            out_clean = model(x)
            clean_correct += (out_clean.argmax(1) == y).sum().item()

        # Robust accuracy: generate adversarial examples and classify
        x_adv = pgd_attack(model, x, y, epsilon, alpha, num_steps, random_start=True)
        with torch.no_grad():
            out_adv = model(x_adv)
            robust_correct += (out_adv.argmax(1) == y).sum().item()

    model.train()
    return clean_correct / total, robust_correct / total


# ─── PGD-AT Training Loop ──────────────────────────────────────────────────────
def train_pgdat(model, train_loader, test_loader, epochs, lr, weight_decay,
                lr_milestones, epsilon, alpha, pgd_steps_train, pgd_steps_eval,
                device):
    """
    PGD-AT training: the outer minimization with PGD inner maximization.

    At each step:
    1. Generate adversarial examples using PGD with pgd_steps_train steps
    2. Compute the loss on the adversarial examples
    3. Backpropagate and update model parameters

    Args:
        model:            Neural network to train
        train_loader:     Training data loader
        test_loader:      Test data loader (for evaluation)
        epochs:           Number of training epochs
        lr:               Initial learning rate
        weight_decay:     L2 regularization strength (λ)
        lr_milestones:    Epochs at which to decay LR by 10x
        epsilon:          L-inf constraint for adversarial examples
        alpha:            PGD step size during training
        pgd_steps_train:  Number of PGD steps during training (inner max)
        pgd_steps_eval:   Number of PGD steps for evaluation (stronger attack)
        device:           torch device

    Returns:
        best_model:       Model checkpoint with highest test robust accuracy
        history:          Training history dict
    """
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=weight_decay)

    # Piecewise constant LR schedule: multiply by 0.1 at each milestone
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=0.1
    )

    criterion = nn.CrossEntropyLoss()

    # For early stopping: track best checkpoint by test robust accuracy
    best_test_robust = 0.0
    best_model_state = None

    history = {
        'train_clean_loss': [],
        'test_clean_acc': [],
        'test_robust_acc': []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # ── Inner Maximization (find adversarial examples) ──────────────
            # We pause gradient tracking for model parameters during attack generation.
            # The attack requires gradients w.r.t. input x, not model parameters.
            x_adv = pgd_attack(
                model, x, y,
                epsilon=epsilon,
                alpha=alpha,
                num_steps=pgd_steps_train,
                random_start=True
            )

            # ── Outer Minimization (update model on adversarial examples) ───
            model.train()
            optimizer.zero_grad()

            output = model(x_adv)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (output.argmax(1) == y).sum().item()
            total += x.size(0)

        scheduler.step()

        # ── Periodic Evaluation ─────────────────────────────────────────────
        if epoch % 10 == 0 or epoch == epochs:
            train_clean_loss = train_loss / total
            train_clean_acc = train_correct / total

            # Evaluate on test set with PGD-pgd_steps_eval
            test_clean_acc, test_robust_acc = evaluate_robustness(
                model, test_loader,
                epsilon=epsilon, alpha=alpha,
                num_steps=pgd_steps_eval, device=device
            )

            history['train_clean_loss'].append(train_clean_loss)
            history['test_clean_acc'].append(test_clean_acc)
            history['test_robust_acc'].append(test_robust_acc)

            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train loss: {train_clean_loss:.4f} | "
                  f"Train clean acc: {train_clean_acc:.3f} | "
                  f"Test clean acc: {test_clean_acc:.3f} | "
                  f"Test robust acc (PGD-{pgd_steps_eval}): {test_robust_acc:.3f}")

            # ── Early Stopping Checkpoint ──────────────────────────────────
            # Save model with the best test robust accuracy
            # This is the key mitigation for robust overfitting
            if test_robust_acc > best_test_robust:
                best_test_robust = test_robust_acc
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  [Checkpoint] New best test robust acc: {best_test_robust:.3f}")

    # Load best checkpoint
    model.load_state_dict(best_model_state)
    print(f"\nTraining complete. Best test robust accuracy: {best_test_robust:.3f}")
    return model, history


# ─── Data Loading ──────────────────────────────────────────────────────────────
def get_cifar10_loaders(batch_size):
    """
    CIFAR-10 data loaders with standard augmentation.
    Augmentation: random horizontal flip + random crop with 4-pixel padding.
    Test set: no augmentation (deterministic evaluation).
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        # Note: we do NOT normalize here, as the PGD attack is defined
        # in [0,1] pixel space. Normalization can be absorbed into the model.
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    return train_loader, test_loader
```

**Typical output during training:**
```
Epoch  10/200 | Train loss: 1.4521 | Train clean acc: 0.621 | Test clean acc: 0.710 | Test robust acc (PGD-20): 0.312
Epoch  20/200 | Train loss: 1.2314 | Train clean acc: 0.671 | Test clean acc: 0.741 | Test robust acc (PGD-20): 0.392
...
Epoch 100/200 | Train loss: 0.6821 | Train clean acc: 0.812 | Test clean acc: 0.838 | Test robust acc (PGD-20): 0.543
  [Checkpoint] New best test robust acc: 0.543
Epoch 110/200 | Train loss: 0.5891 | Train clean acc: 0.836 | Test clean acc: 0.837 | Test robust acc (PGD-20): 0.548
  [Checkpoint] New best test robust acc: 0.548
Epoch 120/200 | Train loss: 0.5712 | Train clean acc: 0.843 | Test clean acc: 0.836 | Test robust acc (PGD-20): 0.543
Epoch 130/200 | Train loss: 0.5643 | Train clean acc: 0.848 | Test clean acc: 0.832 | Test robust acc (PGD-20): 0.539
...
Epoch 200/200 | Train loss: 0.4821 | Train clean acc: 0.892 | Test clean acc: 0.826 | Test robust acc (PGD-20): 0.518
Training complete. Best test robust accuracy: 0.548
```

This illustrates robust overfitting: the best checkpoint (epoch 110, ~54.8% robust accuracy) is significantly better than the final model (epoch 200, ~51.8% robust accuracy).

---

## 6. Why PGD-AT Works: Geometric Intuition

### 6.1 The Decision Boundary Perspective

A neural network classifier partitions the input space $\mathbb{R}^d$ into decision regions $\mathcal{R}_1, \ldots, \mathcal{R}_K$ (one per class). An adversarial example for input $x$ with label $y$ is any point $x + \delta \in \mathcal{R}_{y'} \cup \{y' \neq y\}$ — a point near $x$ that lies in a different decision region.

Standard training does not constrain where decision boundaries are placed relative to training points. If a training point $x$ is close to a decision boundary, adversarial examples with very small $\|\delta\|$ can cross it.

PGD-AT modifies the training objective so that the model is penalized whenever the decision boundary is within $\epsilon$ of a training point. By maximizing the loss over the $\epsilon$-ball and then minimizing the maximum loss, we are pushing the decision boundary *away* from training points by at least $\epsilon$ in all directions.

**Formally:** After PGD-AT converges, for any training point $(x, y)$ and any $\delta$ with $\|\delta\|_\infty \leq \epsilon$:

$$f_\theta(x + \delta) = y \quad \text{(approximately)}$$

This means the L-infinity ball of radius $\epsilon$ around each training point is classified correctly. The decision boundary is at least $\epsilon$ away from the training points.

### 6.2 Why More PGD Steps Give Better Robustness

The inner maximization accuracy (finding the true worst-case example) is bounded by the quality of the PGD approximation. With $K$ PGD steps, we may not find the global maximum of $L(f_\theta(x + \delta), y)$ over $\mathcal{S}$.

If the inner maximization is weak (few steps, poor initialization), we are effectively training on adversarial examples that are not truly worst-case. The model may learn to defend against those specific adversarial examples (the local maxima found by PGD) without being robust against all adversarial examples in the ball.

More PGD steps → better inner approximation → the model must be robust against more adversarial examples → higher true robust accuracy.

However, there are diminishing returns: beyond 20-40 PGD steps, the improvement in finding the adversarial example is small (PGD has already converged locally), so adding more steps provides little benefit.

### 6.3 The Capacity Argument

Madry et al. (2018) observed that **larger models are more robust**: a wider ResNet achieves higher robust accuracy than a narrower one, holding training procedure fixed. This suggests that genuine robustness requires more model capacity than standard accuracy.

**Intuition:** A robust classifier must correctly classify not just the training points but all points within the $\epsilon$-balls around them. This is a much larger set of constraints than standard classification imposes. A model with more parameters can represent more complex decision boundaries that navigate around the extended (ball-shaped) training points.

**Empirical evidence:**
| Architecture | Clean acc | Robust acc (PGD-20) |
|---|---|---|
| ResNet-18 | 84% | 53% |
| WideResNet-34-10 | 87% | 58% |
| WideResNet-70-16 | 88% | 63% |

The WideResNet-70-16 has ~77M parameters vs. ~11M for ResNet-18. The robust accuracy improvement (~10%) suggests that capacity is a meaningful bottleneck for adversarial training.

---

## 7. Fast Adversarial Training Variants

### 7.1 "Free" Adversarial Training (Shafahi et al., 2019)

**Key idea:** During standard PGD-AT, both the model update and the adversarial example generation require gradient computations. Free-AT uses the *same* gradient for both the model update and the perturbation update.

**Algorithm (free-AT, minibatch replay):**

```python
# Free-AT: each minibatch is repeated m times
for x, y in loader:
    delta = torch.zeros_like(x)  # Initialize perturbation

    for replay in range(m):
        x_adv = x + delta

        # Single forward-backward pass
        output = model(x_adv)
        loss = criterion(output, y)
        loss.backward()

        # Use SAME gradient for both:
        # 1. Update model parameters (descend on θ)
        optimizer.step()
        optimizer.zero_grad()

        # 2. Update perturbation (ascend on δ)
        delta = delta + alpha * model_grad_input.sign()  # gradient w.r.t. input
        delta = torch.clamp(delta, -epsilon, epsilon)
```

By replaying each minibatch $m$ times, free-AT achieves $m$ steps of adversarial training at the cost of $m$ forward-backward passes per batch (identical to standard PGD-AT with $m$ steps). But unlike PGD-AT, each pass updates both $\theta$ and $\delta$ simultaneously.

**Claimed benefit:** Same computational cost as standard training with batch size $B/m$ — you just do $m$ gradient steps per batch instead of 1.

**When free-AT doesn't work well:** Shafahi et al. noted that free-AT can be less stable than PGD-AT and may perform slightly worse for the same total computation budget. The reason is that the perturbation $\delta$ accumulates across multiple model updates within a batch, but each model update changes $f_\theta$, so the perturbation is no longer optimal for the updated model.

### 7.2 Fast Adversarial Training (Wong et al., 2020)

Wong et al. (2020) showed that **single-step FGSM training with random initialization (FGSM-RS)** can match the robustness of PGD-AT with 7 steps, at a fraction of the computational cost.

**The FGSM-RS algorithm:**

```python
for x, y in loader:
    # Random initialization in [-epsilon, epsilon]^d (critically important)
    delta = torch.FloatTensor(x.shape).uniform_(-epsilon, epsilon).to(device)
    delta.requires_grad_(True)

    # Single FGSM step (no loop, unlike PGD-AT)
    output = model(x + delta)
    loss = criterion(output, y)
    loss.backward()

    # Update perturbation (single step)
    delta_adv = (delta + alpha * delta.grad.sign()).clamp(-epsilon, epsilon)
    delta_adv = torch.clamp(x + delta_adv, 0, 1) - x

    # Update model
    optimizer.zero_grad()
    output_adv = model(x + delta_adv)
    loss_adv = criterion(output_adv, y)
    loss_adv.backward()
    optimizer.step()
```

**Why random initialization is critical:** Without random initialization (standard FGSM training), the model suffers from "catastrophic overfitting": robust accuracy suddenly collapses to near 0% around epoch 20-30. With random initialization, this catastrophic overfitting does not occur.

**Understanding catastrophic overfitting:** Andriushchenko and Flammarion (2020) analyzed this phenomenon. Without random initialization, FGSM tends to find adversarial examples that are concentrated along a low-dimensional subspace (often just the sign of the gradient). Over many training steps, the model learns to only be robust in this subspace, and the PGD attack (which searches the full ball) easily finds adversarial examples outside this subspace.

**When fast-AT fails:** On harder datasets or with more aggressive threat models, FGSM-RS can be less stable and may underperform PGD-AT. For high-stakes applications, PGD-AT with more steps is preferred.

---

## 8. Worked Numerical Example: One PGD-AT Update

Let's trace through a single PGD-AT update step in detail.

**Setup:**
- 2-class linear classifier: $f_\theta(x) = \text{softmax}(Wx + b)$ with $W \in \mathbb{R}^{2 \times 2}$, $b \in \mathbb{R}^2$
- Single training example: $x_0 = [0.5, 0.5]^\top$, $y = 0$ (class 0)
- $\epsilon = 0.1$ (L-infinity), $\alpha = 0.05$, $K = 2$ PGD steps

**Initial parameters:** $W = \begin{bmatrix} 1 & 1 \\ -1 & -1 \end{bmatrix}$, $b = [0, 0]^\top$

**Step 1: Compute logits at clean input.**

$$z = Wx_0 + b = \begin{bmatrix} 1 & 1 \\ -1 & -1 \end{bmatrix} \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix} = \begin{bmatrix} 1.0 \\ -1.0 \end{bmatrix}$$

$p = \text{softmax}(z) = [\sigma(2), 1 - \sigma(2)] \approx [0.881, 0.119]$

Clean loss: $L = -\log(0.881) \approx 0.127$. The model already classifies $x_0$ correctly.

**Step 2: Random initialization for PGD.**

$\delta^{(0)} = [0.08, -0.07]^\top$ (drawn uniformly from $[-0.1, 0.1]^2$)

**Step 3: PGD step 1.**

Compute $\nabla_\delta L(f_\theta(x_0 + \delta^{(0)}), y=0)$:

$x_0 + \delta^{(0)} = [0.58, 0.43]^\top$

$z' = W(x_0 + \delta^{(0)}) = \begin{bmatrix} 1.01 \\ -1.01 \end{bmatrix}$

$p' = \text{softmax}(z') \approx [0.882, 0.118]$

$\nabla_z L = p' - e_0 = [-0.118, 0.118]^\top$ (gradient of CE loss w.r.t. logits)

$\nabla_x L = W^\top \nabla_z L = \begin{bmatrix} 1 & -1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} -0.118 \\ 0.118 \end{bmatrix} = \begin{bmatrix} -0.236 \\ -0.236 \end{bmatrix}$

Wait, but we want to maximize the loss (ascend), so the update is:

$\delta^{(1)} = \Pi_\mathcal{S}(\delta^{(0)} + \alpha \cdot \text{sign}(\nabla_\delta L))$

$\text{sign}(\nabla_\delta L) = \text{sign}([-0.236, -0.236]) = [-1, -1]^\top$

$\delta^{(1)} = \text{clip}([0.08, -0.07] + 0.05 \cdot [-1, -1], -0.1, 0.1)$
$= \text{clip}([0.03, -0.12], -0.1, 0.1) = [0.03, -0.10]^\top$

**Step 4: PGD step 2.**

$x_0 + \delta^{(1)} = [0.53, 0.40]^\top$

$z'' = W(x_0 + \delta^{(1)}) = [0.93, -0.93]^\top$

$p'' = [0.866, 0.134]$

$\nabla_x L$ again points in $[-1,-1]$ direction.

$\delta^{(2)} = \text{clip}([0.03, -0.10] + 0.05 \cdot [-1, -1], -0.1, 0.1)$
$= \text{clip}([-0.02, -0.15], -0.1, 0.1) = [-0.02, -0.10]^\top$

The adversarial example: $x_{\text{adv}} = x_0 + \delta^{(2)} = [0.48, 0.40]^\top$

**Step 5: Outer minimization step.**

Compute loss on $x_{\text{adv}}$ and update $W, b$ via SGD:

$z_{\text{adv}} = W x_{\text{adv}} = [0.88, -0.88]^\top$

$p_{\text{adv}} = [0.855, 0.145]$

$L_{\text{adv}} = -\log(0.855) \approx 0.157$

$\nabla_W L_{\text{adv}} = \nabla_z L_{\text{adv}} \cdot x_{\text{adv}}^\top = [-0.145, 0.145]^\top \cdot [0.48, 0.40]$

$= \begin{bmatrix} -0.069 & -0.058 \\ 0.069 & 0.058 \end{bmatrix}$

Update: $W \leftarrow W - \eta \nabla_W L_{\text{adv}}$ (with outer learning rate $\eta = 0.1$)

This pushes the model to classify $x_{\text{adv}} = [0.48, 0.40]$ (and similar adversarial examples) correctly.

---

## 9. Key Takeaways

1. **The min-max formulation is the right framework.** Adversarial training minimizes the expected worst-case loss, which is the adversarial risk. This is the correct objective for robust classification.

2. **PGD is the standard approximation for the inner maximization.** 7 PGD steps is the standard training configuration for CIFAR-10; 20 steps is the standard for evaluation.

3. **Robust overfitting is a real phenomenon.** Test robust accuracy peaks early in training and then declines. Early stopping (saving the best checkpoint) is essential.

4. **Clean accuracy and robust accuracy trade off.** PGD-AT costs ~8-10% clean accuracy relative to standard training on CIFAR-10. This tradeoff is the subject of Week 03.

5. **Fast-AT (FGSM with random initialization) works but is less reliable.** It's a practical speedup (5-7x) that approximately matches PGD-AT, but is prone to catastrophic overfitting without careful implementation.

6. **Capacity matters.** Larger models achieve higher robust accuracy under PGD-AT. WideResNet-70-16 is currently the architecture of choice for CIFAR-10 robustness.

7. **Synthetic data augmentation pushes the frontier.** Using diffusion-model-generated synthetic images significantly improves robust accuracy, currently reaching ~63% at $\epsilon = 8/255$ on CIFAR-10.

---

## Discussion Questions

1. Danskin's theorem says the gradient of the max equals the gradient at the maximizer. But we approximate the maximizer with $K$ PGD steps. How does this approximation error affect the outer minimization? Under what conditions is the approximation good enough?

2. The PGD-AT training loop generates fresh adversarial examples for each batch at each epoch. An alternative is to precompute adversarial examples at the start of each epoch (similar to an "adversarial dataset"). Why might this precomputed approach fail to work well?

3. Robust overfitting is mitigated by early stopping. But the optimal early-stopping checkpoint is determined using the test set, which introduces test set leakage into the training process. How would you design an experiment that correctly evaluates robust accuracy without this leakage?

4. Free-AT updates both the model parameters and the adversarial perturbation using the same gradient. This seems problematic: a gradient that's good for the model update might be bad for the perturbation update, and vice versa. Can you construct a scenario where this simultaneous update causes the training to diverge?

5. The standard CIFAR-10 threat model uses $\epsilon = 8/255$ in L-infinity. Consider using an L-2 threat model with $\epsilon_2 = 0.5$ instead. How would you modify the PGD-AT training loop? Is the 7-step convention still appropriate?

---

## References

- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. *ICLR 2018*.
- Rice, L., Wong, E., & Kolter, J.Z. (2020). Overfitting in Adversarially Robust Deep Learning. *ICML 2020*.
- Shafahi, A., et al. (2019). Adversarial Training for Free! *NeurIPS 2019*.
- Wong, E., Rice, L., & Kolter, J.Z. (2020). Fast is Better than Free: Revisiting Adversarial Training. *ICLR 2020*.
- Andriushchenko, M., & Flammarion, N. (2020). Understanding and Improving Fast Adversarial Training. *NeurIPS 2020*.
- Rebuffi, S.A., et al. (2021). Data Augmentation Can Improve Robustness. *NeurIPS 2021*.
- Smith, L.N. (2017). Cyclical Learning Rates for Training Neural Networks. *WACV 2017*.
