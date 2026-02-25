# Week 4: Evasion Attacks I — FGSM and the Geometry of Adversarial Examples

**CS 6800: Security of Machine Learning Systems**
**Graduate Seminar | Spring 2026**

---

## Learning Objectives

By the end of this lecture, students will be able to:

1. Formally state the evasion attack problem in terms of constrained optimization over Lp norm perturbation balls.
2. Define L0, L1, L2, and L∞ norms, describe the geometry of their unit balls, and explain when each is the appropriate threat model.
3. Explain intuitively why neural network decision boundaries are close to natural data points using the geometry of high-dimensional spaces.
4. Describe the key findings of Szegedy et al. (2014) and explain why transferability of adversarial examples was the more consequential discovery.
5. Derive FGSM from first principles using the linearization argument, and explain why the sign of the gradient is used rather than the gradient itself.
6. Distinguish targeted from untargeted attacks at the level of formal optimization formulations.
7. Identify the limitations of FGSM and explain why single-step attacks are insufficient for reliable security evaluation.
8. Explain the label leaking problem and why it matters for correctly implementing attack algorithms.
9. Implement FGSM in PyTorch at the conceptual level, including the clipping operation that enforces the perturbation budget.

---

## 1. The Evasion Attack Problem: Formal Setup

An evasion attack is a test-time manipulation of the input to a trained classifier. The classifier $f: \mathcal{X} \rightarrow \mathcal{Y}$ has fixed weights; the adversary cannot modify the model. Their only lever is the choice of input. The adversary's goal is to find an input $x'$ that is "close" to a natural input $x$ but causes the classifier to produce an incorrect output.

The formal problem is:

$$\text{find } x' \in \mathcal{X} \text{ such that } f(x') \neq y \text{ and } \|x' - x\|_p \leq \epsilon$$

where $y$ is the true label of $x$, $\|\cdot\|_p$ is a norm (typically L2 or L∞), and $\epsilon > 0$ is the perturbation budget. Because we want the "best" adversarial example (often the one with the smallest perturbation, or equivalently the highest confidence mismatch), this is typically rewritten as an optimization problem:

$$\max_{x' : \|x' - x\|_p \leq \epsilon} L(f(x'), y)$$

where $L$ is the training loss (e.g., cross-entropy). Maximizing the loss under the constraint that $x'$ remains in the $\epsilon$-ball around $x$ is equivalent to finding a nearby input that the model is most wrong about.

Two additional constraints are typically imposed for image domains. First, the valid image range: $x' \in [0, 1]^d$ (or $[0, 255]^d$ before normalization). Second, the perturbation constraint: $\|x' - x\|_p \leq \epsilon$. Together these define the feasible set $\mathcal{S}(x, \epsilon) = \{x' : \|x' - x\|_p \leq \epsilon\} \cap [0, 1]^d$.

This formulation has two "knobs" controlled by the threat model designer: the choice of norm $p$ and the perturbation budget $\epsilon$. Both choices have significant consequences for what attacks are possible and what defenses are meaningful.

---

## 2. Lp Norm Perturbation Balls: Geometry and Semantics

The Lp norm is defined as:

$$\|v\|_p = \left(\sum_{i=1}^{d} |v_i|^p\right)^{1/p}$$

for $p \geq 1$. The special case $p = \infty$ is defined as $\|v\|_\infty = \max_i |v_i|$.

The "perturbation ball" of radius $\epsilon$ around $x$ under the Lp norm is the set of all $x'$ such that $\|x' - x\|_p \leq \epsilon$, which is a scaled and translated version of the Lp unit ball centered at $x$.

### 2.1 L0 Norm

The L0 "norm" (technically a pseudonorm, since it does not satisfy the scaling property of a norm) counts the number of non-zero components of a vector:

$$\|v\|_0 = |\{i : v_i \neq 0\}|$$

An L0 constraint of $\epsilon$ means: the adversary may change at most $\epsilon$ coordinates of the input, but may change those coordinates by any amount. For images, this models an adversary who can arbitrarily modify a small number of pixels. This corresponds to scenarios such as adding a small visible patch or sticker to an image.

The L0 ball is geometrically a union of lower-dimensional affine subspaces of $\mathbb{R}^d$ aligned with coordinate axes. Optimizing over L0-constrained perturbations is an NP-hard combinatorial problem in general, which is why L0 attacks (such as the Jacobian Saliency Map Attack, JSMA) use greedy approximations.

### 2.2 L1 Norm

The L1 norm sums the absolute values of components:

$$\|v\|_1 = \sum_{i=1}^{d} |v_i|$$

An L1 constraint of $\epsilon$ means: the total absolute perturbation across all coordinates is bounded by $\epsilon$. This norm promotes sparsity — under L1 constraints, optimal perturbations tend to be concentrated in a small number of coordinates, similar to L0 but with a convex constraint.

The L1 ball in $\mathbb{R}^d$ is a cross-polytope (a d-dimensional generalization of a diamond shape). For $d=2$, it is the familiar diamond-shaped ball with vertices at distance $\epsilon$ along each axis. L1 attacks are less commonly studied but relevant when the adversary must pay a cost proportional to the total magnitude of perturbation.

### 2.3 L2 Norm

The L2 (Euclidean) norm is:

$$\|v\|_2 = \sqrt{\sum_{i=1}^{d} v_i^2}$$

An L2 constraint of $\epsilon$ means: the Euclidean distance between $x$ and $x'$ is at most $\epsilon$. This norm distributes the perturbation across all coordinates in a geometrically uniform way — the ball is a Euclidean sphere. L2-constrained perturbations tend to perturb every pixel by a small amount, rather than perturbing a few pixels by a large amount.

L2 is the norm used in Szegedy et al.'s original attack, in the Carlini-Wagner (CW) attack, and in many other influential attack methods. It is often considered to have good perceptual properties: a perturbation that is small in L2 tends to be visually subtle because no individual pixel is changed dramatically.

For a numerical example: an MNIST image has $d = 28 \times 28 = 784$ pixels, each in $[0, 1]$. An L2 perturbation of $\epsilon = 1.5$ spreads 1.5 units of total Euclidean perturbation across 784 pixels. Each pixel is changed by approximately $1.5 / \sqrt{784} \approx 0.054$, which is barely perceptible. The same total perturbation under L∞ would change every pixel by exactly $\epsilon/\sqrt{d} \approx 0.054$; under L0 with $\epsilon=1$, it would change one pixel by the full 1.5 units.

### 2.4 L∞ Norm

The L∞ norm is:

$$\|v\|_\infty = \max_{i \in \{1,\ldots,d\}} |v_i|$$

An L∞ constraint of $\epsilon$ means: no single coordinate is changed by more than $\epsilon$. This norm permits all $d$ coordinates to be changed simultaneously, each by an amount up to $\epsilon$. The L∞ ball is a hypercube.

L∞ is the most commonly used threat model for adversarial examples research, for several reasons. First, it is the easiest norm to handle analytically and computationally — projection onto the L∞ ball is a simple clipping operation. Second, the FGSM attack (which we derive below) is naturally formulated for L∞. Third, the perceptual interpretation is clean: changing every pixel by at most $\epsilon$ is a reasonable model of imperceptible perturbations for small $\epsilon$ (e.g., $\epsilon = 8/255$ on a 0–255 scale, which is less than 3% of the full pixel range).

For a concrete comparison: a CIFAR-10 image is $32 \times 32 \times 3 = 3072$ pixels. An L∞ perturbation of $\epsilon = 8/255 \approx 0.031$ changes all 3072 pixels by up to 3.1%. An L2 perturbation that changes each pixel by the same amount would have L2 norm $\epsilon \sqrt{d} \approx 0.031 \times \sqrt{3072} \approx 1.72$. This illustrates why L∞ and L2 budgets are not directly comparable: they are parameterizing different geometries of the perturbation ball.

### 2.5 Choosing the Right Norm

The choice of norm should reflect the threat model:

- **L∞** is appropriate when the adversary can spread perturbation across the entire input (e.g., adding a pixel-level noise pattern to a digital image before transmission). It is the most commonly used norm in research and is the standard for adversarial robustness benchmarks.
- **L2** is appropriate when the adversary is constrained to perturbations with bounded total energy (e.g., physical perturbations that add a fixed amount of "distortion"). CW attacks and certified defenses (smoothed classifiers) often use L2.
- **L0** is appropriate when the adversary can modify only a few input components arbitrarily (e.g., changing a few words in a text document, or adding a small patch to an image).
- **L1** is rarely used as a primary constraint but appears in sparse attack formulations.

No single norm is universally "correct." A comprehensive evaluation should test under multiple norms, as a defense that works against L∞ perturbations may be vulnerable to L2 or L0 attacks with the same perceptual impact.

---

## 3. Why Decision Boundaries Are Close to Natural Data: High-Dimensional Geometry

Szegedy et al.'s discovery that adversarial examples exist ubiquitously for correctly classified inputs was initially surprising. If a neural network correctly classifies a natural image, why should there be a nearby adversarial example? The answer lies in the counterintuitive geometry of high-dimensional spaces.

### 3.1 The Blessing and Curse of Dimensionality

Consider a simplified model: a linear classifier in $d$ dimensions. The decision boundary is a hyperplane $w^T x = 0$. If a natural data point $x$ lies at signed distance $\gamma = w^T x / \|w\|_2$ from the hyperplane, then the minimum L2 perturbation needed to cross the boundary is exactly $\gamma$.

For deep neural networks trained on image data, the margin $\gamma$ (the distance from a natural training point to the nearest decision boundary) is typically small relative to the dimensions of the input space. This is because:

1. The number of distinct classes is small (10, 100, or 1000 for typical benchmarks) while the input dimension is large (thousands to millions). The decision boundaries must partition a high-dimensional space into a small number of regions, and by necessity, those boundaries are close to the natural data manifold.

2. The training objective (minimizing cross-entropy loss on natural data) provides no incentive to push decision boundaries away from training points. The model only needs to classify training points correctly, not to maintain large margins around them.

3. The natural data manifold is itself a low-dimensional structure embedded in the high-dimensional input space. Natural images occupy a tiny fraction of all possible pixel configurations. The decision boundaries of a well-trained classifier are shaped around this manifold but do not need to be far from it in the ambient high-dimensional space.

### 3.2 The Concentration of Measure Argument

There is a deeper reason, related to the phenomenon of concentration of measure in high-dimensional geometry. In high dimensions, the volume of a ball is concentrated near its surface. More specifically, for a ball of radius $r$ in $d$ dimensions, the fraction of volume within a thin shell of width $\delta$ near the surface approaches 1 as $d$ increases:

$$\frac{\text{Vol}(B(r, d)) - \text{Vol}(B(r-\delta, d))}{\text{Vol}(B(r, d))} = 1 - \left(1 - \frac{\delta}{r}\right)^d \approx 1 - e^{-d\delta/r}$$

For large $d$, this fraction approaches 1 even for very small $\delta/r$. The implication is that in high dimensions, most of the volume of a ball lies near its boundary, not near its center.

For adversarial examples, this means: when we add a small perturbation to a natural image $x$ in a high-dimensional space, the perturbed point $x + \delta$ is almost certainly in a part of the input space that the training distribution did not cover. The model's behavior in this "shell" near the natural data manifold is unconstrained by training — there were no training examples there. The decision boundary may be arbitrarily close in this shell, and in practice, for the standard norm budgets used in the literature, adversarial examples are found with near-100% success rate for undefended models.

### 3.3 Informal Numerical Illustration

Consider a network trained on $32 \times 32 \times 3$ CIFAR-10 images, and suppose that the decision boundary in the neighborhood of a natural image is approximately linear. If the margin (distance to the decision boundary) is $\gamma = 0.01$ in L2 norm, then any perturbation of L2 norm greater than 0.01 that points toward the decision boundary will cross it. The L∞ ball of radius $\epsilon = 8/255 \approx 0.031$ has L2 radius $0.031 \times \sqrt{3072} \approx 1.72$ — much larger than the margin. There is a great deal of "room" within the L∞ perturbation budget to cross the decision boundary many times.

This calculation is rough because real decision boundaries are nonlinear and the margin varies by direction, but it correctly captures why adversarial examples are found so easily.

---

## 4. Szegedy et al. (2014): Intriguing Properties of Neural Networks

The 2014 paper "Intriguing Properties of Neural Networks" by Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus introduced adversarial examples to the deep learning community and made two fundamental discoveries.

### 4.1 The Attack Method: Box-Constrained L-BFGS

Szegedy et al. found adversarial examples by solving a constrained optimization problem using L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno), a quasi-Newton optimization algorithm. Given an input $x$ and target class $t \neq f(x)$ (in the targeted variant), they minimized:

$$\|x' - x\|_2^2 \quad \text{subject to} \quad f(x') = t, \quad x' \in [0, 1]^d$$

This is computationally expensive — it requires many forward and backward passes through the network — but it reliably finds perturbations with small L2 norm. The "box constraint" ($x' \in [0, 1]^d$) ensures that the perturbed image has valid pixel values.

The numerical results were stark: for a deep network correctly classifying an image of a school bus, they found perturbations with L∞ norm as small as 0.007 (on a [0,1] scale) that caused the network to classify the perturbed image as an ostrich with high confidence. These perturbations were invisibly small to human observers.

### 4.2 First Key Finding: Ubiquity of Adversarial Examples

The first major finding was that adversarial examples could be found for essentially every correctly classified input in their test set. This was not a rare pathology affecting a small fraction of inputs; it appeared to be a systematic property of the trained classifier's decision surface.

Furthermore, adversarial examples existed for networks with very different architectures, training procedures, and hyperparameters. This suggested that the property was not an artifact of a specific network configuration but rather a general property of high-capacity classifiers trained on image data.

### 4.3 Second Key Finding: Transferability

The second major finding was more surprising and, in retrospect, more consequential: adversarial examples transfer across models. An image crafted to fool Network A (trained on the same dataset with a different architecture or different random seed) would often also fool Network B.

This transferability property shatters what might have been a comforting security assumption: "Even if adversarial examples exist in theory, an attacker who doesn't have access to our specific model can't exploit them." Transferability means that an attacker can:

1. Train their own model (or use a public pretrained model) on similar data.
2. Generate adversarial examples against this surrogate model using white-box methods.
3. Use those adversarial examples to attack the target model with a reasonable probability of success.

This attack strategy is called a transfer-based black-box attack, and it is one of the most practically relevant attack scenarios. The degree of transferability depends on several factors — how similar the models are architecturally and in training data, what attack method is used, and what perturbation budget is allowed — but it is nonzero in virtually all studied cases.

Why does transferability occur? The most widely accepted explanation is that different networks trained on the same data learn similar decision boundaries in the regions near the natural data manifold, because those boundaries are constrained by the training data in the same way. Adversarial examples exploit the geometry of this shared decision boundary, and since the geometry is similar across models, perturbations that work against one model tend to work against others.

---

## 5. FGSM: Fast Gradient Sign Method — Full Derivation

The Fast Gradient Sign Method (FGSM) was introduced by Ian Goodfellow, Jonathon Shlens, and Christian Szegedy in "Explaining and Harnessing Adversarial Examples" (2015, ICLR). It is the most important attack method to understand conceptually, not because it is the strongest attack (it is not), but because every iterative attack method is built on it, and its derivation reveals the essential geometry of adversarial perturbation.

### 5.1 The Linearization Argument

The key insight of FGSM is that for small perturbations, a neural network's loss function can be approximated by a linear function of the input.

Let $L(f_\theta(x), y)$ denote the cross-entropy loss of the classifier $f$ with parameters $\theta$ on input $x$ with label $y$. Taking a first-order Taylor expansion of the loss around the natural input $x$:

$$L(f_\theta(x + \delta), y) \approx L(f_\theta(x), y) + \delta^T \nabla_x L(f_\theta(x), y)$$

This linear approximation is valid when $\delta$ is small relative to the scale at which the loss changes, which is the case for small $\epsilon$.

Given this linear approximation, the problem of finding the perturbation $\delta$ that maximizes the loss within the L∞ ball of radius $\epsilon$ becomes:

$$\max_{\delta : \|\delta\|_\infty \leq \epsilon} \delta^T \nabla_x L(f_\theta(x), y)$$

This is an inner product maximization problem subject to a norm constraint, which has a clean analytic solution.

### 5.2 The Sign Gradient: Maximizing Under L∞ Constraint

We want to maximize $\delta^T g$ where $g = \nabla_x L(f_\theta(x), y)$ and $\|\delta\|_\infty \leq \epsilon$.

Since $\|\delta\|_\infty \leq \epsilon$, we have $-\epsilon \leq \delta_i \leq \epsilon$ for each coordinate $i$. The inner product is:

$$\delta^T g = \sum_{i=1}^{d} \delta_i g_i$$

Each term $\delta_i g_i$ is maximized independently by setting $\delta_i = \epsilon \cdot \text{sign}(g_i)$. This gives:

$$\delta_i^* = \epsilon \cdot \text{sign}(g_i)$$

Substituting back:

$$\max_{\|\delta\|_\infty \leq \epsilon} \delta^T g = \sum_{i=1}^d \epsilon |g_i| = \epsilon \|g\|_1$$

The optimal perturbation $\delta^* = \epsilon \cdot \text{sign}(\nabla_x L)$ achieves an increase in the linearized loss of $\epsilon \|g\|_1$.

This is the FGSM perturbation. The full adversarial example is:

$$x' = x + \epsilon \cdot \text{sign}(\nabla_x L(f_\theta(x), y))$$

### 5.3 Why the Sign, Not the Gradient Itself?

It is instructive to compare the FGSM perturbation to what you might naively do: use the gradient direction $\hat{g} = g / \|g\|_2$, normalized to unit L2 norm, scaled by $\epsilon$ to satisfy the L2 constraint.

The gradient direction attack maximizes the linearized loss under an L2 constraint. The maximum increase in linearized loss is $\epsilon \|g\|_2$.

FGSM (the sign) maximizes the linearized loss under an L∞ constraint. The maximum increase in linearized loss is $\epsilon \|g\|_1$.

For a $d$-dimensional gradient, $\|g\|_1 \leq \sqrt{d} \|g\|_2$ by the Cauchy-Schwarz inequality, with equality when all gradient components have the same magnitude. In practice, for image classifiers in high dimensions, $\|g\|_1 / \|g\|_2$ is often much larger than 1, meaning the sign perturbation (used with L∞ budget) achieves a much larger increase in loss than the normalized gradient perturbation (used with an equivalent L2 budget).

The intuition: in high dimensions, the sign gradient uses the entire L∞ budget on every coordinate simultaneously. The L2-normalized gradient concentrates the perturbation in the directions where the gradient is largest and spreads less energy to smaller gradient directions. For classifiers where the gradient has many roughly equal-magnitude components (which is empirically common), the sign gradient dominates.

### 5.4 Complete FGSM Formula and Perturbation Budget

Putting it all together, the FGSM adversarial example for untargeted attack is:

$$x' = \text{Clip}_{[0,1]}\left(x + \epsilon \cdot \text{sign}\left(\nabla_x L(f_\theta(x), y)\right)\right)$$

where $\text{Clip}_{[0,1]}(v) = \max(0, \min(1, v))$ is the element-wise projection onto the valid image range.

The perturbation budget $\epsilon$ controls the tradeoff between attack strength and perturbation visibility:

- **Small $\epsilon$ (e.g., $\epsilon = 2/255 \approx 0.008$):** Perturbations are nearly invisible to human observers. The single FGSM step may not be enough to cross the decision boundary reliably.
- **Moderate $\epsilon$ (e.g., $\epsilon = 8/255 \approx 0.031$):** This is the standard benchmark budget for CIFAR-10. Perturbations are visible upon close inspection but do not distort the semantic content of the image.
- **Large $\epsilon$ (e.g., $\epsilon = 16/255$ or larger):** Perturbations become clearly visible. FGSM is more likely to succeed but the adversarial examples are less realistic.

The perturbation budget must be specified in the same units as the pixel values. If pixels are represented in [0, 255], the budget should be specified in the same range (e.g., $\epsilon = 8$). If pixels are normalized to [0, 1], the budget should be correspondingly scaled (e.g., $\epsilon = 8/255$).

### 5.5 A Numerical Example

Let us trace through FGSM on a small example. Suppose $x$ is an image with $d = 3$ pixels (a highly simplified 1D example), with values $x = [0.5, 0.3, 0.8]$. The classifier is a two-class network predicting spam (y=1) or ham (y=0). The true label is $y = 0$ (ham), but we want to attack it to be classified as spam (y=1).

After a forward pass and backpropagation, suppose we obtain:
$$\nabla_x L(f_\theta(x), y=0) = [0.12, -0.07, 0.21]$$

The sign of the gradient is:
$$\text{sign}(\nabla_x L) = [+1, -1, +1]$$

With $\epsilon = 0.1$:
$$\delta = 0.1 \times [+1, -1, +1] = [0.1, -0.1, 0.1]$$

The adversarial example (before clipping, which is unnecessary here):
$$x' = [0.5 + 0.1, 0.3 - 0.1, 0.8 + 0.1] = [0.6, 0.2, 0.9]$$

Verify the L∞ constraint: $\|x' - x\|_\infty = \max(0.1, 0.1, 0.1) = 0.1 = \epsilon$. Satisfied.

The perturbation adds 0.1 to pixels 1 and 3 (in the direction that increases the loss, i.e., moves toward incorrect classification) and subtracts 0.1 from pixel 2 (also in the direction that increases the loss).

The resulting adversarial example $x' = [0.6, 0.2, 0.9]$ is, under the linear approximation, the input in the $L_\infty$ ball of radius 0.1 around $x$ that maximizes the cross-entropy loss.

---

## 6. Targeted vs. Untargeted Attacks

The formulation above is for an **untargeted attack**: the adversary simply wants the model to output anything other than the correct class $y$. This is the weakest requirement on the attack.

A **targeted attack** specifies a target class $t \neq y$ that the adversary wants the model to output. This is a stronger requirement and, correspondingly, harder to achieve.

### 6.1 Untargeted FGSM

For an untargeted attack, we maximize the loss with respect to the true label $y$:

$$x' = \text{Clip}_{[0,1]}\left(x + \epsilon \cdot \text{sign}\left(\nabla_x L(f_\theta(x), y)\right)\right)$$

The intuition is: move in the direction that most increases the probability of any wrong class.

### 6.2 Targeted FGSM

For a targeted attack with target class $t$, we minimize the loss with respect to the target label $t$, i.e., we maximize the probability of class $t$:

$$x' = \text{Clip}_{[0,1]}\left(x - \epsilon \cdot \text{sign}\left(\nabla_x L(f_\theta(x), t)\right)\right)$$

Note the minus sign: we want to decrease the loss for class $t$ (i.e., make the model more confident about $t$). The gradient $\nabla_x L(f_\theta(x), t)$ points in the direction of increasing the loss for target $t$, so we step in the opposite direction.

Targeted attacks are more useful in practice when the adversary has a specific desired outcome — for example, a targeted attack on a face recognition system that specifically misidentifies an adversary as a specific authorized user, rather than simply any authorized user.

Targeted attacks are harder than untargeted attacks because the adversary must move to a specific region of the output space rather than simply away from the correct class. For FGSM (single step), targeted attacks typically have lower success rates than untargeted attacks for the same $\epsilon$.

---

## 7. Implementing FGSM in PyTorch

The following code walkthrough illustrates how FGSM is implemented in PyTorch. The key requirement is that gradients must be computed with respect to the input (not just the model parameters), which requires the input tensor to track gradients.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

def fgsm_attack(model, loss_fn, images, labels, epsilon):
    """
    Perform FGSM attack on a batch of images.

    Args:
        model: PyTorch model (must output logits).
        loss_fn: Loss function, e.g., nn.CrossEntropyLoss().
        images: Clean input images, shape (N, C, H, W), in [0, 1].
        labels: True integer labels, shape (N,).
        epsilon: Perturbation budget under L_inf norm.

    Returns:
        adversarial_images: Perturbed images, same shape as images.
    """
    # Require gradient tracking on the input tensor.
    # We create a copy and set requires_grad=True.
    images_adv = images.clone().detach().requires_grad_(True)

    # Forward pass with the adversarial (initially clean) images.
    outputs = model(images_adv)

    # Compute loss with respect to true labels.
    # We call model.zero_grad() to clear gradients from any previous step.
    model.zero_grad()
    loss = loss_fn(outputs, labels)

    # Backward pass: compute gradients of loss w.r.t. image pixels.
    loss.backward()

    # Retrieve the gradient of the loss with respect to the input.
    # sign() gives the element-wise sign of the gradient tensor.
    data_grad = images_adv.grad.data
    sign_data_grad = data_grad.sign()

    # Create perturbed image: add epsilon * sign(gradient) to original image.
    perturbed_images = images + epsilon * sign_data_grad

    # Clip to valid pixel range [0, 1] to maintain valid image.
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images.detach()


def evaluate_fgsm(model, data_loader, epsilon, device='cuda'):
    """
    Evaluate model clean accuracy and FGSM robust accuracy.

    Args:
        model: Trained PyTorch model.
        data_loader: DataLoader for the test set.
        epsilon: FGSM perturbation budget.
        device: 'cuda' or 'cpu'.

    Returns:
        clean_acc: Clean accuracy (float).
        robust_acc: FGSM robust accuracy (float).
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    correct_clean = 0
    correct_robust = 0
    total = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        # Clean accuracy
        with torch.no_grad():
            outputs_clean = model(images)
            preds_clean = outputs_clean.argmax(dim=1)
            correct_clean += (preds_clean == labels).sum().item()

        # FGSM adversarial accuracy
        perturbed = fgsm_attack(model, loss_fn, images, labels, epsilon)
        with torch.no_grad():
            outputs_adv = model(perturbed)
            preds_adv = outputs_adv.argmax(dim=1)
            correct_robust += (preds_adv == labels).sum().item()

        total += labels.size(0)

    return correct_clean / total, correct_robust / total
```

Several implementation details deserve attention.

**Gradient with respect to input:** In standard training, we compute gradients with respect to model parameters $\theta$. For FGSM, we compute gradients with respect to the input $x$. In PyTorch, this requires `images.requires_grad_(True)` before the forward pass. Forgetting this step is the most common FGSM implementation error.

**Zero gradients:** We call `model.zero_grad()` to ensure that gradients from any previous batch or step are not accumulated into the current gradient computation.

**Clipping:** After adding the perturbation, we clip to $[0, 1]$ to ensure valid pixel values. This clipping is part of the attack specification: the feasible set is the intersection of the $L_\infty$ ball and the valid image range.

**Model in eval mode:** We typically call `model.eval()` before generating adversarial examples. This is important because batch normalization and dropout behave differently in training vs. evaluation mode, and we want the adversarial example to be optimized for the model's inference-time behavior.

---

## 8. Limitations of FGSM

FGSM is fast (one forward pass and one backward pass per example) and clean to analyze, but it has significant limitations that make it inadequate for serious security evaluation.

### 8.1 Single Step Is Not Optimal

The FGSM perturbation is optimal under the linear approximation of the loss. But the loss function of a deep neural network is not linear in the input — it is a highly nonlinear function with many local maxima within the $\epsilon$-ball. A single gradient step from $x$ may move in a locally optimal direction but fail to find the global maximum of the loss within the $\epsilon$-ball.

Empirically, for the same $\epsilon$, iterative attacks (like PGD, which we cover in Week 5) achieve much higher attack success rates than FGSM. For $\epsilon = 8/255$ on CIFAR-10 against a naturally trained ResNet, FGSM achieves roughly 50–60% attack success rate (i.e., 50–60% of correctly classified images are successfully misclassified), while PGD-20 achieves >99% attack success rate. This gap means that a defense that appears robust against FGSM may be completely non-robust when evaluated against stronger attacks.

### 8.2 Fails at Large $\epsilon$

At large perturbation budgets, FGSM can paradoxically perform worse than at moderate budgets. This is because the loss surface has significant curvature, and a large step in the linearized gradient direction may overshoot the optimal point and land in a region of lower loss than the starting point. This phenomenon, sometimes called "catastrophic overshoot," is documented in Goodfellow et al. (2015) and is one motivation for the development of iterative attacks with smaller step sizes.

### 8.3 Label Leaking

A subtle but important issue in implementing FGSM is the "label leaking" problem, identified by Kurakin et al. (2017). The FGSM formula uses the true label $y$ to compute the gradient. But in practice, we often want to evaluate FGSM attack success on a set of images including those that the model already misclassifies (before the attack). For a model that misclassifies $x$ (i.e., $f(x) \neq y$), computing `loss_fn(model(x), y)` and taking the sign of the gradient moves $x$ away from the true class boundary. If $x$ is already on the wrong side, this may inadvertently move it back toward the correct class.

The fix is to use the "maximum likelihood" label — the model's own predicted label $\hat{y} = \text{argmax} f(x)$ — rather than the true label $y$. This is sometimes called the "Goodfellow-Kurakin attack" or "iterative FGSM with label leaking correction." The formula becomes:

$$x' = \text{Clip}_{[0,1]}\left(x + \epsilon \cdot \text{sign}\left(\nabla_x L(f_\theta(x), \hat{y})\right)\right)$$

However, there is a countervailing consideration: using the model's predicted label rather than the true label can cause the attack to work against wrong predictions in the wrong direction. The standard practice in the research community is to use the true label for FGSM when the goal is untargeted attack, and to exclude already-misclassified images from the attack success rate calculation.

For PGD and other iterative attacks, this distinction matters less because the iterative steps naturally find the loss maximum in the $\epsilon$-ball regardless of the starting gradient direction.

---

## 9. Summary: The Geometry Picture

Let us synthesize the key conceptual points from this lecture into a coherent geometric picture.

A trained image classifier defines a decision boundary in the high-dimensional input space $\mathcal{X} = [0,1]^d$. Natural images lie on (or near) a low-dimensional manifold $\mathcal{M}$ embedded in $\mathcal{X}$. The decision boundary is shaped by the training data on $\mathcal{M}$, but it extends through the much larger ambient space $\mathcal{X}$.

For any natural image $x \in \mathcal{M}$, consider the $L_\infty$ ball of radius $\epsilon$ around $x$: the set of all images that differ from $x$ by at most $\epsilon$ per pixel. This ball has $2^\epsilon$ times the volume of the natural image manifold neighborhood around $x$ (approximately). For moderate $\epsilon$, the ball contains enormous numbers of points not on $\mathcal{M}$ where the model's behavior is unconstrained.

The decision boundary, shaped around $\mathcal{M}$, does not need to maintain any margin within the L∞ ball. As a result, for most natural images, there exist points in the L∞ ball that are on the wrong side of the decision boundary — adversarial examples.

FGSM finds these adversarial examples efficiently by exploiting the local geometry: the gradient of the loss points in the direction of steepest loss increase, and the sign of the gradient is the direction that maximally increases the loss under an L∞ constraint. One step in this direction usually crosses the decision boundary for moderate $\epsilon$ on naturally trained models.

---

## 10. Discussion Questions

1. **Norms and semantics:** We discussed L0, L1, L2, and L∞ norms as threat models for image perturbations. For natural language text, what would be the analogous threat models? How do you define "small" perturbation of a sentence? What constraints make semantic sense?

2. **Transferability:** The transferability of adversarial examples across models suggests that different models learn similar decision boundaries near natural data. Can you think of model design choices that would reduce transferability? What is the tradeoff?

3. **FGSM limitations:** We identified that FGSM is not optimal and can fail at large $\epsilon$ due to curvature. Can you design a simple modification to FGSM (one or two additional steps) that would address the overshoot problem? How does this relate to what PGD does?

4. **Label leaking:** Consider a scenario where a defense against FGSM achieves 95% robust accuracy using true labels in the gradient computation, but only 70% robust accuracy using predicted labels. Is this difference evidence of label leaking, or could there be other explanations? How would you design an experiment to distinguish these possibilities?

5. **Physical perturbations:** The Eykholt stop sign attack (Week 1) used the Expectation over Transformation (EoT) approach to create physically robust adversarial examples. How does this connect to the L∞ formulation of FGSM? What norm or threat model would be most natural for physical adversarial attacks?

---

## 11. Key Takeaways

The evasion attack problem is formally a constrained optimization problem: find a nearby input (in Lp norm) that maximizes the model's loss. The choice of norm $p$ and budget $\epsilon$ defines the threat model and should be chosen to reflect realistic adversary capabilities.

High-dimensional geometry explains why adversarial examples exist ubiquitously: the L∞ ball of even small radius $\epsilon$ covers an enormous volume of the input space where the model's behavior is unconstrained by training, and the decision boundary necessarily passes through this volume.

FGSM is derived from the linearization of the loss function and the analytic solution to the resulting inner product maximization problem. The sign of the gradient is the optimal perturbation direction under an L∞ constraint.

Targeted attacks minimize the loss with respect to the target class (negate the gradient step). Untargeted attacks maximize the loss with respect to the true class.

FGSM's limitations — single step, susceptibility to curvature, label leaking — make it insufficient for serious security evaluation but provide the conceptual foundation for iterative attacks that we will study next week.

---

## Assigned Reading

- Szegedy, C. et al. (2014). "Intriguing properties of neural networks." ICLR 2014.
- Goodfellow, I.J., Shlens, J., & Szegedy, C. (2015). "Explaining and harnessing adversarial examples." ICLR 2015.
- Kurakin, A., Goodfellow, I., & Bengio, S. (2017). "Adversarial examples in the physical world." ICLR 2017 Workshop.

**Problem Set 2** (due in two weeks) will implement FGSM, BIM, and PGD in PyTorch and evaluate them on CIFAR-10. The implementation skeleton will be provided; you must fill in the attack functions and the evaluation loop.

---

*End of Lecture 4 Notes*
*Next lecture: Week 5 — Evasion Attacks II: PGD, Iterative Attacks, and Reliable Evaluation*
