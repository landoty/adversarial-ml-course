# CS 6810 — Adversarial Machine Learning
## Week 06: Black-Box Attacks II — Score-Based Attacks — Zeroth-Order Optimization

**Prerequisites:** Week 01 (taxonomy), Week 05 (transferability), familiarity with gradient descent and Monte Carlo estimation.

**Learning Objectives:**
1. Explain the score-based threat model and why it requires zeroth-order optimization.
2. Derive the finite-difference gradient estimator and analyze its query complexity.
3. Implement the NES attack with antithetic sampling.
4. Compare ZOO, NES, SimBA, and Square Attack on the query efficiency frontier.
5. Reason about the SPSA estimator and its variance compared to Gaussian antithetic sampling.

---

## 1. The Score-Based Threat Model

### 1.1 Formal Definition

In the score-based (or soft-label black-box) threat model:
- The attacker can submit arbitrary inputs to the target model and observe the full probability vector (or logit vector) output.
- The attacker has NO access to the model's weights, architecture, or gradients.
- Each model query is counted toward a "query budget."

**Notation:** Let $p : \mathbb{R}^n \to \Delta^K$ be the target model's probability output (where $\Delta^K$ is the $(K-1)$-simplex). The attacker observes $p(x)$ for any queried $x$ but cannot compute $\nabla_x p(x)$.

**Why this matters:** Many commercial ML APIs (Google Cloud Vision, Amazon Rekognition, Microsoft Azure Vision) return probability scores, not just hard labels. This enables score-based attacks.

**Query budget:** APIs often rate-limit or charge per query. Real-world query budgets are on the order of $10^3$–$10^5$ queries. A successful score-based attack must work within this budget.

### 1.2 Relationship to Other Threat Models

```
White-box ──────────────────────── most powerful
    ↓ remove gradient access
Score-based (soft labels) ──────── medium
    ↓ remove probability scores
Decision-based (hard labels) ───── least powerful
    ↓ add surrogate model
Transfer-based ───────────────────── no query access
```

Score-based attacks are strictly weaker than white-box (no gradient computation) but strictly stronger than decision-based (can observe confidence information).

---

## 2. Zeroth-Order Gradient Estimation

### 2.1 The Basic Problem

We want to compute $\nabla_x \mathcal{L}(x)$ where $\mathcal{L}(x) = \mathcal{L}(p(x), y)$ (some loss applied to the model's output). But we cannot compute gradients through $p$.

The fundamental tool: the **finite difference approximation**. For a smooth function $\mathcal{L} : \mathbb{R}^n \to \mathbb{R}$, the partial derivative in coordinate $i$ is:

$$\frac{\partial \mathcal{L}}{\partial x_i} \approx \frac{\mathcal{L}(x + h \cdot e_i) - \mathcal{L}(x - h \cdot e_i)}{2h} \tag{1}$$

where $e_i$ is the $i$-th standard basis vector and $h > 0$ is a small step size.

**Error analysis:** By Taylor expansion:

$$\mathcal{L}(x + h e_i) = \mathcal{L}(x) + h \frac{\partial \mathcal{L}}{\partial x_i} + \frac{h^2}{2} \frac{\partial^2 \mathcal{L}}{\partial x_i^2} + O(h^3)$$

$$\mathcal{L}(x - h e_i) = \mathcal{L}(x) - h \frac{\partial \mathcal{L}}{\partial x_i} + \frac{h^2}{2} \frac{\partial^2 \mathcal{L}}{\partial x_i^2} + O(h^3)$$

Subtracting: $\frac{\mathcal{L}(x + h e_i) - \mathcal{L}(x - h e_i)}{2h} = \frac{\partial \mathcal{L}}{\partial x_i} + O(h^2)$. The approximation has error $O(h^2)$ — better than the forward-difference approximation which has error $O(h)$.

**Choosing $h$:** Too small: numerical instability from floating-point errors in the probability output. Too large: the Taylor expansion breaks down (high-order terms matter). Typical: $h = 0.001$–$0.01$.

### 2.2 The Query Complexity Catastrophe

To estimate the full gradient $\nabla_x \mathcal{L}(x) \in \mathbb{R}^n$ by coordinate-wise finite differences, we need $2n$ model queries per gradient step (two queries per coordinate, plus a baseline query).

**For CIFAR-10 ($32 \times 32 \times 3$):** $n = 3072$, so $2 \times 3072 = 6144$ queries per gradient step. For 100 gradient steps (like PGD-100), that's $614,400$ queries — impractical for most APIs.

**For ImageNet ($224 \times 224 \times 3$):** $n = 150,528$, so $301,056$ queries per gradient step — hopeless with coordinate-wise finite differences.

This query explosion motivates the more efficient methods in the following sections.

---

## 3. ZOO: Zeroth-Order Optimization

### 3.1 Core Algorithm

ZOO (Chen et al. 2017) applies the C&W attack framework with a zeroth-order gradient estimator. Instead of computing the exact gradient, it estimates it coordinate-by-coordinate, but with two key efficiency improvements:

**Key idea 1 — Coordinate-wise update:** Rather than estimating all $n$ gradient components and then taking a step, ZOO updates *one coordinate at a time*, using a standard ADAM update for that coordinate. This is coordinate-wise descent with Adam.

**Key idea 2 — Importance sampling:** Not all pixel coordinates are equally important. Pixels in the background (for a classification attack) have small gradient magnitude. ZOO spends more queries on "important" pixels.

### 3.2 ZOO-ADAM Update

For coordinate $i$, the zeroth-order gradient estimate is:

$$\hat{g}_i = \frac{\mathcal{L}(x + h e_i) - \mathcal{L}(x - h e_i)}{2h} \tag{2}$$

This requires 2 queries. The ADAM state (first and second moment estimates $m_i$ and $v_i$) is updated for coordinate $i$ only. The coordinate $i$ to update at each step is chosen by one of two strategies:

**ZOO-ADAM (random coordinate):** Sample $i \sim \text{Uniform}(1, \ldots, n)$ at each step.

**ZOO-ADAM with importance sampling:** Maintain an importance weight $\omega_i$ for each coordinate. At each step, sample $i$ proportional to $|\hat{g}_i|^2$ (more queries to high-gradient coordinates).

### 3.3 Query Complexity of ZOO

For $T$ optimization steps, ZOO needs $2T$ queries (regardless of $n$, since we only estimate one coordinate per step). However, convergence requires $T \gg n$ steps to visit each coordinate enough times. In practice:

- ZOO requires roughly $2n$ to $10n$ queries to converge.
- For CIFAR-10: $\approx 6000$–$30,000$ queries per adversarial example.
- For ImageNet (with downsampled attack): $\approx 50,000$–$200,000$ queries.

ZOO uses two additional tricks for efficiency:
1. **Hierarchical coordinate selection:** Divide the image into blocks; estimate gradient at the block level first; then refine within important blocks.
2. **Attack in image space, not pixel space:** Reduce dimensionality by attacking a compressed representation (e.g., DCT coefficients).

### 3.4 ZOO Loss Function

ZOO uses the C&W loss (equation 1 of Week 03):

$$\mathcal{L}_{\text{ZOO}}(x') = \|\delta\|_2^2 + c \cdot g(x') \tag{3}$$

where $g$ is the C&W hinge loss. The distortion term $\|\delta\|_2^2$ does not require model queries (it only depends on $x'$ and $x_0$); the attack-loss term $g(x')$ requires 2 model queries per coordinate update.

---

## 4. NES: Natural Evolution Strategies

### 4.1 From Optimization to Evolution

Natural Evolution Strategies (Wierstra et al. 2014) views the gradient estimation problem differently. Instead of estimating $\nabla_x \mathcal{L}(x)$ directly, NES estimates the gradient of the *expected loss* under a distribution $p_\psi$ centered at $x$:

$$\nabla_\psi \mathbb{E}_{z \sim p_\psi}[\mathcal{L}(z)] = \mathbb{E}_{z \sim p_\psi}\!\left[\mathcal{L}(z) \cdot \nabla_\psi \log p_\psi(z)\right] \tag{4}$$

This is the "REINFORCE" / score function gradient estimator. For $p_\psi = \mathcal{N}(\psi, \sigma^2 I)$ (Gaussian centered at $\psi = x$):

$$\nabla_\psi \mathbb{E}_{z \sim p_\psi}[\mathcal{L}(z)] = \frac{1}{\sigma} \mathbb{E}_{\delta \sim \mathcal{N}(0,I)}\!\left[\mathcal{L}(x + \sigma\delta) \cdot \delta\right] \tag{5}$$

This is a gradient estimate in the *natural evolution* sense: sample perturbations, evaluate the loss, weight the perturbations by the loss.

### 4.2 NES for Adversarial Attacks

Ilyas et al. (2018) applied NES to adversarial attacks. The estimate of $\nabla_x \mathcal{L}(x)$ using $n_s$ samples is:

$$\hat{g} = \frac{1}{n_s \sigma} \sum_{i=1}^{n_s} \mathcal{L}(x + \sigma \delta_i) \cdot \delta_i, \quad \delta_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, I) \tag{6}$$

**Query count:** $n_s$ queries per gradient estimate (one query per sample).

**PGD with NES gradient:**

$$x_{t+1} = \text{Clip}_{x, \epsilon}\!\left(x_t + \alpha \cdot \text{sign}(\hat{g}_t)\right) \tag{7}$$

This is exactly PGD, but with the gradient replaced by the NES estimate.

### 4.3 Antithetic Sampling: Cutting Query Count in Half

**Key trick:** Instead of drawing $n_s$ independent Gaussian samples, use *antithetic pairs*: $(\delta_i, -\delta_i)$. This gives:

$$\hat{g} = \frac{1}{n_s \sigma} \sum_{i=1}^{n_s/2} (\mathcal{L}(x + \sigma\delta_i) - \mathcal{L}(x - \sigma\delta_i)) \cdot \delta_i \tag{8}$$

**Why this helps (variance reduction):** The estimator (6) has high variance because $\mathcal{L}(x + \sigma\delta_i)$ includes a large "baseline" component ($\mathcal{L}(x)$) that cancels when using antithetic pairs. Antithetic sampling approximately doubles the effective sample size for the same number of queries.

**Formal variance comparison:**
- Independent sampling (equation 6): $\text{Var}[\hat{g}_i] \propto \mathbb{E}[\mathcal{L}^2]/n_s$
- Antithetic sampling (equation 8): $\text{Var}[\hat{g}_i] \propto \mathbb{E}[(\Delta\mathcal{L})^2]/(4n_s)$ where $\Delta\mathcal{L} = \mathcal{L}(x+\sigma\delta) - \mathcal{L}(x-\sigma\delta)$

If $\mathcal{L}(x) \gg \Delta\mathcal{L}$ (the loss has large baseline relative to its variation under perturbation), antithetic sampling can reduce variance by an order of magnitude.

### 4.4 SPSA: Simultaneous Perturbation Stochastic Approximation

SPSA (Uesato et al. 2018 in adversarial context) uses Rademacher (Bernoulli $\pm 1$) random vectors instead of Gaussian:

$$\hat{g} = \frac{1}{n_s \sigma} \sum_{i=1}^{n_s/2} (\mathcal{L}(x + \sigma\delta_i) - \mathcal{L}(x - \sigma\delta_i)) \cdot \delta_i, \quad \delta_i \sim \text{Rademacher}(n) \tag{9}$$

where each component of $\delta_i$ is $\pm 1$ with equal probability.

**Comparison to Gaussian NES:**

| Property | Gaussian NES | SPSA |
|----------|-------------|------|
| Distribution of $\delta_i$ | $\mathcal{N}(0, I)$ | Rademacher ($\pm 1$) |
| $\mathbb{E}[\delta_i \delta_i^\top]$ | $I$ | $I$ |
| Estimator bias | Unbiased (for smooth $\mathcal{L}$) | Unbiased |
| Variance per sample | $O(1/n)$ | $O(1/n)$ |
| Implementation | Draw Gaussian, normalize | Draw $\pm 1$ per coordinate |

Both are unbiased estimators of $\nabla_x \mathcal{L}(x)$ with the same asymptotic variance. The practical difference is minimal; SPSA is slightly simpler to implement.

### 4.5 NES Attack: Hyperparameter Analysis

**Smoothing parameter $\sigma$:** Controls the scale of the gradient estimate.
- Too small: The estimate $\hat{g}$ approximates $\nabla_x \mathcal{L}$ but is high-variance (noisy).
- Too large: The estimate is low-variance but biased — it estimates the gradient of a smoothed version of $\mathcal{L}$.
- Typical: $\sigma = 0.01$–$0.1$ (normalized to $[0,1]$ pixel range).

**Samples per gradient $n_s$:** Controls variance of the gradient estimate.
- More samples → lower variance → better optimization per query, but more queries per step.
- Typical: $n_s = 20$–$100$ (i.e., $20$–$100$ queries per gradient step).

**Number of steps $T$:**
- Typical: $T = 100$–$500$ gradient steps.
- Total queries: $n_s \times T = 2000$–$50,000$.

**Step size $\alpha$:**
- Typical: $\alpha = \epsilon/T$ (decaying step size as in PGD).

---

## 5. SimBA: Simple Black-Box Adversarial Attack

### 5.1 Algorithm

SimBA (Guo et al. 2019) is a conceptually different approach: instead of estimating gradients, it directly searches for perturbations in a structured subspace of the pixel space.

**Algorithm:**
1. Initialize $x_0^{\text{adv}} = x_0$.
2. At each step $t$, pick a direction $q_t$ from a predefined orthonormal basis $\{q_1, q_2, \ldots\}$.
3. Try the two perturbations $x_t^{\text{adv}} \pm \epsilon' q_t$.
4. Accept the one that increases the loss more: $x_{t+1}^{\text{adv}} = x_t^{\text{adv}} + \epsilon' q_t$ if $\mathcal{L}(x_t + \epsilon' q_t) > \mathcal{L}(x_t - \epsilon' q_t)$, else $x_{t+1}^{\text{adv}} = x_t^{\text{adv}} - \epsilon' q_t$.
5. Reject if neither perturbation increases the loss (total norm constraint).

### 5.2 Frequency Domain SimBA

Guo et al. propose using the 2D Discrete Cosine Transform (DCT) basis as the orthonormal basis $\{q_i\}$. The DCT basis vectors are ordered by frequency:
- Low-frequency DCT vectors: smooth, large-scale intensity changes.
- High-frequency DCT vectors: fine-grained, spatially localized changes.

**Why DCT?** Adversarial perturbations are not uniformly distributed in frequency space. Empirically, the most effective adversarial directions lie in the low-to-medium frequency range (consistent with studies of adversarial perturbation spectra). The DCT basis orders directions by frequency, allowing SimBA to first try the most effective directions.

**Random SimBA:** Pick $q_t$ uniformly at random from $\{q_1, \ldots, q_n\}$ (random pixel-space basis). This requires no frequency knowledge but converges slower.

### 5.3 Query Complexity of SimBA

Each step requires **2 queries** (one for $x + \epsilon' q_t$ and one for $x - \epsilon' q_t$). After $T$ steps, SimBA has used $2T$ queries.

SimBA has no convergence guarantee for arbitrary target models, but empirically achieves:
- ~100–300 queries for $>90\%$ attack success on ResNet-50 at $\epsilon = 8/255$ (CIFAR-10).
- ~1000–5000 queries for ImageNet.

This is significantly more query-efficient than ZOO and comparable to NES.

---

## 6. Query Budget Analysis

### 6.1 Formal Query Budget

Let $Q(\text{attack}, \epsilon, \text{SR})$ be the number of queries needed to achieve success rate SR against a standard ResNet classifier at perturbation budget $\epsilon$.

**Theoretical lower bound:** Any score-based attack requires at least $\Omega(n / \epsilon^2)$ queries to achieve constant success rate on worst-case inputs (Theorem from Chen et al. 2020). For CIFAR-10 with $\epsilon = 8/255$: $\Omega(3072 / (8/255)^2) \approx 3 \times 10^6$ queries in the worst case. In practice, non-adversarial inputs are much easier.

### 6.2 Empirical Comparison Table

**Setting:** CIFAR-10, ResNet-18, $\epsilon = 8/255$, untargeted attack, 90% success rate target.

| Attack | Queries (median) | Queries (90th pct.) | Success Rate @1000q | Success Rate @10000q |
|--------|-----------------|---------------------|--------------------|--------------------|
| ZOO (random coord.) | 8,500 | 18,000 | 35% | 78% |
| ZOO (importance sampling) | 5,200 | 12,000 | 48% | 85% |
| NES ($n_s=50$, antithetic) | 2,100 | 6,000 | 62% | 91% |
| SPSA ($n_s=50$) | 2,300 | 6,500 | 60% | 90% |
| SimBA (DCT) | 800 | 2,500 | 80% | 97% |
| Square Attack | 500 | 1,800 | 84% | 98% |

Note: These numbers are order-of-magnitude estimates from the literature; exact values depend on implementation details and random seeds.

### 6.3 Why Square Attack is Most Query-Efficient

The Square Attack (Andriushchenko et al. 2020) is a score-based attack that samples random square-shaped perturbations rather than Gaussian vectors. Its efficiency comes from:

1. **Structured search space:** Square perturbations have spatial coherence (neighboring pixels change together), which aligns with how neural networks process images.
2. **Adaptive step size:** The square size decreases over iterations, providing a coarse-to-fine search.
3. **No gradient estimation:** Square Attack does not estimate gradients; it directly samples perturbations and accepts/rejects them based on loss improvement (a form of coordinate block descent).

**Query counting:** At each step, Square Attack samples 1 perturbation and queries the model once. After $T$ steps: $T$ queries.

---

## 7. Complete NES Attack Implementation (Annotated)

```python
import torch
import numpy as np

def nes_attack(model, x, y, epsilon=8/255, n_steps=300, n_samples=50,
               sigma=0.01, step_size=2/255, targeted=False, target=None):
    """
    NES (Natural Evolution Strategies) black-box adversarial attack.

    Parameters:
    -----------
    model : callable, takes batch of images [B, C, H, W], returns logits [B, K]
    x : torch.Tensor [C, H, W], original image in [0, 1]
    y : int, true class label
    epsilon : float, L-infinity perturbation budget
    n_steps : int, number of gradient steps
    n_samples : int, samples per gradient estimate (must be even for antithetic)
    sigma : float, NES smoothing parameter
    step_size : float, PGD step size
    targeted : bool, if True, attack to make model predict `target`
    target : int, target class (required if targeted=True)

    Returns:
    --------
    x_adv : torch.Tensor [C, H, W], adversarial example
    queries : int, total number of model queries used
    success : bool, whether the attack succeeded
    """
    assert n_samples % 2 == 0, "n_samples must be even for antithetic sampling"
    n_pairs = n_samples // 2

    x_adv = x.clone()
    queries = 0
    device = x.device

    for step in range(n_steps):
        # --- Gradient estimation via antithetic NES ---
        grad_estimate = torch.zeros_like(x_adv)  # [C, H, W]

        for _ in range(n_pairs):
            # Sample a random Gaussian perturbation
            delta = torch.randn_like(x_adv)  # [C, H, W]

            # Evaluate loss at x + sigma*delta
            x_plus = torch.clamp(x_adv + sigma * delta, 0, 1)
            with torch.no_grad():
                logits_plus = model(x_plus.unsqueeze(0))  # [1, K]
            queries += 1

            # Evaluate loss at x - sigma*delta
            x_minus = torch.clamp(x_adv - sigma * delta, 0, 1)
            with torch.no_grad():
                logits_minus = model(x_minus.unsqueeze(0))  # [1, K]
            queries += 1

            # Compute loss difference (antithetic)
            if targeted:
                # Targeted: minimize loss of target class (maximize logit of target)
                # Loss = -logit[target]; we want to maximize logit[target]
                loss_plus = -logits_plus[0, target].item()
                loss_minus = -logits_minus[0, target].item()
            else:
                # Untargeted: maximize loss of true class
                loss_plus = -logits_plus[0, y].item()   # CE loss approximation
                loss_minus = -logits_minus[0, y].item()

            # Antithetic gradient estimate accumulation
            # From equation (8): sum_i (L(x+sigma*delta) - L(x-sigma*delta)) * delta
            grad_estimate += (loss_plus - loss_minus) * delta

        # Normalize: divide by (n_pairs * sigma) from equation (8)
        grad_estimate = grad_estimate / (n_pairs * sigma)

        # --- PGD step with NES gradient ---
        x_adv = x_adv + step_size * torch.sign(grad_estimate)

        # Project onto L-inf ball around x
        x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
        # Project onto valid pixel range
        x_adv = torch.clamp(x_adv, 0, 1)

        # --- Early termination check ---
        with torch.no_grad():
            logits = model(x_adv.unsqueeze(0))
        queries += 1
        pred = logits.argmax(dim=1).item()

        if targeted and pred == target:
            return x_adv, queries, True
        elif not targeted and pred != y:
            return x_adv, queries, True

    # Check final result
    with torch.no_grad():
        logits = model(x_adv.unsqueeze(0))
    queries += 1
    pred = logits.argmax(dim=1).item()
    success = (targeted and pred == target) or (not targeted and pred != y)
    return x_adv, queries, success
```

**Key implementation details:**
1. The `n_pairs = n_samples // 2` ensures we always draw antithetic pairs.
2. The loss is computed as negative logit (not cross-entropy through softmax) to avoid softmax saturation gradient masking.
3. The early termination query counts the evaluation — total queries include estimation + evaluation.
4. For targeted attacks, we minimize the loss (maximize logit) of the target class, so we use negative logit.

---

## 8. Score-Based Attacks vs. Decision-Based Attacks

It is instructive to compare score-based attacks with the even-harder decision-based setting.

**Decision-based attacks:** The attacker observes only $C(x) = \arg\max_k p(x)_k$ — the hard class label. No probability scores are available. This is the threat model for HopSkipJump (Chen et al. 2019) and the Boundary Attack (Brendel et al. 2018).

In the decision-based setting:
- Gradient estimation requires $O(n)$ queries per step just to distinguish the sign of the gradient.
- HopSkipJump uses a gradient estimation strategy that requires $O(n \log n / \epsilon^2)$ queries for a target distortion of $\epsilon$.
- Typical query budgets: $10^4$–$10^5$ queries for CIFAR-10 at $\epsilon = 0.5$ L-2.

The comparison between score-based and decision-based attacks illustrates that each bit of information from the model significantly reduces the required query budget.

---

## 9. Practical Considerations

### 9.1 Which Loss to Use?

The loss function choice matters significantly for score-based attacks because gradients are estimated through a high-variance process. Recommendations:

- **Negative logit of the target/true class:** Most stable, avoids softmax saturation. Preferred for NES.
- **Cross-entropy:** Standard, but softmax can mask gradient signal. Works when probabilities are well-calibrated.
- **CW hinge loss:** Requires computing $\max_{i \neq y} p(x)_i$, which requires observing all logits. More informative but potentially noisier.

### 9.2 Handling APIs That Return Top-5 Probabilities

Many commercial APIs return only top-$k$ probabilities (e.g., $k=5$), not the full probability vector. This limits which loss functions can be computed. Adaptation:

- Use the top-1 probability as a proxy for the true-class logit.
- Use indicator functions: $\mathcal{L}(x) = \mathbf{1}[\hat{y}(x) \neq y]$ — this reduces to decision-based.
- Use rank as a pseudo-loss: minimize the rank of the true class among the top-$k$ results.

### 9.3 API Defenses Against Score-Based Attacks

Practical defenses targeted at score-based attackers:

1. **Rate limiting:** Limit to $N$ queries per IP/hour. Effectiveness depends on $N$ relative to the attack's query budget.
2. **Query detection:** Flag unusual query patterns (e.g., a sequence of similar images — the attacker's gradient estimation queries).
3. **Adding noise to probabilities:** Returning $p(x) + \mathcal{N}(0, \sigma^2 I)$ requires the attacker to use more samples per gradient estimate (more queries).
4. **Returning rounded probabilities:** Reducing the precision of returned probabilities (e.g., top-2 decimal places) limits gradient estimation accuracy.

**Effectiveness of noise:** If the API adds Gaussian noise $\mathcal{N}(0, \sigma_{\text{API}}^2)$ to the returned probability, the attacker's NES gradient estimator has higher variance. The attacker must increase $n_s$ to compensate:

$$n_s^{\text{noisy}} = n_s^{\text{clean}} \cdot \left(1 + \frac{\sigma_{\text{API}}^2}{\sigma_{\text{loss}}^2}\right)$$

where $\sigma_{\text{loss}}^2$ is the variance of the loss across the gradient-estimation samples.

---

## 10. Worked Example: NES on a 2D Classifier

**Setup:** 2D binary classifier $p_1(x) = \sigma(2x_1 + x_2)$ (probability of class 1). Original input $x_0 = [-0.5, -0.5]$ (class 0, since $p_1(x_0) = \sigma(-1.5) \approx 0.18$). Goal: untargeted attack to make class 1 win.

**True gradient (for comparison):** $\nabla_{x_1} (-p_1) = -2\sigma(1-\sigma) = -2 \times 0.18 \times 0.82 \approx -0.296$.

**NES gradient estimate** with $\sigma = 0.1$, $n_s = 4$ (2 antithetic pairs):

Sample 1: $\delta_1 = [0.42, -0.71]$ (random Gaussian).
- $x + \sigma \delta_1 = [-0.458, -0.571]$, $\mathcal{L}(x + \sigma\delta_1) = -p_1([-0.458, -0.571]) = -\sigma(-1.487) = -0.184$.
- $x - \sigma \delta_1 = [-0.542, -0.429]$, $\mathcal{L}(x - \sigma\delta_1) = -\sigma(-1.513) = -0.180$.
- Contribution: $(-0.184 - (-0.180)) \times [0.42, -0.71] / (2 \times 0.1) = -0.004 \times [0.42, -0.71] / 0.2 = [-0.0084, 0.0142]$.

Sample 2: $\delta_2 = [-0.88, 0.34]$.
- (Similar computation — omitted for brevity.)
- Contribution: approximately $[0.052, -0.020]$.

Sum and normalize: $\hat{g} \approx [-0.0084 + 0.052, 0.0142 - 0.020] / 1 = [0.044, -0.006]$.

**True gradient:** $\nabla_x (-p_1(x_0)) \approx [-0.296, -0.148]$.

The NES estimate $[0.044, -0.006]$ is noisy (only 4 samples). Sign: $\text{sign}(\hat{g}) = [+1, -1]$.

True sign: $[-1, -1]$. With 4 samples, the sign estimate for $x_1$ is *wrong* (estimated $+1$, true is $-1$). This is why NES needs many samples per gradient estimate.

With $n_s = 100$ antithetic samples, the estimator converges to the true gradient sign with high probability.

**PGD step (correct direction):** $x_1^{\text{adv}} = \text{clip}(x_0 + \alpha \cdot [-1, -1], x_0 - \epsilon, x_0 + \epsilon) = [-0.5 - \alpha, -0.5 - \alpha]$. Moving in direction $[-1, -1]$ increases $2x_1 + x_2$ by... wait: gradient of $p_1 = \sigma(2x_1 + x_2)$ is $[2\sigma(1-\sigma), \sigma(1-\sigma)]$ — positive in both directions. So increasing $x_1$ and $x_2$ increases $p_1$ (class 1 probability). The correct untargeted attack direction is $+[1, 1]$, not $[-1, -1]$.

This illustrates that with few samples, NES can get the gradient sign wrong and take a counterproductive step. The attack still converges eventually because subsequent correct steps outweigh incorrect ones in expectation.

---

## 11. Discussion Questions

1. **Query complexity:** Prove that the antithetic NES estimator (equation 8) is an unbiased estimator of $\nabla_x \mathcal{L}(x)$ when $\mathcal{L}$ is differentiable and $\delta_i \sim \mathcal{N}(0, I)$. (Hint: use the fact that $\mathbb{E}[\delta_i] = 0$ and $\mathbb{E}[\delta_i \delta_i^\top] = I$, and expand to first order in $\sigma$.)

2. **Variance bound:** The variance of the single-sample NES estimator $\hat{g}_1^{(i)} = (\mathcal{L}(x + \sigma\delta_1) \cdot \delta_1^{(i)})/(n_s \sigma)$ depends on $\mathbb{E}[\mathcal{L}^2]$. For a fixed-accuracy gradient estimate (say, we need $\text{Var}[\hat{g}] \leq \tau$), how does the required $n_s$ scale with the dimension $n$? With $\sigma$? Interpret your result: for what types of loss functions is NES most query-efficient?

3. **SimBA optimality:** Prove or disprove: SimBA with the optimal ordering of basis vectors (ordered by true gradient magnitude) requires at most half the queries of random-order SimBA, on average, to achieve the same attack success rate.

4. **API noise defense:** An API returns soft labels with additive Gaussian noise $\mathcal{N}(0, \sigma_{\text{noise}}^2 I)$. An attacker uses NES with antithetic sampling. Derive the optimal $n_s$ (samples per gradient estimate) as a function of $\sigma_{\text{noise}}$ and the desired gradient estimation error $\tau$. At what $\sigma_{\text{noise}}$ does the attack require more than 100,000 total queries for CIFAR-10?

5. **ZOO vs. NES in high dimensions:** For ImageNet (n=150,528), compute the expected number of queries for:
   (a) ZOO with random coordinate sampling to achieve an expected gradient error of $\tau = 0.01$ per coordinate.
   (b) NES with $n_s = 200$ antithetic samples to achieve the same.
   Which is more query-efficient in high dimensions, and why?

6. **Targeted score-based attacks:** Score-based attacks for *targeted* misclassification are significantly harder than untargeted, because the attacker must find a specific direction in the $K$-dimensional probability simplex. Quantify this: for a $K$-class uniform random classifier, what is the expected number of queries for NES to succeed at a targeted attack vs. untargeted? How does this scale with $K$?

7. **Loss function selection:** Suppose a classifier uses temperature scaling: $p_k(x) = e^{f_k(x)/T} / \sum_j e^{f_j(x)/T}$ with high temperature $T = 10$. This makes the output probabilities nearly uniform (low confidence). How does this affect:
   (a) The NES gradient estimate (variance, bias).
   (b) The expected query budget.
   (c) The recommended NES hyperparameters ($\sigma$, $n_s$).

---

## 12. Summary: Score-Based Attack Comparison

| Attack | Gradient Est. | Queries/Step | Strengths | Weaknesses |
|--------|--------------|-------------|-----------|------------|
| ZOO | Coordinate-wise FD | 2 per coord | Simple, principled | Slow in high dims |
| NES | Gaussian antithetic | $n_s$ | Low variance, parallelizable | Requires tuning $\sigma$, $n_s$ |
| SPSA | Rademacher antithetic | $n_s$ | Similar to NES, simple | Same as NES |
| SimBA | Orthogonal search | 2 | No gradient estimation needed | Order-dependent, no convergence guarantee |
| Square | Random block | 1 | Most query-efficient, no gradients | Stochastic, harder to analyze |

---

## 13. Further Reading

**Required:**
- Ilyas et al. (2018). "Black-box Adversarial Attacks with Limited Queries and Information." ICML. [NES attack]
- Chen et al. (2017). "ZOO: Zeroth Order Optimization Based Black-box Attacks to Deep Neural Networks." AISec. [ZOO attack]
- Guo et al. (2019). "Simple Black-box Adversarial Attacks." ICML. [SimBA]
- Andriushchenko et al. (2020). "Square Attack: a query-efficient black-box adversarial attack via random search." ECCV. [Square Attack]

**Background:**
- Wierstra et al. (2014). "Natural Evolution Strategies." JMLR. [NES foundations]
- Uesato et al. (2018). "Adversarial Risk and the Dangers of Evaluating Against Weak Attacks." ICML. [SPSA in adversarial context]
- Chen et al. (2020). "Hopskipjumpattack: A query-efficient decision-based attack." IEEE S&P. [Decision-based comparison]
- Brendel et al. (2018). "Decision-based adversarial attacks: Reliable attacks against black-box machine learning models." ICLR. [Boundary Attack]
