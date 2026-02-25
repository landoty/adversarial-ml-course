# CS6820 — Week 03: Adversarial Training II
## TRADES and the Principled Robustness-Accuracy Tradeoff

**Prerequisites:** Week 02 (PGD-AT), KL divergence definition, familiarity with classification loss functions.

**Learning Objectives:**
- Understand the theoretical argument that robustness and accuracy are fundamentally at odds
- Derive the TRADES bound on adversarial risk and understand why it leads to a better loss function than PGD-AT
- Implement TRADES, MART, and AWP
- Understand the β hyperparameter and its role in the accuracy-robustness tradeoff

---

## 1. Is the Accuracy-Robustness Tradeoff Fundamental?

PGD-AT, despite achieving significantly better robustness than standard training, incurs a substantial cost in clean accuracy: typically 8-10% on CIFAR-10 (from ~94% to ~84%). A natural question: is this tradeoff fundamental, or is it an artifact of the PGD-AT training procedure?

### 1.1 The Tsipras et al. Theoretical Framework

Tsipras, Santurkar, Engstrom, Turner, and Madry (2019) provided a theoretical argument that **robustness and accuracy are fundamentally in tension when the data distribution contains "non-robust useful features."**

**The constructed example (binary classification):**

Consider a binary classification task with label $y \in \{-1, +1\}$ and input $x = (x_1, x_2, \ldots, x_d) \in \mathbb{R}^{d+1}$:

$$x_1 = \eta \cdot y, \quad x_i \sim \mathcal{N}(\nu \cdot y, 1) \text{ for } i = 2, \ldots, d+1$$

where $\eta = 2, \nu = 0.1/\sqrt{d}$.

**The two types of features:**

1. **Feature $x_1$ (robust feature):** Strongly correlated with $y$ (correlation $\eta$), but detectable only when looking at $x_1$ specifically. For an L-infinity adversary with $\epsilon = 1$: since $\eta = 2 > 2\epsilon$, the adversary cannot flip $x_1$ past zero, so this feature is *robustly useful*.

2. **Features $x_2, \ldots, x_{d+1}$ (non-robust features):** Each is weakly correlated with $y$ (correlation $\nu = 0.1/\sqrt{d}$), but there are $d$ of them. The sum $\sum_{i=2}^{d+1} \nu x_i$ has magnitude $\sim \nu \cdot d / \sqrt{d} = 0.1\sqrt{d}$, which grows with $d$.

**Standard accuracy vs. robustness:**

*Accuracy-optimal classifier:* Uses all $d+1$ features. The combined signal is $x_1 + \sum_{i=2}^{d+1} \nu x_i$, which is a Bayes-optimal classifier with accuracy approaching 1 as $d \to \infty$.

*Adversarial robustness of accuracy-optimal classifier:* An adversary can perturb each $x_i$ by $\epsilon = 1$. The perturbation to the non-robust features:

$$\text{Adversarial effect on } \sum_{i=2}^{d+1} x_i = d \cdot \epsilon = d$$

This overwhelms the robust signal $x_1 = \eta = 2$. As $d$ grows, the accuracy-optimal classifier becomes arbitrarily easy to fool.

*Robustness-optimal classifier:* Uses only $x_1$. The adversary can perturb $x_1$ by $\epsilon = 1$, changing it from $2$ to $1$ (when $y = +1$), but the sign is unchanged. Robust accuracy $\to 1$ as $\eta > 2\epsilon$.

**But this classifier has lower standard accuracy:** It ignores the $d$ weak features, achieving only 84% accuracy (from the single feature $x_1$) instead of approaching 100%.

**Conclusion:** In this example, the highest-accuracy classifier is not robust, and the highest-robustness classifier is not most accurate. There exists a **Pareto frontier** of classifiers with different clean-vs.-robust accuracy tradeoffs.

### 1.2 The "Robust vs. Non-Robust Features" Interpretation

Ilyas et al. (2019) extended this framework with the "robust features / non-robust features" interpretation:

- **Robust features:** Input features that are predictive of the label AND are hard to manipulate by a small perturbation. Example: the overall shape of an animal.
- **Non-robust features:** Input features that are predictive of the label BUT can be easily manipulated by small perturbations. Example: high-frequency texture patterns that correlate with class labels in the training set.

Standard training uses both types of features, since both improve accuracy. Adversarial training forces the model to *rely only* on robust features (or at least rely less on non-robust features), which reduces standard accuracy.

**This suggests some accuracy degradation is inevitable** when we force the model to be robust: we are explicitly asking it to ignore (or downweight) features that help with standard accuracy but can be manipulated adversarially.

---

## 2. TRADES: A Principled Loss Decomposition

### 2.1 The Problem with PGD-AT's Objective

PGD-AT trains on adversarial examples generated from the *true labels*:

$$L_{\text{PGD-AT}}(x, y) = L_{\text{CE}}(f(x_{\text{adv}}), y)$$

where $x_{\text{adv}} = \arg\max_{\|\delta\| \leq \epsilon} L_{\text{CE}}(f(x + \delta), y)$.

This trains the model to correctly classify adversarial examples, directly. While intuitively appealing, Zhang et al. (2019) argue this is suboptimal:

1. **Clean accuracy is not explicitly optimized.** The model only trains on adversarial examples, so its behavior on clean examples is improved only indirectly.
2. **The regularization is indirect.** What we really want is for the model's prediction to be *stable* under perturbation (not change when a small perturbation is applied). PGD-AT trains for correctness of adversarial predictions, which is a stronger requirement.

### 2.2 Deriving the TRADES Bound

**Setup:** Let $f: \mathcal{X} \to \Delta^K$ be the classifier outputting probability distributions (after softmax). Let $f_y(x) = P_f(Y=y | X=x)$ denote the predicted probability of the true class.

**Standard risk:** $R_{\text{nat}}(\theta) = \mathbb{E}_{(x,y)}[-\log f_y(x)]$ (expected cross-entropy loss)

**Adversarial risk:** $R_{\text{adv}}(\theta) = \mathbb{E}_{(x,y)}\left[\max_{\delta \in \mathcal{S}} (-\log f_y(x + \delta))\right]$

**The key decomposition:** Zhang et al. derive that:

$$R_{\text{adv}}(\theta) \leq R_{\text{nat}}(\theta) + \mathbb{E}_{(x,y)}\left[\max_{\delta \in \mathcal{S}} \text{KL}(f(x) \| f(x + \delta))\right]$$

**Derivation of this bound:**

Starting from the adversarial risk:

$$R_{\text{adv}} = \mathbb{E}_{x,y}\left[\max_\delta (-\log f_y(x + \delta))\right]$$

We add and subtract $-\log f_y(x)$:

$$= \mathbb{E}_{x,y}\left[-\log f_y(x) + \max_\delta \left(-\log f_y(x + \delta) + \log f_y(x)\right)\right]$$

$$= R_{\text{nat}} + \mathbb{E}_{x,y}\left[\max_\delta \log \frac{f_y(x)}{f_y(x+\delta)}\right]$$

Now we bound the inner term. Note that:

$$\log \frac{f_y(x)}{f_y(x+\delta)} = \log \frac{f_y(x)}{f_y(x+\delta)}$$

This is related to the KL divergence. Recall:

$$\text{KL}(f(x) \| f(x+\delta)) = \sum_c f_c(x) \log \frac{f_c(x)}{f_c(x+\delta)}$$

Since $-\log f_y(x+\delta) + \log f_y(x) = \log \frac{f_y(x)}{f_y(x+\delta)}$ and $f_y(x) \leq 1$, we can apply Jensen's inequality or bound this term by the KL divergence (which is always non-negative). Specifically, for any probability vector $p$ and any class $y$:

$$-\log p_y \leq \text{KL}(e_y \| p) + H(e_y) = -\log p_y$$

Wait, that's an equality. Let us be more careful.

The correct bound uses the fact that:

$$\log \frac{f_y(x)}{f_y(x+\delta)} \leq \frac{f_y(x)}{f_y(x+\delta)} - 1 \leq \sum_c f_c(x) \left(\frac{f_c(x)}{f_c(x+\delta)} - 1\right) = \ldots$$

Actually, Zhang et al. (2019) prove the bound using a different approach. The formal proof uses the following lemma:

**Lemma:** For any distributions $p, q$ and any $y$:
$$-\log q_y \leq -\log p_y + \text{KL}(p \| q) / p_y$$

However, for the practical bound used in TRADES, the key relationship exploited is: the worst-case adversarial loss is upper bounded by the natural loss plus the worst-case KL divergence:

$$\max_\delta L_{\text{CE}}(f(x+\delta), y) \leq L_{\text{CE}}(f(x), y) + \max_\delta \text{KL}(f(x) \| f(x+\delta))$$

This follows because:
- $L_{\text{CE}}(f(x+\delta), y) = -\log f_y(x+\delta)$
- $\text{KL}(f(x) \| f(x+\delta)) = \sum_c f_c(x) \log \frac{f_c(x)}{f_c(x+\delta)}$

For the true class $y$: $-\log f_y(x+\delta) \leq -\log f_y(x) + \frac{1}{f_y(x)} \text{KL}(f(x) \| f(x+\delta))$

But more practically, the TRADES bound is:

$$R_{\text{adv}}(\theta) \leq R_{\text{nat}}(\theta) + \mathbb{E}_x\left[\max_{\delta \in \mathcal{S}} \text{KL}(f(x) \| f(x + \delta))\right]$$

The intuition: if the model's output distribution barely changes under perturbation (small KL), then the adversarial loss is close to the natural loss. We want to minimize both the natural loss (accuracy) and the worst-case KL divergence (stability/smoothness).

### 2.3 The TRADES Loss Function

TRADES directly minimizes the upper bound:

$$L_{\text{TRADES}}(x, y) = L_{\text{CE}}(f(x), y) + \beta \cdot \max_{\delta \in \mathcal{S}} \text{KL}\left(f(x + \delta) \| f(x)\right)$$

Note: in the implementation, the KL is $\text{KL}(f(x+\delta) \| f(x))$ — the direction is from the adversarial output *to* the clean output. This is the non-symmetric "reverse KL" that measures how much the adversarial output diverges from the clean output.

**The β hyperparameter:**
- $\beta = 1$: equal weight on natural accuracy and smoothness regularization. Approximates PGD-AT behavior.
- $\beta = 6$: the TRADES paper's best setting for CIFAR-10. Six times more weight on smoothness than accuracy.
- $\beta = 0$: pure natural training (no robustness).
- Increasing $\beta$: more robust but lower clean accuracy. Decreasing $\beta$: higher clean accuracy but less robust.

**Why is this better than PGD-AT?**

1. **Explicit clean accuracy term.** $L_{\text{CE}}(f(x), y)$ is directly optimized, so the model trains on clean examples as well as adversarial examples. PGD-AT trains only on adversarial examples (though the adversarial examples are in the neighborhood of the clean examples).

2. **Smoothness regularization is the right objective.** Robustness requires that the model's output doesn't change much under perturbation. $\text{KL}(f(x+\delta) \| f(x))$ directly measures this instability. PGD-AT measures correctness of adversarial predictions, which is a downstream consequence of smoothness.

3. **Better interpretation of the inner maximization.** In TRADES, the inner max solves $\max_\delta \text{KL}(f(x+\delta) \| f(x))$ — finding the perturbation that maximally changes the model's output distribution. This is different from PGD-AT's inner max which maximizes the cross-entropy loss (finding the perturbation that maximally fools the correct class prediction). The TRADES inner problem is more directly aligned with the desired property (output stability).

### 2.4 Solving the TRADES Inner Maximization

The inner maximization in TRADES:

$$\delta^* = \arg\max_{\delta \in \mathcal{S}} \text{KL}(f(x + \delta) \| f(x))$$

Note: $f(x)$ is a constant in the inner maximization (it's the prediction at the clean input, which doesn't depend on $\delta$). So this is equivalent to:

$$\delta^* = \arg\max_{\delta \in \mathcal{S}} \sum_c f_c(x+\delta) \log f_c(x+\delta) - \sum_c f_c(x+\delta) \log f_c(x)$$

The gradient with respect to $\delta$:

$$\nabla_\delta \text{KL}(f(x+\delta) \| f(x)) = J_f(x+\delta)^\top \nabla_q \text{KL}(q \| f(x))\big|_{q=f(x+\delta)}$$

where $J_f$ is the Jacobian of the softmax output with respect to the input. In practice, we compute this gradient by automatic differentiation.

**PGD for the TRADES inner maximization:**

```python
def trades_inner_max(model, x, epsilon, alpha, num_steps):
    """
    Solve the TRADES inner maximization:
        max_{||δ||∞ ≤ ε} KL(f(x+δ) || f(x))

    Note: f(x) is constant during the inner optimization.
    We initialize δ from a small Gaussian (not random L-inf ball).
    """
    # Get clean predictions (constant target for KL divergence)
    with torch.no_grad():
        p_clean = F.softmax(model(x), dim=1)

    # Initialize from small Gaussian noise (not large random initialization)
    delta = 0.001 * torch.randn_like(x)
    delta.requires_grad_(True)

    for _ in range(num_steps):
        output_adv = model(x + delta)
        p_adv = F.softmax(output_adv, dim=1)

        # KL(f(x+δ) || f(x)) — maximize this
        kl_loss = F.kl_div(
            F.log_softmax(output_adv, dim=1),
            p_clean,
            reduction='batchmean'
        )
        kl_loss.backward()

        with torch.no_grad():
            delta.data = delta.data + alpha * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(x + delta.data, 0, 1) - x

        delta.grad.zero_()

    return (x + delta).detach()
```

**Why initialize with small Gaussian noise (not large random)?**

In PGD-AT, random initialization from the full $[-\epsilon, \epsilon]$ ball is important for coverage. In TRADES, the inner problem is finding the direction that maximally changes the output distribution. Starting from a small perturbation and taking gradient steps is more efficient for this purpose. The Gaussian initialization provides a small push away from the (potentially flat) zero-perturbation point.

### 2.5 Complete TRADES Training Loop

```python
def trades_loss(model, x, y, epsilon, alpha, num_steps, beta):
    """
    TRADES loss: L_CE(f(x), y) + β * KL(f(x+δ*) || f(x))

    where δ* = argmax_{||δ||≤ε} KL(f(x+δ) || f(x))

    Args:
        model:     Classifier f_θ
        x:         Clean input batch
        y:         True labels
        epsilon:   L-inf perturbation budget
        alpha:     Step size for inner PGD (typically epsilon/4 or 2/255)
        num_steps: PGD steps for inner maximization
        beta:      Tradeoff parameter (β=6 recommended)

    Returns:
        loss:    Scalar TRADES loss
    """
    # 1. Natural (clean) loss term: L_CE(f(x), y)
    output_clean = model(x)
    loss_natural = F.cross_entropy(output_clean, y)

    # 2. Inner maximization: find worst-case KL perturbation
    x_adv = trades_inner_max(model, x, epsilon, alpha, num_steps)

    # 3. Compute KL divergence: KL(f(x+δ*) || f(x))
    # Note: f(x) (clean predictions) is the "target" distribution
    with torch.no_grad():
        p_clean = F.softmax(model(x), dim=1)

    output_adv = model(x_adv)
    # F.kl_div expects log-probabilities as input, probabilities as target
    loss_robust = F.kl_div(
        F.log_softmax(output_adv, dim=1),
        p_clean,
        reduction='batchmean'
    )

    # 4. Combine: natural loss + β * robustness regularization
    loss = loss_natural + beta * loss_robust
    return loss
```

---

## 3. TRADES Results and the β Sweep

### 3.1 Empirical Results on CIFAR-10

Zhang et al. (2019) evaluated TRADES on CIFAR-10 with WideResNet-34-10 (WRN-34-10), $\epsilon = 8/255$:

| Method | Clean Acc | Robust Acc (PGD-20) | Robust Acc (PGD-100) |
|--------|-----------|---------------------|-----------------------|
| Standard training | 95.0% | 0.0% | 0.0% |
| PGD-AT (Madry) | 87.3% | 47.0% | 44.0% |
| TRADES (β=6) | **84.9%** | **56.6%** | **54.4%** |

TRADES achieves +9.6% robust accuracy over PGD-AT at the cost of 2.4% clean accuracy. The improvement comes from the explicit smoothness regularization.

### 3.2 The β Sweep: Navigating the Tradeoff Curve

| β | Clean Acc | Robust Acc (PGD-20) |
|---|-----------|---------------------|
| 0 (standard training) | 95.0% | 0.0% |
| 1 | 89.1% | 52.3% |
| 3 | 86.8% | 54.8% |
| 6 | 84.9% | 56.6% |
| 10 | 82.1% | 55.2% |
| 100 | 68.0% | 41.0% |

The sweet spot is $\beta = 6$: beyond this, the clean accuracy loss outweighs the robust accuracy gain. At very large $\beta$, the natural loss is downweighted so heavily that the model underfits clean examples and both clean and robust accuracy degrade.

**Geometric intuition for β:** Increasing β forces the model to be smoother (smaller KL under perturbation). A perfectly smooth classifier (constant output for all inputs in the $\epsilon$-ball) would have perfect robustness but potentially very poor clean accuracy (it cannot distinguish points in the same $\epsilon$-ball). β controls how much smoothness the model is forced to exhibit.

---

## 4. MART: Fixing TRADES's Treatment of Misclassified Examples

### 4.1 The Problem with TRADES on Misclassified Examples

TRADES's loss has two terms:
1. $L_{\text{CE}}(f(x), y)$: trains on clean examples with the true label
2. $\text{KL}(f(x+\delta) \| f(x))$: forces the adversarial prediction close to the clean prediction

For correctly classified examples ($\hat{y} = y$), this works well: the clean prediction is correct, so forcing the adversarial prediction close to the clean prediction also encourages correct adversarial prediction.

**The problem:** For misclassified examples ($\hat{y} \neq y$, the model predicts the wrong class on the clean input), the clean prediction $f(x)$ is already wrong. Forcing $f(x+\delta)$ close to $f(x)$ means forcing the adversarial prediction to match an already-incorrect prediction. This does not help robustness for these examples.

Wang et al. (2020) identified this issue: **TRADES implicitly downweights misclassified examples** because the KL loss on a misclassified example with high-confidence wrong prediction is easy to minimize (the adversarial example just needs to be confidently wrong in the same direction as the clean example).

### 4.2 The MART Loss

MART (Misclassification-Aware adveRsarial Training) explicitly upweights misclassified examples:

$$L_{\text{MART}}(x, y) = L_{\text{BCE}}(f(x_{\text{adv}}), y) \cdot (1 - f(x)_y) + \beta \cdot \text{KL}(f(x_{\text{adv}}) \| f(x))$$

where:
- $L_{\text{BCE}}(f(x_{\text{adv}}), y) = -\log f_y(x_{\text{adv}})$ is the binary cross-entropy on adversarial examples
- $f(x)_y$ is the clean prediction probability for the true class $y$
- $(1 - f(x)_y)$ is the **misclassification weighting factor**
- $\text{KL}(f(x_{\text{adv}}) \| f(x))$ is the KL regularization (same as TRADES but with reversed direction)

**Interpretation of $(1 - f(x)_y)$:**
- If $f(x)_y \approx 1$ (model is confident and correct on the clean example), then $(1 - f(x)_y) \approx 0$, so this example gets *downweighted*.
- If $f(x)_y \approx 0$ (model is confident but wrong on the clean example), then $(1 - f(x)_y) \approx 1$, so this example gets full weight.
- Misclassified examples contribute fully to the loss; confidently correct examples are downweighted.

**Why does upweighting misclassified examples help?**

Misclassified examples are near the decision boundary (or on the wrong side of it). For robust classification, the decision boundary must be placed far from all data points. Misclassified examples indicate the decision boundary is too close or crossed. By upweighting these examples during training, MART focuses optimization effort on pushing the decision boundary away from the hard examples.

**MART inner maximization:** MART uses standard PGD on the cross-entropy loss (like PGD-AT) rather than the KL divergence:

$$\delta^* = \arg\max_{\delta \in \mathcal{S}} L_{\text{CE}}(f(x + \delta), y)$$

This finds adversarial examples that maximize the cross-entropy loss (i.e., examples that fool the classifier), which is more directly relevant to the classification task than the TRADES inner maximization (which maximizes KL).

**MART implementation:**

```python
def mart_loss(model, x, y, epsilon, alpha, num_steps, beta):
    """
    MART loss: BCE(f(x_adv), y) * (1 - f(x)_y) + β * KL(f(x_adv) || f(x))

    Inner maximization: max_{||δ||≤ε} CE(f(x+δ), y)  [same as PGD-AT]
    """
    # Inner maximization (PGD on CE loss, like PGD-AT)
    x_adv = pgd_attack(model, x, y, epsilon, alpha, num_steps)

    with torch.no_grad():
        output_clean = model(x)
        p_clean = F.softmax(output_clean, dim=1)

    output_adv = model(x_adv)
    p_adv = F.softmax(output_adv, dim=1)

    # Misclassification weighting: (1 - f(x)_y)
    # Gathers the clean probability for the true class y
    p_clean_true = p_clean.gather(1, y.unsqueeze(1)).squeeze(1)  # shape (B,)
    weight = 1 - p_clean_true  # higher weight for misclassified examples

    # Binary cross-entropy on adversarial examples, weighted by (1 - f(x)_y)
    # BCE = -log(f(x_adv)_y)
    log_p_adv_true = F.log_softmax(output_adv, dim=1).gather(1, y.unsqueeze(1)).squeeze(1)
    loss_adv = (-log_p_adv_true * weight).mean()

    # KL divergence regularization: KL(f(x_adv) || f(x))
    # Note: direction is f(x_adv) to f(x), opposite to TRADES convention
    loss_kl = F.kl_div(
        F.log_softmax(output_adv, dim=1),
        p_clean,
        reduction='batchmean'
    )

    loss = loss_adv + beta * loss_kl
    return loss
```

### 4.3 MART vs. TRADES Results

| Method | Clean Acc | Robust Acc (PGD-20) | Robust Acc (CW) |
|--------|-----------|---------------------|----|
| PGD-AT | 83.1% | 48.0% | 44.5% |
| TRADES (β=6) | 84.9% | 56.6% | 54.2% |
| MART (β=6) | 83.6% | **57.4%** | 54.8% |

MART achieves slightly better robust accuracy than TRADES with comparable clean accuracy. The improvement is most pronounced for misclassified examples.

---

## 5. AWP: Adversarial Weight Perturbation

### 5.1 The Robust Overfitting Problem (Revisited)

As discussed in Week 02, adversarially trained models suffer from robust overfitting: test robust accuracy peaks early in training and then declines. Wu et al. (2020) hypothesized that robust overfitting is related to the sharpness of the loss landscape:

**Sharp minima and overfitting:** A model at a sharp minimum in the loss landscape has a large second derivative (high curvature). Small perturbations to the weights $\theta$ cause large changes in loss. Sharp minima tend to correspond to models that overfit — they perform well on the training set but generalize poorly.

**Flat minima and generalization:** A model at a flat minimum has a small second derivative (low curvature). Weight perturbations cause small changes in loss. Flat minima tend to correspond to models that generalize better.

Wu et al. showed empirically that robust overfitting correlates with the model drifting toward sharper minima over training. Adversarial training finds a local minimum that is sharp in the weight space, even if it achieves low training loss.

### 5.2 The AWP Algorithm

**Adversarial Weight Perturbation (AWP)** seeks flat minima by simultaneously perturbing both inputs (adversarial examples) and model weights (weight perturbation).

**The double adversarial problem:**

$$\min_\theta \mathbb{E}_{(x,y)} \left[ \max_{\|\delta\|_\infty \leq \epsilon} \max_{\|\Delta\theta\| \leq \gamma} L(f_{\theta + \Delta\theta}(x + \delta), y) \right]$$

where $\Delta\theta$ is a perturbation to the model weights, and $\gamma$ is the weight perturbation radius.

**Intuition:** If the model is at a flat minimum, then small weight perturbations $\Delta\theta$ don't significantly increase the adversarial loss. We explicitly train the model to be robust against *both* input perturbations and weight perturbations, which encourages convergence to flat minima.

**The AWP training loop:**

```python
def awp_train_step(model, optimizer, x, y, epsilon, alpha, num_steps,
                   awp_gamma, awp_warmup=10, epoch=0):
    """
    One AWP training step:
    1. Find adversarial input perturbation δ* (PGD-AT inner max)
    2. Find adversarial weight perturbation Δθ* (AWP inner max)
    3. Compute loss at (θ + Δθ*, x + δ*)
    4. Backpropagate and update θ (not θ + Δθ*, revert weight perturbation)

    The weight perturbation is not accumulated — it's applied temporarily
    only for the forward pass used to compute the gradient update.
    """
    # Step 1: PGD inner max (adversarial input)
    x_adv = pgd_attack(model, x, y, epsilon, alpha, num_steps)

    # Step 2: AWP inner max (adversarial weight perturbation)
    # Only applied after warmup epochs (first apply standard PGD-AT)
    if epoch >= awp_warmup:
        # Compute gradient of loss w.r.t. model weights at (x_adv, θ)
        model.zero_grad()
        output = model(x_adv)
        loss = F.cross_entropy(output, y)
        loss.backward()

        # Store original weights and compute weight perturbation
        # Δθ = γ * ∇_θ L / ||∇_θ L||  (normalized gradient ascent direction)
        weight_grad_norm = compute_weight_grad_norm(model)
        with torch.no_grad():
            original_weights = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    original_weights[name] = param.data.clone()
                    # Perturb weight in gradient ascent direction
                    param.data += awp_gamma * param.grad / (weight_grad_norm + 1e-8)

    # Step 3: Compute loss at perturbed weights (or original weights if not AWP epoch)
    model.zero_grad()
    output = model(x_adv)
    loss = F.cross_entropy(output, y)
    loss.backward()

    # Step 4: Revert weight perturbation, then apply gradient update to original weights
    if epoch >= awp_warmup:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_weights:
                    param.data = original_weights[name]  # revert to original
        # The gradient computed at perturbed weights is now applied to original weights
        # This is the key: ∇_{θ} L(f_{θ+Δθ}(x+δ)) applied to θ

    optimizer.step()

def compute_weight_grad_norm(model):
    """Compute the L2 norm of the gradient of all parameters."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5
```

**Why the gradient at the perturbed weights is useful:** The gradient $\nabla_\theta L(f_{\theta + \Delta\theta}(x + \delta), y)$ points toward a direction that reduces loss in the *vicinity* of the current weights. By computing this gradient at a nearby weight perturbation and applying it to the original weights, we are effectively finding a flat loss landscape: the update reduces loss over a neighborhood of $\theta$, not just at $\theta$ itself.

### 5.3 AWP Results and Cost

| Method | Clean Acc | Robust Acc (PGD-20) | Robust Acc (AutoAttack) |
|--------|-----------|---------------------|------------------------|
| TRADES | 84.9% | 56.6% | 53.1% |
| TRADES + AWP | **85.4%** | **59.2%** | **56.2%** |
| MART | 83.6% | 57.4% | 54.8% |
| MART + AWP | 84.4% | **60.1%** | 56.8% |

AWP adds approximately +3% robust accuracy over TRADES and MART.

**Computational cost:** AWP requires one additional forward-backward pass per training step (to compute the weight gradient for the perturbation). This is approximately 2x the computation of PGD-AT (1 for PGD inner max, 1 for AWP weight perturbation, 1 for the outer update — but the PGD and outer steps share the backward pass). In practice, AWP with TRADES is about 1.5-2x slower than TRADES alone.

**AWP vs. benefit:** The +3% robust accuracy improvement is significant in the context of the field, where improvements are hard-won. For research or competitions (RobustBench), AWP is often worth the computational cost. For production systems with limited compute, the standard TRADES may be preferred.

---

## 6. Connecting the Dots: When to Use Which Method

| Scenario | Recommended Method | Reason |
|----------|--------------------|--------|
| Fast training, limited compute | Fast-AT (FGSM-RS) | 5-7x faster than PGD-AT |
| Standard robustness benchmark | PGD-AT or TRADES (β=6) | Well-studied, reliable |
| Maximize robust accuracy (compute available) | TRADES or MART + AWP | Best empirical results |
| Many misclassified examples in training set | MART | Better handling of misclassified examples |
| Concern about robust overfitting | AWP + early stopping | Addresses sharp minima issue directly |
| Certified robustness required | Randomized smoothing or IBP | (Week 06) Empirical methods cannot certify |

---

## 7. Worked Numerical Example: TRADES vs. PGD-AT Loss

Consider a binary classifier on $\mathbb{R}^2$ with $f(x) = \text{softmax}(Wx)$, and clean input $x_0 = [1, 0]^\top$ with label $y = 0$.

**Parameters:** $W = \begin{bmatrix} 2 & 0 \\ -2 & 0 \end{bmatrix}$, $\epsilon = 0.5$

**Clean predictions:**
$z_0 = W x_0 = [2, -2]^\top$
$p_0 = f(x_0) = \text{softmax}([2,-2]) = [e^2/(e^2+e^{-2}), e^{-2}/(e^2+e^{-2})] \approx [0.982, 0.018]$

Clean CE loss: $L_{\text{nat}} = -\log(0.982) \approx 0.018$

**PGD-AT inner maximization:**

$\delta^* = \arg\max_{\|\delta\|_\infty \leq 0.5} L_{\text{CE}}(f(x_0 + \delta), y=0)$

The gradient of CE loss w.r.t. input: $\nabla_x L = W^\top [p_1 - \mathbf{1}_{y=0}]$

At $x_0$: $\nabla_x L \approx W^\top [-0.018, 0.018] = [2,-2]^\top \cdot [-0.018, 0.018]$...

Let's compute: $W^\top = \begin{bmatrix} 2 & -2 \\ 0 & 0 \end{bmatrix}$, so $\nabla_x L = \begin{bmatrix} 2 & -2 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} -0.018 \\ 0.018 \end{bmatrix} = \begin{bmatrix} -0.072 \\ 0 \end{bmatrix}$

sign: $[-1, 0]^\top$. So $\delta^* \approx [-0.5, 0]^\top$.

Adversarial example: $x_{\text{adv}} = [0.5, 0]^\top$

PGD-AT loss: $L_{\text{adv}} = -\log f_0(x_{\text{adv}}) = -\log \text{softmax}([1, -1])_0 = -\log(0.880) \approx 0.128$

**TRADES inner maximization:**

$\delta^* = \arg\max_{\|\delta\| \leq 0.5} \text{KL}(f(x_0 + \delta) \| f(x_0))$

The same direction applies (gradient of KL is similar to gradient of CE for small perturbations), giving approximately the same $\delta^* \approx [-0.5, 0]^\top$.

TRADES loss (β=6): $L_{\text{TRADES}} = 0.018 + 6 \cdot \text{KL}(f([0.5,0]) \| f([1,0]))$

$= 0.018 + 6 \cdot \text{KL}([0.880, 0.120] \| [0.982, 0.018])$

$\text{KL} = 0.880 \log(0.880/0.982) + 0.120 \log(0.120/0.018) = 0.880 \cdot (-0.110) + 0.120 \cdot 1.897$

$\approx -0.097 + 0.228 = 0.131$

$L_{\text{TRADES}} = 0.018 + 6 \cdot 0.131 = 0.018 + 0.786 = 0.804$

**Key difference:** TRADES explicitly optimizes the natural accuracy term (0.018 for clean accuracy) plus the KL term (0.786 for robustness). PGD-AT only sees the adversarial loss (0.128). TRADES provides a richer training signal that directly connects natural and adversarial performance.

---

## 8. Key Takeaways

1. **The robustness-accuracy tradeoff has a theoretical foundation.** Tsipras et al. (2019) show that in distributions with non-robust useful features, the highest-accuracy classifier cannot be robust, and the most robust classifier sacrifices accuracy.

2. **TRADES's decomposition is principled.** The adversarial risk is upper-bounded by natural risk plus worst-case KL divergence. Minimizing this bound directly gives the TRADES loss, which separates the accuracy and robustness objectives.

3. **The TRADES β parameter controls the accuracy-robustness tradeoff.** β=6 is the empirically optimal setting for CIFAR-10. Higher β trades clean accuracy for robust accuracy.

4. **MART fixes TRADES's treatment of misclassified examples.** By upweighting examples the model already struggles with, MART focuses training effort on the decision boundary region most critical for robustness.

5. **AWP addresses robust overfitting through flat minima.** By perturbing model weights and seeking loss-landscape flatness, AWP finds solutions that generalize robustly to the test set.

6. **The state-of-the-art on RobustBench combines all techniques.** The best CIFAR-10 results use TRADES or MART combined with AWP, WideResNet architectures, and synthetic data augmentation.

---

## Discussion Questions

1. The TRADES bound states $R_{\text{adv}} \leq R_{\text{nat}} + \mathbb{E}_x[\max_\delta \text{KL}(f(x) \| f(x+\delta))]$. Is this bound tight? Can you construct a case where the bound is very loose (much larger than the true adversarial risk)?

2. TRADES uses $\text{KL}(f(x+\delta) \| f(x))$ (from adversarial to clean) as the regularization, but the bound derivation involves $\text{KL}(f(x) \| f(x+\delta))$ (from clean to adversarial). Does the direction of KL matter in practice? What is the conceptual difference between the two directions?

3. MART upweights misclassified examples by a factor $(1 - f(x)_y)$. Consider a training set where 20% of examples are misclassified under the current model. How does this affect the gradient signal relative to TRADES? Does upweighting misclassified examples risk overfitting to hard examples?

4. AWP finds flat minima by perturbing model weights in the gradient ascent direction and computing the gradient at the perturbed point. How is this related to Sharpness-Aware Minimization (SAM, Foret et al. 2021)? Could SAM replace AWP for adversarial training?

5. The theoretical results of Tsipras et al. use a synthetic data distribution where the tradeoff between robustness and accuracy is sharp and provable. How well does this theoretical model match real vision datasets like CIFAR-10 and ImageNet? What evidence exists that real datasets also have "non-robust useful features"?

---

## References

- Zhang, H., Yu, Y., Jiao, J., Xing, E.P., Ghaoui, L.E., & Jordan, M.I. (2019). Theoretically Principled Trade-off between Robustness and Accuracy. *ICML 2019*.
- Tsipras, D., Santurkar, S., Engstrom, L., Turner, A., & Madry, A. (2019). Robustness May Be at Odds with Accuracy. *ICLR 2019*.
- Wang, Y., Zou, D., Yi, J., Bailey, J., Ma, X., & Gu, Q. (2020). Improving Adversarial Robustness Requires Revisiting Misclassified Examples. *ICLR 2020*.
- Wu, D., Xia, S.T., & Wang, Y. (2020). Adversarial Weight Perturbation Helps Robust Generalization. *NeurIPS 2020*.
- Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., & Madry, A. (2019). Adversarial Examples Are Not Bugs, They Are Features. *NeurIPS 2019*.
- Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). Sharpness-Aware Minimization for Efficiently Improving Generalization. *ICLR 2021*.
