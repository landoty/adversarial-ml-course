# CS6820 — Week 06: Certified Defenses II
## Randomized Smoothing — Theory and Guarantees

**Prerequisites:** Weeks 01-03 (empirical defenses), probability theory (Normal distribution, hypothesis testing), basic calculus.

**Learning Objectives:**
- Understand why certified defenses are qualitatively different from empirical defenses
- Derive the main certification theorem (Cohen et al. 2019) from the Neyman-Pearson lemma
- Implement the Monte Carlo certification algorithm with Clopper-Pearson confidence intervals
- Understand the limitations of randomized smoothing and where it fits relative to other certified defenses

---

## 1. Motivation: Why Certification Matters

### 1.1 The Fundamental Limitation of Empirical Robustness

Empirical robustness evaluation — testing a model against PGD, AutoAttack, C&W — can only tell us that specific attacks fail. It cannot tell us that all attacks fail. As established in Week 01, adaptive attackers can design new attacks specifically tailored to any defense. Even if AutoAttack fails against a defense, a sufficiently motivated adversary might find a novel attack that succeeds.

This is the fundamental limitation of empirical robustness: **negative results (attacks failing) do not imply security.**

**Certified robustness** breaks out of this limitation by providing a mathematical proof: for every adversarial example within the threat model, the certified classifier's prediction is guaranteed to be correct. No adaptive attack within the threat model can fool a certified classifier at certified inputs.

Formally, a certified classifier $g$ provides a certificate: for input $x$ with label $y$, there exists a certified radius $R(x)$ such that:

$$\forall \delta : \|\delta\|_2 \leq R(x) \implies g(x + \delta) = y$$

This is an unconditional guarantee — it holds for all possible attacks, including adaptive attacks not yet invented.

### 1.2 The Tradeoff: What Certification Costs

Certified defenses come with costs:
1. **Reduced clean accuracy:** Certification requires the classifier to be "smoother," which often means lower accuracy on clean examples.
2. **Certification radius limitations:** The maximum achievable certification radius is bounded by the method.
3. **Computational cost:** Certification requires additional computation at test time (e.g., sampling thousands of noise vectors).
4. **Norm restriction:** Most certified methods certify only one norm (randomized smoothing certifies L2, IBP certifies L-infinity).

Despite these costs, certified defenses are the gold standard for high-stakes applications where a formal security guarantee is required.

---

## 2. The Randomized Smoothing Idea

### 2.1 Informal Overview

Cohen, Rosenfeld, and Kolter (2019) introduced **randomized smoothing** as a general method for constructing certifiably robust classifiers. The key idea is simple:

Given **any** base classifier $f: \mathbb{R}^d \to \mathcal{Y}$ (any neural network, without any special architecture requirements), define the **smoothed classifier**:

$$g(x) = \arg\max_{c \in \mathcal{Y}} P_{\varepsilon \sim \mathcal{N}(0, \sigma^2 I)}\left[f(x + \varepsilon) = c\right]$$

In words: $g(x)$ predicts whichever class the base classifier $f$ predicts most often when the input $x$ is randomly perturbed by Gaussian noise.

**Why does this help?** If the noise $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$ is large relative to the adversarial perturbation $\delta$, then the adversarial perturbation $\delta$ becomes "lost in the noise" — the distribution of $f(x + \delta + \varepsilon)$ is very close to the distribution of $f(x + \varepsilon)$, because $\delta$ is small relative to $\sigma$.

Formally, adding a perturbation $\delta$ to the input shifts the noise distribution from $\mathcal{N}(x, \sigma^2 I)$ to $\mathcal{N}(x + \delta, \sigma^2 I)$. The total variation distance between these distributions is small when $\|\delta\|_2 / \sigma$ is small, which limits how much $\delta$ can change the prediction $g(x)$.

### 2.2 Why Gaussian Noise Specifically?

The key properties of Gaussian noise that make the analysis work:

1. **Isotropy:** $\mathcal{N}(0, \sigma^2 I)$ is rotationally symmetric. The noise has the same standard deviation in all directions, which means the robustness guarantee is uniform across all directions — consistent with the L2 norm.

2. **Gaussian noise and L2 shifts:** The total variation distance between $\mathcal{N}(\mu_1, \sigma^2 I)$ and $\mathcal{N}(\mu_2, \sigma^2 I)$ is determined by $\|\mu_1 - \mu_2\|_2 / \sigma$. For L2 shifts (adversarial perturbations bounded in L2 norm), Gaussian noise gives tight robustness certificates.

3. **The isoperimetric inequality:** For Gaussian measures, there is a sharp isoperimetric inequality (Borell's inequality) that characterizes which sets achieve the minimum and maximum probability under a shift. This is the technical foundation of the certification theorem.

4. **Why not uniform noise?** Uniform noise $\text{Uniform}([-\sigma, \sigma]^d)$ certifies L-infinity robustness, but only with much weaker guarantees (Lecuyer et al. 2019). The Gaussian structure is critical for tight certification.

---

## 3. The Certification Theorem

### 3.1 Theorem Statement (Cohen et al. 2019, Theorem 1)

**Theorem:** Let $f: \mathbb{R}^d \to \mathcal{Y}$ be any deterministic or stochastic function, and let $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$. Let $g$ be the smoothed classifier:

$$g(x) = \arg\max_{c \in \mathcal{Y}} P[f(x + \varepsilon) = c]$$

Suppose that for input $x$, the top class $c_A \in \mathcal{Y}$ satisfies:

$$p_A := P[f(x + \varepsilon) = c_A] \geq \underline{p_A} \geq \overline{p_B} \geq \max_{c \neq c_A} P[f(x + \varepsilon) = c] =: p_B$$

(That is, we have lower bound $\underline{p_A}$ on the top-class probability and upper bound $\overline{p_B}$ on the runner-up probability.) Then:

$$\forall \delta : \|\delta\|_2 < R \implies g(x + \delta) = c_A$$

where:
$$R = \frac{\sigma}{2}\left(\Phi^{-1}(\underline{p_A}) - \Phi^{-1}(\overline{p_B})\right)$$

and $\Phi^{-1}$ is the inverse CDF (quantile function) of the standard normal $\mathcal{N}(0, 1)$.

### 3.2 Full Proof

The proof uses two key ingredients: the Neyman-Pearson lemma and the isoperimetric inequality for Gaussian measures (Borell's lemma).

**Setup and notation:**
- Let $x \in \mathbb{R}^d$ be the input, $\delta \in \mathbb{R}^d$ be any perturbation with $\|\delta\|_2 = r < R$.
- Let $\mu_0 = x$ and $\mu_1 = x + \delta$ be the clean and perturbed inputs.
- The noise distribution: $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$.
- So $x + \varepsilon \sim \mathcal{N}(\mu_0, \sigma^2 I)$ and $x + \delta + \varepsilon \sim \mathcal{N}(\mu_1, \sigma^2 I)$.

**Step 1: The Neyman-Pearson Lemma applied to class sets.**

For any class $c$, define the set $S_c = \{z : f(z) = c\} \subset \mathbb{R}^d$ (the region of input space that $f$ maps to class $c$). The probability that the smoothed classifier predicts $c$ under the clean distribution is:

$$P[f(x + \varepsilon) = c] = P_{\varepsilon \sim \mathcal{N}(0, \sigma^2 I)}[x + \varepsilon \in S_c] = \mathcal{N}(\mu_0, \sigma^2 I)(S_c)$$

We want to show that $g(x + \delta) = c_A$, which requires:

$$\mathcal{N}(\mu_1, \sigma^2 I)(S_{c_A}) > \mathcal{N}(\mu_1, \sigma^2 I)(S_{c_B})$$

for all $c_B \neq c_A$.

**The Neyman-Pearson Lemma** characterizes the trade-off between the type-I and type-II errors for binary hypothesis testing. Applied to our setting:

For two Gaussian distributions $P = \mathcal{N}(\mu_0, \sigma^2 I)$ and $Q = \mathcal{N}(\mu_1, \sigma^2 I)$, and any measurable set $S$:

$$\text{If } P(S) \geq t, \text{ then } Q(S) \geq Q(S^*)$$

where $S^* = \{z : \frac{dP}{dQ}(z) \geq \lambda\}$ for some $\lambda$ that satisfies $P(S^*) = t$.

That is, **among all sets with $P$-measure at least $t$, the set $S^*$ achieves the minimum $Q$-measure**. The set $S^*$ is a half-space (because the likelihood ratio $dP/dQ$ for Gaussians is a linear function of $z$).

**Step 2: Computing the optimal set $S^*$.**

The likelihood ratio between $P = \mathcal{N}(\mu_0, \sigma^2 I)$ and $Q = \mathcal{N}(\mu_1, \sigma^2 I)$:

$$\frac{dP}{dQ}(z) = \exp\left(-\frac{\|z - \mu_0\|^2 - \|z - \mu_1\|^2}{2\sigma^2}\right) = \exp\left(\frac{(\mu_0 - \mu_1)^\top z + (\|\mu_1\|^2 - \|\mu_0\|^2)/2}{\sigma^2}\right)$$

This is a monotone function of $(\mu_0 - \mu_1)^\top z$, which is a linear functional of $z$. So the level set $\{z : dP/dQ(z) \geq \lambda\}$ is a half-space:

$$S^* = \{z : (\mu_0 - \mu_1)^\top z \geq c_\lambda\}$$

for some threshold $c_\lambda$. Since $\mu_1 - \mu_0 = \delta$, this is:

$$S^* = \{z : -\delta^\top z \geq c_\lambda\} = \{z : \delta^\top z \leq -c_\lambda\}$$

**Step 3: Computing $P(S^*)$ and $Q(S^*)$.**

The linear functional $\delta^\top z$ under $P = \mathcal{N}(\mu_0, \sigma^2 I)$ follows:

$$\delta^\top z \sim \mathcal{N}(\delta^\top \mu_0, \sigma^2 \|\delta\|_2^2)$$

Let $r = \|\delta\|_2$. Then $\delta^\top z / (r\sigma) \sim \mathcal{N}(\delta^\top \mu_0 / (r\sigma), 1)$. The set $S^*$ is defined by $\delta^\top z \leq -c_\lambda$.

Under $P$: $P(S^*) = \Phi\left(\frac{-c_\lambda - \delta^\top \mu_0}{r\sigma}\right) = \Phi(\alpha)$ for some $\alpha$.

Under $Q = \mathcal{N}(\mu_1, \sigma^2 I) = \mathcal{N}(\mu_0 + \delta, \sigma^2 I)$:

$\delta^\top z$ under $Q$ has mean $\delta^\top (\mu_0 + \delta) = \delta^\top \mu_0 + r^2$.

$$Q(S^*) = P_Q[\delta^\top z \leq -c_\lambda] = \Phi\left(\frac{-c_\lambda - \delta^\top \mu_0 - r^2}{r\sigma}\right) = \Phi\left(\alpha - \frac{r}{\sigma}\right)$$

**Step 4: Applying the Neyman-Pearson bound.**

By Neyman-Pearson, for any set $S$ with $P(S) \geq p_A$:

$$Q(S) \geq Q(S^*) = \Phi\left(\Phi^{-1}(p_A) - \frac{r}{\sigma}\right)$$

(Here we set $\alpha = \Phi^{-1}(p_A)$ so that $P(S^*) = p_A$.)

**Similarly, for any set $S$ with $P(S) \leq p_B$:**

By a symmetric argument (the complement), we get:

$$Q(S) \leq \Phi\left(\Phi^{-1}(p_B) + \frac{r}{\sigma}\right)$$

**Step 5: Ensuring $c_A$ remains the top class.**

We want $Q(S_{c_A}) > Q(S_{c_B})$ for all $c_B \neq c_A$.

From Step 4:
- $Q(S_{c_A}) \geq \Phi\left(\Phi^{-1}(p_A) - r/\sigma\right)$ (using $P(S_{c_A}) = p_A$)
- $Q(S_{c_B}) \leq \Phi\left(\Phi^{-1}(p_B) + r/\sigma\right)$ (using $P(S_{c_B}) = p_B$)

We need: $\Phi\left(\Phi^{-1}(p_A) - r/\sigma\right) > \Phi\left(\Phi^{-1}(p_B) + r/\sigma\right)$

Since $\Phi$ is monotone increasing, this is equivalent to:

$$\Phi^{-1}(p_A) - \frac{r}{\sigma} > \Phi^{-1}(p_B) + \frac{r}{\sigma}$$

$$\Phi^{-1}(p_A) - \Phi^{-1}(p_B) > \frac{2r}{\sigma}$$

$$r < \frac{\sigma}{2}\left(\Phi^{-1}(p_A) - \Phi^{-1}(p_B)\right) = R$$

This completes the proof. The classifier $g$ is robust for all $\|\delta\|_2 < R$. $\square$

### 3.3 Corollary 1: Simplified Form

When there are $K > 2$ classes, the runner-up class probability is bounded by $p_B \leq 1 - p_A$ (since all probabilities sum to 1 and the top class has probability $p_A$). Setting $p_B = 1 - p_A$ gives the loosest possible bound, which simplifies to:

$$R = \frac{\sigma}{2}\left(\Phi^{-1}(p_A) - \Phi^{-1}(1 - p_A)\right) = \frac{\sigma}{2} \cdot 2\Phi^{-1}(p_A) = \sigma \cdot \Phi^{-1}(p_A)$$

(using the symmetry of the normal distribution: $\Phi^{-1}(1-p) = -\Phi^{-1}(p)$).

**Corollary 1:** If $P[f(x + \varepsilon) = c_A] \geq p_A$, then:

$$g \text{ is certified at radius } R = \sigma \cdot \Phi^{-1}(p_A)$$

(with the understanding that $R > 0$ requires $p_A > 0.5$.)

**Numerical examples:**

| $p_A$ | $\Phi^{-1}(p_A)$ | $R$ (with $\sigma = 0.25$) | $R$ (with $\sigma = 0.5$) |
|--------|------------------|--------------------------|--------------------------|
| 0.51 | 0.025 | 0.006 | 0.013 |
| 0.60 | 0.253 | 0.063 | 0.127 |
| 0.70 | 0.524 | 0.131 | 0.262 |
| 0.80 | 0.842 | 0.211 | 0.421 |
| 0.90 | 1.282 | 0.321 | 0.641 |
| 0.95 | 1.645 | 0.411 | 0.823 |
| 0.99 | 2.326 | 0.582 | 1.163 |

**The maximum possible certification radius:** Since $p_A \leq 1$, the maximum is $R_{\max} = \sigma \cdot \Phi^{-1}(1) = +\infty$. But in practice $p_A < 1$ (the base classifier makes mistakes), so the maximum achievable radius is bounded by the base classifier's accuracy under noise.

**If the base classifier is perfect under noise ($p_A \to 1$):** $R \to \sigma \cdot \Phi^{-1}(1) = +\infty$. But this is theoretical; real classifiers always have errors.

**If $p_A = 0.99$ (top class probability 99%):** $R = \sigma \cdot 2.326$. For $\sigma = 0.5$: $R \approx 1.163$.

---

## 4. The Certification Algorithm

### 4.1 The Problem: Estimating $p_A$ from Samples

The certification theorem requires knowing $p_A = P[f(x + \varepsilon) = c_A]$ exactly. In practice, this is unknown and must be estimated by Monte Carlo sampling:

1. Sample $n$ noise vectors: $\varepsilon_1, \ldots, \varepsilon_n \sim \mathcal{N}(0, \sigma^2 I)$
2. Evaluate the base classifier: $c_i = f(x + \varepsilon_i)$ for each $i$
3. Count votes: $V_c = |\{i : c_i = c\}|$ for each class $c$
4. The top class is $\hat{c}_A = \arg\max_c V_c$

The estimated top-class probability is $\hat{p}_A = V_{\hat{c}_A} / n$. But this is an estimate with statistical uncertainty. We need a **confidence interval** on $p_A$ to give a certifiably correct result.

### 4.2 The Clopper-Pearson Confidence Interval

The Clopper-Pearson (1934) confidence interval is an **exact** (conservative) confidence interval for a binomial proportion. Given $k$ successes in $n$ trials, the $(1-\alpha)$-confidence lower bound on the true probability $p$ is:

$$\underline{p} = \text{Beta}_{\alpha/2}(k, n-k+1)$$

where $\text{Beta}_{\alpha/2}$ is the $\alpha/2$ quantile of the Beta distribution with parameters $k$ and $n-k+1$.

**Why Clopper-Pearson?** It provides exact coverage: $P[\underline{p} \leq p_A] \geq 1 - \alpha$ with guarantee. Conservative bounds are essential for safety — we would rather underestimate $R$ (conservative) than overestimate it (incorrect certificate).

**In Python (scipy):**

```python
from scipy.stats import binom

def clopper_pearson_lower(k, n, alpha=0.001):
    """
    Compute the lower bound of the Clopper-Pearson CI for p given k successes in n trials.

    This is the exact lower confidence bound at level α:
        P[p >= lower_bound] >= 1 - α

    Args:
        k:     Number of successes (votes for top class)
        n:     Total number of trials (noise samples)
        alpha: Confidence level (1-alpha is the coverage probability)

    Returns:
        lower_bound: Lower bound on the true top-class probability p_A
    """
    # The lower Clopper-Pearson bound is the α/2 quantile of Beta(k, n-k+1)
    if k == 0:
        return 0.0
    lower = binom.ppf(alpha / 2, n, k / n) / n
    # Equivalent: scipy.stats.beta.ppf(alpha/2, k, n-k+1)
    from scipy.stats import beta
    return beta.ppf(alpha / 2, k, n - k + 1)
```

### 4.3 The Full Certification Algorithm

The certification algorithm for a single input $x$:

```python
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import beta, norm

ABSTAIN = -1  # sentinel value when certification fails

def certify(model, x, sigma, n0, n, alpha, device):
    """
    Certify the smoothed classifier g at input x.

    Two-phase algorithm:
    Phase 1: Use n0 samples to identify the most likely class (selection phase).
             Using fewer samples reduces cost of the selection phase.
    Phase 2: Use n samples to estimate p_A with a tight confidence interval.

    Args:
        model:   Base classifier f (returns logits)
        x:       Input to certify (C, H, W) or (D,), values in [0, 1]
        sigma:   Gaussian noise standard deviation
        n0:      Number of samples for class selection (typically 100)
        n:       Number of samples for certification (typically 100,000)
        alpha:   Failure probability (the CI holds with probability >= 1-alpha)
        device:  torch device

    Returns:
        (prediction, radius):
            prediction = predicted class (ABSTAIN = -1 if abstaining)
            radius     = certified L2 radius (0 if abstaining)
    """
    model.eval()
    x = x.to(device)
    num_classes = get_num_classes(model, x, device)

    # ── Phase 1: Class Selection ─────────────────────────────────────────────
    # Sample n0 noisy copies of x and take a majority vote to select class.
    counts0 = sample_and_count(model, x, sigma, n0, num_classes, device)
    top_class = counts0.argmax().item()

    # ── Phase 2: Confidence Interval ─────────────────────────────────────────
    # Sample n noisy copies to get a tight estimate of P[f(x+ε) = top_class].
    counts = sample_and_count(model, x, sigma, n, num_classes, device)
    k_A = counts[top_class].item()  # votes for the top class

    # Clopper-Pearson lower bound on p_A
    p_A_lower = beta.ppf(alpha, k_A, n - k_A + 1)

    # Abstain if the lower bound is <= 0.5 (cannot certify majority class)
    if p_A_lower <= 0.5:
        return ABSTAIN, 0.0

    # Certified radius: R = σ * Φ^{-1}(p_A_lower)
    radius = sigma * norm.ppf(p_A_lower)
    return top_class, radius


def sample_and_count(model, x, sigma, num_samples, num_classes, device,
                     batch_size=1000):
    """
    Sample num_samples noisy copies of x and return class vote counts.

    Args:
        model:        Base classifier
        x:            Input (C, H, W), values in [0, 1]
        sigma:        Gaussian noise std
        num_samples:  Number of noise samples
        num_classes:  Number of classes (K)
        device:       torch device
        batch_size:   Number of samples to process per GPU batch

    Returns:
        counts:  torch.Tensor of shape (num_classes,), vote counts per class
    """
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    remaining = num_samples
    with torch.no_grad():
        while remaining > 0:
            this_batch = min(batch_size, remaining)
            remaining -= this_batch

            # Add Gaussian noise to x: x_noisy = x + ε, ε ~ N(0, σ²I)
            x_batch = x.unsqueeze(0).expand(this_batch, *x.shape)
            noise = torch.randn_like(x_batch) * sigma
            x_noisy = torch.clamp(x_batch + noise, 0.0, 1.0)

            # Get predictions from base classifier
            logits = model(x_noisy)
            preds = logits.argmax(dim=1)

            # Accumulate votes
            counts.scatter_add_(0, preds, torch.ones_like(preds))

    return counts


def get_num_classes(model, x, device):
    """Infer number of classes from model output on a single input."""
    with torch.no_grad():
        logits = model(x.unsqueeze(0).to(device))
    return logits.shape[1]
```

### 4.4 The Abstain Option

The algorithm may **abstain** — refuse to make a certified prediction — when the Clopper-Pearson lower bound on $p_A$ is at most 0.5. This happens when:
- The base classifier does not have a clear majority (many classes have similar probabilities under noise)
- The input is far from the training distribution
- The noise level $\sigma$ is too large relative to the underlying structure of the data

The abstain option is correct: it means we cannot certify robustness at this input, not that the input is adversarial. A good smoothed classifier should abstain infrequently on clean test examples.

**Abstain rate vs. $\sigma$:** As $\sigma$ increases, the smoothed classifier is less confident (the noise washes out fine-grained distinctions between classes), so the abstain rate increases. This is one of the key limitations of randomized smoothing.

### 4.5 Certified Accuracy

**Certified accuracy at radius $r$:** The fraction of test examples $(x, y)$ for which:
1. The smoothed classifier $g$ correctly predicts $y$ (does not abstain), AND
2. The certified radius $R(x) \geq r$

$$\text{CertAcc}(r) = \frac{1}{n_{\text{test}}} \sum_{(x,y) \in \mathcal{D}_{\text{test}}} \mathbf{1}\left[g(x) = y \text{ and } R(x) \geq r\right]$$

Note: $\text{CertAcc}(0) = \text{clean accuracy}$ (since $R(x) \geq 0$ is always true when $g(x) = y$, and the constraint becomes just correct prediction).

**Typical certified accuracy curves on CIFAR-10 (Cohen et al. 2019):**

| $\sigma$ | CertAcc at $r=0$ | CertAcc at $r=0.5$ | CertAcc at $r=1.0$ |
|----------|-----------------|---------------------|---------------------|
| 0.12 | 71% | 33% | 0% |
| 0.25 | 63% | 43% | 0% |
| 0.50 | 57% | 46% | 28% |
| 1.00 | 44% | 36% | 29% |

Observations:
- Larger $\sigma$ → lower clean accuracy but higher certified accuracy at large radii
- The optimal $\sigma$ depends on the certification radius of interest
- There's a fundamental tradeoff between clean accuracy and large-radius certification

---

## 5. Training the Base Classifier

### 5.1 Gaussian Noise Augmentation

To achieve high certified accuracy, the base classifier $f$ must perform well on noisy inputs (since the smoothed classifier evaluates $f$ on inputs corrupted by $\mathcal{N}(0, \sigma^2 I)$ noise). The training procedure adds Gaussian noise to all training images:

```python
def train_smoothed_base_classifier(model, train_loader, sigma, epochs, lr, device):
    """
    Train the base classifier f with Gaussian noise augmentation.

    Each training batch:
    1. Take clean images x from the loader
    2. Add Gaussian noise: x_noisy = x + σ * randn(x.shape)
    3. Clip to [0, 1]: x_noisy = clip(x_noisy, 0, 1)
    4. Train on (x_noisy, y) with standard cross-entropy

    This is equivalent to data augmentation with Gaussian noise, which
    trains the model to be robust to noise of magnitude σ.

    Args:
        model:        Neural network to train as base classifier
        train_loader: DataLoader for training data (clean images)
        sigma:        Noise standard deviation (matches certification σ)
        epochs:       Number of training epochs
        lr:           Learning rate
        device:       torch device
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Add Gaussian noise (the key augmentation step)
            noise = torch.randn_like(x) * sigma
            x_noisy = torch.clamp(x + noise, 0.0, 1.0)

            optimizer.zero_grad()
            output = model(x_noisy)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}: Loss = {total_loss/len(train_loader):.4f}")
```

### 5.2 Why Noise-Augmented Training is Necessary

Without noise-augmented training, the base classifier is trained on clean images and performs poorly on noisy inputs (the base classifier's predictions under noise are nearly random). This means:
- The vote counts are nearly uniform across classes
- The top class probability $p_A \approx 1/K$
- The certified radius $R = \sigma \cdot \Phi^{-1}(1/K) < 0$ (cannot certify anything)

With noise-augmented training, the model learns to classify correctly even under noise of magnitude $\sigma$, achieving $p_A >> 1/2$ on most clean inputs.

### 5.3 Optimal σ Selection

The noise level $\sigma$ is a hyperparameter that controls the tradeoff:
- Small $\sigma$: High base classifier accuracy under noise, but small maximum certification radius.
- Large $\sigma$: Low base classifier accuracy (noise overwhelms the signal), but large possible certification radius when the model does predict correctly.

**Rule of thumb:** Choose $\sigma$ approximately equal to the certification radius you want to achieve. Specifically:
- To certify at $r = 0.5$: use $\sigma = 0.25$ or $\sigma = 0.5$
- To certify at $r = 1.0$: use $\sigma = 0.5$ or $\sigma = 1.0$

The Cohen et al. (2019) paper trains separate models for each $\sigma$ value and reports the certified accuracy at each.

---

## 6. Worked Numerical Example: Full Certification

Let's trace through the complete certification algorithm on a single CIFAR-10 image.

**Setup:**
- 10-class classifier, $\sigma = 0.5$
- We use $n_0 = 100$ samples for class selection, $n = 10,000$ samples for certification
- Confidence level: $\alpha = 0.001$ (0.1% failure probability)

**Phase 1: Class selection (n0 = 100 samples)**

Sample 100 noisy copies of $x$ and classify each. Suppose the vote counts are:

| Class | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-------|---|---|---|---|---|---|---|---|---|---|
| Votes | 8 | 6 | **51** | 9 | 4 | 7 | 5 | 4 | 3 | 3 |

Top class: $c_A = 2$ (class "bird" in CIFAR-10), with 51 votes out of 100.

**Phase 2: Certification (n = 10,000 samples)**

Sample 10,000 noisy copies and count votes for class 2. Suppose $k_A = 7,312$ votes.

**Clopper-Pearson lower bound:**

$k_A = 7312$, $n = 10000$, $\alpha = 0.001$

$$\underline{p_A} = \text{Beta}_{0.001/2}(7312, 10000 - 7312 + 1) = \text{Beta}_{0.0005}(7312, 2689)$$

Using scipy: `scipy.stats.beta.ppf(0.0005, 7312, 2689) ≈ 0.721`

(The sample proportion is $\hat{p}_A = 7312/10000 = 0.7312$, and the Clopper-Pearson lower bound is about 0.014 below the sample proportion.)

**Certified radius:**

$$R = \sigma \cdot \Phi^{-1}(\underline{p_A}) = 0.5 \cdot \Phi^{-1}(0.721) = 0.5 \cdot 0.588 = 0.294$$

**Interpretation:** The smoothed classifier certifies that $g(x + \delta) = 2$ (class "bird") for all perturbations $\delta$ with $\|\delta\|_2 < 0.294$. This certification holds with probability at least $1 - 0.001 = 99.9\%$ over the randomness of the Monte Carlo sampling.

**Error check:** Could this certification be wrong? The Clopper-Pearson interval guarantees that $p_A \geq 0.721$ with probability 99.9%. If $p_A \geq 0.721 > 0.5$, the class selection is correct ($c_A$ is indeed the top class). And if $p_A \geq 0.721$, the certified radius is $R = 0.5 \cdot \Phi^{-1}(0.721) = 0.294 > 0$. So the certification is correct with 99.9% probability.

---

## 7. Key Limitations of Randomized Smoothing

### 7.1 Clean Accuracy Degradation

The smoothed classifier inevitably has lower clean accuracy than the base classifier trained on clean images, because:
1. Gaussian noise augmentation during training reduces clean image performance.
2. The smoothed classifier abstains on examples where the vote is not decisive.

**Typical accuracy degradation on CIFAR-10:**
- Standard training (no noise): 95% clean accuracy
- Noise augmentation ($\sigma = 0.25$): 87% base classifier accuracy, ~72% smoothed classifier accuracy (due to abstentions and noise confusion)
- Noise augmentation ($\sigma = 0.5$): 82% base classifier accuracy, ~63% smoothed classifier accuracy

This is the fundamental cost of certification: the model must be more "cautious" (smoother) to be certifiable, which reduces its sharpness and accuracy.

### 7.2 L2 Norm Restriction

The certification theorem only applies to L2 perturbations. The Gaussian noise structure is specifically designed for the L2 norm (isotropic noise is invariant to L2-ball-preserving transformations). There is no analogous result for L-infinity perturbations with Gaussian noise.

For L-infinity certification, different methods are needed:
- **IBP (Interval Bound Propagation):** Certifies L-infinity robustness by propagating intervals through the network. Efficient but requires special training.
- **Randomized smoothing with Uniform noise:** Provides L-infinity certificates but with much weaker guarantees (Lecuyer et al., 2019).

**Why does the L2 restriction matter?** L-infinity perturbations ($\|\delta\|_\infty \leq \epsilon$) are the standard threat model in adversarial ML. Randomized smoothing's certification is for L2 perturbations, which is a different (larger at the same nominal radius) threat model. An L2 certificate at radius $R$ does not certify against an L-infinity adversary with budget $\epsilon$, because an L-infinity ball of radius $\epsilon$ has L2 norm up to $\epsilon \sqrt{d}$ (where $d$ is the input dimension), which can be much larger than $R$.

### 7.3 Maximum Certification Radius

The maximum certified radius is bounded by the noise level $\sigma$ and the base classifier's accuracy under noise. For a base classifier with top-class probability $p_A$ under noise:

$$R \leq \sigma \cdot \Phi^{-1}(p_A) \leq \sigma \cdot \Phi^{-1}(p_A^{\text{max}})$$

where $p_A^{\text{max}}$ is the best achievable top-class probability for the given $\sigma$.

**In practice:** If the base classifier is 99% accurate under noise ($p_A \approx 0.99$), then $R \leq \sigma \cdot \Phi^{-1}(0.99) = \sigma \cdot 2.33$. For $\sigma = 0.5$: $R \leq 1.16$.

This means that for large perturbations (e.g., $\|\delta\|_2 > 1.16$ on CIFAR-10 at $\sigma = 0.5$), no certification is achievable regardless of how good the base classifier is. The maximum certifiable radius is fundamentally limited.

### 7.4 Computational Cost at Test Time

Certification requires evaluating the base classifier $n$ times per input ($n = 100,000$ for tight certification). For a ResNet-50 on a modern GPU (inference at ~1000 samples/second), $n = 100,000$ evaluations take approximately 100 seconds per input. This is computationally prohibitive for large-scale deployment.

**Mitigation:** Use smaller $n$ at the cost of looser (wider) confidence intervals, or use approximate certification that trades off guarantees for speed.

### 7.5 Comparison to IBP

| Property | Randomized Smoothing | IBP |
|----------|---------------------|-----|
| Threat model | L2 norm | L-infinity norm |
| Scales to large models | Yes (ResNet-50+) | Limited (small networks) |
| Clean accuracy drop | Moderate (5-15%) | Large (10-30%) |
| Certification tightness | Moderate | Can be tight for small ε |
| Test-time cost | High (10,000-100,000 samples) | Low (single forward pass) |
| Training method | Noise augmentation | IBP training |

**When to use randomized smoothing:** Large models (ResNet-50+) on complex datasets (CIFAR-100, ImageNet), when L2 certification is acceptable.

**When to use IBP:** Small networks on simple datasets (MNIST, small CIFAR), when L-infinity certification is needed, and when test-time efficiency is important.

---

## 8. State-of-the-Art Results

### 8.1 CIFAR-10 Certified Accuracy (Cohen et al. 2019, with ResNet-110)

| σ | CertAcc at r=0 | CertAcc at r=0.5 | CertAcc at r=1.0 | CertAcc at r=1.5 |
|---|---|---|---|---|
| 0.12 | 71% | 31% | 0% | 0% |
| 0.25 | 63% | 43% | 0% | 0% |
| 0.50 | 57% | 46% | 28% | 0% |
| 1.00 | 44% | 36% | 29% | 14% |

### 8.2 ImageNet Certified Accuracy (Cohen et al. 2019, with ResNet-50)

| σ | CertAcc at r=0 | CertAcc at r=0.5 | CertAcc at r=1.0 | CertAcc at r=2.0 |
|---|---|---|---|---|
| 0.25 | 67% | 49% | 0% | 0% |
| 0.50 | 57% | 46% | 28% | 0% |
| 1.00 | 44% | 38% | 30% | 11% |

**A key observation:** Randomized smoothing achieves non-trivial certified accuracy even on ImageNet — a 1000-class dataset with high-resolution images. This is remarkable because IBP-based certified defenses struggle to scale beyond MNIST. Randomized smoothing's architecture-agnosticism is its key strength.

### 8.3 Improved Results: Denoised Smoothing (Salman et al. 2020)

Salman et al. (2020) showed that prepending a pretrained denoiser $d$ before the base classifier significantly improves certified accuracy. The smoothed classifier becomes:

$$g(x) = \arg\max_c P_\varepsilon[f(d(x + \varepsilon)) = c]$$

The denoiser removes the Gaussian noise before classification, allowing a cleaner signal to reach the classifier. With a denoiser trained on $\sigma = 0.5$ noise, CIFAR-10 certified accuracy at $r = 1.0$ improves from 28% to 40%.

---

## 9. Key Takeaways

1. **Certified robustness provides unconditional guarantees.** For certified inputs, no adaptive attack can change the prediction. This is qualitatively different from empirical robustness.

2. **The certification theorem is tight via Neyman-Pearson.** The Gaussian isoperimetric inequality gives exactly the maximum certification radius achievable by any smoothing-based method with Gaussian noise.

3. **The Corollary 1 formula $R = \sigma \Phi^{-1}(p_A)$ is the key practical result.** To certify at a given radius, you need the top-class probability to satisfy $p_A \geq \Phi(r/\sigma)$.

4. **Monte Carlo certification requires Clopper-Pearson confidence intervals.** Accurate probability estimation requires $n = 10,000$–$100,000$ samples per input, making certification computationally expensive.

5. **There is a fundamental tradeoff between σ, clean accuracy, and certification radius.** Larger $\sigma$ allows certifying larger radii but reduces clean accuracy. The optimal $\sigma$ depends on the target radius.

6. **Randomized smoothing certifies L2 robustness, not L-infinity.** This is a significant limitation for the standard CIFAR-10 threat model.

7. **Randomized smoothing scales to large models (ImageNet).** This is its key advantage over IBP, which is limited to small networks.

---

## Discussion Questions

1. The proof uses the Neyman-Pearson lemma to show that the half-space $S^*$ minimizes the $Q$-measure among all sets with $P$-measure $\geq p_A$. The optimal $S^*$ is a half-space (a hyperplane that partitions $\mathbb{R}^d$). Does this correspond to any natural structure of the base classifier's decision regions?

2. The Clopper-Pearson interval uses $n = 100,000$ samples for tight certification. Could we use $n = 1,000$ samples and get a valid (but looser) certificate? What is the tradeoff between sample size and certification radius tightness?

3. Randomized smoothing certifies L2 robustness. Consider an adversary with an L2 budget of $r = 1.0$. In image space, what does an L2 perturbation of magnitude 1.0 look like? Is this a realistic threat model for vision classifiers?

4. The smoothed classifier $g$ may abstain. Consider designing a defense that uses $g$ to abstain on inputs it cannot certify, and falls back to an uncertified classifier $f$ otherwise. Is this composition safe? What happens if the adversary can detect when the system abstains?

5. Salman et al. (2020) showed that denoising before classification improves certified accuracy. But the denoiser is also a neural network that can potentially be fooled. Does using a denoiser invalidate the certification guarantee? Why or why not?

---

## References

- Cohen, J., Rosenfeld, E., & Kolter, J.Z. (2019). Certified Adversarial Robustness via Randomized Smoothing. *ICML 2019*.
- Lecuyer, M., Atlidakis, V., Geambasu, R., Hsu, D., & Jana, S. (2019). Certified Robustness to Adversarial Examples with Differential Privacy. *IEEE S&P 2019*.
- Clopper, C.J., & Pearson, E.S. (1934). The Use of Confidence or Fiducial Limits. *Biometrika*.
- Salman, H., et al. (2020). Denoised Smoothing: A Provable Defense for Pretrained Classifiers. *NeurIPS 2020*.
- Gowal, S., et al. (2018). On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models. *arXiv:1810.12715*.
- Raghunathan, A., Steinhardt, J., & Liang, P. (2018). Certified Defenses against Adversarial Examples. *ICLR 2018*.
- Singh, G., Gehr, T., Püschel, M., & Vechev, M. (2019). An Abstract Domain for Certifying Neural Networks. *POPL 2019*.
