# CS 6820 — Week 10 Lecture
# Differential Privacy in Machine Learning: DP-SGD and the Privacy-Utility Tradeoff

---

## Learning Objectives

By the end of this lecture, students will be able to:
1. State the formal (ε, δ)-differential privacy definition and explain the privacy guarantee it provides to individual training examples
2. Derive the sensitivity of a clipped gradient and explain why per-sample gradient clipping is necessary (not batch-gradient clipping)
3. Trace through the DP-SGD algorithm step by step, identifying where noise is added and how the noise scale σ is selected
4. Explain Rényi DP and why it gives tighter privacy accounting than naive composition
5. Reason about the privacy-utility tradeoff: why high-dimensional data, small datasets, and low ε are the hardest settings

---

## 1. Motivation: Why Train a Private Model?

In the previous weeks, we studied membership inference attacks — the attacker queries the model and infers whether a specific record was in the training set. We also saw (Week 4, CS 6800) that LLMs can memorize and reproduce verbatim training data. Both of these are privacy violations:

- **Membership inference:** Does the model reveal that Alice was in the training dataset?
- **Reconstruction:** Does the model reproduce Alice's medical record or private communication?

The threat model for **private model training** is:

> An adversary observes the deployed model's weights (white-box) or its predictions (black-box). The adversary wants to learn something about any individual training example — their presence (membership), their features, or their exact data.

The standard tool for providing a rigorous guarantee against this class of attacks is **differential privacy (DP)**.

---

## 2. Differential Privacy: The Definition

**Definition (Dwork, McSherry, Nissim, Smith, 2006):**

A randomized algorithm $\mathcal{M}: \mathcal{D} \to \mathcal{R}$ is **$(\varepsilon, \delta)$-differentially private** if for all pairs of *neighboring* datasets $D, D' \in \mathcal{D}$ (differing in exactly one record) and for all subsets $S \subseteq \mathcal{R}$:

$$\Pr[\mathcal{M}(D) \in S] \leq e^\varepsilon \cdot \Pr[\mathcal{M}(D') \in S] + \delta$$

**Interpretation:**
- The distributions of outputs on $D$ and $D'$ are nearly indistinguishable.
- $\varepsilon$ (the "privacy budget"): Smaller = more private. $\varepsilon = 0$ means the output is completely independent of any individual record. $\varepsilon = 1$ means the ratio of probabilities is bounded by $e^1 \approx 2.72$.
- $\delta$: The "failure probability" — with probability at most $\delta$, the $\varepsilon$ bound may not hold. Typically $\delta = 1/n$ where $n$ is dataset size, or $\delta = 10^{-5}$.
- **Pure DP** ($\delta = 0$): Stronger; harder to achieve.
- **Approximate DP** ($\delta > 0$): Necessary for DP-SGD to be computationally tractable.

**Neighboring datasets:** Two datasets $D$ and $D'$ are neighboring if $D' = D \setminus \{x_i\}$ for some record $x_i$ (unbounded DP, commonly used) or $D'$ replaces one record with another (bounded DP).

### Why This Is a Strong Guarantee

DP is a worst-case guarantee: it holds for *every possible* adjacent dataset pair, not just on average. It implies:

- **Membership inference resistance:** An adversary cannot significantly improve their ability to determine whether any particular record $x_i$ was in the training set.
- **Composition immunity:** Any post-processing of $\mathcal{M}(D)$ (e.g., fine-tuning, distillation) cannot increase the privacy cost.
- **Group privacy:** A dataset differing in $k$ records is protected at $(\varepsilon k, \delta k e^{\varepsilon(k-1)})$ (weaker, but still bounded).

### Privacy Budget Values in Practice

| $\varepsilon$ | Privacy Level | Typical Use |
|---------------|--------------|-------------|
| $\varepsilon \leq 1$ | Strong | Medical, financial data |
| $1 < \varepsilon \leq 3$ | Moderate | General user data |
| $3 < \varepsilon \leq 10$ | Weak | "Demonstrates DP" |
| $\varepsilon > 10$ | Negligible | Practically equivalent to no DP |

---

## 3. Sensitivity: Bounding the Impact of One Record

DP requires us to add noise calibrated to **how much one record can change the output**. This is called the *sensitivity*.

**Definition (L2 sensitivity):** For a function $f: \mathcal{D} \to \mathbb{R}^d$:

$$\Delta_2 f = \max_{D, D' \text{ neighboring}} \| f(D) - f(D') \|_2$$

For gradient-based learning, the "output" is the gradient $g_i = \nabla_\theta \ell(\theta; x_i, y_i)$ computed on a single sample.

**The sensitivity problem:** Without bounding, a single sample's gradient can have arbitrarily large norm — a single outlier training example could have $\|g_i\|_2 = 10^6$. Adding noise proportional to this sensitivity would completely destroy utility.

**Solution: Per-sample gradient clipping.** Before aggregating, clip each gradient to a maximum L2 norm $C$:

$$\tilde{g}_i = g_i \cdot \min\!\left(1,\; \frac{C}{\|g_i\|_2}\right)$$

This ensures $\|\tilde{g}_i\|_2 \leq C$, which bounds the L2 sensitivity of the sum to $C$.

**Critical point:** We must clip *per-sample* gradients, not the *batch mean* gradient. Consider two batches:
- Batch $D$: contains sample $x_j$ with gradient $g_j = [10^6, 0, \ldots]$
- Batch $D'$: $D$ with $x_j$ removed

If we clip the batch mean, the influence of $x_j$ could be diluted but not bounded — removing $x_j$ could change the mean by $g_j / B$, which for large $g_j$ is still large. Clipping per-sample first guarantees each sample's contribution to the sum is at most $C$, regardless of batch size.

---

## 4. The Gaussian Mechanism

Given a clipped gradient sum with sensitivity $C$, we add Gaussian noise to make it private.

**Gaussian Mechanism:** Given $f: \mathcal{D} \to \mathbb{R}^d$ with $\Delta_2 f \leq C$, the mechanism:

$$\mathcal{M}(D) = f(D) + \mathcal{N}(0, \sigma^2 C^2 I)$$

is $(\varepsilon, \delta)$-DP for $\sigma \geq \frac{\sqrt{2 \ln(1.25/\delta)}}{\varepsilon}$.

This formula means: to achieve lower $\varepsilon$ (stronger privacy), we need larger $\sigma$ (more noise). The $\delta$ dependence is logarithmic — halving $\delta$ only slightly increases $\sigma$.

---

## 5. DP-SGD Algorithm (Abadi et al., 2016)

DP-SGD (Differentially Private Stochastic Gradient Descent) applies differential privacy to neural network training by modifying the SGD update rule.

### Algorithm

**Input:** Dataset $D = \{(x_1, y_1), \ldots, (x_n, y_n)\}$, loss $\ell$, parameters $\theta_0$, learning rate $\eta$, noise multiplier $\sigma$, clipping norm $C$, lot size $L$, number of steps $T$.

**For** $t = 1, \ldots, T$:
1. **Sample a minibatch** $\mathcal{L}_t$ by including each example independently with probability $q = L/n$ (Poisson sampling; approximated by fixed batches in practice)
2. **Compute per-sample gradients:**
   $$g_i \leftarrow \nabla_\theta \ell(\theta_t; x_i, y_i) \quad \forall i \in \mathcal{L}_t$$
3. **Clip each gradient:**
   $$\tilde{g}_i \leftarrow g_i \cdot \min\!\left(1, \frac{C}{\|g_i\|_2}\right)$$
4. **Aggregate and add noise:**
   $$\tilde{g} \leftarrow \frac{1}{|\mathcal{L}_t|} \left(\sum_{i \in \mathcal{L}_t} \tilde{g}_i + \mathcal{N}(0,\; \sigma^2 C^2 I)\right)$$
5. **Update parameters:**
   $$\theta_{t+1} \leftarrow \theta_t - \eta \cdot \tilde{g}$$

**Output:** $\theta_T$ with $(\varepsilon, \delta)$-DP guarantee (computed via privacy accountant).

### Implementation Notes

- **Per-sample gradients in PyTorch:** Standard autograd computes gradients summed over the batch. To get per-sample gradients, one must either: (a) process each sample individually in a loop, (b) use `functorch.grad` with `vmap`, or (c) use the Opacus library, which implements efficient per-sample gradient computation via custom `grad_sample` functions for each layer type.
- **Noise calibration:** The noise multiplier $\sigma$ is chosen to achieve a target $(\varepsilon, \delta)$ after $T$ steps. This is computed by the privacy accountant (Opacus: `RDPAccountant` or `PRVAccountant`).
- **Batch size matters for privacy:** Larger batches $L$ increase the sampling probability $q = L/n$, which increases the privacy cost per step — but also reduces the noise-to-signal ratio. In practice, larger batches improve utility without necessarily hurting privacy if the accountant is used correctly.

### The Clipping Norm $C$ as a Hyperparameter

The clipping norm $C$ controls the trade-off between information retention and sensitivity:

- **Too small $C$:** Most gradients are clipped to $C$, losing their direction and magnitude information. Learning slows dramatically.
- **Too large $C$:** Sensitivity bound is loose; need very large $\sigma$ to achieve target $\varepsilon$; noise dominates. Learning still fails.
- **Optimal $C$:** Approximately equal to the median gradient norm of the unclipped gradients. This way, about half the gradients are clipped (losing some info) and half are kept (preserving direction).

**Tuning strategy:** Run a few non-private training steps, log the distribution of gradient norms, and set $C$ to the 50th–75th percentile.

---

## 6. Privacy Accounting: How Budget Accumulates

DP-SGD runs for $T$ steps. Each step consumes some privacy budget. The total budget is determined by **composition** — how DP guarantees accumulate over multiple operations.

### Naive Composition (Too Pessimistic)

The basic composition theorem: $T$ applications of an $(\varepsilon_0, \delta_0)$-DP mechanism compose to $(T \varepsilon_0,\; T \delta_0)$-DP.

For $T = 1000$ steps of SGD, each with $\varepsilon_0 = 0.1$: naive composition gives $\varepsilon = 100$. This is useless — a budget of 100 provides no meaningful privacy.

### Moments Accountant (Abadi et al., 2016)

Abadi et al. introduced the *moments accountant*, tracking the moment-generating function of the privacy loss random variable. This gives a tighter bound: for Poisson-sampled DP-SGD with $T$ steps, the effective $\varepsilon$ grows as $O(q\sqrt{T})$ rather than $O(qT)$, where $q = L/n$ is the sampling rate.

**Practical implication:** With $n=60,000$ (CIFAR-10), $L=256$, $T=4700$ steps (≈ 20 epochs), $\sigma=1.1$, the moments accountant gives $\varepsilon \approx 3$ at $\delta = 10^{-5}$. Naive composition would give $\varepsilon \approx 200$.

### Rényi Differential Privacy (Mironov, 2017)

RDP provides even tighter accounting by working with Rényi divergences.

**Rényi divergence of order $\alpha$:** $D_\alpha(P \| Q) = \frac{1}{\alpha-1} \log \mathbb{E}_{x \sim Q}\!\left[\left(\frac{P(x)}{Q(x)}\right)^\alpha\right]$

**$(\alpha, \varepsilon)$-RDP:** $\mathcal{M}$ satisfies $(\alpha, \varepsilon)$-RDP if $D_\alpha(\mathcal{M}(D) \| \mathcal{M}(D')) \leq \varepsilon$ for all neighboring $D, D'$.

**Composition:** $(\alpha, \varepsilon_1)$-RDP and $(\alpha, \varepsilon_2)$-RDP compose to $(\alpha, \varepsilon_1 + \varepsilon_2)$-RDP. Addition, not multiplication — this is the advantage over $(\varepsilon, \delta)$-DP composition.

**Converting RDP to $(\varepsilon, \delta)$-DP:**
$$\varepsilon(\delta) = \min_{\alpha > 1} \left( \varepsilon_{\text{RDP}}(\alpha) + \frac{\log((\alpha-1)/\alpha) - \log\delta}{\alpha - 1} \right)$$

Opacus uses this conversion internally. After training, you call `accountant.get_epsilon(delta=1e-5)` and get the final $(\varepsilon, \delta)$ guarantee.

**Gaussian mechanism under RDP:** The Gaussian mechanism with noise multiplier $\sigma$ satisfies $(\alpha, \alpha / (2\sigma^2))$-RDP (for the non-subsampled case). For the subsampled case (DP-SGD), the amplification by subsampling reduces the RDP cost to approximately $(\alpha,\; q^2 \alpha / (2\sigma^2))$.

---

## 7. The Privacy-Utility Tradeoff

### Empirical Observation

| Setting | ε = ∞ (no DP) | ε = 10 | ε = 3 | ε = 1 |
|---------|--------------|--------|-------|-------|
| MNIST (LeNet-5) | 99.2% | 98.8% | 97.5% | 94.1% |
| CIFAR-10 (ResNet-18) | 93.1% | 88.2% | 82.3% | 68.7% |
| ImageNet (ResNet-50) | 76.1% | 68.2% | 55.1% | 38.4% |

*Approximate values from Opacus benchmarks and literature.*

**Key observations:**
1. MNIST suffers minimally from DP — it's a simple task with clear gradients
2. CIFAR-10 at $\varepsilon = 3$ degrades by ~11% absolute — significant but usable
3. ImageNet at $\varepsilon = 1$ is nearly useless — complex task, small per-class sample count hurts

### Why High Dimensionality Hurts DP

The noise added to gradients has norm $\sim \sigma C \sqrt{d}$ where $d$ is the parameter count. For ResNet-18: $d \approx 11 \times 10^6$. The signal (gradient) has norm $O(C)$. The noise-to-signal ratio is $\sigma \sqrt{d}$, which is $O(10^3)$ for large models. Achieving $\varepsilon = 1$ requires $\sigma \approx 10$, meaning the noise norm is $10 \times C \times 3000 = 30,000 C$ — completely dominated by noise.

**Mitigations:**
- **Public pre-training + private fine-tuning:** Fine-tune only the last few layers privately. Only the fine-tuned parameters need noise. Li et al. (2022) shows this dramatically improves the DP tradeoff.
- **Transfer learning:** Use a publicly-trained feature extractor and only privately train the classification head ($d \approx 1000 \times 10 = 10,000$).
- **Large batches:** Increasing batch size $L$ reduces the noise relative to the gradient signal (gradient norm scales as $\sqrt{L}$ while noise adds as $\sqrt{L}$, so SNR improves as $\sqrt{L}$).

### Membership Inference Advantage as a Function of ε

A key sanity check for DP: does it actually reduce membership inference success?

**Empirical relationship:** For loss-threshold membership inference attacks:
$$\text{MI Advantage} = |\text{TPR} - \text{FPR}| \approx \frac{e^\varepsilon - 1}{e^\varepsilon + 1}$$

At $\varepsilon = 3$: MI advantage $\leq 0.46$ (compared to 0.0 for perfect DP)
At $\varepsilon = 10$: MI advantage $\leq 0.82$ — nearly as bad as no protection

This illustrates why $\varepsilon > 10$ provides negligible practical privacy against membership inference.

---

## 8. Using Opacus: Practical DP-SGD in PyTorch

[Opacus](https://opacus.ai/) is Meta's DP training library for PyTorch. It handles:
- Per-sample gradient computation via hooks
- Gradient clipping
- Noise injection
- Privacy accounting (RDP accountant)

### Minimal Example

```python
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# 1. Define model (must be compatible with Opacus)
model = LeNet5()
model = ModuleValidator.fix(model)  # Replaces BatchNorm → GroupNorm, etc.

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
train_loader = ...  # Your DataLoader

# 2. Attach PrivacyEngine
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=20,
    target_epsilon=3.0,     # Target ε
    target_delta=1e-5,       # Target δ
    max_grad_norm=1.0,       # Clipping norm C
)

# 3. Standard training loop — Opacus handles the rest automatically
for epoch in range(20):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()  # Opacus clips and adds noise here

    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f'Epoch {epoch}: ε = {epsilon:.2f}')
```

### Opacus Compatibility Notes
- **BatchNorm incompatibility:** BatchNorm requires access to batch statistics, which leaks cross-sample information. Opacus replaces it with GroupNorm automatically via `ModuleValidator.fix()`.
- **Grad accumulation:** Standard PyTorch grad accumulation is not compatible with per-sample gradients. Use `optimizer.step()` every batch.
- **Custom layers:** If you define custom layers, you must register a `grad_sampler` that computes per-sample gradients.

---

## 9. Federated Learning Connection

DP-SGD is naturally compatible with **federated learning** (covered in Week 11), where:
- Each client trains on their local data and sends gradients to the server
- The server aggregates gradients (secure aggregation)
- Adding DP noise before uploading (local DP) or at the server (central DP) provides privacy

The tension: local DP (noise added at each client) requires much more noise than central DP (noise added once at the server after aggregation) to achieve the same $\varepsilon$, because the noise must hide the influence of each client's *entire* dataset rather than just one record.

---

## 10. Discussion Questions

1. **Hyperparameter attack:** An adversary watches the training process and can observe the publicly reported $(\varepsilon, \delta)$ guarantee. Does this reveal anything about the model, and does it help the adversary in any way?

2. **The DP guarantee is for one record.** If an attacker wants to infer whether a *group* of 10 people were in the training set, how does the group privacy property of DP bound their advantage? Is this bound practically useful?

3. **Transfer learning and DP:** Suppose you use a public ImageNet-pretrained ResNet-50 as a fixed feature extractor and train only the final FC layer with DP. How does this change the noise-to-signal analysis? What assumptions does this strategy require about the pre-training data?

4. **DP vs. empirical defenses:** We've seen that empirical defenses (detection, preprocessing) can be broken by adaptive attacks. DP is a certified guarantee — does this mean DP is always preferable to adversarial training for privacy? What are the cases where adversarial training is still preferred?

5. **Reconstruction attacks:** The DP guarantee bounds membership inference, but recent work (Balle et al. 2022, Carlini et al. 2022) shows that even at moderate $\varepsilon$, reconstruction of individual training examples from gradients (gradient inversion attacks in federated learning) is still possible. Does this violate DP? Why or why not?

---

## Key Takeaways

1. **$(\varepsilon, \delta)$-DP** provides a worst-case guarantee: the output distribution barely changes when any single training record is added or removed.

2. **DP-SGD** achieves DP by: (a) per-sample gradient clipping (bounds sensitivity to $C$) and (b) adding Gaussian noise scaled to $\sigma C$ (privatizes the sum).

3. **Privacy accounting** (Rényi DP) enables tight accumulation tracking over $T$ steps, making meaningful privacy ($\varepsilon = 1$–$3$) achievable with large datasets.

4. **The privacy-utility tradeoff** is real and fundamental: small $\varepsilon$ requires large noise, which hurts accuracy, especially on complex tasks and large models.

5. **Mitigation strategies:** Public pre-training + private fine-tuning, large batches, and feature-space DP are practical approaches that dramatically improve the tradeoff.
