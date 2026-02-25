# CS 6810 — Adversarial Machine Learning
## Week 05: Black-Box Attacks I — Transferability — Why Adversarial Examples Cross Model Boundaries

**Prerequisites:** Week 01 (taxonomy), Week 03 (C&W, PGD), familiarity with ensemble methods and model architectures (ResNets, ViTs, MobileNets).

**Learning Objectives:**
1. Explain the transferability phenomenon and the main theoretical accounts for it.
2. Derive the MI-FGSM update rule from first principles.
3. Implement the MI-TI-DI-FGSM combined attack.
4. Reason about which surrogate model choices maximize transfer.
5. Critically evaluate the "aligned gradients" hypothesis using Demontis et al.'s analysis.

---

## 1. The Transferability Phenomenon

### 1.1 Discovery and Basic Observation

In 2013, Szegedy et al. made two observations in the same paper:
1. Deep networks are vulnerable to imperceptible perturbations (adversarial examples exist).
2. Adversarial examples crafted on *one* model fool *different* models — even models with different architectures trained on the same dataset.

The second observation was initially treated as a curiosity. Its implications became clear only gradually: an attacker who trains their own model on the same task (or even a related task) can craft adversarial examples that fool a *completely unknown* target model, with no access to the target's weights, architecture, or gradients.

**Formal statement of transferability:** Let $f_A$ and $f_B$ be two classifiers. An adversarial example $x'$ for $f_A$ satisfies $f_A(x') \neq y$. Transferability is the observation that $\Pr[f_B(x') \neq y \mid f_A(x') \neq y] > \Pr[f_B(x') \neq y \mid x' = x]$, i.e., adversarial examples for $f_A$ fool $f_B$ at above-chance rates.

**Magnitude of the effect (empirical):** On ImageNet:
- Adversarial examples for ResNet-50 (white-box attack) transfer to Inception-v3 at 40-60% rates using FGSM.
- Stronger iterative attacks (PGD-100) transfer at lower rates (20-40%) because they overfit to the specific local geometry of $f_A$.
- MI-FGSM and its variants achieve 60-90% transfer rates.

The counterintuitive observation that *stronger* white-box attacks transfer *worse* is crucial. We return to it in Section 4.

### 1.2 Why Transferability Matters Practically

Transferability enables realistic black-box attacks against commercial ML APIs:
- The attacker trains a local surrogate model on public data (same distribution as the target's training data).
- The attacker crafts adversarial examples on the surrogate using any white-box attack.
- The adversarial examples are submitted to the target API.

This was demonstrated by Papernot et al. (2017) against the Clarifai image recognition API, and by Ilyas et al. (2018) against the Google Cloud Vision API.

---

## 2. Why Transferability Happens: Three Theories

### 2.1 The Linearity Hypothesis (Goodfellow et al. 2014)

Goodfellow et al. proposed that adversarial examples exist because neural networks behave approximately linearly in high-dimensional input spaces. If two models $f_A$ and $f_B$ are approximately linear in a neighborhood of $x$:

$$f_A(x + \delta) \approx f_A(x) + \nabla_x f_A(x)^\top \delta$$
$$f_B(x + \delta) \approx f_B(x) + \nabla_x f_B(x)^\top \delta$$

A perturbation $\delta = \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}_A(x, y))$ that maximally increases $f_A$'s loss will *also* increase $f_B$'s loss if:

$$\nabla_x \mathcal{L}_A(x, y) \cdot \nabla_x \mathcal{L}_B(x, y) > 0$$

i.e., if the gradients of the two models are positively correlated.

**Prediction:** Transfer rates should correlate with gradient alignment. Empirically, this holds roughly: models trained on the same data with similar architectures have more correlated gradients and higher transfer rates.

**Limitation:** This theory doesn't explain why adversarial examples transfer between very different architectures (CNN → Transformer) or different training objectives. In those cases, gradient alignment is much lower but transfer still occurs.

### 2.2 Aligned Decision Boundaries in High Dimensions

A more structural theory: models trained on the same data distribution (by ERM) tend to learn similar decision boundaries, even if the specific functions are different. This is because:

1. The task has a "true" decision boundary induced by the data distribution.
2. Any model with low generalization error must approximate this true boundary.
3. Adversarial examples near the true boundary transfer between any two models that approximate it.

**Formalization (informal):** Let $P(x, y)$ be the data distribution. The Bayes-optimal classifier $f^*(x) = \arg\max_k P(y=k|x)$ has a fixed decision boundary. If $f_A$ and $f_B$ both achieve low generalization error on $P$, they must agree on most of the data manifold and have similar boundaries in high-density regions.

**Prediction:** Transfer should be higher on in-distribution inputs than out-of-distribution inputs. Transfer should be higher when both models achieve similar accuracy on the task. Empirically: yes.

**Limitation:** This theory doesn't explain *why* adversarial examples tend to be near the boundary and *why* they approach the true boundary rather than spurious local minima.

### 2.3 Shared Non-Robust Features (Ilyas et al. 2019)

Ilyas et al. (2019) proposed that adversarial examples exploit *non-robust features* — features that are predictive of the label but that change quickly under small perturbations. Their argument:

1. Real datasets contain both "robust features" (stable, human-recognizable patterns) and "non-robust features" (high-frequency patterns, imperceptible correlations with labels).
2. ERM-trained models use *both* types of features, since non-robust features are predictive.
3. An adversarial perturbation adds non-robust features of the *wrong* class while preserving the visual appearance (which depends on robust features).
4. Different models share the same non-robust features (they are features of the data, not the model).

**Key experiment:** Ilyas et al. construct a "non-robust" training dataset where labels are flipped according to adversarial perturbations. Models trained on this dataset classify images "correctly" (according to the non-robust features) but classify the original images "incorrectly" by human standards. They achieve non-trivial accuracy on the test set.

**Implication for transferability:** If adversarial examples exploit shared non-robust features, they should transfer between any two models trained on the same data distribution — which is empirically confirmed.

---

## 3. Empirical Transferability Patterns

### 3.1 Architecture Families Matter

Not all pairs of models transfer equally. Empirical findings (Liu et al. 2016, Demontis et al. 2019):

| Source → Target | Transfer Rate |
|----------------|--------------|
| ResNet-50 → ResNet-101 | High (70-85%) |
| ResNet-50 → DenseNet-121 | Medium-High (55-70%) |
| ResNet-50 → VGG-16 | Medium (45-60%) |
| ResNet-50 → ViT-B/16 | Low-Medium (30-50%) |
| CNN (any) → ViT (any) | Low (20-40%) |
| ViT → CNN | Low (25-45%) |

**Why CNN → ViT transfers poorly:** CNNs and Vision Transformers process images fundamentally differently. CNNs build features via local convolutions (spatial inductive bias). ViTs use global attention (no spatial inductive bias). The adversarial directions in their input spaces are very different. A perturbation that manipulates local high-frequency features (effective against CNNs) may have no impact on the global attention patterns of ViTs.

This creates a potential defense strategy: use an ensemble of heterogeneous models. But heterogeneous ensembles are expensive, and more sophisticated attacks have been developed to handle this.

### 3.2 Training Procedure Matters

Models trained with:
- **Adversarial training:** Lower transfer rates received (more robust to transferred attacks) but *higher* transfer rates sent (adversarial examples crafted on them are more transferable). The latter is because adversarially trained models have smoother loss landscapes with more meaningful gradients.
- **Data augmentation (mixup, cutmix):** Reduced transfer received.
- **Different optimizers (SGD vs. Adam):** Minor effect on transfer rate.
- **Different random seeds:** Low but nonzero transfer effect (models with same architecture and same data but different seeds transfer at ~40-50% rates with PGD, higher with MI-FGSM).

---

## 4. MI-FGSM: Momentum Iterative FGSM

### 4.1 Why Iterative Attacks Transfer Poorly

PGD (iterative FGSM) transfers worse than FGSM despite being a stronger white-box attack. This is the "overfitting" phenomenon in transfer attacks:

Iterative attacks follow the loss surface of $f_A$ too closely. After many iterations, they find a point in the loss surface of $f_A$ that happens to fool $f_A$ via a model-specific quirk (local gradient direction) rather than via the shared structure that would fool $f_B$. The perturbation has "overfit" to $f_A$'s specific local minima.

**Intuition:** Think of iterative attacks as gradient descent on the loss surface of $f_A$. Early iterations move in a direction correlated with the average gradient across models (high transfer). Later iterations follow $f_A$-specific gradient directions (low transfer).

### 4.2 The MI-FGSM Update Rule

Dong et al. (2018) proposed accumulating momentum in the gradient direction to smooth the update and avoid overfit local optima. The key insight: gradient momentum approximates the *average* gradient direction across the trajectory, smoothing out per-step model-specific noise.

**MI-FGSM update:**

$$g_{t+1} = \mu \cdot g_t + \frac{\nabla_x \mathcal{L}(x_t^{\text{adv}}, y)}{\|\nabla_x \mathcal{L}(x_t^{\text{adv}}, y)\|_1} \tag{1}$$

$$x_{t+1}^{\text{adv}} = \text{Clip}_{x,\epsilon}\!\left(x_t^{\text{adv}} + \alpha \cdot \text{sign}(g_{t+1})\right) \tag{2}$$

where:
- $g_t \in \mathbb{R}^n$ is the accumulated momentum gradient.
- $\mu \in [0, 1]$ is the momentum decay factor (typical: $\mu = 1.0$).
- $\alpha = \epsilon / T$ is the step size ($\epsilon$ is the total budget, $T$ the number of steps).
- $\text{Clip}_{x,\epsilon}$ clips to the L-infinity ball of radius $\epsilon$ around $x$ intersected with $[0,1]^n$.

**Normalization by $\ell_1$:** The gradient is normalized by its $\ell_1$ norm before accumulation, so each gradient has the same total magnitude regardless of the loss scale. This prevents early large-gradient steps from dominating.

**Initialization:** $g_0 = 0$, $x_0^{\text{adv}} = x$.

### 4.3 Why Momentum Helps Transfer

Claim: The momentum gradient $g_t$ approximates an average over the trajectory:

$$g_t \approx \sum_{s=0}^{t} \mu^{t-s} \cdot \frac{\nabla \mathcal{L}(x_s, y)}{\|\nabla \mathcal{L}(x_s, y)\|_1}$$

For $\mu = 1$: $g_t = \sum_{s=0}^{t} \frac{\nabla \mathcal{L}(x_s, y)}{\|\nabla \mathcal{L}(x_s, y)\|_1}$ — an equal-weight sum of all gradients seen so far.

**Geometric interpretation:** The gradient at each step varies as $x_t^{\text{adv}}$ moves. If the loss surface is locally smooth, nearby gradients point in similar directions. The sum amplifies the common direction (the one that all gradients agree on) and cancels the noise (model-specific gradient directions that vary across steps). The result is a gradient that:
1. Points toward high-loss regions that are robust across the trajectory.
2. Does not follow model-specific narrow canyons in the loss surface.

**Empirical evidence:** Dong et al. (2018) show MI-FGSM increases transfer from:
- 41.8% → 78.5% on average across 8 black-box models on ImageNet.
- The improvement is largest when transferring between architecturally dissimilar models.

### 4.4 TI-FGSM: Translation Invariant FGSM

Dong et al. (2019) proposed that adversarial examples should be robust to spatial translations of the input — because a translated version of the adversarial image might also fool the target model.

**Key observation:** The gradient $\nabla_x \mathcal{L}(x_t, y)$ is a function of the specific pixel arrangement of $x_t$. If we instead optimize for an adversarial example that is effective even after small translations, the resulting perturbation relies on structural features (patterns that appear at multiple locations) rather than pixel-specific features (features tied to an exact location).

**TI-FGSM update:** Convolve the gradient with a Gaussian kernel $W$ before taking the sign:

$$x_{t+1}^{\text{adv}} = \text{Clip}_{x,\epsilon}\!\left(x_t^{\text{adv}} + \alpha \cdot \text{sign}(W * \nabla_x \mathcal{L}(x_t, y))\right) \tag{3}$$

where $*$ denotes 2D convolution and $W$ is a Gaussian kernel (e.g., $7 \times 7$, $\sigma = 3$).

**Why this works:** Convolving the gradient with a Gaussian smooths it spatially, making each gradient element a weighted average of its neighbors. The resulting perturbation is more spatially coherent — it relies on regional patterns rather than individual pixel values. Such patterns are more transferable because they align with image features that different models have learned.

**Mathematical perspective:** Convolving the gradient corresponds to averaging the gradient over a distribution of small translations:

$$W * g \approx \mathbb{E}_{\Delta \sim P_\Delta}\!\left[\nabla_x \mathcal{L}(T_\Delta(x), y)\right]$$

where $T_\Delta$ is a translation by $\Delta$ and $P_\Delta$ is a distribution over small displacements. This is an instance of Expectation Over Transformation (EOT).

### 4.5 DI-FGSM: Diverse Inputs FGSM

Xie et al. (2019) proposed using input diversity — applying random transformations to the input before computing the gradient. The gradient is more robust to such transformations, so the adversarial examples rely on features that survive the transformations.

**DI-FGSM update:**

$$x_{t+1}^{\text{adv}} = \text{Clip}_{x,\epsilon}\!\left(x_t^{\text{adv}} + \alpha \cdot \text{sign}(\nabla_x \mathcal{L}(T(x_t^{\text{adv}};\; p), y))\right) \tag{4}$$

where $T(\cdot; p)$ is a random transformation applied with probability $p$ (else identity):
- Random resizing to a size in $[224, 331]$ (for $224 \times 224$ inputs), then zero-padding back to $331 \times 331$.
- Probability $p$ of applying (default $p = 0.5$).

**Why this helps:** Each step computes the gradient at a slightly different input. The optimization implicitly averages over a family of transformations, producing perturbations effective across this family. This reduces model-specific overfitting.

**Trade-off:** Transformation-averaged gradients are less powerful for white-box attack (the attack "wastes" budget on transformation robustness). But they are more transferable because the resulting adversarial example is effective against any model that fails on the family of transformations.

### 4.6 MI-TI-DI-FGSM: The Combined Attack

The attacks are composable. The full MI-TI-DI-FGSM combines:

$$g_{t+1} = \mu \cdot g_t + \frac{W * \nabla_x \mathcal{L}(T(x_t;\; p), y)}{\|W * \nabla_x \mathcal{L}(T(x_t;\; p), y)\|_1} \tag{5}$$

$$x_{t+1}^{\text{adv}} = \text{Clip}_{x,\epsilon}\!\left(x_t^{\text{adv}} + \alpha \cdot \text{sign}(g_{t+1})\right) \tag{6}$$

Each component addresses a different cause of poor transfer:
- **MI:** Smooths gradient direction over iterations (avoids model-specific local optima).
- **TI:** Smooths gradient spatially (avoids spatially local features).
- **DI:** Smooths gradient over transformations (avoids transformation-specific features).

**Empirical result (Dong et al. 2019, Xie et al. 2019):** MI-TI-DI-FGSM achieves ~80-95% untargeted transfer rates on standard ImageNet evaluation with $\epsilon = 16/255$, compared to ~40-60% for basic FGSM.

---

## 5. Ghost Networks: Ensemble-Based Transfer

### 5.1 The Idea

Li et al. (2020) proposed "Ghost Networks" — a method to simulate an ensemble of diverse models at inference time using a single model by applying random transformations to the model's intermediate representations.

**Mechanism:** Apply random erasing of activations (dropout) and random skip connections in intermediate layers of a ResNet. Each forward pass uses a different random mask, effectively creating a "ghost" of the original network with slightly different computation paths.

**Transfer advantage:** Adversarial examples crafted against an ensemble of ghost networks tend to transfer better because they must fool multiple distinct computation paths. The result is a perturbation that exploits features common to many network architectures, not just the specific paths of the surrogate.

### 5.2 Implementation

```python
class GhostNetwork(nn.Module):
    def __init__(self, model, skip_prob=0.3, dropout_prob=0.1):
        super().__init__()
        self.model = model
        self.skip_prob = skip_prob  # prob of activating skip connections
        self.dropout_prob = dropout_prob  # activation dropout rate

    def forward(self, x):
        # Apply random dropout to intermediate activations
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                h = module.register_forward_hook(
                    lambda m, inp, out: F.dropout(out, p=self.dropout_prob, training=True)
                )
                hooks.append(h)
        out = self.model(x)
        for h in hooks:
            h.remove()
        return out
```

---

## 6. Feature-Level Transfer Attacks

### 6.1 Motivation

Pixel-level adversarial perturbations may fool one model but not another because different models extract different pixel-level features. However, if two models are trained on the same task, their *intermediate feature representations* may be more similar than their input-space gradients suggest.

**Feature-level transfer attack:** Craft a perturbation that maximizes the change in intermediate feature representations:

$$\max_{\|\delta\|_\infty \leq \epsilon} \; \mathcal{L}_{\text{feat}}(x + \delta, x) = \|F_A(x + \delta) - F_A(x)\|_2^2 \tag{7}$$

where $F_A(x)$ is the activation of a specific layer of model $A$ at input $x$. The intuition is that changing the feature representation maximally will also change the features in model $B$ if the two models have learned similar features at that level.

### 6.2 Intermediate-Level Attack (ILA)

Huang et al. (2019) proposed the Intermediate Level Attack: first craft a regular adversarial perturbation $\delta_0$ (using FGSM), then fine-tune $\delta$ to maximize the inner product of $\delta$ with the feature-space perturbation direction of $\delta_0$:

$$\max_{\|\delta\|_\infty \leq \epsilon} \; \langle F_A(x + \delta) - F_A(x),\; \Delta_0 \rangle \tag{8}$$

where $\Delta_0 = F_A(x + \delta_0) - F_A(x)$ is the feature-space displacement of the initial attack.

The idea: $\Delta_0$ points in the direction that the features change under adversarial attack. We want to maximize the projection of the new feature change onto this direction, encouraging the model to make the same "mistake" in feature space.

---

## 7. The Surrogate Model Problem

### 7.1 What Makes a Good Surrogate?

The surrogate model is the model on which the attacker crafts adversarial examples. Key properties:
1. **Similar architecture:** ResNet surrogates transfer better to ResNet targets than to ViT targets.
2. **Similar training distribution:** A surrogate trained on ImageNet transfers better to a target trained on ImageNet than to one trained on a proprietary dataset.
3. **Similar output space:** A surrogate must have the same (or overlapping) class set.
4. **Smoother loss landscape:** Adversarially trained surrogates often produce more transferable attacks because their loss landscapes have sharper, more informative gradients.

### 7.2 Multiple Surrogate Ensemble

A classic strategy (Liu et al. 2016): attack an ensemble of surrogates:

$$\max_{\|\delta\|_\infty \leq \epsilon} \; \sum_{i=1}^M w_i \cdot \mathcal{L}(f_i(x + \delta), y) \tag{9}$$

where $\{f_1, \ldots, f_M\}$ are diverse surrogate models and $w_i$ are weights (often uniform). An adversarial example that fools all surrogates is more likely to transfer to an unknown target.

**Practical guideline:** Using 3-5 diverse surrogates (e.g., ResNet-50, DenseNet-121, VGG-19) typically saturates transfer improvement; adding more models yields diminishing returns.

---

## 8. Demontis et al. (2019): What Makes a Surrogate Good?

### 8.1 The Theoretical Framework

Demontis et al. (2019) provide a formal analysis of transferability by deriving bounds on the expected transfer rate.

**Setup:** Source model $f_A$, target model $f_B$, adversarial perturbation $\delta^* = \arg\min_{\|\delta\|_\infty \leq \epsilon} \mathcal{L}_A(x + \delta, y)$.

**Main theorem (informal):** The transfer rate from $f_A$ to $f_B$ depends on:
1. **Gradient alignment:** $\cos(\nabla_x \mathcal{L}_A, \nabla_x \mathcal{L}_B)$ — how aligned the gradients are in the direction of the attack.
2. **Input loss variability of $f_A$:** $\text{Var}_x[\mathcal{L}_A(x, y)]$ — how much the loss varies across inputs. Models with more uniform loss distributions transfer better.
3. **Lipschitz constant of $f_B$:** How smooth $f_B$ is. A smoother target model is harder to fool but also less affected by transferred perturbations.

### 8.2 The Variability Insight

The most actionable finding: surrogates with **lower input loss variability** produce more transferable attacks. A surrogate where the loss is "everywhere high" (not concentrated in a few input directions) produces perturbations that are effective in many directions, making them more likely to align with the target model's vulnerability.

**Practical recommendation:** Prefer surrogates with:
- Lower training accuracy on clean inputs (they have higher and more spread-out loss gradients).
- Trained with data augmentation (more uniform loss landscape).
- Adversarially trained (smoother loss landscape with informative gradients).

### 8.3 Gradient Alignment in Practice

For ResNet-50 vs. ResNet-101 (same architecture, similar weights): gradient cosine similarity $\approx 0.7$–0.9. High transfer.

For ResNet-50 vs. ViT-B/16 (different architecture): gradient cosine similarity $\approx 0.1$–0.3. Low transfer.

The MI-FGSM techniques in Section 4 effectively increase the *expected gradient alignment* by smoothing the gradient direction over multiple steps and transformations, making it more likely to align with the target model's gradient.

---

## 9. Worked Example: Computing MI-FGSM Transfer

### Setup
- Source: ResNet-18 (weights from torchvision, fine-tuned on CIFAR-10).
- Target: DenseNet-121 (separate training run on CIFAR-10).
- Input: CIFAR-10 test image, class "airplane", true label $y = 0$.
- Budget: $\epsilon = 8/255$, step size $\alpha = 2/255$, $T = 10$ iterations, $\mu = 1.0$.

### Step-by-Step Trace

**Initialize:** $x_0^{\text{adv}} = x$, $g_0 = 0$.

**Iteration 1:**
1. Compute $\nabla_x \mathcal{L}_{\text{CE}}(x_0^{\text{adv}}, y)$ using ResNet-18. Let $\bar{g}_1 = \nabla \mathcal{L} / \|\nabla \mathcal{L}\|_1$.
2. $g_1 = 1.0 \cdot 0 + \bar{g}_1 = \bar{g}_1$.
3. $x_1^{\text{adv}} = \text{Clip}_{x,8/255}(x_0^{\text{adv}} + (2/255) \cdot \text{sign}(g_1))$.

**Iteration 5 (typical state):**
- $g_5 = \sum_{s=1}^{5} \bar{g}_s$ (sum of normalized gradients, since $\mu = 1$).
- The direction of $g_5$ is a vote among 5 gradient directions. If all gradients point roughly in the same direction, $\|g_5\|$ is large and the sign update is well-defined.
- If gradients oscillate, $g_5$ is small in magnitude and the sign is unreliable — but this is also a signal that the optimization is at a flat region.

**At $T = 10$:** $x_{10}^{\text{adv}}$ is the adversarial example. Evaluate: $C_{A}(x_{10}^{\text{adv}}) \neq y$ (white-box success) and $C_{B}(x_{10}^{\text{adv}}) \neq y$ (transfer success).

**Typical results on CIFAR-10:**

| Attack | White-box (ResNet-18) | Transfer to DenseNet-121 |
|--------|----------------------|--------------------------|
| FGSM ($T=1$, no momentum) | 55% | 22% |
| PGD-10 | 85% | 18% |
| MI-FGSM ($T=10$, $\mu=1$) | 75% | 38% |
| MI-TI-DI-FGSM ($T=10$) | 70% | 52% |

Note the trade-off: MI-FGSM is weaker in white-box than PGD-10, but much better in transfer.

---

## 10. Discussion Questions

1. **Overfitting paradox:** PGD-100 is a stronger white-box attack than FGSM but transfers worse. Formalize this as an overfitting argument: define what it means for an attack to "overfit" to a surrogate model. Under what conditions would adding more iterations to PGD improve transfer instead of hurting it?

2. **Gradient alignment:** Compute the cosine similarity between:
   - $\text{sign}(\nabla_x \mathcal{L}_{f_A})$ and $\nabla_x \mathcal{L}_{f_B}$ for a 2-class linear classifier where $f_A(x) = w_A^\top x$ and $f_B(x) = w_B^\top x$, with $w_A = [1, 0, \ldots, 0]$ and $w_B = [\cos\theta, \sin\theta, 0, \ldots, 0]$.
   - What is the transfer rate of FGSM on $f_A$ to $f_B$ as a function of $\theta$? At what angle $\theta$ does transfer fail?

3. **DI-FGSM diversity:** The DI-FGSM applies random resizing and padding. Explain why zero-padding (not reflective or circular padding) is preferred. How does the choice of padding affect the frequency content of the adversarial perturbation?

4. **TI-FGSM kernel:** The TI-FGSM convolves with a Gaussian kernel. What is the effect of increasing $\sigma$ (kernel width) on:
   (a) The frequency content of the adversarial perturbation.
   (b) The white-box attack success rate.
   (c) The transfer success rate.
   Is there an optimal $\sigma$ for a given pair of source/target architectures?

5. **Feature vs. pixel attacks:** Suppose you know the target model uses ResNet-50 and the intermediate features after layer 3 are publicly accessible (e.g., via an embedding API). Design a feature-level attack that uses these features. How does this compare to a pixel-level transfer attack in terms of expected success rate?

6. **Heterogeneous ensemble surrogate:** You have access to five ResNets and one ViT. How would you weight them in the ensemble attack objective (equation 9) to maximize transfer to an unknown target? What information about the target (if any) would change your weighting strategy?

7. **Physical transfer:** Suppose you print a transferred adversarial patch on a stop sign. The patch was crafted using MI-TI-DI-FGSM with $\epsilon = 32/255$. List three physical effects (viewing angle variation, lighting changes, camera sensor noise, etc.) that could destroy the adversarial property. For each, propose a modification to the attack that would make it more robust to that effect.

---

## 11. Further Reading

**Required:**
- Dong et al. (2018). "Boosting Adversarial Attacks with Momentum." CVPR. [MI-FGSM]
- Dong et al. (2019). "Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks." CVPR. [TI-FGSM]
- Xie et al. (2019). "Improving Transferability of Adversarial Examples with Input Diversity." CVPR. [DI-FGSM]
- Demontis et al. (2019). "Why Do Adversarial Attacks Transfer? Explaining Transferability of Evasion and Poisoning Attacks." USENIX Security. [Theoretical analysis]

**Recommended:**
- Szegedy et al. (2013/2014). "Intriguing Properties of Neural Networks." ICLR 2014. [Original transferability observation]
- Liu et al. (2016). "Delving into Transferable Adversarial Examples and Black-box Attacks." ICLR 2017. [Ensemble surrogates]
- Ilyas et al. (2019). "Adversarial Examples Are Not Bugs, They Are Features." NeurIPS. [Non-robust features hypothesis]
- Papernot et al. (2017). "Practical Black-Box Attacks Against Machine Learning." ASIACCS. [Real-world transferability demonstration]
- Li et al. (2020). "Learning Transferable Adversarial Examples via Ghost Networks." AAAI. [Ghost networks]
- Huang et al. (2019). "Enhancing Adversarial Example Transferability with an Intermediate Level Attack." ICCV. [Feature-level ILA]
