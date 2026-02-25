# PA1: Implementing C&W L2 and L∞ Attacks from Scratch on CIFAR-10

**Course:** CS 6810 — Adversarial ML: Attacks
**Assignment:** Programming Assignment 1

---

## Abstract

This report presents an implementation of the Carlini & Wagner (C&W) L2 and L∞ adversarial attack algorithms from scratch, benchmarked against FGSM, BIM, and PGD-40 on the CIFAR-10 dataset using a pretrained ResNet-18 classifier. The C&W attacks are formulated as constrained optimization problems solved via Adam with a tanh reparameterization to enforce box constraints and a binary search procedure over the trade-off constant $c$. Experiments on 1000 CIFAR-10 test images demonstrate that C&W L2 achieves a 91.3% attack success rate with a mean L2 distortion of 0.43, substantially lower than PGD-40's mean L2 distortion of 0.82 at a comparable 79.8% success rate. These results confirm the key insight from Carlini & Wagner (2017): directly minimizing perturbation norm as an objective rather than treating it as a constraint yields significantly smaller distortions. We verify our implementation against CleverHans, finding success rates within 1%, confirming correctness. An ablation study examines the effect of binary search iterations, learning rate, and the confidence margin hyperparameter $\kappa$ on attack performance.

---

## 1. Introduction

Adversarial attacks on neural networks expose a fundamental tension in deep learning: models that achieve near-human accuracy on clean data can be fooled by imperceptible perturbations. The Goodfellow et al. (2015) Fast Gradient Sign Method (FGSM) and its iterative extensions — BIM and PGD — formulate the attack as finding a perturbation $\delta$ with bounded $L\infty$ norm that maximizes classification loss. These methods are computationally efficient and widely used in adversarial training pipelines. However, they are not designed to minimize perturbation magnitude; they merely respect a prescribed budget $\varepsilon$.

Carlini & Wagner (2017) introduced a different objective: rather than maximizing loss subject to a norm constraint, directly minimize the perturbation norm plus a margin-based objective that ensures misclassification. This formulation finds adversarial examples that are, in a precise sense, the *closest* adversarial examples to the original input — making C&W attacks particularly powerful evaluation tools. In the years since, C&W has been used as a standard benchmark and as a component of evaluation frameworks such as AutoAttack.

**Contributions of this work:**

1. A clean, well-documented PyTorch implementation of C&W L2 and C&W L∞ from scratch, without relying on adversarial robustness libraries.
2. A systematic comparison of C&W against FGSM, BIM, and PGD-40 on CIFAR-10.
3. An ablation study of binary search iterations, Adam learning rate, and the confidence margin $\kappa$.
4. Verification against CleverHans to confirm implementation correctness.

---

## 2. Methods

### 2.1 C&W L2 Attack Formulation

The C&W L2 attack solves the following optimization problem:

$$\min_{\delta} \|\delta\|_2^2 + c \cdot f(x + \delta)$$

subject to $x + \delta \in [0, 1]^d$, where the objective function $f$ is defined as:

$$f(x') = \max\!\left(\max_{j \neq t} Z(x')_j - Z(x')_t,\ -\kappa\right)$$

Here $Z(x')_j$ denotes the pre-softmax logit for class $j$, $t$ is the target class, $\kappa \geq 0$ is a confidence margin hyperparameter, and $c > 0$ is a trade-off constant balancing distortion against the misclassification objective. When $f(x') < 0$, the input $x'$ is already classified as class $t$ with margin at least $\kappa$, so the attack objective is satisfied.

**Box Constraint via Tanh Reparameterization.** To enforce $x' \in [0,1]^d$ without projected gradient descent, C&W introduces a change of variables:

$$x' = \frac{1}{2}\left(\tanh(w) + 1\right), \qquad \delta = x' - x$$

The optimization variable is $w \in \mathbb{R}^d$, which is unconstrained. The inverse mapping from a valid input $x$ to $w$ is $w = \tanh^{-1}(2x - 1)$. Gradients flow through the tanh without requiring projection steps.

**Binary Search over $c$.** The constant $c$ governs the trade-off between minimizing distortion and achieving misclassification. A single fixed value of $c$ may be too aggressive (high distortion) or too lenient (attack fails). We perform binary search over $c$ for each input: starting with bounds $[c_{\min}, c_{\max}] = [10^{-3}, 10^{10}]$, we run the inner Adam optimization for each candidate $c$ and update the bounds based on whether the attack succeeded.

**Inner Optimization.** For each value of $c$, we run 1000 iterations of Adam with learning rate $\alpha = 0.01$. We track the best adversarial example found (lowest $\|\delta\|_2$ subject to $f(x') < 0$) across all iterations.

### 2.2 C&W L∞ Attack Formulation

The C&W L∞ attack minimizes $\|\delta\|_\infty$ subject to misclassification. This is harder to optimize directly because the $L\infty$ norm is not differentiable everywhere. The original C&W paper introduces a slack variable formulation: for a threshold $\tau$, define the penalized objective

$$\min_{w} c \cdot f(x') + \sum_i \max(|x'_i - x_i| - \tau,\ 0)$$

The attack iteratively decreases $\tau$ until the smallest achievable $L\infty$ distortion that still fools the classifier is found. Each stage runs Adam optimization with the slack penalty; if the attack succeeds at threshold $\tau$, we reduce $\tau$ by a factor and repeat.

### 2.3 Experimental Setup

- **Dataset:** 1000 randomly sampled CIFAR-10 test images (100 per class), all correctly classified by the clean model.
- **Model:** Pretrained ResNet-18 achieving 93.4% clean accuracy on CIFAR-10.
- **Attack budgets:** $\varepsilon \in \{8/255, 16/255\}$ for FGSM/BIM/PGD; C&W is unconstrained but we report distortions.
- **C&W hyperparameters:** Binary search over $c$ with 10 iterations; Adam LR = 0.01; 1000 inner iterations; $\kappa = 0$.
- **Baseline attacks:** FGSM with $\varepsilon = 8/255$; BIM with $\varepsilon = 8/255$, step size $2/255$, 40 steps; PGD-40 with $\varepsilon = 8/255$, step size $2/255$, random restarts = 1.
- **Success criterion:** Attack succeeds if the model misclassifies the adversarial example (untargeted).

---

## 3. Results

### 3.1 Attack Performance Comparison

| Attack | Success Rate (%) | Mean $L_2$ Distortion | Mean $L_\infty$ Distortion | Iterations |
|---|---|---|---|---|
| FGSM ($\varepsilon = 8/255$) | 43.2 | 0.89 | 0.031 | 1 |
| BIM ($\varepsilon = 8/255$) | 71.4 | 0.76 | 0.031 | 40 |
| PGD-40 ($\varepsilon = 8/255$) | 79.8 | 0.82 | 0.031 | 40 |
| C&W L2 ($c = 1.0$) | 91.3 | **0.43** | 0.018 | 1000 |
| C&W L∞ ($\tau = 0.03$) | 88.7 | 0.51 | **0.028** | 1000 |

C&W L2 achieves the highest success rate (91.3%) while simultaneously producing the smallest mean L2 distortion (0.43), nearly half that of BIM (0.76) and PGD-40 (0.82). This validates the core claim: directly minimizing distortion as an objective yields qualitatively smaller perturbations than $L\infty$-constrained attacks.

### 3.2 Convergence Analysis

The C&W objective function value $\|\delta\|_2^2 + c \cdot f(x')$ decreases monotonically over the 1000 Adam iterations, with rapid initial descent in the first 200 iterations followed by a slower refinement phase. The binary search over $c$ exhibits the expected behavior: early binary search iterations use large $c$, achieving misclassification quickly but with large distortion; later iterations reduce $c$, tightening the distortion while maintaining success. The final binary search iteration typically finds $c \in [0.5, 2.0]$ for CIFAR-10 inputs.

---

## 4. Ablation Study

### 4.1 Effect of Binary Search Iterations

| Binary Search Iters | Success Rate (%) | Mean $L_2$ Distortion |
|---|---|---|
| 5 | 89.1 | 0.51 |
| 10 | 91.3 | 0.43 |
| 20 | 91.6 | 0.41 |

Increasing binary search iterations from 5 to 10 provides a meaningful improvement in distortion quality (0.51 → 0.43) with modest cost. The improvement from 10 to 20 iterations is marginal, suggesting 10 iterations is a practical sweet spot for CIFAR-10 scale experiments.

### 4.2 Effect of Adam Learning Rate

Learning rates in $\{0.001, 0.005, 0.01, 0.05\}$ were evaluated. The rate $\alpha = 0.01$ achieves the best balance: lower rates ($\alpha = 0.001$) fail to converge in 1000 iterations, while higher rates ($\alpha = 0.05$) overshoot and yield noisier perturbations. The success rate varies by approximately 3% across the range.

### 4.3 Effect of Confidence Margin $\kappa$

| $\kappa$ | Success Rate (%) | Mean $L_2$ Distortion | Transferability (to VGG-16) |
|---|---|---|---|
| 0 | 91.3 | 0.43 | 31.2% |
| 5 | 90.8 | 0.56 | 44.7% |
| 20 | 87.4 | 0.71 | 51.3% |

Higher $\kappa$ requires the attack to push the target logit further above all others, increasing distortion but producing adversarial examples that transfer better to other architectures. This reflects the known phenomenon that high-confidence adversarial examples tend to be more transferable, as they must exploit more generalizable features of the input.

---

## 5. Comparison with CleverHans

To verify correctness, we ran CleverHans's C&W L2 implementation on the same 1000 images with identical hyperparameters ($c = 1.0$, $\kappa = 0$, 1000 iterations, Adam LR = 0.01). Our implementation achieves 91.3% success rate versus CleverHans's 91.8% — a difference of 0.5 percentage points, attributable to minor numerical differences in the tanh reparameterization and floating point ordering. Mean L2 distortions agree to within 0.02. We consider this sufficient to confirm correctness.

---

## 6. Discussion

**Why C&W finds smaller distortions than PGD.** PGD attacks are formulated to maximize loss subject to an $L\infty$ norm constraint. The optimal PGD solution fills the entire $\varepsilon$-ball as needed; it does not minimize the perturbation size. C&W, by contrast, directly minimizes $\|\delta\|_2^2$ as part of the objective, so the optimizer is actively pushed toward smaller perturbations. Even accounting for the different norm ($L_2$ vs. $L\infty$), C&W finds adversarial examples that are substantially closer to the original in both norms.

**Role of the confidence parameter $\kappa$.** The parameter $\kappa \geq 0$ controls how far into the target class the attack pushes the example. With $\kappa = 0$, the attack stops as soon as the target logit marginally exceeds all others — the minimum requirement for misclassification. Higher $\kappa$ produces "deeply adversarial" examples that the model classifies with high confidence in the wrong class, at the cost of larger distortions. Practitioners running evaluations should use $\kappa = 0$ for distortion measurement and higher $\kappa$ when studying transferability.

**Computational cost.** C&W is significantly more expensive than single-step or 40-step attacks. At 1000 iterations per example with 10 binary search steps, the wall-clock time is approximately 150× that of PGD-40 for the same batch. This makes C&W impractical for adversarial training (which requires attacks at every training step) but appropriate for evaluation, where attacks are computed once.

---

## 7. Conclusion

We implemented C&W L2 and L∞ attacks from scratch in PyTorch and benchmarked them against FGSM, BIM, and PGD-40 on CIFAR-10. C&W L2 achieves a 91.3% attack success rate with mean L2 distortion 0.43 — the highest success rate and lowest distortion among all methods tested. The tanh reparameterization cleanly enforces box constraints, and binary search over $c$ reliably finds a good trade-off constant without manual tuning. Our implementation matches CleverHans to within 1%, confirming correctness. These results underscore the importance of using optimization-based attacks like C&W for evaluation, as norm-constrained attacks such as PGD underestimate model vulnerability when distortion quality is the criterion of interest.

---

## References

1. Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks. *IEEE Symposium on Security and Privacy*.
2. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. *ICLR*.
3. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. *ICLR*.
4. Kurakin, A., Goodfellow, I. J., & Bengio, S. (2017). Adversarial examples in the physical world. *ICLR Workshop*.
5. Papernot, N., Faghri, F., Carlini, N., Goodfellow, I., Feinman, R., Kurakin, A., ... & McDaniel, P. (2018). Technical report on the CleverHans v2.1.0 adversarial examples library. *arXiv:1610.00768*.
