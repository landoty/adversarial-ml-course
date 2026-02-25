# PS1 Lab Report: PGD-AT vs. TRADES Adversarial Training on CIFAR-10

**Course:** CS 6820 — Defenses and Robustness in ML
**Assignment:** Problem Set 1
**Dataset:** CIFAR-10
**Architecture:** ResNet-18

---

## Abstract

We implement and compare two foundational adversarial training methods — PGD-AT (Madry et al., 2018) and TRADES (Zhang et al., 2019) — on CIFAR-10 using a ResNet-18 architecture. Training is performed at L∞ perturbation budgets of ε = 2/255 and ε = 8/255 for PGD-AT, and ε = 8/255 with β ∈ {1, 6, 10} for TRADES. Models are evaluated on natural accuracy and robust accuracy under PGD-20 and AutoAttack. TRADES with β = 6 achieves the best robust accuracy (45.8% AutoAttack) while maintaining 82.8% natural accuracy, outperforming PGD-AT at ε = 8/255 (42.1% AutoAttack, 84.6% natural) on the robustness axis. Increasing β from 6 to 10 yields marginal additional robustness (46.2%) at a cost of 3.4 percentage points of natural accuracy. We analyze the natural-robust accuracy tradeoff across methods and discuss practical deployment recommendations.

---

## Introduction

Standard deep neural networks are highly vulnerable to adversarial examples — imperceptible perturbations to inputs that cause confident misclassification (Goodfellow et al., 2015; Szegedy et al., 2014). A ResNet-18 trained standardly on CIFAR-10 achieves 93.4% natural accuracy but drops to near 0% robust accuracy under even a weak 20-step PGD attack at ε = 8/255, as demonstrated in our experiments. This gap motivates the development of training procedures that incorporate adversarial examples during optimization.

**Adversarial training** is the most practically successful empirical defense against adversarial examples. Rather than modifying the model architecture or adding preprocessing, adversarial training modifies the training objective to minimize loss over adversarially perturbed inputs, forcing the model to learn representations that are robust to worst-case perturbations within a specified threat model.

### PGD-AT

Madry et al. (2018) formalize adversarial training as the following min-max optimization:

$$\min_\theta \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \max_{\|\delta\|_\infty \leq \varepsilon} L(f_\theta(x + \delta), y) \right]$$

The inner maximization is approximated using Projected Gradient Descent (PGD), which iteratively applies:
$$\delta^{t+1} = \Pi_{\|\cdot\|_\infty \leq \varepsilon}\left(\delta^t + \alpha \cdot \text{sign}(\nabla_\delta L(f_\theta(x + \delta^t), y))\right)$$

The outer minimization then trains the model to correctly classify the worst-case perturbed inputs found by the inner PGD. This approach treats adversarial examples as hard negatives and uses them directly as training data.

**Key properties of PGD-AT:**
- Inner loop PGD generates adversarial examples using the current model weights, creating a curriculum that adapts to the model's current vulnerabilities.
- Training with hard adversarial labels (i.e., using the true label y for the perturbed input) can cause label leakage for examples where the perturbation crosses a decision boundary.
- Computationally, each training step requires T inner PGD steps, increasing wall-clock time by a factor of roughly T relative to standard training.

### TRADES

Zhang et al. (2019) observe that PGD-AT conflates natural accuracy optimization with robustness optimization, leading to suboptimal tradeoffs. TRADES separates these objectives explicitly:

$$\min_\theta \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ L_{CE}(f_\theta(x), y) + \beta \cdot \max_{\|\delta\|_\infty \leq \varepsilon} \text{KL}(f_\theta(x+\delta) \| f_\theta(x)) \right]$$

The first term optimizes natural accuracy on clean inputs using the standard cross-entropy loss. The second term penalizes the KL divergence between the model's predictions on perturbed and clean inputs, promoting local smoothness. The hyperparameter β controls the tradeoff: larger β places more weight on robustness regularization at the cost of natural accuracy.

**Key differences from PGD-AT:**
- TRADES uses the clean prediction `f_θ(x)` as the reference distribution in the KL term, not the hard label y. This avoids the label leakage problem because the adversarial perturbation is directed at maximizing divergence from the clean prediction rather than inducing misclassification.
- The decomposed objective allows independent analysis and optimization of natural accuracy and robustness components.
- β provides a continuous knob to navigate the Pareto frontier between natural and robust accuracy.

---

## Methods

### Training Setup

All models use a ResNet-18 architecture. Training uses SGD with momentum 0.9, weight decay 5 × 10⁻⁴, initial learning rate 0.1, and a cosine annealing learning rate schedule over 200 epochs. Batch size is 128. CIFAR-10 standard data augmentation (random horizontal flip and random crop with 4-pixel padding) is applied to clean inputs in all conditions.

### PGD-AT Configuration

Inner PGD: 10 steps, step size α = 2/255, perturbation budget ε ∈ {2/255, 8/255}, random initialization within the ε-ball. The training loss is the cross-entropy on adversarial examples: `L_train = CE(f_θ(x + δ*), y)` where `δ* = argmax_{‖δ‖∞ ≤ ε} CE(f_θ(x + δ), y)`.

### TRADES Configuration

Inner PGD: 10 steps, step size α = 2/255, perturbation budget ε = 8/255, random initialization. The inner maximization targets KL divergence: `δ* = argmax_{‖δ‖∞ ≤ ε} KL(f_θ(x+δ) ‖ f_θ(x))`. β ∈ {1, 6, 10}; β = 6 is the canonical recommendation from Zhang et al. (2019).

### Evaluation

Trained models are evaluated on:
- **Natural accuracy:** test set accuracy on clean (unperturbed) inputs.
- **PGD-20 robust accuracy:** accuracy under 20-step PGD, 5 random restarts, step size 2/255, ε = 8/255.
- **AutoAttack robust accuracy:** accuracy under the full AutoAttack ensemble (APGD-CE, APGD-DLR, FAB, Square Attack), which provides a reliable upper bound on robust accuracy without cherry-picking attack parameters.

Training time is measured on a single NVIDIA RTX 3090.

---

## Results

| Method | ε (train) | Natural Acc (%) | PGD-20 Acc (%) | AutoAttack Acc (%) | Train Time (hrs) |
|--------|-----------|----------------|----------------|-------------------|-----------------|
| Standard | — | 93.4 | 0.2 | 0.0 | 1.2 |
| PGD-AT | 2/255 | 90.1 | 74.3 | 67.8 | 4.1 |
| PGD-AT | 8/255 | 84.6 | 47.2 | 42.1 | 4.3 |
| TRADES β=1 | 8/255 | 87.3 | 44.1 | 39.7 | 4.5 |
| TRADES β=6 | 8/255 | 82.8 | 50.3 | 45.8 | 4.5 |
| TRADES β=10 | 8/255 | 79.4 | 51.7 | 46.2 | 4.6 |

### Natural-Robust Accuracy Tradeoff (Pareto Frontier)

A key empirical phenomenon visible in these results is the **natural-robust accuracy tradeoff**: no method simultaneously achieves the highest natural accuracy and the highest robust accuracy. The Pareto frontier traces the achievable combinations:

- Standard training: (93.4% natural, 0.0% AutoAttack robust) — maximum natural accuracy, zero robustness.
- PGD-AT ε=2/255: (90.1%, 67.8%) — mild robustness at small perturbation budget, natural accuracy well preserved.
- PGD-AT ε=8/255: (84.6%, 42.1%) — robustness against ε=8/255, natural accuracy drops ~9 points.
- TRADES β=6: (82.8%, 45.8%) — slightly lower natural accuracy than PGD-AT but higher robustness.
- TRADES β=10: (79.4%, 46.2%) — further shift toward robustness; diminishing returns in robustness, accelerating cost in natural accuracy.

TRADES with β = 6 and β = 10 lie on or near the Pareto frontier at ε = 8/255, while PGD-AT at ε = 8/255 is slightly dominated by TRADES β=6 (higher robustness at similar natural accuracy). TRADES β=1 is dominated: it achieves neither the natural accuracy of standard training nor the robustness of PGD-AT ε=8/255, suggesting the regularization weight is too low to meaningfully shape the loss landscape.

---

## Analysis

### Why TRADES Outperforms PGD-AT on Robust Accuracy

The superior robust accuracy of TRADES (β=6: 45.8%) over PGD-AT (42.1%) at the same perturbation budget can be attributed to two mechanisms:

**1. KL Divergence vs. Hard Labels for Adversarial Examples.**
PGD-AT trains on the cross-entropy loss `CE(f_θ(x + δ*), y)` where y is the true label. When x + δ* crosses a decision boundary (i.e., the natural prediction is already wrong), the gradient signal is inconsistent: the model is asked to be confident in the true label for an input that, geometrically, might lie in another class's region. This is the **label leakage** problem identified by Zhang et al. (2019), and it creates conflicting gradient signals that slow convergence and reduce robustness.

TRADES replaces this with `KL(f_θ(x+δ) ‖ f_θ(x))`, which asks the model to be locally consistent — specifically, to produce similar predictions on perturbed and clean inputs — regardless of the label. This is a smoother, more geometrically coherent objective: the model only needs to be "locally Lipschitz" around each clean example, not necessarily confident in the true label everywhere in the perturbation ball. The result is a smoother decision surface that is harder to exploit.

**2. Separation of Natural and Robust Objectives.**
In PGD-AT, the model is updated only on adversarial examples `x + δ*`, with the implicit assumption that robustness on adversarial examples implies accuracy on clean examples. In practice, this conflation can cause the model to overfit to the adversarial distribution at the expense of the clean distribution. TRADES's explicit `L_CE(f_θ(x), y)` term ensures that the model is also updated on clean examples at every step, preserving the natural accuracy signal throughout training.

### The β Tradeoff: Robustness vs. Natural Accuracy

Increasing β shifts the training objective toward robustness regularization and away from natural accuracy:

- β=1: Insufficient regularization — the KL term is too weak to produce a meaningfully smoother model, yielding robustness (39.7%) below PGD-AT and natural accuracy (87.3%) that is also below standard training. This suggests the model learns neither clean features nor robust ones effectively.
- β=6: Recommended setting — provides the best robust accuracy (45.8%) with a natural accuracy penalty of ~10.6 points relative to standard training. The KL regularization is strong enough to create smooth local decision boundaries without catastrophically degrading clean feature learning.
- β=10: Marginal robustness gain (+0.4% AutoAttack over β=6) at the cost of 3.4 points of natural accuracy and slightly higher training time. The diminishing returns suggest the KL term begins to dominate excessively, overly smoothing the model and erasing natural accuracy signals.

The relationship between β and the natural-robust accuracy tradeoff is approximately monotone in this range, consistent with the theoretical analysis in Zhang et al. (2019) that shows the TRADES objective upper-bounds the robustness error by a term proportional to β⁻¹.

### Computational Overhead

All adversarial training methods incur roughly 3–4× training time overhead relative to standard training, due to the inner PGD loop. PGD-AT at ε=2/255 and ε=8/255 have nearly identical wall-clock times (4.1 vs. 4.3 hours), as both use 10 inner steps. TRADES is slightly slower (4.5–4.6 hours) due to the separate KL loss computation and an additional forward pass for the clean prediction. The differences between adversarial training variants are small (< 15%) relative to the baseline overhead; the dominant cost is the inner PGD loop, which is shared across all methods.

### PGD-20 vs. AutoAttack Robustness Gap

A consistent pattern is that PGD-20 robust accuracy is 4–8 percentage points higher than AutoAttack robust accuracy across all methods. This gap reflects the fact that PGD-20 is an adaptive but single-attack evaluation, while AutoAttack combines multiple parameter-free attacks including Square Attack (black-box), which does not use gradient information. Models that partially mask gradients or exhibit narrow loss landscape structure can fool PGD-20 but not AutoAttack's diverse attack ensemble. AutoAttack is therefore the more reliable robustness metric and should be the primary evaluation standard for published results.

---

## Conclusion

TRADES with β = 6 provides the best practical balance between natural accuracy (82.8%) and robust accuracy (45.8% AutoAttack) on CIFAR-10 at ε = 8/255, outperforming PGD-AT at the same perturbation budget on the robustness axis while incurring only marginally higher training time. The β = 10 setting yields negligible additional robustness at a meaningful accuracy cost and is not recommended for deployment. PGD-AT at ε = 2/255 is appropriate for settings where robustness is needed only against very small perturbations and natural accuracy is a priority (90.1% natural, 67.8% robust at ε = 2/255).

For deployment, we recommend TRADES β = 6 trained at ε = 8/255 as the default adversarially trained model for CIFAR-10-scale vision tasks. AutoAttack evaluation should be used as the standard robustness benchmark, and PGD-20 results should be reported alongside AutoAttack for reproducibility but not treated as the primary measure of robustness. Future work should evaluate TRADES and PGD-AT on larger architectures (e.g., WideResNet-28-10) and larger datasets (CIFAR-100, ImageNet), where the natural-robust accuracy tradeoff may differ substantially.

---

*References:*
- Goodfellow, I.J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. ICLR 2015.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR 2018.
- Szegedy, C., et al. (2014). Intriguing Properties of Neural Networks. ICLR 2014.
- Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., & Jordan, M. (2019). Theoretically Principled Trade-off between Robustness and Accuracy. ICML 2019.
- Croce, F., & Hein, M. (2020). Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks. ICML 2020.
