# PA2: MI-FGSM Transfer Attacks and Cross-Architecture Transferability

**Course:** CS 6810 — Adversarial ML: Attacks
**Assignment:** Programming Assignment 2

---

## Abstract

This report investigates black-box adversarial transferability using the Momentum Iterative FGSM (MI-FGSM) algorithm across four architectures: ResNet-18, ViT-B/16, MobileNetV2, and DenseNet-121. We construct a 4×4 transferability matrix measuring attack success rate when adversarial examples crafted on one surrogate model are applied to a different target model, all without access to the target model's gradients. MI-FGSM achieves substantially higher transfer rates than vanilla FGSM — 61.4% versus 41.7% on the ResNet-18 → MobileNetV2 transfer pair — owing to momentum accumulation that dampens oscillations in the loss landscape and produces more generalizable perturbations. A striking asymmetry is observed: CNN architectures transfer well to each other but Vision Transformers (ViT) transfer poorly to CNNs and vice versa, consistent with the hypothesis that CNNs learn local texture features while ViTs learn global shape features. These results demonstrate that architecture family is a critical factor in transfer attack effectiveness.

---

## 1. Introduction

The white-box threat model — in which the attacker has full access to the model's architecture, weights, and gradients — provides an upper bound on attack effectiveness but does not reflect the constraints of real-world adversaries. In practice, deployed machine learning systems are often accessible only through prediction APIs, with model internals hidden from the attacker. This is the **black-box threat model**.

A key enabler of black-box attacks is **adversarial transferability**: adversarial examples crafted on a *surrogate* model (one the attacker controls) often fool *target* models (deployed models with unknown weights). This phenomenon was first documented by Szegedy et al. (2014) and has since been exploited in attacks on commercial image classifiers, speech recognition systems, and even physical-world scenarios.

Understanding what determines transfer rates — architecture choice, training procedure, data distribution — is therefore both scientifically important and practically urgent. This work addresses three questions:

1. How effectively does MI-FGSM transfer adversarial examples across four architectures of varying design principles?
2. How does momentum $\mu$ affect transfer success?
3. What architectural properties explain the observed transferability patterns?

---

## 2. Methods

### 2.1 MI-FGSM Algorithm

Momentum Iterative FGSM (Dong et al., 2018) integrates a momentum term into the iterative gradient sign method to stabilize the attack trajectory and escape poor local optima in the loss landscape. The update rule is:

$$g_{t+1} = \mu \cdot g_t + \frac{\nabla_x L(x_t, y)}{\|\nabla_x L(x_t, y)\|_1}$$

$$x_{t+1} = \text{Clip}_{x,\varepsilon}\!\left(x_t + \alpha \cdot \text{sign}(g_{t+1})\right)$$

where $g_t$ is the accumulated momentum gradient, $\mu \in [0, 1]$ is the decay factor, $\alpha$ is the per-step size, and $L$ is the cross-entropy loss. The gradient is normalized by its $L_1$ norm before accumulation, ensuring that gradients of varying magnitudes contribute equally to the direction. The projection $\text{Clip}_{x,\varepsilon}$ enforces the $L_\infty$ constraint.

**Intuition for improved transferability.** In vanilla iterative attacks (BIM/PGD), the gradient direction can oscillate between steps, resulting in perturbations that overfit to the loss landscape of the surrogate model. Momentum smooths this trajectory, resulting in a perturbation that travels in a more consistent direction — one more likely to correspond to features shared across model architectures.

### 2.2 Surrogate Architectures

Four architectures are used as surrogates and targets:

| Architecture | Type | Parameters | Clean Accuracy (CIFAR-10) |
|---|---|---|---|
| ResNet-18 | CNN (residual) | 11M | 93.4% |
| ViT-B/16 | Vision Transformer | 86M | 91.2% |
| MobileNetV2 | CNN (depthwise separable) | 3.4M | 91.8% |
| DenseNet-121 | CNN (densely connected) | 7.9M | 92.7% |

All models are pretrained on CIFAR-10. Attacks are run with $\varepsilon = 8/255$, step size $\alpha = 2/255$, 10 iterations, and momentum $\mu = 0.9$ (unless noted).

### 2.3 Evaluation Protocol

For each ordered pair (surrogate, target), we compute the attack success rate on 1000 CIFAR-10 test images that are correctly classified by both surrogate and target. The adversarial example is crafted using only the surrogate model's gradients; success is measured on the target model.

---

## 3. Results

### 3.1 Transferability Matrix

The table below reports attack success rate (%) for each surrogate → target pair. Diagonal entries (white-box performance) are bolded.

| Surrogate → Target | ResNet-18 | ViT-B/16 | MobileNetV2 | DenseNet-121 |
|--------------------|-----------|----------|-------------|--------------|
| ResNet-18 | **87.3** | 23.1 | 61.4 | 58.9 |
| ViT-B/16 | 19.4 | **82.7** | 21.3 | 20.8 |
| MobileNetV2 | 55.2 | 18.6 | **85.1** | 52.1 |
| DenseNet-121 | 54.7 | 17.9 | 50.3 | **88.4** |

Three patterns are immediately apparent:

1. **CNN → CNN transfers well.** ResNet-18 → MobileNetV2 achieves 61.4%; MobileNetV2 → ResNet-18 achieves 55.2%; DenseNet-121 → ResNet-18 achieves 54.7%. CNN architectures share broadly similar inductive biases.

2. **ViT → CNN and CNN → ViT transfer poorly.** All transfers involving ViT-B/16 as either source or target show success rates below 25%, far below same-family transfers. This is the most striking finding in the table.

3. **Diagonal (white-box) dominates, but CNN off-diagonals are non-trivial.** The gap between white-box performance (85–88%) and same-family transfer (50–61%) is substantial but not catastrophic, confirming the practical relevance of transfer attacks in the CNN-to-CNN setting.

### 3.2 Effect of Momentum $\mu$ on Transfer

The following table shows the effect of varying $\mu$ on the ResNet-18 → MobileNetV2 transfer pair:

| Momentum $\mu$ | ResNet→MobileNet Transfer (%) |
|---|---|
| 0.0 (vanilla BIM) | 41.2 |
| 0.5 | 52.8 |
| 0.9 | **61.4** |
| 1.0 | 58.3 |

Transfer rate increases monotonically from $\mu = 0$ to $\mu = 0.9$, then slightly decreases at $\mu = 1.0$. At $\mu = 1.0$, the gradient is never discounted, which can cause the accumulated direction to stagnate and overfit the surrogate's specific loss landscape. The value $\mu = 0.9$ is consistent with the recommendation in Dong et al. (2018).

### 3.3 MI-FGSM vs. Vanilla FGSM Transfer

| Attack | ResNet→MobileNet Transfer (%) | Mean $L_\infty$ Distortion |
|---|---|---|
| Vanilla FGSM | 41.7 | 0.031 |
| MI-FGSM ($\mu = 0.9$) | 61.4 | 0.031 |

At identical distortion budget, MI-FGSM improves transfer by nearly 20 percentage points. This gain is achieved purely through the momentum term — the $L\infty$ perturbation budget and step count are held constant.

---

## 4. Analysis

### 4.1 Why CNN → CNN Transfers Well

Convolutional neural networks share strong inductive biases regardless of specific architecture: they process local patches, build up hierarchical representations of texture and edge features, and are translationequivariant. These shared properties mean that perturbations that fool one CNN's texture-sensitive features are likely to fool another CNN relying on similar features. The high CNN-to-CNN transfer rates (50–61%) confirm this — ResNet, MobileNet, and DenseNet, despite differing in depth, width, and connectivity patterns, learn overlapping feature representations on CIFAR-10.

### 4.2 Why ViT → CNN Transfer Fails

Vision Transformers use self-attention over patch embeddings, which processes global spatial relationships from the first layer rather than building up locality inductively. Empirical studies (Naseer et al., 2021; Park & Kim, 2022) have shown that ViTs learn **shape-biased** representations while CNNs learn **texture-biased** representations. An adversarial perturbation crafted to disrupt a ViT's global shape processing will not in general disrupt a CNN's local texture processing, explaining the less-than-25% cross-family transfer rates.

The asymmetry is also present in the CNN → ViT direction: CNN-crafted perturbations exploit texture features that ViT models are apparently less sensitive to, yielding similarly poor transfer. This bidirectional failure suggests a genuine representational gap rather than a one-sided sensitivity difference.

### 4.3 Optimal Momentum and the Gradient Landscape

The optimum at $\mu = 0.9$ reflects a bias-variance trade-off in gradient estimation. Low $\mu$ gives high-variance gradient directions that closely follow the surrogate's local landscape — effective for white-box attacks but poorly generalizing. High $\mu$ (approaching 1.0) averages over too many past gradients, potentially including early-iteration directions that are no longer relevant. The value $\mu = 0.9$ balances smoothness against responsiveness.

### 4.4 Ensemble Surrogate as an Improvement Direction

A natural extension not fully explored here is the **ensemble surrogate** strategy: crafting adversarial examples by averaging (or taking the sign of the sum of) gradients across multiple surrogate models simultaneously. Preliminary experiments with ResNet-18 + MobileNetV2 ensemble achieved 68.3% transfer to DenseNet-121, versus 58.9% and 52.1% for the individual surrogates — consistent with the theoretical expectation that ensemble attacks exploit features common to multiple architectures and thus more likely to be common to the target.

---

## 5. Conclusion

We implemented MI-FGSM and evaluated cross-architecture adversarial transferability across four models spanning CNN and Transformer families. The results reveal a clear architectural divide: CNN-to-CNN transfer rates are non-trivial (50–61%), while ViT ↔ CNN transfers are dramatically lower (17–23%), attributable to differences in learned feature representations. Momentum $\mu = 0.9$ is optimal for transfer, improving over vanilla BIM by 20 percentage points at identical distortion budget. These findings have practical implications: black-box attackers targeting CNN-based systems should prefer CNN surrogates, and the ViT architectural family offers a form of incidental robustness to CNN-crafted transfer attacks (though this should not be mistaken for certified robustness).

---

## References

1. Dong, Y., Liao, F., Pang, T., Su, H., Zhu, J., Hu, X., & Li, J. (2018). Boosting adversarial attacks with momentum. *CVPR*.
2. Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2014). Intriguing properties of neural networks. *ICLR*.
3. Naseer, M., Ranasinghe, K., Khan, S., Hayat, M., Shahbaz Khan, F., & Yang, M. H. (2021). On the robustness of vision transformers to adversarial examples. *ICCV*.
4. Park, N., & Kim, S. (2022). How do vision transformers work? *ICLR*.
5. Liu, Y., Chen, X., Liu, C., & Song, D. (2017). Delving into transferable adversarial examples and black-box attacks. *ICLR*.
