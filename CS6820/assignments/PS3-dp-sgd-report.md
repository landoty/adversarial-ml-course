# PS3 Lab Report: DP-SGD Training with Opacus and Membership Inference on MNIST

**Course:** CS 6820 — Defenses and Robustness in ML
**Assignment:** Problem Set 3
**Dataset:** MNIST
**Framework:** PyTorch + Opacus
**Topics:** Differential Privacy, DP-SGD, Membership Inference Attacks

---

## Abstract

We implement Differentially Private Stochastic Gradient Descent (DP-SGD) using the Opacus library and evaluate its effectiveness as a defense against membership inference (MI) attacks on MNIST. A loss-threshold MI attack is used to measure privacy leakage across a range of privacy budgets ε ∈ {0.5, 1.0, 3.0, 10.0, ∞} at fixed δ = 10⁻⁵. We find that DP-SGD substantially reduces MI advantage (from 18.3% without DP to 4.2% at ε=3, and to 0.3% at ε=0.5) with modest accuracy costs (99.2% → 93.1% at ε=1, 84.2% at ε=0.5). A clipping norm sweep at fixed ε=3 reveals an optimal clipping norm of C=1.0, with both smaller and larger values reducing effective utility. We conclude that ε=3 provides the best practical privacy-utility tradeoff for MNIST-scale tasks, and discuss the semantic interpretation of the ε budget, privacy composition, and the distinction between DP-SGD's privacy guarantee and the MI attack as a measurement instrument.

---

## Introduction

### Differential Privacy: Definition and Motivation

A randomized mechanism M: D → R satisfies **(ε, δ)-differential privacy** if for all adjacent datasets S, S' (differing in one record) and all measurable output sets O ⊆ R:

$$\Pr[M(S) \in O] \leq e^\varepsilon \cdot \Pr[M(S') \in O] + \delta$$

Intuitively, this bounds how much the output distribution of M can change when any single individual's data is added or removed from the dataset. The parameter ε ("privacy budget") controls the tightness of this bound: smaller ε means the output is nearly indistinguishable whether or not any individual's data is included, providing stronger privacy. The parameter δ allows a small probability of the guarantee failing; typically δ = 10⁻⁵ ≪ 1/|dataset|.

### Why DP Matters for ML: Membership Inference

Machine learning models trained on sensitive data can inadvertently memorize and leak information about their training set. Shokri et al. (2017) introduced the **membership inference (MI) attack**, which demonstrates that an adversary with black-box query access to a trained model can often determine whether a specific record was in the training set. The attack exploits a well-known overfitting signal: models tend to assign lower loss (higher confidence) to training examples than to held-out test examples.

MI attacks are practically significant in settings where training set membership itself is sensitive: medical records (membership implies a patient has a disease), financial data (membership implies a financial transaction occurred), or any dataset with legal data minimization requirements (GDPR Article 5).

**Why does MI succeed on standard models?** The train-test loss gap is the primary signal. A classifier trained without privacy regularization will typically achieve near-zero training loss while having nonzero test loss; an adversary who can query the model's loss on a target record and observe low loss can infer membership with above-chance accuracy. More sophisticated attacks also use shadow models or confidence vectors, but the basic loss-threshold attack already exposes significant leakage on standard models.

### DP-SGD as a Defense

DP-SGD (Abadi et al., 2016) extends stochastic gradient descent to satisfy (ε, δ)-DP by applying two modifications at each training step:

1. **Per-sample gradient clipping:** The gradient of the loss for each individual example is clipped to have L2 norm at most C: `g_i ← g_i / max(1, ‖g_i‖₂ / C)`. This bounds the sensitivity of the gradient to any single training example.

2. **Gaussian noise addition:** Calibrated Gaussian noise is added to the (sum of clipped) per-sample gradients before the parameter update: the effective gradient for the batch is `(Σ_i g_i) + N(0, σ²C²I)`. The noise scale σ determines the privacy-utility tradeoff.

By bounding each individual's contribution to the gradient (clipping) and adding calibrated noise, DP-SGD ensures that the model's parameters — and thus its predictions — are approximately as likely to be produced from a training set that does or does not include any specific individual. This directly limits MI attack success because the model's per-example loss is no longer a reliable membership signal.

**Privacy accounting.** Each training step consumes a portion of the total privacy budget. The Rényi Differential Privacy (RDP) accountant (Mironov, 2017) and the moments accountant track the cumulative privacy expenditure across steps, and Opacus uses these to calibrate σ such that training for the specified number of epochs uses exactly the target (ε, δ) budget.

---

## Methods

### DP-SGD Implementation with Opacus

We use Opacus v1.x with a 4-layer convolutional network on MNIST (two conv layers followed by two FC layers). Key implementation details:

- `PrivacyEngine.make_private_with_epsilon()` calibrates the noise multiplier σ to achieve the target (ε, δ) over the full training run (50 epochs, batch size 256).
- Per-sample gradients are computed using Opacus's hooks on supported layer types (Conv2d, Linear, Embedding).
- Clipping norm C is set to 1.0 in the main experiment and swept in the ablation.
- Privacy budget δ = 10⁻⁵ throughout; ε ∈ {0.5, 1.0, 3.0, 10.0} for the main experiment.
- The baseline (ε = ∞, no DP) uses standard SGD with momentum.
- Training uses the Adam optimizer; Opacus requires replacing it with a DP-compatible optimizer via `make_private_with_epsilon`.

### Membership Inference Attack: Loss-Threshold Attack

The membership inference attack is a **loss-threshold attack** (Yeom et al., 2018):

1. Train the target model on a training set S of size 50,000.
2. Collect a challenge set of 10,000 members (drawn from S) and 10,000 non-members (from held-out test set, never seen during training).
3. Query the target model's cross-entropy loss on each challenge example.
4. For threshold τ: predict "member" if loss < τ, "non-member" otherwise.
5. Compute TPR(τ) = fraction of members correctly identified at threshold τ; FPR(τ) = fraction of non-members incorrectly labeled as members at threshold τ.
6. **MI Advantage** = max_τ |TPR(τ) − FPR(τ)|: the maximum gap between true positive rate and false positive rate over all thresholds. A perfectly private model would have MI Advantage = 0 (the attack cannot do better than random). A perfectly leaky model would have MI Advantage = 1.
7. **MI AUC** = area under the ROC curve for the binary membership prediction task. A random classifier achieves AUC = 0.5; a perfect attack achieves AUC = 1.0.

The loss-threshold attack is a simple baseline; more powerful attacks (shadow model attacks, adversarial MI attacks) would achieve higher MI advantage on non-DP models. However, for measuring the effect of DP-SGD, the loss-threshold attack is appropriate because DP-SGD directly controls the train-test loss gap that this attack exploits.

### Opacus Privacy Calibration

```python
from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    target_epsilon=epsilon,      # e.g., 3.0
    target_delta=1e-5,
    epochs=50,
    max_grad_norm=C,             # clipping norm
)
```

Opacus computes the required noise multiplier σ using the RDP accountant, which provides tighter composition than basic (ε, δ)-DP composition. The resulting σ is reported in Table 1.

---

## Results

### Table 1: Privacy-Utility-MI Tradeoff

| ε | δ | Test Accuracy (%) | MI Advantage (%) | MI AUC | Noise Multiplier σ |
|---|---|-------------------|-----------------|--------|--------------------|
| ∞ (no DP) | — | 99.2 | 18.3 | 0.623 | 0 |
| 10.0 | 1e-5 | 98.7 | 9.1 | 0.558 | 0.42 |
| 3.0 | 1e-5 | 97.4 | 4.2 | 0.524 | 0.89 |
| 1.0 | 1e-5 | 93.1 | 1.8 | 0.509 | 1.73 |
| 0.5 | 1e-5 | 84.2 | 0.3 | 0.501 | 3.21 |

### Table 2: Clipping Norm Sweep (ε=3, δ=1e-5)

| Clipping Norm C | Test Accuracy (%) | MI Advantage (%) |
|-----------------|-------------------|-----------------|
| 0.1 | 91.2 | 2.8 |
| 0.5 | 96.8 | 3.9 |
| 1.0 | 97.4 | 4.2 |
| 5.0 | 95.1 | 5.7 |

---

## Analysis

### Privacy-Utility Tradeoff: Choosing ε

The primary result in Table 1 is a clear privacy-utility tradeoff: as ε decreases (stronger privacy guarantee, more noise added), test accuracy decreases and MI advantage decreases toward zero.

**No DP (ε=∞):** The baseline model achieves 99.2% test accuracy but 18.3% MI advantage and MI AUC of 0.623. An MI advantage of 18.3% means that at the optimal threshold, the attacker correctly labels 18.3% more of the members-vs-nonmembers than chance would predict. An AUC of 0.623 (compared to the random baseline of 0.5) indicates substantial membership leakage even from a simple loss-threshold attack.

**ε=10:** Test accuracy drops only 0.5 points (98.7%) while MI advantage halves to 9.1% and AUC drops to 0.558. The low noise multiplier (σ=0.42) provides loose privacy protection. This setting is appropriate only when a very weak privacy guarantee is acceptable.

**ε=3:** The recommended setting for MNIST-scale tasks. Test accuracy drops 1.8 points (97.4%) relative to the no-DP baseline — a small and likely acceptable cost in most practical settings. MI advantage drops to 4.2% and MI AUC to 0.524 (near the random baseline of 0.5). The MI attack retains some signal but is substantially degraded compared to the no-DP baseline. Noise multiplier σ=0.89 adds substantial gradient noise without overwhelming the gradient signal entirely for this well-structured dataset.

**ε=1:** Test accuracy drops to 93.1% (6.1 points below baseline), MI advantage is 1.8%, and MI AUC is 0.509 — nearly indistinguishable from random. Privacy is strong, but the 6-point accuracy cost may be unacceptable for accuracy-critical deployments.

**ε=0.5:** MI advantage approaches zero (0.3%) and AUC is 0.501 (statistically indistinguishable from random), indicating the MI attack has no useful signal. However, test accuracy drops to 84.2% — a 15-point reduction relative to the baseline. For MNIST, this is a severe degradation; for harder tasks (CIFAR-10, ImageNet), DP at ε=0.5 is currently infeasible without substantial model scale.

**Practical interpretation of ε.** The ε parameter has a concrete worst-case interpretation: an adversary who observes the trained model gains at most an e^ε multiplicative advantage in distinguishing "trained with record x" from "trained without record x" for any individual x, with probability 1−δ over training randomness. At ε=3: e³ ≈ 20.1×. At ε=1: e¹ ≈ 2.7×. At ε=0.5: e^0.5 ≈ 1.6×. The gap between ε=3 and ε=1 in terms of MI advantage (4.2% vs. 1.8%) is meaningful in practice even if both seem small, because the DP guarantee applies to all possible attacks — not just the loss-threshold attack evaluated here.

### Optimal Clipping Norm: C=1.0

The clipping norm C controls the tradeoff between gradient bias and gradient noise:

**C too small (C=0.1):** Per-sample gradients are clipped to very small norms, meaning only their direction is preserved and magnitude information is lost. This introduces high bias in the gradient estimate: the effective learning signal is dominated by the direction of each gradient, not its magnitude, and gradients from examples with naturally small magnitude (well-classified examples) are scaled up to the clipping norm, while gradients from hard examples are scaled down. Result: test accuracy drops to 91.2%, worse than C=1.0 at the same ε. Interestingly, MI advantage is slightly lower (2.8%) because the high gradient bias means the model is learning less effectively, reducing overfitting further.

**C=0.5 and C=1.0:** Both provide good accuracy (96.8% and 97.4%) with MI advantage in the 3.9–4.2% range. C=1.0 is modestly better on both metrics, suggesting that the gradient clipping at this norm preserves sufficient useful gradient information while the added noise provides adequate privacy.

**C too large (C=5.0):** Clipping at a large norm means fewer gradients are clipped — for well-trained models where most gradient norms are < 5, clipping is nearly inactive. This reduces gradient bias but increases noise: the noise term N(0, σ²C²I) scales with C², so at C=5.0 the noise is 25× larger in variance than at C=1.0, given the same σ. For a fixed ε budget, Opacus calibrates σ to the same level regardless of C, so higher C directly increases the noise variance. Result: test accuracy drops to 95.1% (worse than C=1.0) and MI advantage increases to 5.7%, likely because the high-noise gradients make learning noisier without reducing the memorization as effectively.

The C=1.0 optimum is consistent with findings in the Opacus literature and likely reflects the typical distribution of gradient norms for this architecture: most per-sample gradient norms fall near 1.0, so clipping at C=1.0 clips the outlier gradients (which would dominate the update) while preserving the typical ones.

### MI Advantage Approaching Zero: ε → 0.5

As ε decreases toward 0.5, MI advantage approaches zero (0.3%), and MI AUC approaches 0.5 (random). This demonstrates that DP-SGD effectively closes the train-test loss gap that the MI attack exploits: with sufficient noise, the model's per-example loss is no longer a reliable membership signal. The loss distributions for members and non-members become statistically indistinguishable.

However, MI advantage = 0.3% does not imply perfect privacy — it means this specific attack cannot distinguish members from non-members, but more powerful attacks or attacks that exploit different model properties might. The (ε, δ)-DP guarantee is the formal statement of what is guaranteed; MI advantage is a measurement of how much this specific attack succeeds, and it will always be ≤ the DP bound.

### Privacy Budget Composition Across Training Runs

An important practical consideration not captured in Table 1 is **privacy composition**: if the same dataset is used to train multiple models (hyperparameter search, retraining with different seeds, model ensembles), the privacy budgets compose. Under basic composition, k training runs each using (ε, δ)-DP combine for (kε, kδ)-DP. Under advanced composition (RDP), the combination is tighter but still grows with k. Organizations deploying DP-ML systems must track cumulative ε across all training runs on the same dataset to maintain a meaningful privacy guarantee, particularly when iterating during development.

For the experiments in this report, each model is trained once. If hyperparameter tuning were included (e.g., the clipping norm sweep in Table 2, which involves 4 training runs), the total privacy expenditure would be higher than the reported ε for any single run.

---

## Conclusion

DP-SGD implemented via Opacus effectively mitigates loss-threshold membership inference attacks on MNIST. The privacy-utility analysis indicates that ε=3 provides the best practical tradeoff for this task: test accuracy degrades by only 1.8 points (97.4% vs. 99.2% baseline) while MI advantage drops from 18.3% to 4.2% and MI AUC approaches the random baseline (0.524 vs. 0.5). Stronger privacy (ε=0.5) nearly eliminates MI success (MI advantage 0.3%, AUC 0.501) but incurs a 15-point accuracy reduction that would be unacceptable in most practical settings.

The clipping norm sweep identifies C=1.0 as the optimal hyperparameter for this architecture, where the bias-variance tradeoff in the gradient estimate is best balanced. Both C too small (high bias) and C too large (high noise variance relative to fixed ε budget) degrade accuracy without commensurate privacy gains.

For deployment, we recommend: (1) use ε=3 for MNIST-scale tasks where a 2% accuracy budget is acceptable; (2) use the Rényi DP accountant for tight composition; (3) track cumulative ε across all training runs using the same dataset to maintain the stated guarantee; (4) use the MI attack as a measurement tool to empirically verify that the DP guarantee is effective against the specific threat model, while recognizing that the DP guarantee is the formal and complete statement of the protection. On harder tasks (CIFAR-10, language models), ε values in the range 3–10 are typically the practical frontier; very strong privacy (ε < 1) is currently achievable only at the cost of substantial accuracy degradation or very large model scale.

---

*References:*
- Abadi, M., Chu, A., Goodfellow, I., McMahan, H.B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep Learning with Differential Privacy. CCS 2016.
- Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating Noise to Sensitivity in Private Data Analysis. TCC 2006.
- Mironov, I. (2017). Rényi Differential Privacy of the Gaussian Mechanism. CSF 2017.
- Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership Inference Attacks Against Machine Learning Models. IEEE S&P 2017.
- Subramani, P., Vadivelu, N., & Kamath, G. (2021). Enabling Fast Differentially Private SGD via Just-in-Time Compilation and Vectorization. NeurIPS 2021.
- Yousefpour, A., Shilov, I., Sablayrolles, A., Testuggine, D., Prasad, K., Malek, M., Nguyen, J., Ghosh, S., Bharadwaj, A., Zhao, J., Bhatt, U., & Mironov, I. (2021). Opacus: User-Friendly Differential Privacy Library in PyTorch. arXiv:2109.12298.
- Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018). Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting. CSF 2018.
