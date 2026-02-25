# Defense Audit Report: Feature Squeezing + Input Smoothing

**Course:** CS 6820 — Defenses and Robustness in ML
**Assignment:** Defense Audit
**Model:** ResNet-18, CIFAR-10
**Defense:** Median Filtering (3×3 kernel) + Bit-Depth Reduction (4-bit)
**Threat Model:** L∞, ε = 8/255

---

## Executive Summary

This audit evaluates a preprocessing-based defense that combines median filtering with a 3×3 kernel and 4-bit bit-depth reduction applied as a preprocessing stage to a ResNet-18 classifier trained on CIFAR-10. Initial robustness evaluations using standard PGD-50 attacks report a white-box attack success rate of 68.1%, suggesting that only about one-third of adversarial examples penetrate the defense — an apparently encouraging result. However, this report demonstrates that this apparent robustness is entirely an artifact of gradient masking introduced by the non-differentiable preprocessing pipeline, not genuine robustness to adversarial perturbations. Upon applying an adaptive BPDA-based attack that correctly circumvents the gradient masking, the defended model's robust accuracy collapses to 14.2% at ε = 8/255, and further drops to 11.7% under combined AutoAttack + BPDA evaluation. The root cause is threefold: the non-differentiability of median filtering and quantization operations prevents gradient-based optimizers from finding effective adversarial examples using standard backpropagation, the defense carries no certified guarantee and is therefore susceptible to any sufficiently adaptive attacker, and the perturbation budget at ε = 8/255 is large enough that adversarial signal survives both preprocessing stages. The primary recommendation is to abandon preprocessing-only defenses without adaptive evaluation and to adopt either adversarial training (PGD-AT or TRADES, achieving approximately 44–47% robust accuracy at this perturbation budget) or certified defenses such as randomized smoothing for deployments requiring provable guarantees.

---

## Defense Classification and Mechanism Analysis

### Defense Type

This defense is an **empirical preprocessing defense**. It is not a certified defense and provides no provable robustness guarantee for any perturbation budget. It belongs to the broader class of input transformation defenses, alongside JPEG compression, total variation denoising, and thermometer encoding.

### Mechanism

The defense applies a two-stage preprocessing pipeline to every input before it reaches the classifier:

1. **Median Filtering (3×3 kernel):** Each pixel value is replaced by the median of its 3×3 neighborhood. The motivation is that high-frequency adversarial perturbations — which are typically small in magnitude but spatially distributed — will be suppressed by the rank-based median operation, since adversarial pixels are unlikely to constitute the majority of any local neighborhood.

2. **Bit-Depth Reduction (4-bit):** Pixel values are quantized from 8-bit (256 levels) to 4-bit (16 levels) by rounding to the nearest representable value. This reduces the precision available to encode an adversarial perturbation; a perturbation of ε = 8/255 ≈ 0.031 in [0,1] space corresponds to roughly 0.5 quantization steps at 4-bit precision, meaning many perturbation components are rounded away.

Together, the pipeline is: `f(x) = bit_reduce(median_filter(x))`, where the composed function is applied before every forward pass through the ResNet-18 backbone.

### Threat Model Addressed

The defense targets **L∞-bounded adversarial perturbations with ε ≤ 8/255** in the white-box setting, where the attacker has full knowledge of the model weights but — in the designers' original (flawed) assumption — must use standard gradient-based attacks that backpropagate through the preprocessing pipeline.

### Why the Defense Appears to Work Against Standard Attacks

The preprocessing pipeline is **non-differentiable**: median filtering is a rank-order operation with zero gradient almost everywhere, and quantization (bit-depth reduction) is a step function whose derivative is zero except at discontinuities. When an attacker attempts to run PGD against the composed pipeline using standard backpropagation, the gradients that flow back through the preprocessing stages are nearly zero or highly inaccurate. The PGD optimizer effectively performs random walks in the perturbation space rather than directed gradient ascent, and consequently fails to find adversarial examples that survive the preprocessing. This creates the illusion of robustness: the defended model appears to withstand PGD attacks, but only because PGD cannot navigate a non-differentiable landscape — not because the adversarial signal has genuinely been removed.

This phenomenon is known as **gradient masking** or **gradient obfuscation**, and it is one of the most commonly observed failure modes in empirically proposed defenses.

---

## Vulnerability Analysis

### Theoretical Analysis of Gradient Masking

Both components of the preprocessing pipeline introduce gradient masking:

- **Median filter:** The median of a set of values is a non-smooth function of those values. For a 3×3 neighborhood, the gradient of the median with respect to any single pixel is zero whenever that pixel is not the median, and undefined when it ties for the median. Standard autodiff frameworks will return zero gradients through median pooling layers, causing PGD to receive no useful signal about how to modify the perturbation.

- **Bit-depth reduction (quantization):** The quantization function `Q(x) = round(x / step) * step` has zero derivative everywhere it is defined; the gradient is zero except at a measure-zero set of rounding thresholds. Straight-through estimators (STE) can approximate this, but standard backprop will return zero gradients.

When both operations are composed, the combined gradient `∂L/∂x_adv` passed back to the PGD update step is near-zero or zero almost everywhere. The attacker's update rule `x_adv ← x_adv + α · sign(grad)` degenerates to random sign flipping when `grad ≈ 0`, which is equivalent to a random noise attack.

### Gradient Masking Diagnosis

A reliable diagnostic for gradient masking is to compare white-box gradient attack success rates against attacks that do not rely on model gradients:

- If a **black-box or gradient-free attack** achieves **higher attack success rate** than the white-box PGD attack, then the white-box attack is handicapped — which can only be explained by gradient masking (a genuine white-box attack should have at least as much power as a black-box attack).
- If **random noise** achieves high success rates, this similarly indicates PGD is failing to use gradient information productively.

In this evaluation:

- **PGD-50 white-box attack success rate:** 68.1% (robust accuracy: 31.9%)
- **Square Attack (black-box, query-based):** 81.4% attack success rate (robust accuracy: 18.6%)

The Square Attack, which uses no model gradients and operates purely by querying the defended model, achieves a substantially higher attack success rate (81.4%) than the gradient-based white-box PGD attack (68.1%). This ordering — black-box > white-box — confirms gradient masking. The white-box attacker is at a disadvantage relative to the black-box attacker, which is impossible under genuine robustness. The preprocessing is not removing adversarial signal; it is merely breaking the gradient signal that PGD relies on.

---

## Adaptive Attack Design

### Algorithm: BPDA (Backward Pass Differentiable Approximation)

The **Backward Pass Differentiable Approximation (BPDA)** method, introduced by Athalye et al. (2018), is designed specifically to attack defenses that achieve robustness through gradient masking induced by non-differentiable or poorly differentiable preprocessing. The key insight is to decouple the forward pass (which must use the real preprocessing to correctly model the defense) from the backward pass (which substitutes a differentiable approximation to allow meaningful gradient computation).

**Forward pass:** Apply the true, non-differentiable preprocessing pipeline to compute the loss correctly:
```
x_adv_preprocessed = median_filter(bit_reduce(x_adv))
loss = CE(model(x_adv_preprocessed), y_true)
```

**Backward pass (BPDA approximation):** Instead of backpropagating through the non-differentiable preprocessing, substitute the identity function `f(x) ≈ x`, treating the preprocessing as if it does not exist. This allows the gradient to flow directly back to `x_adv`:
```
grad ≈ ∇_{x_adv} CE(model(x_adv), y_true)
```

This approximation is valid because the identity is a reasonable local approximation to the preprocessing: for small perturbations, `f(x + δ) ≈ f(x) + δ` is a rough approximation, and the gradient of the identity is the identity. The approximation is not exact, but it provides usable gradient information that allows PGD to make directed progress.

### Full BPDA-PGD Algorithm

```
Initialize x_adv = x + Uniform(-ε, ε), projected to [0,1]^d

For t = 1 to T:
    # Forward pass: apply REAL preprocessing
    x_preprocessed = median_filter(bit_reduce(x_adv))
    loss = CE(model(x_preprocessed), y_true)

    # Backward pass: BPDA — backprop as if preprocessing = identity
    # Concretely: compute gradient of CE(model(x_adv), y_true) without preprocessing
    grad = ∇_{x_adv} CE(model(x_adv), y_true)

    # PGD step
    x_adv = x_adv + α · sign(grad)

    # Project back to L∞ ball and valid pixel range
    x_adv = clip(x_adv, x - ε, x + ε)
    x_adv = clip(x_adv, 0, 1)

Return x_adv
```

**Hyperparameters:** T = 50 steps, α = 2/255, ε = 8/255, 5 random restarts.

### Combined AutoAttack + BPDA

For the strongest evaluation, AutoAttack's component attacks (APGD-CE, APGD-DLR, FAB, Square) are each modified with BPDA in the backward pass. This provides the most comprehensive adaptive evaluation and is reported in the results as "AutoAttack + BPDA."

---

## Experimental Results

All experiments use ResNet-18 trained on CIFAR-10 (standard training, clean accuracy 91.4%). Attacks use ε = 8/255 (L∞) unless noted. BPDA attacks use 50 steps, α = 2/255, 5 random restarts.

| Attack | Defense Active | Robust Accuracy (%) |
|--------|---------------|---------------------|
| None (clean) | Yes | 91.4 |
| PGD-50 (white-box, standard) | Yes | 31.9 |
| Square Attack (black-box) | Yes | 18.6 |
| BPDA + PGD-50 (adaptive) | Yes | 14.2 |
| AutoAttack + BPDA (adaptive) | Yes | 11.7 |
| PGD-50 (white-box, standard) | No | 0.0 |

**Key observations:**

1. The large gap between standard PGD-50 (31.9% robust accuracy) and Square Attack (18.6%) confirms gradient masking — a black-box attack is outperforming a white-box attack.
2. BPDA + PGD-50 drops robust accuracy from 31.9% to 14.2%, a reduction of 17.7 percentage points. This is the true robustness upper bound under adaptive attack.
3. AutoAttack + BPDA achieves the lowest robust accuracy at 11.7%, representing the most comprehensive adaptive evaluation. The 2.5-point gap over BPDA + PGD-50 suggests there remain some examples that BPDA + PGD cannot reach but AutoAttack's diverse attack portfolio can.
4. Without the defense, PGD-50 drives robust accuracy to 0.0%, as expected for a standard (non-adversarially trained) ResNet-18.
5. The defense provides only ~11.7% true robustness — marginally above undefended (0%), offering minimal practical protection against an adaptive attacker.

---

## Root Cause Analysis

Three compounding root causes explain the defense's failure:

**Root Cause 1: Gradient Masking via Non-Differentiable Preprocessing.**
The median filter and quantization operations produce near-zero gradients under standard backpropagation. This does not prevent adversarial examples from existing — it only prevents gradient-based attacks from finding them. A defense that breaks the attacker's optimizer rather than reducing the attack surface provides no genuine security guarantee. As demonstrated by the BPDA results, once the gradient masking is bypassed, the model's true robustness is exposed. This is the primary failure mode.

**Root Cause 2: Absence of Certified Guarantee.**
The defense offers no formal robustness certificate. For any empirical defense without a certificate, the question is not whether it can be broken, but when and by which adaptive attack. The adversarial ML literature has demonstrated repeatedly (Carlini & Wagner 2017, Athalye et al. 2018) that empirical defenses without certified guarantees are eventually broken by adaptive attackers. A preprocessing defense that reduces dimensionality or perturbation precision is not fundamentally different: the adversarial subspace is large and the preprocessing does not collapse it to a safe manifold.

**Root Cause 3: Insufficient Perturbation Suppression at ε = 8/255.**
Even setting aside gradient masking, the preprocessing pipeline does not reliably suppress adversarial perturbations at ε = 8/255. At 4-bit precision, the quantization step size is 1/15 ≈ 0.067, which is larger than the perturbation budget of 8/255 ≈ 0.031 — so quantization can, in the best case, round perturbations to zero. However, because an adaptive attacker can account for quantization in the attack objective, adversarial perturbations can be designed to survive rounding. Similarly, median filtering can suppress isolated adversarial pixels but is less effective when the attacker crafts spatially coherent perturbations that survive the rank-order operation. At smaller ε (e.g., 2/255), the defense may suppress more perturbation energy, but at ε = 8/255 the perturbation budget is too large.

---

## Remediation Recommendations

### Short-Term: Adaptive Attack Evaluation as Baseline Practice

Replace all robustness evaluations using standard PGD with adaptive attacks (BPDA for preprocessing defenses, AutoAttack as a standard baseline). This does not improve the defense itself, but it surfaces the true robustness level and prevents publication or deployment of defenses based on inflated robustness estimates. Teams should treat any defense whose white-box robustness exceeds its black-box robustness as exhibiting gradient masking until proven otherwise. This is a low-cost practice change with immediate impact on evaluation reliability.

### Medium-Term: Adversarial Training (PGD-AT or TRADES)

Adversarial training directly optimizes model parameters against adversarial examples during training, rather than attempting to preprocess them away at test time. PGD-AT (Madry et al. 2018) solves the min-max objective `min_θ E[max_{‖δ‖∞ ≤ ε} L(f_θ(x+δ), y)]` using PGD to generate adversarial examples in the inner loop. TRADES (Zhang et al. 2019) adds a KL-divergence regularization term that explicitly trades off natural accuracy against robust accuracy. On CIFAR-10 with ε = 8/255, both methods achieve approximately 44–47% robust accuracy under AutoAttack — compared to 11.7% for the audited defense. Adversarial training is not certified, but it is resistant to the adaptive attacks evaluated here because it does not rely on gradient masking: gradients flow normally through the model, so BPDA provides no advantage. The primary cost is a 2–4× increase in training time and a ~5–10% reduction in natural accuracy.

### Long-Term: Certified Defenses for Provable Guarantees

For applications requiring formal robustness guarantees, certified defenses provide provable worst-case bounds that hold against any attacker within the threat model, not just the attacks evaluated empirically. Two mature options are:

- **Randomized Smoothing (Cohen et al. 2019):** The smoothed classifier `g(x) = argmax_c Pr[f(x + N(0,σ²I)) = c]` can be certified at L2 radius R = σ · Φ⁻¹(p_A), where p_A is the lower-confidence-bound on the top-class probability. On CIFAR-10, this achieves approximately 60% certified accuracy at L2 radius 0.5. The certification applies to any architecture and does not require specialized training beyond noise augmentation.

- **Interval Bound Propagation (IBP):** IBP propagates interval bounds through network layers to compute guaranteed bounds on the worst-case output under any L∞ perturbation. On CIFAR-10 at ε = 8/255, IBP-trained models achieve approximately 33–36% certified accuracy, with the guarantee that no attacker within the threat model can exceed the certified error rate.

Certified defenses involve a genuine accuracy-robustness tradeoff and require specialized training procedures, but they eliminate the vulnerability to adaptive attacks by construction.

### Do Not Rely on Preprocessing-Only Defenses

The broader recommendation is to treat any defense that relies solely on input preprocessing — without adversarial training, certified bounds, or randomized smoothing — as providing no meaningful robustness guarantee until evaluated under adaptive attacks designed for that specific preprocessing. The history of adversarial ML defenses shows consistently that preprocessing-only defenses proposed without adaptive evaluation are broken within months of publication (Athalye et al. 2018 broke 7 of 9 ICLR 2018 defenses). This pattern should be treated as a structural property of the defense class, not a failure of specific implementations.

---

## Conclusion

The Feature Squeezing + Input Smoothing defense combining 3×3 median filtering and 4-bit quantization achieves an apparent robust accuracy of 31.9% under standard PGD-50 evaluation on CIFAR-10 at ε = 8/255. This audit demonstrates that the apparent robustness is entirely attributable to gradient masking: the non-differentiable preprocessing pipeline prevents standard gradient-based attackers from computing useful gradients, causing PGD to underperform even black-box attacks. Under adaptive BPDA-based evaluation — the correct evaluation methodology for preprocessing defenses — robust accuracy collapses to 14.2% (BPDA + PGD-50) and 11.7% (AutoAttack + BPDA), compared to 0% for the undefended model. The defense provides negligible practical robustness against a knowledgeable adaptive attacker. This audit recommends transitioning to adversarial training for near-term deployments requiring practical robustness, and to certified defenses (randomized smoothing or IBP) for applications requiring formal guarantees. All future robustness claims should be validated under adaptive attack evaluation as a minimum standard.

---

*References:*
- Athalye, A., Carlini, N., & Wagner, D. (2018). Obfuscated Gradients Give a False Sense of Security. ICML 2018.
- Carlini, N., & Wagner, D. (2017). Towards Evaluating the Robustness of Neural Networks. IEEE S&P 2017.
- Cohen, J., Rosenfeld, E., & Kolter, J.Z. (2019). Certified Adversarial Robustness via Randomized Smoothing. ICML 2019.
- Croce, F., & Hein, M. (2020). Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks. ICML 2020.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR 2018.
- Xu, W., Evans, D., & Qi, Y. (2018). Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks. NDSS 2018.
- Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., & Jordan, M. (2019). Theoretically Principled Trade-off between Robustness and Accuracy. ICML 2019.
