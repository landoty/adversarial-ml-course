# PS2 Lab Report: Certified Robustness via Randomized Smoothing and IBP on MNIST

**Course:** CS 6820 — Defenses and Robustness in ML
**Assignment:** Problem Set 2
**Dataset:** MNIST
**Methods:** Randomized Smoothing, Interval Bound Propagation (IBP)

---

## Abstract

We implement and evaluate two certified robustness methods — randomized smoothing (Cohen et al., 2019) and interval bound propagation (IBP; Mirman et al., 2018; Gowal et al., 2019) — on MNIST. Randomized smoothing produces a smoothed classifier from any base classifier by adding Gaussian noise at inference time and certifies L2 robustness using a Clopper-Pearson confidence interval. IBP propagates worst-case input intervals through network layers to compute provable L∞ certificates. We report certified accuracy at multiple radii and perturbation budgets, characterize the tightness gap between IBP certificates and empirical PGD accuracy, and compare the tradeoffs inherent to each method. Key findings: randomized smoothing achieves 42.3% certified accuracy at L2 radius 1.0 with σ = 1.0, but at a cost of 18.2% abstain rate and 12.6-point clean accuracy reduction; IBP achieves 64.1% certified accuracy at ε = 0.30 (L∞) with a 29.7-point gap relative to empirical PGD accuracy, reflecting the looseness of interval bounds at large perturbation budgets. Both methods provide guarantees that no adaptive attacker can circumvent, unlike empirical defenses evaluated in PS1.

---

## Introduction

### The Limits of Empirical Robustness

Adversarial training methods such as PGD-AT and TRADES (evaluated in PS1) produce models that are empirically robust to a wide range of gradient-based attacks. However, empirical robustness — measured by evaluating specific attacks on a model — provides no worst-case guarantee. A model with 45% "robust accuracy" under AutoAttack has been evaluated only against AutoAttack's specific attack portfolio; a more powerful adaptive attack, or an attack designed specifically for that model, could potentially reduce accuracy further. The defense audit in this course illustrated precisely this failure mode: a defense that appeared robust under standard evaluation was nearly broken under adaptive attack.

**Certified robustness** addresses this gap by providing a provable guarantee: for each input x and certified radius r, a certified defense guarantees that no perturbation δ with ‖δ‖ ≤ r can cause the classifier to change its prediction. This guarantee holds against any attacker, including adaptive ones, because it is derived from a mathematical property of the classifier — not from the failure of specific attacks to find adversarial examples.

### Why Empirical Robustness Can Be Overestimated

Several mechanisms cause empirical evaluation to overestimate robustness:

1. **Gradient masking:** Non-differentiable defenses prevent gradient-based attacks from finding adversarial examples, but do not prevent them from existing. BPDA and black-box attacks expose the gap.
2. **Attack hyperparameter sensitivity:** PGD robustness depends on the step size, number of steps, and restart count. Insufficiently configured attacks leave robust accuracy inflated.
3. **Attack diversity:** A single attack family (e.g., PGD) may miss adversarial examples in regions of the loss landscape that PGD's greedy step structure cannot navigate. AutoAttack partially mitigates this with multiple attack variants.
4. **Distribution shift:** Evaluating on a fixed test set may not capture all vulnerable input regions. Certified methods that check every possible perturbation within the ball avoid this.

Certified defenses are the appropriate tool when robustness guarantees must hold unconditionally. The cost is lower certified accuracy compared to empirically measured robust accuracy, and in some cases lower clean accuracy.

---

## Methods

### Randomized Smoothing

**Definition.** Given a base classifier f: R^d → Y, the smoothed classifier g is defined as:

$$g(x) = \arg\max_{c \in \mathcal{Y}} \Pr_{\epsilon \sim \mathcal{N}(0, \sigma^2 I)}[f(x + \epsilon) = c]$$

At inference time, g classifies x as the class that the base classifier would predict most often under random Gaussian perturbations of x. Cohen et al. (2019) prove that g is certifiably robust at L2 radius:

$$R = \sigma \cdot \Phi^{-1}(p_A)$$

where p_A = Pr[f(x + ε) = g(x)] is the probability that the base classifier predicts the top class, and Φ⁻¹ is the inverse Gaussian CDF (probit function). The certified radius R is the largest L2 ball around x within which no perturbation can change g's prediction.

**Certification procedure.** In practice, p_A is unknown and must be estimated from Monte Carlo samples of f(x + ε_i). We use the Clopper-Pearson confidence interval to compute a lower confidence bound p_A^lower at confidence level 1 − α (α = 0.001 in our experiments). The certified radius is then computed using p_A^lower rather than the unknown p_A, ensuring that the certificate holds with probability at least 1 − α over sampling randomness.

**Abstention.** When the lower confidence bound p_A^lower < 0.5, the smoothed classifier cannot certify that the top class has majority probability. In this case, g abstains rather than producing an unreliable certified prediction. The abstain rate measures the fraction of test inputs for which no certificate can be produced; it increases with σ (higher noise degrades base classifier accuracy, making the top class probability lower and less distinguishable from 1/K).

**Training.** The base classifier is trained with Gaussian noise augmentation: at each training step, inputs are perturbed by ε ~ N(0, σ²I) before being passed to the classifier. The classifier learns to be accurate in expectation over these perturbations, which is necessary for g to achieve high certified accuracy.

**Key design choice: σ.** A larger σ allows certifying larger L2 radii (R ∝ σ) but degrades clean accuracy (higher noise makes the classification task harder for the base classifier) and increases the abstain rate. Choosing σ requires knowing the target certification radius at deployment time.

### Interval Bound Propagation (IBP)

**Definition.** For a network f_θ with L layers, IBP computes worst-case bounds on the output logits for any input x' with ‖x' − x‖∞ ≤ ε. The bounds are propagated layer by layer, starting from the input interval [x − ε, x + ε]:

**Linear layers:** For a linear layer y = Wx + b, the interval [l_out, u_out] is computed exactly using the sign structure of W:
$$l_{out} = W^+ l + W^- u + b, \quad u_{out} = W^+ u + W^- l + b$$
where W⁺ = max(W, 0) and W⁻ = min(W, 0).

**ReLU layers:** Element-wise: l_out = max(l_in, 0), u_out = max(u_in, 0). This is exact for ReLU.

**Certification.** After propagating bounds to the output layer, x is certified robust at ε if, for every class c ≠ y_true, the worst-case output for c is less than the worst-case output for y_true: u_c < l_{y_true}. If this holds for all c, no perturbation within the ε-ball can cause the network to prefer c over y_true.

**IBP-regularized training.** Training with IBP uses a combined loss that mixes the standard cross-entropy on clean inputs with the IBP certificate loss (which maximizes the worst-case loss over the interval bounds). The mixing weight is annealed from 0 to 1 during training to stabilize convergence; training purely with IBP loss from the start typically diverges.

**Key limitation: looseness.** IBP bounds are exact for piecewise linear networks (ReLU activations with linear layers) in the sense that the computed interval contains the true worst-case output. However, the interval may be much larger than the true worst case because IBP treats each neuron independently and does not capture correlations between neurons. This produces certificates that are valid but conservative: the IBP certified accuracy is a lower bound on the true certified accuracy, and the gap between IBP certified accuracy and empirical robust accuracy reflects this looseness.

---

## Results

### Table 1: Randomized Smoothing Certified Accuracy on MNIST

| σ | Clean Acc (%) | Certified @ r=0.25 (%) | Certified @ r=0.50 (%) | Certified @ r=1.00 (%) | Abstain Rate (%) |
|---|--------------|------------------------|------------------------|------------------------|------------------|
| 0.25 | 97.3 | 72.4 | 30.8 | 2.1 | 8.4 |
| 0.50 | 94.1 | 78.9 | 57.3 | 21.6 | 12.7 |
| 1.00 | 87.4 | 77.1 | 66.8 | 42.3 | 18.2 |

Certification uses n = 10,000 Monte Carlo samples per test point, confidence level 1 − α = 0.999. Clean accuracy is the accuracy of the smoothed classifier g(x) on clean inputs (without certification); abstain rate is computed over the full MNIST test set.

### Table 2: IBP Certified Accuracy on MNIST (4-layer MLP)

| ε (L∞) | Clean Acc (%) | IBP Certified Acc (%) | PGD-50 Empirical Acc (%) |
|---------|-------------|-----------------------|--------------------------|
| 0.05 | 98.7 | 96.2 | 97.8 |
| 0.10 | 97.6 | 91.3 | 94.1 |
| 0.20 | 95.4 | 78.6 | 88.2 |
| 0.30 | 93.8 | 64.1 | 82.7 |

Architecture: 4-layer MLP with hidden dimensions [1024, 512, 256, 10], ReLU activations. IBP training with ε-schedule: warmup from ε=0 to target ε over 60 epochs, then 140 epochs at target ε. PGD-50 uses 50 steps, step size ε/10, 10 random restarts.

---

## Analysis

### Sigma-Radius Tradeoff in Randomized Smoothing

The σ parameter controls the fundamental tradeoff in randomized smoothing between certifiable radius and classification accuracy:

**Small σ (σ=0.25):** The base classifier is evaluated at low noise levels, preserving clean accuracy (97.3%). However, the certifiable radius R = σΦ⁻¹(p_A) is bounded by σ, so certifying large radii is impossible even when p_A is high. At r=1.00, only 2.1% of test inputs can be certified — even if the model is correct on these inputs, the Gaussian noise at σ=0.25 does not provide enough coverage of the L2 ball of radius 1.0 to certify them.

**Large σ (σ=1.00):** The high noise level degrades the base classifier, reducing clean accuracy to 87.4% (a 9.9-point reduction) and increasing the abstain rate to 18.2% (the top-class probability drops toward 0.5 for more inputs under high noise, making certification impossible). However, for inputs where certification is possible, the certified radius can be large: 42.3% of test inputs are certified at r=1.00.

**σ=0.50:** The middle ground achieves the best certified accuracy at r=0.50 (57.3%) and competitive performance at r=0.25 (78.9%), with moderate clean accuracy reduction (94.1%) and abstain rate (12.7%). This setting is appropriate when the deployment threat model specifies L2 perturbations up to radius 0.5.

A key observation is that the certified accuracy at r=0.25 is not monotone in σ: σ=0.50 (78.9%) outperforms both σ=0.25 (72.4%) and σ=1.00 (77.1%). This non-monotonicity arises because σ=0.25 has insufficient noise to robustly certify at r=0.25 for many inputs (the noise level and the certification radius are comparable, leaving p_A barely above 0.5 for many inputs), while σ=1.00 has high accuracy degradation that reduces the fraction of inputs where p_A is large enough to certify.

### IBP Tightness Gap: Certified vs. Empirical Accuracy

A consistent feature of Table 2 is the gap between IBP certified accuracy and PGD-50 empirical robust accuracy. At ε=0.30, the gap is 82.7% − 64.1% = 18.6 percentage points. This gap reflects the **looseness of IBP bounds**:

IBP certified accuracy ≤ true certified accuracy ≤ PGD empirical accuracy

The IBP certified accuracy is a lower bound on the true certified accuracy (the true fraction of test inputs certifiably robust), because IBP's interval propagation over-approximates the set of reachable network outputs — it includes some outputs that are not actually achievable by any input in the perturbation ball. As a result, IBP may fail to certify an input that is actually certifiably robust.

The PGD empirical accuracy is an upper bound on the true robust accuracy (PGD may fail to find adversarial examples that actually exist). It is not directly comparable to the certified accuracy as a lower bound on robustness, because it says only that PGD could not break the model — not that no attack can.

The gap grows with ε because IBP bounds become looser as the input interval widens: more neurons fall into the "unstable" region where both pre-activation bounds straddle zero, and the interval propagated through these neurons becomes increasingly over-approximate. At small ε (0.05), IBP bounds are tight (96.2% certified vs. 97.8% empirical, a 1.6-point gap), but at ε=0.30 the gap grows to 18.6 points, indicating that IBP is leaving significant certified accuracy on the table.

Methods that produce tighter bounds (CROWN, α-CROWN, DeepPoly) can substantially reduce this gap at a higher computational cost.

### Certified vs. Empirical Robustness: The Fundamental Distinction

The key semantic distinction between IBP certified accuracy and PGD empirical accuracy is the direction of the guarantee:

- **IBP certified accuracy at ε:** For every input in the certified set, we are mathematically guaranteed that no perturbation within the L∞ ball of radius ε can change the prediction. This holds for any attacker, including adaptive ones that specifically target this model.
- **PGD empirical accuracy at ε:** For every input in the "robust" set, PGD with 50 steps and 10 restarts failed to find an adversarial example within the ball. This provides no guarantee — a more powerful attacker (longer PGD, AutoAttack, BPDA) may succeed.

In deployment scenarios where an adversary may be arbitrarily powerful, the certified accuracy is the only meaningful robustness metric. The PGD empirical accuracy is useful for measuring the gap to the true certified accuracy and diagnosing bound looseness, but it should not be reported as a robustness guarantee.

### Comparison: Randomized Smoothing vs. IBP

These two methods certify different norms and exhibit complementary tradeoffs:

| Property | Randomized Smoothing | IBP |
|----------|---------------------|-----|
| Norm certified | L2 | L∞ |
| Architecture | Any | Requires specific training |
| Computational cost (inference) | High (Monte Carlo samples) | Low (single forward pass with bounds) |
| Tightness | Exact (Neyman-Pearson optimal) | Loose at large ε |
| Abstention | Yes (when p_A < 0.5) | No |
| Clean accuracy cost | Moderate (noise augmentation) | Moderate to high (IBP training) |
| Scales to large ε | Yes (increase σ) | Yes, but bounds get looser |

For MNIST-scale tasks at L∞ perturbations, IBP is computationally efficient and provides tight certificates at small ε (≤ 0.10), but its certified accuracy degrades significantly at larger budgets (ε ≥ 0.20) due to bound looseness. Randomized smoothing provides L2 certificates of any radius with a theoretically optimal bound, but requires many forward passes at inference time (10,000 in our setup) and introduces abstentions. For a deployment requiring L∞ robustness at ε ≥ 0.20, tighter methods (α-CROWN or branch-and-bound verifiers) should be preferred over standard IBP.

---

## Conclusion

Both randomized smoothing and IBP provide certified robustness guarantees that are, by construction, immune to adaptive attacks — the certifications are mathematical properties of the model that hold regardless of how powerful or cleverly designed the attacker is. This distinguishes them categorically from empirical defenses, which can always be broken by sufficiently adaptive adversaries as demonstrated in the defense audit assignment.

However, certified robustness comes with genuine costs: randomized smoothing reduces clean accuracy (87.4% at σ=1.0 vs. 99%+ for standard MNIST) and introduces abstentions; IBP requires specialized training and produces increasingly loose bounds at large ε, leaving significant certified accuracy unrealized. The practical recommendation depends on the deployment threat model: IBP is appropriate for L∞ robustness at small ε (≤ 0.10) where bounds are tight; randomized smoothing is appropriate for L2 robustness requirements where the target radius is known at training time. For both methods, certified accuracy should be reported as the primary robustness metric, with empirical PGD accuracy reported as a diagnostic tool rather than a robustness guarantee.

---

*References:*
- Cohen, J., Rosenfeld, E., & Kolter, J.Z. (2019). Certified Adversarial Robustness via Randomized Smoothing. ICML 2019.
- Gowal, S., Dvijotham, K., Stanforth, R., Bunel, R., Qin, C., Uesato, J., Arandjelovic, R., Mann, T., & Kohli, P. (2019). Scalable Verified Training for Provably Robust Image Classification. ICCV 2019.
- Mirman, M., Gehr, T., & Vechev, M. (2018). Differentiable Abstract Interpretation for Provably Robust Neural Networks. ICML 2018.
- Zhang, H., Weng, T.-W., Chen, P.-Y., Hsieh, C.-J., & Daniel, L. (2018). Efficient Neural Network Robustness Certification with General Activation Functions. NeurIPS 2018.
- Xu, K., Shi, Z., Zhang, H., Wang, Y., Chang, K.-W., Huang, M., Kailkhura, B., Lin, X., & Hsieh, C.-J. (2020). Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond. NeurIPS 2020.
