# CS6820 — Week 01: Why Defenses Fail
## A History of Broken Defenses and the Adaptive Adversary Principle

**Prerequisites:** Familiarity with FGSM and PGD attacks, basic neural network training, L-infinity and L-2 norms as threat models.

**Learning Objectives:**
- Understand why the majority of proposed ML defenses have been broken by adaptive attacks
- Recognize the three types of gradient obfuscation and how each is circumvented
- Apply the Carlini 10 evaluation guidelines to critically assess a defense paper
- Understand BPDA and EOT as general-purpose techniques for attacking preprocessor-based defenses
- Distinguish empirical robustness claims from certified robustness claims

---

## 1. The Fundamental Challenge: The Adaptive Adversary

The central difficulty in adversarial ML defense is not finding a clever transformation that fools attackers using today's attacks — it is finding a defense that remains robust against an attacker who knows the defense and adapts their attack to it specifically.

This is the **adaptive adversary principle**: in evaluating any defense, we must assume the attacker has complete knowledge of the defense mechanism and crafts their attack specifically against it. This is sometimes called the Kerckhoffs's principle analog for ML security, borrowed from cryptography.

The history of adversarial ML defenses since 2014 has been largely a history of defenses proposed, a brief period of apparent success, and then a demonstrative adaptive attack that breaks the defense entirely. The pattern repeats so reliably that it has become one of the most important lessons in the field.

### 1.1 Why "Security Through Obscurity" Fails

In classical security, "security through obscurity" refers to the practice of hiding implementation details as a security measure. This approach is broadly rejected in cryptography: a cipher that is only secure as long as the adversary doesn't know how it works provides very weak guarantees, because:

1. **The algorithm will eventually become public.** Either through reverse engineering, insider threat, or eventual publication.
2. **Attackers adapt.** Once the algorithm is known, the security collapses completely.

The exact same failure mode applies to ML defenses. Consider a defense that applies a secret transformation $T$ to input $x$ before feeding it to the classifier $f$. If an attacker doesn't know $T$, they cannot adapt their attack to it. But:

- If $T$ is published (as in academic papers), the attacker immediately knows it.
- Even if $T$ is kept secret, a sufficiently motivated attacker can probe the system (query the defended model) to learn $T$'s behavior.
- The defense provides no mathematical guarantee of robustness — only a practical barrier that might be circumvented at any time.

This is fundamentally different from a **certified defense**, which provides a mathematical proof that no attack within the threat model can succeed, regardless of how adaptive or sophisticated it is.

### 1.2 The Evaluation Trap

When a defense is proposed, the authors typically evaluate it against existing attacks: FGSM, BIM, C&W, PGD. If the defense reduces the success rate of these attacks, the authors report high "robustness." But this robustness is illusory if:

1. The existing attacks are not the best possible attacks against this specific defense.
2. The defense introduces gradient obfuscation that causes gradient-based attacks to fail for reasons other than genuine robustness.
3. The evaluation does not account for the attacker's ability to design an adaptive attack.

The pattern has been so consistent that Athalye, Carlini, and Wagner (2018) were able to break 7 of the 9 defenses accepted at ICLR 2018 — the top machine learning venue — using a common set of techniques for handling gradient obfuscation.

---

## 2. A History of Broken Defenses

### 2.1 Defensive Distillation (Papernot et al., 2016)

**What was the defense?**

Defensive distillation applies knowledge distillation (Hinton et al.) to make a neural network more robust to adversarial examples. The procedure is:

1. Train a teacher network $f^T$ normally on labeled data, obtaining soft probability outputs.
2. Train a student network $f^S$ on the same data, but using the soft probability outputs of $f^T$ as training targets instead of hard one-hot labels.
3. At test time, apply the student network $f^S$ at a high temperature $T$ in the softmax: $\text{softmax}(z/T)$.

The intuition was that the soft targets smooth the output probability landscape, reducing the magnitude of gradients used to craft adversarial examples. Specifically, with high-temperature softmax, the output probabilities are very flat (close to uniform), and the gradient of the loss with respect to the input $\nabla_x L(f^S(x), y)$ becomes very small.

**What did it claim?**

Papernot et al. (2016) reported that defensive distillation reduced the success rate of FGSM and JSMA attacks from over 90% to under 0.5% on MNIST and CIFAR-10. This was presented as a near-complete defense against adversarial examples.

**How was it broken?**

Carlini and Wagner (2016) introduced the C&W attack and used it to break defensive distillation immediately. The key insight was:

The high-temperature softmax does not make the classifier more robust — it only makes the gradient of the softmax output very small. The logit values $z$ (pre-softmax activations) are unaffected by the temperature scaling. The C&W attack operates directly on the logit space, minimizing:

$$\min_\delta \|\delta\|_2 + c \cdot \max(z_{y_{true}}(x+\delta) - \max_{j \neq y_{true}} z_j(x+\delta), -\kappa)$$

where $z_j$ are the logit values (pre-softmax). By working in logit space, the attack completely bypasses the vanishing gradient problem caused by temperature scaling. Carlini and Wagner found 100% attack success rate against defensively distilled networks, with very small perturbations.

**The lesson:** If a defense works by making gradients small (or zero) without genuinely changing the decision boundary, an attacker can circumvent the gradient issue and find attacks using a different loss formulation or optimization approach.

---

### 2.2 MagNet (Meng and Chen, 2017)

**What was the defense?**

MagNet uses autoencoders (reconstruction networks) as a preprocessing defense. The idea is to train one or more autoencoders on clean training data. These autoencoders learn to reconstruct clean images well. At test time:

1. The input $x$ is passed through the autoencoder to produce a reconstruction $\hat{x} = \text{AE}(x)$.
2. If $\|x - \hat{x}\|$ (the reconstruction error) exceeds a threshold, the input is detected as adversarial and rejected.
3. Otherwise, $\hat{x}$ (the reconstruction) is passed to the classifier.

The intuition is that adversarial perturbations lie off the manifold of natural images, so the autoencoder will fail to reconstruct them faithfully and project them back toward the natural image manifold.

**What did it claim?**

MagNet reported high detection rates against FGSM, JSMA, and Deepfool attacks, with the system achieving over 95% detection on CIFAR-10 and MNIST.

**How was it broken?**

Carlini and Wagner (2017b) broke MagNet using adaptive attacks that account for the autoencoder preprocessing. The attack has two components:

*Attack on the detection component:* To avoid detection, the adversary crafts a perturbation $\delta$ such that $\|x + \delta - \text{AE}(x + \delta)\|$ is small. Since the autoencoder reconstructs clean images well, if $x + \delta$ is close to the natural image manifold, the reconstruction error will be small. The attack explicitly minimizes reconstruction error as an auxiliary objective.

*Attack on the reformer component:* Even if the input passes detection, the adversarial example must survive the autoencoder reconstruction. The attack must therefore be found in the "reformer space" — it must be adversarial even after passing through the autoencoder. This is achieved by including the autoencoder in the attack's computational graph:

$$\min_\delta \|\delta\|_2 + c \cdot L_{\text{adv}}(f(\text{AE}(x + \delta)), y)$$

This composite loss creates adversarial examples that fool the classifier even after reconstruction, and also avoid triggering the detector.

The result: Carlini and Wagner achieved near-100% attack success rates against MagNet on both MNIST and CIFAR-10, with perturbations indistinguishable from those that fool undefended networks.

**The lesson:** Preprocessing defenses (detection + reconstruction) must be evaluated with attacks that explicitly include the preprocessing step in the optimization. An attacker who knows the preprocessing exists can simply differentiate through it.

---

### 2.3 Thermometer Encoding (Buckman et al., 2018)

**What was the defense?**

Thermometer encoding discretizes pixel values into a one-hot representation. Each pixel value $p \in [0, 1]$ is encoded as a vector $t(p) \in \{0, 1\}^K$ where $t(p)_k = 1$ if $p \geq k/K$ and 0 otherwise. This creates a step-function input representation.

The intuition is that this non-differentiable discretization step prevents gradient-based attacks. The gradient of the loss with respect to the input is zero almost everywhere (where the step function is constant) and undefined at the step boundaries. An attacker who computes $\nabla_x L(f(x), y)$ through the thermometer encoding gets useless zero gradients.

**What did it claim?**

Buckman et al. (2018) reported 79.0% robust accuracy on CIFAR-10 at $\epsilon = 8/255$ under L-infinity PGD attacks, while the undefended network had near-zero robust accuracy. This was one of the highest reported robustness numbers at the time.

**How was it broken?**

Athalye et al. (2018) identified thermometer encoding as a "shattered gradients" defense — one where gradients are zero or incorrect due to non-differentiable operations. They applied **BPDA (Backward Pass Differentiable Approximation)**, which we discuss in detail in Section 4.

The idea: even though the forward pass through the thermometer encoding is non-differentiable, we can substitute a differentiable approximation (the identity function) in the backward pass. Concretely:

- **Forward pass:** $y = \text{ThermEncode}(x)$ (the actual thermometer encoding)
- **Backward pass:** approximate $\partial y / \partial x \approx I$ (identity gradient)

This allows gradient-based attacks to run effectively through the non-differentiable layer. Athalye et al. reduced the reported 79.0% robust accuracy to approximately 0% using BPDA-enabled PGD.

**The lesson:** Non-differentiability is not security. An attacker can always substitute a differentiable approximation in the backward pass.

---

### 2.4 Feature Squeezing (Xu et al., 2018)

**What was the defense?**

Feature squeezing reduces the precision of input features to detect adversarial examples. Two squeezers were proposed:

1. **Bit-depth reduction:** Reduce pixel precision from 8-bit to 1, 2, 4 bits (rounding to the nearest representable value).
2. **Spatial smoothing:** Apply a median filter to the image.

Detection works as follows: feed the original input $x$ and the squeezed input $q(x)$ to the classifier, and compare predictions. If $\|f(x) - f(q(x))\|_1 > \tau$, flag as adversarial.

The intuition: adversarial perturbations are carefully tuned to fool the classifier. When the input is smoothed or reduced in precision, the perturbation is destroyed, but the prediction on the clean image changes little. An adversarial example, however, will show a large change in prediction when squeezed.

**What did it claim?**

Xu et al. reported high detection rates across multiple attack types and datasets, presenting feature squeezing as a practical, lightweight defense requiring no retraining.

**How was it broken?**

Feature squeezing is broken by adaptive attacks that simultaneously minimize adversarial loss and maintain consistency across all squeezers. The adversary solves:

$$\min_\delta \|\delta\|_\infty \quad \text{subject to} \quad f(x+\delta) \neq y, \quad \forall q \in Q: \|f(x+\delta) - f(q(x+\delta))\|_1 \leq \tau$$

where $Q$ is the set of squeezers. An adversarial example that simultaneously fools the classifier on both the original and all squeezed inputs will evade detection.

The construction of such examples: the smoothing operations (median filter, bit reduction) are approximately differentiable (or can be treated as identity for BPDA purposes), so the constraint can be incorporated as a regularization term in the attack loss.

Additionally, Carlini and Wagner observed that the detection approach suffers from a fundamental statistical problem: with enough queries, an attacker can estimate the threshold $\tau$ and craft examples just below it.

**The lesson:** Detection-based defenses that compare predictions across multiple views of an input can be evaded by crafting perturbations that are consistent across all views.

---

### 2.5 Stochastic Activation Pruning (Dhillon et al., 2018)

**What was the defense?**

Stochastic Activation Pruning (SAP) randomly drops neurons during inference by sampling a binary mask $m \sim \text{Bernoulli}(p)$ for each activation and zeroing out the pruned activations. Each inference pass uses a different random mask. The claimed intuition is that adversarial perturbations are precisely tuned to fool a fixed network, so a randomly pruned network breaks the adversarial perturbation.

**What did it claim?**

Dhillon et al. reported 48.0% robust accuracy on CIFAR-10 at $\epsilon = 8/255$, suggesting significant robustness from the stochastic pruning.

**How was it broken?**

SAP is a stochastic defense. Athalye et al. applied **Expectation over Transformation (EOT)**:

The expected gradient of the loss over the randomness is:

$$\mathbb{E}_{m \sim \text{Bernoulli}(p)^n}\left[\nabla_x L(f_m(x), y)\right]$$

This expectation is well-defined even though each individual gradient $\nabla_x L(f_m(x), y)$ is random. By averaging gradients across many samples of the mask $m$, an attacker gets a reliable gradient estimate pointing toward adversarial examples. PGD with EOT-estimated gradients successfully fools the stochastically pruned network.

The key point: even though the gradient from any single forward-backward pass is noisy (due to the random mask), the average gradient is a good estimate of the true gradient of the expected loss. An attacker with enough compute can simply sample many masks per PGD step and average the gradients.

**The lesson:** Stochastic defenses do not prevent gradient-based attacks — they only add noise to the gradient. The expected gradient remains informative, and EOT recovers it.

---

### 2.6 Randomized Layer (Xie et al., 2018)

**What was the defense?**

Xie et al. proposed two randomization techniques added to the beginning of a ResNet:

1. **Random resizing:** The input image is randomly resized to a size in $[299, 331]$ before the network.
2. **Random padding:** The resized image is randomly padded to $331 \times 331$.

The idea is that adversarial perturbations are crafted for a fixed input size, and the random transformations break the perturbation's effectiveness.

**What did it claim?**

Xie et al. reported significantly reduced attack success rates against Iterative FGSM and other gradient-based attacks.

**How was it broken?**

Again, EOT is the key technique. The attacker computes:

$$\nabla_x \mathbb{E}_{t \sim T}\left[L(f(t(x)), y)\right] = \mathbb{E}_{t \sim T}\left[\nabla_x L(f(t(x)), y)\right]$$

where $T$ is the distribution over transformations (resize + pad). By sampling many transformations and averaging the gradients, the attacker obtains a reliable signal for crafting adversarial examples that are robust across transformations.

In practice, averaging over 30-50 random transformations per PGD step is sufficient to recover a good gradient estimate. Athalye et al. demonstrated near-100% attack success rates.

**The lesson:** Random input transformations are broken by EOT — simply average gradients over many samples of the transformation.

---

## 3. The Obfuscated Gradients Taxonomy

Athalye, Carlini, and Wagner (2018) introduced a systematic taxonomy of "obfuscated gradients" — defense mechanisms that work by corrupting the gradient signal available to attackers, without providing genuine robustness. They identified three types:

### 3.1 Shattered Gradients

**Definition:** The defense introduces a non-differentiable operation into the forward pass, causing gradients to be zero, NaN, or numerically incorrect.

**Examples:**
- Thermometer encoding (step function discretization)
- JPEG compression preprocessing (non-differentiable quantization step)
- Nearest-neighbor rounding
- Argmax operations (rounding to the nearest class)

**Why it fails:** Non-differentiability only prevents naive use of automatic differentiation. An attacker can apply **BPDA (Backward Pass Differentiable Approximation)**: during the forward pass, use the actual (non-differentiable) operation; during the backward pass, substitute a differentiable approximation. The simplest approximation is the identity function, which corresponds to pretending the non-differentiable layer is not there and backpropagating through the rest of the network. More sophisticated approximations can be constructed by fitting a smooth function to the non-differentiable operation.

**Formal description:** Let $g: \mathbb{R}^n \to \mathbb{R}^m$ be the non-differentiable preprocessing operation. In BPDA:
- Forward pass: $\hat{x} = g(x)$
- Backward pass: $\frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial \hat{x}} \cdot \hat{g}'(\hat{x})$

where $\hat{g}'$ is the derivative of a differentiable surrogate $\hat{g}$ that approximates $g$. For the identity surrogate: $\hat{g}'(\hat{x}) = I$.

**Detection signal:** A defense has shattered gradients if:
- Attack success improves significantly when using BPDA vs. direct gradient computation
- The gradient norm $\|\nabla_x L\|$ is suspiciously small near adversarial examples

---

### 3.2 Stochastic Gradients

**Definition:** The defense introduces randomness into the forward pass, causing gradient estimates to be high-variance and unreliable when computed naively (with a single forward-backward pass).

**Examples:**
- Stochastic activation pruning (random neuron dropout at inference)
- Random input transformations (random resize, crop, padding)
- Random noise injection (adding noise to activations)

**Why it fails:** Randomness does not eliminate gradient information — it only adds noise to gradient estimates. The **Expectation over Transformation (EOT)** technique computes the gradient of the expected loss:

$$\nabla_x \mathbb{E}_{r \sim \mathcal{R}}[L(f_r(x), y)]$$

where $r$ is the randomness and $f_r$ is the network under randomness $r$. By the linearity of expectation and the chain rule:

$$\mathbb{E}_{r \sim \mathcal{R}}[\nabla_x L(f_r(x), y)]$$

This can be estimated by Monte Carlo: sample $N$ random seeds $r_1, \ldots, r_N$ and average the gradients:

$$\hat{\nabla} = \frac{1}{N} \sum_{i=1}^N \nabla_x L(f_{r_i}(x), y)$$

With $N = 30$ to $N = 100$ samples, this provides a reliable gradient estimate for PGD attacks. The computational cost increases by a factor of $N$, but this is well within the capability of a motivated attacker.

**Statistical intuition:** Consider the stochastic gradient estimator. By the central limit theorem, the variance of the estimate scales as $\sigma^2/N$. With large enough $N$, the estimate converges to the true gradient of the expected loss. A defense that relies on the attacker not being able to estimate this expectation is assuming an attacker with limited compute — a weak assumption.

---

### 3.3 Exploding and Vanishing Gradients

**Definition:** The defense causes gradients to explode or vanish as they backpropagate through the defense mechanism, making optimization-based attacks numerically unstable.

**Examples:**
- Defensive distillation (high temperature softmax causes gradients to vanish)
- Deep preprocessing networks with poor conditioning
- Defense mechanisms that include division by very small numbers

**Why it fails:** Modern neural network optimization has well-developed techniques for handling gradient scaling issues (gradient clipping, careful initialization, alternative loss formulations). An attacker can:

1. **Use a logit-space attack formulation** (as in C&W) that avoids the vanishing gradient from softmax temperature.
2. **Use black-box attacks** that don't require gradients at all (NES, SPSA, boundary attack).
3. **Reparameterize the attack** in a space where gradients are well-conditioned.

---

### 3.4 Diagnosing Gradient Obfuscation in Practice

Athalye et al. provide a checklist for detecting gradient obfuscation in claimed defenses:

| Sign | Explanation |
|------|-------------|
| One-step attacks perform better than multi-step attacks | Gradients are pointing in the wrong direction; taking multiple steps makes things worse, not better |
| Black-box attacks outperform white-box attacks | The gradient information is misleading; random/black-box methods work better than gradient following |
| Unbounded attacks fail to reach 100% attack success | A defense that provides no bound on perturbation size should be 100% attackable; if not, the attack is broken |
| Random search outperforms gradient-based search | Gradients are uninformative |

If a defense exhibits any of these signs, it is almost certainly relying on gradient obfuscation and does not provide genuine robustness.

---

## 4. The BPDA and EOT Techniques in Detail

### 4.1 Backward Pass Differentiable Approximation (BPDA)

BPDA was introduced by Athalye et al. (2018) to attack defenses that use non-differentiable preprocessing.

**The formal setup:** Let $g: \mathbb{R}^n \to \mathbb{R}^n$ be a non-differentiable preprocessing function, and let $f: \mathbb{R}^n \to \Delta^K$ be a differentiable classifier. The defended classifier is $h(x) = f(g(x))$. The attacker wants to solve:

$$\max_{\|\delta\|_\infty \leq \epsilon} L(h(x + \delta), y) = \max_{\|\delta\|_\infty \leq \epsilon} L(f(g(x + \delta)), y)$$

The problem: $g$ is non-differentiable, so $\nabla_x L(f(g(x)), y)$ cannot be computed via automatic differentiation.

**BPDA solution:** Define a differentiable surrogate $\hat{g}$ that approximates $g$ (in the sense that $\hat{g}(x) \approx g(x)$ for all $x$ of interest). In the BPDA forward-backward computation:

$$\text{Forward: } \hat{x} \leftarrow g(x), \quad \text{then} \quad L \leftarrow L(f(\hat{x}), y)$$
$$\text{Backward: } \frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial \hat{x}} \cdot \hat{g}'(x)$$

The backward pass uses the gradient of $\hat{g}$ instead of the (undefined) gradient of $g$.

**Choice of surrogate:**

*Identity surrogate:* $\hat{g}(x) = x$. This is the simplest choice: pretend the preprocessing is not there. This is appropriate when $g$ is "close to" the identity (e.g., JPEG compression at high quality factors, or thermometer encoding with fine granularity). The backward pass becomes:

$$\frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial \hat{x}}$$

*Learned surrogate:* Train a smooth neural network $\hat{g}_\phi$ to approximate $g$ by minimizing $\sum_{x \sim D} \|g(x) - \hat{g}_\phi(x)\|_2^2$. This provides a better gradient approximation when $g$ is far from the identity.

**Why BPDA works:** The attack is solving an approximate optimization problem. Even if the gradient of $\hat{g}$ is not the exact gradient of $g$, it still points (approximately) in a direction that increases the loss. As long as the approximation is reasonable, PGD with BPDA-estimated gradients will make progress toward adversarial examples.

**Mathematical justification:** Consider the function value of the attack objective after one BPDA-PGD step. If the surrogate $\hat{g}$ is a good approximation ($g(x) \approx \hat{g}(x)$), then the angle between the true gradient and the BPDA-estimated gradient is small. If this angle is less than $90°$, the update step increases the true objective.

More formally, if $\cos(\theta) = \langle \nabla_x L_{\text{true}}, \nabla_x L_{\text{BPDA}} \rangle / (\|\nabla_x L_{\text{true}}\| \|\nabla_x L_{\text{BPDA}}\|) > 0$, then the BPDA gradient is a valid ascent direction for the true objective.

---

### 4.2 Expectation over Transformation (EOT)

EOT was introduced by Athalye et al. (2018) to attack defenses that use random transformations.

**The formal setup:** Let $t \sim T$ be a random transformation drawn from a distribution $T$. The defended classifier applies a random transformation before prediction: $h(x) = f(t(x))$. The attacker wants an adversarial example $x + \delta$ that fools the model in expectation over the randomness:

$$\max_{\|\delta\|_\infty \leq \epsilon} \mathbb{E}_{t \sim T}[L(f(t(x + \delta)), y)]$$

**EOT gradient estimation:** The gradient of the expected loss is:

$$\nabla_x \mathbb{E}_{t \sim T}[L(f(t(x)), y)] = \mathbb{E}_{t \sim T}[\nabla_x L(f(t(x)), y)]$$

(assuming sufficient regularity to exchange differentiation and expectation, which holds for smooth $f$ and $t$). This expectation is estimated by Monte Carlo with $N$ samples:

$$\hat{g} = \frac{1}{N} \sum_{i=1}^N \nabla_x L(f(t_i(x)), y), \quad t_1, \ldots, t_N \sim T$$

**EOT-PGD algorithm:**

```
Input: x, y, f, T, ε, η (step size), N (samples per step), T_max (steps)
Initialize: δ = 0
For t = 1, ..., T_max:
    g = 0
    For i = 1, ..., N:
        Sample t_i ~ T
        g += ∇_x L(f(t_i(x + δ)), y)
    g /= N
    δ = Π_{||δ||∞ ≤ ε}(δ + η · sign(g))   # L∞ PGD step
Return x + δ
```

**Computational cost:** Each PGD step requires $N$ forward-backward passes instead of 1. For $N = 30$ samples and 40 PGD steps, this is $30 \times 40 = 1200$ forward-backward passes per adversarial example. For a ResNet-18 on a modern GPU, this takes about 10-30 seconds per example — expensive but entirely feasible for evaluating robustness.

**Why EOT works despite the randomness:** A defense that relies on random transformations is essentially adding noise to the gradient: $\nabla_x L(f(t(x)), y) = \nabla_x L(f(x), y) + \text{noise}$ (approximately). The noise does not change the *expected* gradient — it only increases its variance. By averaging over enough samples, the variance of the Monte Carlo estimate $\hat{g}$ becomes small enough that the PGD steps reliably increase the adversarial loss.

The fundamental limitation: if the adversary is computationally unbounded, no defense based on randomness can be secure, because the expected gradient is always well-defined and estimable.

---

## 5. The Carlini 10 Guidelines for Evaluating Adversarial Robustness

Nicholas Carlini's "A Complete List of All (arXiv) Adversarial Example Papers" includes a widely-referenced list of 10 guidelines for evaluating adversarial robustness. We discuss each one in depth.

### Guideline 1: Evaluate with the strongest possible attack

**Motivation:** A defense that reduces attack success rates against *some* attacks is not necessarily robust — it may simply be robust against those specific attacks due to gradient obfuscation or other defense-specific factors. The evaluation must include the strongest known attacks, including:

- PGD with many steps (PGD-100, PGD-1000)
- The C&W attack with large optimization budget
- AutoAttack (which combines four complementary attacks)
- Adaptive attacks designed specifically against this defense

**Violation example:** "We evaluate our defense against FGSM and L-BFGS attacks and achieve 95% robustness." FGSM is a single-step attack — a multi-step PGD attack might break the defense easily. Reporting only single-step attack results is insufficient.

### Guideline 2: Evaluate against an unbounded attacker

**Motivation:** If a defense is claimed to be robust at $\epsilon = 8/255$, the evaluator should check what happens for $\epsilon \to \infty$ (or at very large $\epsilon$ values). An undefended network should approach 0% accuracy as $\epsilon$ grows — if the reported defense maintains high "robust accuracy" even at very large $\epsilon$, this is a strong sign of gradient obfuscation: the attack is failing to find adversarial examples, not the defense preventing their existence.

**Violation example:** "Our defense achieves 50% robust accuracy at $\epsilon = 0.3$ (L-infinity)." But at $\epsilon = 1.0$ (which allows arbitrary input modification), the defense should trivially be broken, because the attacker can move the input anywhere. If the attacker cannot break the defense at $\epsilon = 1.0$, something is wrong with the attack, not the defense.

### Guideline 3: Include white-box and black-box evaluations

**Motivation:** If a defense performs *worse* against black-box attacks (which don't use the model's gradients) than white-box attacks (which do), this is a strong signal of gradient obfuscation. A black-box attack relies on transferability or random search, which is generally weaker than white-box gradient-based attacks. A defense that appears to resist white-box attacks but fails against black-box attacks has manipulated gradient information in a way that protects against gradient-based attacks specifically — not against the adversarial threat in general.

**How to perform black-box evaluation:** Use a surrogate model to generate adversarial examples (transfer attack), or use score-based black-box attacks like NES (Natural Evolution Strategies) or SPSA (Simultaneous Perturbation Stochastic Approximation).

### Guideline 4: Use standard benchmarks

**Motivation:** Comparing against non-standard setups makes it hard to assess whether robustness improvements are genuine. Standard benchmarks include:

- CIFAR-10 with $\epsilon = 8/255$ (L-infinity), ResNet-18 or WideResNet-34-10
- MNIST with $\epsilon = 0.3$ (L-infinity), small CNN
- RobustBench as the standard leaderboard (Croce et al., 2021)

If a paper uses non-standard $\epsilon$ values, non-standard architectures, or non-standard datasets, the results are hard to compare and may be cherry-picked.

### Guideline 5: Ensure the defense doesn't break the attack's optimizer

**Motivation:** Many gradient-based attacks rely on the attack's optimizer (e.g., Adam, PGD with momentum) converging to a good solution. If the defense introduces non-differentiable operations that break the optimizer (cause NaN gradients, zero gradients, or oscillating gradients), the attack fails not because the defense is robust, but because the optimization is broken.

**How to check:** Monitor the attack loss during optimization. If the loss does not increase monotonically (or at least consistently) over PGD steps, the attack optimizer has likely been broken by the defense.

### Guideline 6: Report the *mean* attack loss across the test set, not just accuracy

**Motivation:** Reporting only accuracy (percentage of examples successfully defended) hides important information. A defense that correctly classifies 50% of adversarial examples but has very low confidence on the other 50% is different from a defense that has moderate confidence on all examples. Mean adversarial loss and mean adversarial confidence are more informative metrics.

### Guideline 7: Use confidence intervals, report variance

**Motivation:** Many robustness results have high variance across random seeds, model initializations, and evaluation batch selections. A paper that reports a single number without confidence intervals may be overfitting the evaluation to a specific setup. Best practice: report mean ± standard deviation across multiple random seeds.

### Guideline 8: Evaluate the defense on a held-out test set, not the training set

**Motivation:** Robust overfitting (Sect. 2 of this lecture series; Rice et al. 2020) is a real phenomenon: adversarially trained models can memorize the adversarial examples used during training and exhibit much lower robust accuracy on the held-out test set. The evaluation must use the test set.

### Guideline 9: Compare to the adversarial training baseline

**Motivation:** Adversarial training (PGD-AT or TRADES) is the gold-standard empirical defense. Any new defense must be compared against it. A defense that achieves 70% robust accuracy but is compared only against an undefended baseline is presenting a misleading picture — adversarial training already achieves ~56% robust accuracy (ResNet-18 on CIFAR-10), so the improvement is 14%, not 70%.

### Guideline 10: Design adaptive attacks first, then evaluate

**Motivation:** This is the most important guideline. Before reporting any robustness results, the evaluator must design and test attacks that are specifically tailored to the defense. If the evaluator cannot break the defense with adaptive attacks, they should describe the adaptive attacks they attempted in the paper. The burden of proof is on the defense to show it resists adaptive attacks, not on the attacker to find them post-publication.

**Tramèr et al. (2020) "On Adaptive Attacks to Adversarial Example Defenses"** provides a systematic methodology for designing adaptive attacks.

---

## 6. Tramèr's Adaptive Attack Methodology

Tramèr et al. (2020) systematically broke 13 defenses that had been published as "robust" by designing careful adaptive attacks. Their paper provides a blueprint for how to evaluate any defense adaptively.

### 6.1 The Core Methodology

For any defense $h(x) = f(g(x))$ (classifier $f$ with preprocessing $g$), the adaptive attack design proceeds as:

**Step 1: Identify the type of defense.**
- Is it a preprocessing defense (input transformation)?
- Is it a detection defense (separate detector network)?
- Is it an inference-time defense (stochastic components)?
- Does it involve gradient masking?

**Step 2: Identify the gradient obfuscation mechanism.**
- Is the preprocessing differentiable?
- Does the defense introduce randomness?
- Does the defense use non-differentiable operations?

**Step 3: Choose the appropriate adaptive technique.**
- Non-differentiable preprocessing → BPDA
- Stochastic defense → EOT
- Composite defense → Combine BPDA and EOT

**Step 4: Validate that the adaptive attack works.**
- Check that the attack loss increases monotonically over PGD steps.
- Verify that unbounded attacks achieve ~100% success.
- Confirm that the attack transfers across different random seeds.

**Step 5: Report results honestly.**
- Report both the non-adaptive and adaptive attack results.
- Report the attack loss trajectory, not just final accuracy.

### 6.2 Case Study: Breaking a Stochastic Detection Defense

Suppose a defense applies random noise to the input, then uses a separate detector network to decide whether to accept or reject the input:

$$h(x) = \begin{cases} f(x + \eta), \eta \sim \mathcal{N}(0, \sigma^2 I) & \text{if } D(x) = \text{accept} \\ \text{reject} & \text{otherwise} \end{cases}$$

**Naive attack (fails):** Apply PGD directly to $f(x + \eta)$ with a single noise sample. The gradient is noisy and the attacker cannot guarantee the detector will accept the input.

**Adaptive attack:** Simultaneously:
1. Use EOT to compute a reliable gradient estimate over $\eta \sim \mathcal{N}(0, \sigma^2 I)$.
2. Include the detector loss in the attack objective: minimize the probability of rejection.
3. Project onto the feasible set $\|\delta\|_\infty \leq \epsilon$ at each step.

The combined loss:

$$L_{\text{adaptive}}(x, \delta) = \mathbb{E}_\eta[L_{\text{adv}}(f(x + \delta + \eta), y)] - \lambda \cdot \mathbb{E}_\eta[D(x + \delta + \eta)]$$

where $D(x) \in [0, 1]$ is the detector's "accept" probability. The first term encourages adversarial misclassification; the second encourages acceptance.

---

## 7. What a Trustworthy Defense Looks Like

Given the history of broken defenses, what should practitioners trust?

### 7.1 Certified Defenses

A **certified defense** provides a mathematical guarantee that no adversary within the threat model (e.g., $\|\delta\|_2 \leq R$) can change the prediction. Certified defenses include:

- **Randomized smoothing** (Cohen et al. 2019): certifies L2 robustness via Gaussian smoothing and the Neyman-Pearson lemma. Week 06 covers this in full.
- **Interval Bound Propagation (IBP)** (Gowal et al. 2018): propagates intervals through the network to certify L-infinity robustness. Scalable to moderate-size networks.
- **Semidefinite relaxations** (Raghunathan et al. 2018): tighter bounds but computationally expensive.
- **Abstract interpretation** (Singh et al. 2019, AI2): general framework for certified robustness.

Certified defenses do not report "robust accuracy against PGD" — they report "certified accuracy at radius $\epsilon$," meaning the fraction of test examples for which the certification holds. Certified accuracy is always a *lower bound* on true robust accuracy (the certificate is sufficient but not necessary for robustness).

### 7.2 Empirical Defenses with Rigorous Evaluation

When certification is computationally infeasible (e.g., large neural networks), the next best option is empirical robustness evaluated according to the 10 guidelines above. Key markers of a rigorous empirical evaluation:

1. **AutoAttack evaluation.** AutoAttack (Croce and Hein 2020) is a parameter-free ensemble of four complementary attacks (APGD-CE, APGD-T, FAB, Square Attack). RobustBench uses AutoAttack as the standard evaluation. If a defense reports robustness under AutoAttack, it has passed a significantly higher bar than if it only reports PGD robustness.

2. **Comparison to PGD-AT baseline.** The defense should be benchmarked against PGD-AT (Madry et al. 2018) which is the well-understood, widely-reproduced baseline.

3. **Adaptive attack evaluation.** The paper must include an adaptive attack evaluation specifically designed for the defense.

### 7.3 The Trust Hierarchy

From most to least trustworthy:

| Trust Level | Defense Type | Example |
|-------------|-------------|---------|
| Highest | Certified defense with formal proof | Randomized smoothing (Cohen et al. 2019) |
| High | Adversarial training + AutoAttack evaluation | TRADES on RobustBench |
| Medium | Adversarial training + PGD evaluation only | Basic PGD-AT |
| Low | Heuristic defense + adaptive attack evaluation | Feature squeezing + BPDA eval |
| Very Low | Heuristic defense + non-adaptive evaluation | Early defenses (2017-2018) |
| None | Heuristic defense + no evaluation | Do not trust |

---

## 8. Worked Example: Identifying and Breaking Gradient Obfuscation

**Problem statement:** You are given a classifier that applies JPEG compression (quality factor 75) as a preprocessing step before a ResNet-18 classifier. The combined system is reported to achieve 65% robust accuracy at $\epsilon = 8/255$ L-infinity on CIFAR-10 under PGD-7 attacks. Is this defense trustworthy?

**Step 1: Classify the defense type.**

JPEG compression is a non-differentiable preprocessing step. The JPEG encoding procedure involves discrete cosine transform (DCT), quantization of DCT coefficients (the lossy step), and entropy coding. The quantization step is a rounding operation, which has zero gradient almost everywhere and undefined gradient at integer boundaries. This is a **shattered gradients** defense.

**Step 2: Diagnose the robustness claim.**

The defense claims 65% robust accuracy. Let's apply the diagnostic criteria from Athalye et al.:

- Does single-step FGSM perform better than multi-step PGD? Let's check: FGSM might achieve 30% attack success, while PGD-7 achieves only 35% attack success (65% robust accuracy). If adding more PGD steps doesn't significantly improve attack success, the gradient signal is corrupted.

- Does a black-box transfer attack perform comparably? Transfer attacks from an undefended ResNet-18 might achieve 40% attack success — similar to or better than PGD-7 against the defended model. This suggests the gradients computed through JPEG are unreliable.

**Step 3: Design the adaptive attack (BPDA).**

Apply BPDA with the identity surrogate for the JPEG preprocessing:

```python
class BPDAJPEGCompression(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Actual JPEG compression in forward pass
        x_np = x.detach().cpu().numpy()
        compressed = []
        for img in x_np:
            # Apply JPEG compression at quality 75
            buf = io.BytesIO()
            pil_img = Image.fromarray((img.transpose(1,2,0)*255).astype(np.uint8))
            pil_img.save(buf, format='JPEG', quality=75)
            buf.seek(0)
            decompressed = np.array(Image.open(buf)).astype(np.float32)/255.0
            compressed.append(decompressed.transpose(2,0,1))
        return torch.tensor(np.array(compressed), dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        # Identity surrogate gradient in backward pass
        return grad_output  # Pretend JPEG is identity
```

**Step 4: Apply BPDA-PGD and evaluate.**

Using BPDA-PGD with 50 steps and step size $\eta = 2/255$ on CIFAR-10:

| Attack | Attack Success Rate | Robust Accuracy |
|--------|--------------------|----|
| Undefended (no JPEG) | 95% | 5% |
| Naive PGD-7 with JPEG | 35% | 65% |
| BPDA-PGD-50 with JPEG | 92% | **8%** |
| Black-box transfer | 40% | 60% |

The BPDA attack reveals that the true robust accuracy is approximately 8%, not 65%. The defense was providing only the appearance of robustness through gradient obfuscation.

**Step 5: Interpret the results.**

The 65% reported "robust accuracy" was entirely due to gradient obfuscation from JPEG's non-differentiable quantization. The BPDA attack, which provides reliable gradient information by approximating the JPEG preprocessing as identity in the backward pass, achieves 92% attack success — only marginally lower than the undefended model's 95%.

**Recommendation:** JPEG compression preprocessing does not provide meaningful adversarial robustness. The defense should be replaced with adversarial training (PGD-AT or TRADES) or a certified defense.

---

## 9. Key Takeaways

1. **The adaptive adversary principle is fundamental.** Any defense that is public knowledge can be attacked adaptively. Security through obscurity fails, always.

2. **Gradient obfuscation is not security.** Defenses that work by corrupting gradient information (shattered gradients, stochastic gradients, vanishing gradients) can be circumvented by BPDA, EOT, and alternative attack formulations.

3. **BPDA breaks shattered gradient defenses.** By substituting a differentiable surrogate in the backward pass (typically the identity), an attacker recovers reliable gradient information through non-differentiable preprocessing.

4. **EOT breaks stochastic defenses.** By averaging gradients over many samples of the randomness, an attacker estimates the gradient of the expected loss, which is well-defined despite the defense's randomness.

5. **The Carlini 10 guidelines are the gold standard for evaluation.** Any paper claiming adversarial robustness must address all 10 guidelines, especially adaptive attack evaluation.

6. **Certified defenses are the only way to escape this cycle.** Randomized smoothing and IBP provide mathematical guarantees that no adaptive attacker can violate. However, they come at the cost of reduced clean accuracy and computational overhead.

7. **AutoAttack on RobustBench is the community standard.** If a defense is not on RobustBench and has not been evaluated with AutoAttack, treat the robustness claims with significant skepticism.

---

## Discussion Questions

1. Consider a defense that uses a randomly initialized, untrained neural network as a preprocessing step. An attacker knows this defense exists but does not know the random initialization. Is this defense secure? How would you attack it? (Hint: consider the relationship between black-box transfer attacks and the specific preprocessing network.)

2. The Carlini 10 guidelines require evaluating with an "unbounded attacker." But if the attacker can perturb $x$ arbitrarily (large $\epsilon$), can't they always trivially fool any classifier by moving to a region of the input space that is all zeros? Why is this a useful diagnostic?

3. Defensive distillation was designed to make the loss surface smoother (smaller gradients). Carlini and Wagner bypassed this by attacking in logit space. Can you think of another defense mechanism based on gradient size reduction, and an attack that would bypass it?

4. EOT requires averaging over $N = 30$–$100$ random transformations per PGD step. Consider a defense where the transformation is sampled from a very high-dimensional distribution. Would EOT remain practical? At what point would the variance of the gradient estimate be too high to be useful?

5. A defense claims certified L2 robustness of $R = 0.5$ on CIFAR-10 (using randomized smoothing with $\sigma = 0.25$). The certified accuracy at $R = 0.5$ is 40%. Is this a "trustworthy" defense? What additional information would you want to see?

---

## References

- Athalye, A., Carlini, N., & Wagner, D. (2018). Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples. *ICML 2018*.
- Buckman, J., Roy, A., Raffel, C., & Goodfellow, I. (2018). Thermometer Encoding: One Hot Way To Resist Adversarial Examples. *ICLR 2018*.
- Carlini, N., & Wagner, D. (2016). Evaluating Neural Network Robustness to Adversarial Examples. *arXiv:1608.04644*.
- Carlini, N., & Wagner, D. (2017). Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods. *AISec 2017*.
- Cohen, J., Rosenfeld, E., & Kolter, J.Z. (2019). Certified Adversarial Robustness via Randomized Smoothing. *ICML 2019*.
- Croce, F., & Hein, M. (2020). Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks. *ICML 2020*.
- Dhillon, G.S., et al. (2018). Stochastic Activation Pruning for Robust Adversarial Defense. *ICLR 2018*.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. *ICLR 2018*.
- Meng, D., & Chen, H. (2017). MagNet: A Two-Pronged Defense Against Adversarial Examples. *CCS 2017*.
- Papernot, N., et al. (2016). Distillation as a Defense to Adversarial Perturbations Against Deep Neural Networks. *IEEE S&P 2016*.
- Tramèr, F., et al. (2020). On Adaptive Attacks to Adversarial Example Defenses. *NeurIPS 2020*.
- Xie, C., et al. (2018). Mitigating Adversarial Effects Through Randomization. *ICLR 2018*.
- Xu, W., et al. (2018). Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks. *NDSS 2018*.
