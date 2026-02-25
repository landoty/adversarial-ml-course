# Week 5: Evasion Attacks II — PGD, Iterative Attacks, and Reliable Evaluation

**CS 6800: Security of Machine Learning Systems**
**Graduate Seminar | Spring 2026**

---

## Learning Objectives

By the end of this lecture, students will be able to:

1. Derive the Basic Iterative Method (BIM) as a natural extension of FGSM and explain the role of the projection (Clip) operation in each step.
2. Formulate Projected Gradient Descent (PGD) as a constrained optimization problem and state the Madry et al. theorem establishing PGD as the "strongest first-order attack."
3. Explain why random restarts are essential for PGD and reason about the number of restarts required for a given confidence level.
4. Derive the PGD update rule for both L∞ and L2 perturbation balls, and explain how the projection operation differs between the two.
5. Articulate the limitations of PGD as an evaluation tool and explain what the "gradient masking" problem is and how to detect it.
6. Explain the key design choices in AutoAttack (APGD-CE, APGD-DLR, Square Attack) and why the ensemble design addresses weaknesses of single-attack evaluation.
7. Trace through a 3-step PGD example on a toy 2D problem with concrete numerical values.
8. Select an appropriate attack configuration for evaluating a given defense and justify the choice.

---

## 1. From FGSM to Iterative Attacks: The Basic Iterative Method

In Week 4, we derived FGSM as the optimal single-step attack under an L∞ constraint, given the linearization approximation. The key limitation we identified was that a single gradient step is not optimal for nonlinear loss functions: the loss surface has curvature, and a single large step may miss the maximum of the loss within the $\epsilon$-ball.

The natural remedy is to take multiple smaller steps, recomputing the gradient at each new point. This is the Basic Iterative Method (BIM), introduced by Kurakin, Goodfellow, and Bengio in "Adversarial Examples in the Physical World" (2017).

### 1.1 BIM Update Rule

BIM applies FGSM iteratively with a smaller step size $\alpha < \epsilon$. Starting from the natural input $x_0 = x$, the BIM update at step $t$ is:

$$x_{t+1} = \text{Clip}_{x,\epsilon}\left(x_t + \alpha \cdot \text{sign}\left(\nabla_x L(f_\theta(x_t), y)\right)\right)$$

where $\text{Clip}_{x,\epsilon}$ denotes projection onto the $L_\infty$ ball of radius $\epsilon$ centered at the original input $x$:

$$\text{Clip}_{x,\epsilon}(v) = x + \text{clip}(v - x, -\epsilon, \epsilon)$$

This is equivalent to $x + \text{clip}(v - x, -\epsilon, \epsilon)$, which ensures that the accumulated perturbation $v - x$ never exceeds $\epsilon$ in any coordinate, regardless of how many steps are taken.

The step size $\alpha$ is typically chosen much smaller than $\epsilon$: common choices are $\alpha = \epsilon/T$ (where $T$ is the number of steps) or $\alpha = 2\epsilon/T$. The intuition is that each step should be small enough to follow the curvature of the loss surface rather than overshooting.

### 1.2 The Role of the Clip (Projection) Operation

The clipping operation in BIM serves two purposes:

**Valid image range:** It ensures $x_{t+1} \in [0, 1]^d$ at every step by clipping to the valid pixel range. This is the same clipping we applied in FGSM.

**Perturbation budget enforcement:** It ensures $\|x_{t+1} - x\|_\infty \leq \epsilon$ at every step, not just at the final step. Without this constraint, the accumulated perturbation from $T$ steps of size $\alpha$ could be as large as $T \cdot \alpha$ in any coordinate, which may far exceed $\epsilon$. The projection enforces the constraint at each step.

The combined clip operation is:

$$\text{Clip}_{x,\epsilon}(v)_i = \max\left(\max(0, x_i - \epsilon), \min(\min(1, x_i + \epsilon), v_i)\right)$$

That is: for each coordinate $i$, clamp $v_i$ to the range $[\max(0, x_i - \epsilon), \min(1, x_i + \epsilon)]$.

### 1.3 BIM vs. FGSM: When Does BIM Help?

BIM strictly dominates FGSM in attack success rate when the number of steps $T > 1$: more iterations always allow BIM to find equal or better adversarial examples. The gain is largest when:

1. The loss surface has significant curvature (highly nonlinear model).
2. The budget $\epsilon$ is large relative to the distance to the decision boundary from the natural input.
3. The gradient direction changes significantly along the path from $x$ to the adversarial example.

Empirically, on CIFAR-10 with $\epsilon = 8/255$ against a naturally trained ResNet:
- FGSM (1 step): ~55% attack success rate
- BIM-10 (10 steps, $\alpha = \epsilon/10$): ~95% attack success rate
- BIM-50 (50 steps, $\alpha = \epsilon/50$): ~98% attack success rate

The diminishing returns past $\sim 20$ steps reflect that the attack has largely found the loss maximum within the $\epsilon$-ball by that point.

---

## 2. Projected Gradient Descent: The Principled Formulation

Projected Gradient Descent (PGD), introduced by Madry, Makelov, Schmidt, Tsipras, and Vladu in "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018), reframes the adversarial example problem as a constrained optimization problem and applies standard projected gradient ascent to solve it.

### 2.1 The Optimization Problem

Recall the evasion attack objective from Week 4:

$$\max_{x' \in \mathcal{S}(x, \epsilon)} L(f_\theta(x'), y)$$

where $\mathcal{S}(x, \epsilon) = \{x' : \|x' - x\|_\infty \leq \epsilon\} \cap [0, 1]^d$ is the feasible set (the $\epsilon$-ball intersected with the valid image range).

Projected Gradient Descent (or more precisely, Projected Gradient Ascent, since we are maximizing) solves this problem iteratively. At each step, we take a gradient step in the direction of increasing loss, then project the result back onto the feasible set $\mathcal{S}(x, \epsilon)$:

$$x_{t+1} = \Pi_{\mathcal{S}(x,\epsilon)}\left(x_t + \alpha \cdot \nabla_x L(f_\theta(x_t), y)\right)$$

where $\Pi_{\mathcal{S}(x,\epsilon)}$ is the Euclidean projection operator onto $\mathcal{S}(x, \epsilon)$.

For the L∞ ball, the Euclidean projection is exactly the clipping operation:

$$\Pi_{\mathcal{S}(x,\epsilon)}(v)_i = \text{clip}(v_i, x_i - \epsilon, x_i + \epsilon)$$

clipped further to $[0, 1]$. This is why the L∞ PGD update looks identical to BIM with the gradient (not sign of gradient) step:

$$x_{t+1} = \text{Clip}_{x,\epsilon}\left(x_t + \alpha \cdot \nabla_x L(f_\theta(x_t), y)\right)$$

Wait — should it be the gradient or the sign of the gradient? In Madry et al.'s original formulation, they use $\text{sign}(\nabla_x L)$ for L∞ (i.e., PGD-L∞ is exactly BIM). For L2, they use the normalized gradient. The two variants differ only in the step direction within the feasible set.

**PGD-L∞ update:**
$$x_{t+1} = \text{Clip}_{x,\epsilon}\left(x_t + \alpha \cdot \text{sign}\left(\nabla_x L(f_\theta(x_t), y)\right)\right)$$

**PGD-L2 update:**
$$x_{t+1} = \Pi_{\mathcal{S}_2(x,\epsilon)}\left(x_t + \alpha \cdot \frac{\nabla_x L(f_\theta(x_t), y)}{\|\nabla_x L(f_\theta(x_t), y)\|_2}\right)$$

where $\Pi_{\mathcal{S}_2(x,\epsilon)}$ is the projection onto the L2 ball of radius $\epsilon$ around $x$ (intersected with $[0,1]^d$).

For the L2 projection: if $v$ is the intermediate point before projection, the projected point is:

$$\Pi_{\mathcal{S}_2(x,\epsilon)}(v) = x + \epsilon \cdot \frac{v - x}{\max(\|v - x\|_2, \epsilon)}$$

(followed by clipping to $[0,1]^d$). This moves $v$ toward $x$ if it lies outside the L2 ball, or leaves it unchanged if it is already inside.

### 2.2 PGD as the "Strongest First-Order Attack"

The Madry et al. paper contains a theorem that establishes the theoretical status of PGD among first-order attacks:

**Theorem (informal):** For a loss function that is locally concave within the perturbation set $\mathcal{S}(x, \epsilon)$, PGD with sufficient iterations converges to the global maximum of the loss within $\mathcal{S}(x, \epsilon)$.

The key phrase is "locally concave within the perturbation set." For general nonlinear neural networks, the loss surface is not globally concave and PGD may converge to local maxima. However, in practice, PGD with random restarts (see Section 2.4) reliably finds high-loss adversarial examples, and the remaining gap from the true maximum is small relative to the noise in the evaluation.

The informal version of the theorem is often stated as: "PGD is the strongest attack that uses only gradient information." This distinguishes first-order attacks (which use gradients) from zero-order or decision-based attacks (which use only predictions). The theoretical significance is that a model trained to be robust against PGD attacks provides a strong robustness guarantee against all gradient-based attacks.

This is the theoretical motivation for PGD adversarial training: by training the model to minimize the loss on PGD adversarial examples (rather than natural examples), we produce a model that, in principle, is robust against all gradient-based attacks.

### 2.3 Step Size Selection

The step size $\alpha$ in PGD is a critical hyperparameter. Too large, and each step overshoots the local optimum and the iterates oscillate. Too small, and the attack may converge slowly and not find the loss maximum within the budget of $T$ steps.

A common heuristic is $\alpha = \epsilon / T \cdot C$ where $C$ is a constant factor (often 1–2). For $\epsilon = 8/255$ and $T = 40$ steps:

$$\alpha = \frac{8/255}{40} \approx 0.00078 \approx \frac{2}{255} \cdot \frac{1}{2}$$

A common alternative is $\alpha = 2.5 \cdot \epsilon / T$, which is used in several prominent papers. For $\epsilon = 8/255$, $T = 40$:

$$\alpha = \frac{2.5 \times 8/255}{40} = \frac{20}{255 \times 40} = \frac{1}{510} \approx 0.00196$$

These are rule-of-thumb values; in practice, the step size should be tuned for the specific model and perturbation budget. The key insight is that using too few steps (e.g., $T = 1$ with $\alpha = \epsilon$) degenerates to FGSM, while using too many steps (e.g., $T = 1000$ with tiny $\alpha$) is computationally wasteful since the attack converges well before $T$ steps.

### 2.4 Random Restarts

A single run of PGD, starting from a specific initialization, may converge to a local maximum of the loss rather than the global maximum. To improve coverage, PGD is typically run multiple times from different random starting points, and the best adversarial example found across all runs is retained.

The standard PGD initialization is uniform random: sample $\delta_0 \sim \text{Uniform}(-\epsilon, \epsilon)^d$ (element-wise uniform) and start from $x_0 = \text{Clip}_{[0,1]}(x + \delta_0)$.

The number of restarts required depends on the loss landscape. For naturally trained models, a single restart usually suffices to find a strong adversarial example. For robustly trained models or models with complex loss surfaces, more restarts are needed.

A useful criterion: if the attack success rate increases noticeably with additional restarts, you need more restarts. If it plateaus, you have enough. For PGD-40 on CIFAR-10 against PGD adversarially trained models, 10 random restarts is standard practice; for more careful evaluation, 50 restarts may be used.

The reason random restarts help is that the PGD objective (maximizing loss in the $\epsilon$-ball) is non-convex for nonlinear networks, and has multiple local maxima. Starting from the origin of the $\epsilon$-ball (i.e., starting from $x$ itself) biases the optimization toward local maxima that are reachable from $x$ by a monotone path, which may not include the global maximum.

---

## 3. Numerical Example: 3 Steps of PGD-L∞ on a Toy 2D Example

To build intuition, we trace through three steps of PGD-L∞ on a simple two-class classification problem in $\mathbb{R}^2$. This example uses simplified numbers but correctly illustrates the structure of the algorithm.

### 3.1 Setup

Consider a binary classifier $f: \mathbb{R}^2 \rightarrow \{0, 1\}$ with a sigmoidal output. The logit function is $z(x) = w^T x + b$ where:

$$w = \begin{bmatrix} 1.5 \\ -1.0 \end{bmatrix}, \quad b = 0.2$$

The predicted probability of class 1 is $\sigma(z(x)) = 1/(1 + e^{-z(x)})$. The cross-entropy loss for the true label $y = 0$ (class 0) is:

$$L(x, y=0) = -\log(1 - \sigma(z(x))) = \log(1 + e^{z(x)})$$

The gradient of this loss with respect to $x$ is:

$$\nabla_x L(x, y=0) = \sigma(z(x)) \cdot w$$

Let the natural input be $x_0 = [0.3, 0.5]^T$. The perturbation budget is $\epsilon = 0.4$ (L∞). The step size is $\alpha = 0.2$. We initialize with a random perturbation: $\delta_0 = [-0.2, 0.1]^T$, so $x_0^{\text{adv}} = [0.1, 0.6]^T$.

### 3.2 Step 1

Compute $z(x_0^{\text{adv}})$:

$$z([0.1, 0.6]) = 1.5 \times 0.1 + (-1.0) \times 0.6 + 0.2 = 0.15 - 0.6 + 0.2 = -0.25$$

Compute $\sigma(z) = \sigma(-0.25) = 1/(1 + e^{0.25}) \approx 1/(1 + 1.284) \approx 0.438$.

Compute gradient:

$$\nabla_x L = \sigma(z) \cdot w = 0.438 \times \begin{bmatrix} 1.5 \\ -1.0 \end{bmatrix} = \begin{bmatrix} 0.657 \\ -0.438 \end{bmatrix}$$

Compute sign of gradient: $\text{sign}(\nabla_x L) = [+1, -1]^T$.

Take step:

$$x_1^{\text{adv}} = x_0^{\text{adv}} + \alpha \cdot \text{sign}(\nabla_x L) = [0.1, 0.6] + 0.2 \times [1, -1] = [0.3, 0.4]$$

Project onto L∞ ball of radius $\epsilon = 0.4$ around original $x_0 = [0.3, 0.5]$:
- Feasible range for coordinate 1: $[0.3 - 0.4, 0.3 + 0.4] = [-0.1, 0.7]$. Value 0.3 is in range.
- Feasible range for coordinate 2: $[0.5 - 0.4, 0.5 + 0.4] = [0.1, 0.9]$. Value 0.4 is in range.

$x_1^{\text{adv}} = [0.3, 0.4]^T$. Perturbation so far: $x_1^{\text{adv}} - x_0 = [0.0, -0.1]^T$, $\|[0.0, -0.1]\|_\infty = 0.1 \leq 0.4$. Constraint satisfied.

Current loss: $z([0.3, 0.4]) = 1.5 \times 0.3 - 1.0 \times 0.4 + 0.2 = 0.45 - 0.4 + 0.2 = 0.25$. Loss = $\log(1 + e^{0.25}) \approx \log(1.284) \approx 0.250$.

### 3.3 Step 2

Compute $z([0.3, 0.4]) = 0.25$ (computed above).

Compute $\sigma(0.25) \approx 0.562$.

Compute gradient:

$$\nabla_x L = 0.562 \times \begin{bmatrix} 1.5 \\ -1.0 \end{bmatrix} = \begin{bmatrix} 0.843 \\ -0.562 \end{bmatrix}$$

Sign: $[+1, -1]^T$.

Take step:

$$x_2^{\text{adv}} = [0.3, 0.4] + 0.2 \times [1, -1] = [0.5, 0.2]$$

Project onto L∞ ball of radius 0.4 around $x_0 = [0.3, 0.5]$:
- Coordinate 1: range $[-0.1, 0.7]$. Value 0.5 is in range.
- Coordinate 2: range $[0.1, 0.9]$. Value 0.2 is in range.

$x_2^{\text{adv}} = [0.5, 0.2]^T$. Perturbation: $[0.2, -0.3]^T$, $\|[0.2, -0.3]\|_\infty = 0.3 \leq 0.4$. Satisfied.

Current loss: $z([0.5, 0.2]) = 1.5 \times 0.5 - 1.0 \times 0.2 + 0.2 = 0.75 - 0.2 + 0.2 = 0.75$. Loss = $\log(1 + e^{0.75}) \approx \log(3.117) \approx 1.138$. Loss has increased significantly.

### 3.4 Step 3

Compute $\sigma(0.75) \approx 0.679$.

Compute gradient:

$$\nabla_x L = 0.679 \times \begin{bmatrix} 1.5 \\ -1.0 \end{bmatrix} = \begin{bmatrix} 1.019 \\ -0.679 \end{bmatrix}$$

Sign: $[+1, -1]^T$.

Take step:

$$x_3^{\text{adv}} = [0.5, 0.2] + 0.2 \times [1, -1] = [0.7, 0.0]$$

Project onto L∞ ball of radius 0.4 around $x_0 = [0.3, 0.5]$:
- Coordinate 1: range $[-0.1, 0.7]$. Value 0.7 is exactly at boundary. Clip to 0.7.
- Coordinate 2: range $[0.1, 0.9]$. Value 0.0 is below the lower bound. Clip to 0.1.

$x_3^{\text{adv}} = [0.7, 0.1]^T$. Perturbation: $[0.4, -0.4]^T$, $\|[0.4, -0.4]\|_\infty = 0.4 = \epsilon$. Constraint tight in both dimensions.

Final loss: $z([0.7, 0.1]) = 1.5 \times 0.7 - 1.0 \times 0.1 + 0.2 = 1.05 - 0.1 + 0.2 = 1.15$. Loss = $\log(1 + e^{1.15}) \approx \log(4.158) \approx 1.424$.

### 3.5 Summary of the 3-Step Example

| Step | $x^{\text{adv}}$ | $z(x^{\text{adv}})$ | $\sigma(z)$ | Loss |
|------|------------------|----------------------|-------------|------|
| Init | [0.1, 0.6] | -0.25 | 0.438 | 0.568 |
| 1    | [0.3, 0.4] | 0.25  | 0.562 | 0.250 |
| 2    | [0.5, 0.2] | 0.75  | 0.679 | 1.138 |
| 3    | [0.7, 0.1] | 1.15  | 0.760 | 1.424 |

The loss increases monotonically across steps. The adversarial example at step 3 has $\sigma(z) = 0.760$, meaning the classifier assigns 76% probability to class 1 (wrong class) for an input that should be class 0. The decision boundary ($\sigma = 0.5$, $z = 0$) was crossed between step 1 and step 2.

The key observation: the projection at step 3 clips both coordinates to the boundary of the $\epsilon$-ball. This is characteristic of PGD convergence — the optimal perturbation (for L∞ constrained linear classifiers) is always at the corner of the $\epsilon$-ball, and iterative attacks tend to push toward the boundary of the feasible set.

---

## 4. The Gradient Masking Problem

Perhaps the most consequential practical issue in adversarial ML evaluation is the "gradient masking" or "obfuscated gradients" problem, documented comprehensively by Athalye, Carlini, and Wagner in "Obfuscated Gradients Give a False Sense of Security" (2018). The problem is this: many published defenses appear to dramatically reduce the success rate of PGD attacks, but this apparent robustness is an artifact of the defense's effect on the gradient, not genuine robustness.

### 4.1 What Is Gradient Masking?

A defense achieves gradient masking when it modifies the model's computation in a way that makes the gradient of the loss with respect to the input uninformative — very small, numerically zero, or pointing in a direction that does not increase the loss — even though adversarial examples with significant loss increase exist and can be found by other means.

The gradient is the primary signal used by PGD (and FGSM) to find adversarial examples. If the gradient is masked, PGD produces small perturbations that don't increase the loss much, and the defense appears robust. But the adversarial examples are still there — they just can't be found by gradient-based attacks.

### 4.2 Common Gradient Masking Mechanisms

Athalye et al. identified three primary mechanisms through which defenses inadvertently (or deliberately) mask gradients:

**Shattered gradients:** The defense introduces non-differentiable operations (e.g., thresholding, quantization, or non-differentiable preprocessing) that cause the gradient to be zero or undefined almost everywhere. Gradient-based attacks cannot navigate a region where the gradient is zero, so they make no progress.

**Stochastic gradients:** The defense introduces randomness into the forward pass (e.g., adding random noise to the input or to intermediate activations) such that the gradient at any single forward pass is a noisy estimate of the "expected gradient." PGD, which recomputes the gradient at each step, follows this noisy estimate and makes slow progress. However, the attack can be made effective by computing the expectation over many samples (Expectation over Transformation).

**Vanishing / exploding gradients:** The defense causes the gradient to be numerically very small (due to very flat loss regions) or very large (due to sharp loss surfaces), making gradient-based optimization unreliable.

### 4.3 Detecting Gradient Masking

Several empirical tests can detect gradient masking:

1. **Transfer attack:** If the model appears robust against white-box PGD but is vulnerable to transfer attacks from a substitute model, gradient masking is likely. A truly robust model should be robust against transfer attacks too (since genuinely far decision boundaries are hard to cross even via transferability).

2. **Loss vs. iterations:** For a well-behaved loss surface, loss should increase monotonically with PGD iterations. If loss peaks early and then decreases, or if loss is flat across iterations, the gradient is likely masked.

3. **Attack with random target labels:** If the model is robust against a targeted PGD attack for a specific target class but not for others, gradient masking near specific class boundaries is likely.

4. **Black-box attack:** If a black-box query-based attack (which does not rely on gradients) achieves high success rate against a model that appears robust to white-box PGD, the model is gradient-masked.

5. **Bounded input sensitivity:** A model with masked gradients should have near-zero output change for all perturbations within the $\epsilon$-ball. If the model's output changes significantly when a large perturbation is applied (larger than $\epsilon$), the gradient was masked, not the decision boundary.

### 4.4 The BPDA Workaround

Athalye et al. also developed a general technique for attacking defenses with shattered gradients: Backward Pass Differentiable Approximation (BPDA). The idea is simple: if the forward pass involves a non-differentiable operation $g(x)$, replace it with a differentiable approximation $\hat{g}(x)$ during the backward pass (gradient computation) while using the true $g(x)$ during the forward pass.

For example, if the defense preprocesses the input with a non-differentiable quantization function $q(x)$ (rounding pixel values to 8 discrete levels), BPDA uses the identity function $\hat{q}(x) = x$ during the backward pass. The gradient computed with this approximation is a valid gradient of the identity-approximated network, which often closely tracks the true gradient for small perturbations.

Using BPDA, Athalye et al. broke 7 of 9 defenses published at ICLR 2018 that had claimed significant improvements in adversarial robustness.

---

## 5. Loss Function Choices for Attacks

The loss function used in the attack objective (the function $L$ in $\max_{x' \in \mathcal{S}} L(f(x'), y)$) is not simply fixed at the standard cross-entropy loss. The choice of loss function significantly affects the attack's effectiveness, particularly against robustly trained models.

### 5.1 Cross-Entropy Loss (CE)

The standard cross-entropy loss is:

$$L_{\text{CE}}(f(x), y) = -\log\left(\frac{e^{f(x)_y}}{\sum_{j=1}^{C} e^{f(x)_j}}\right) = -f(x)_y + \log\sum_{j=1}^C e^{f(x)_j}$$

where $f(x)_j$ is the $j$-th logit. CE loss is unbounded above — as the model becomes more confident about the wrong class, the loss grows without limit. This is useful for attacks: the gradient of CE loss provides a strong signal even far from the decision boundary.

However, CE loss can have numerical issues. When the model assigns very high confidence to the correct class (e.g., logit $f(x)_y = 100$, all other logits near 0), the softmax saturates to near 1, and the gradient of the CE loss becomes very small. This saturation can impede gradient-based attacks against robustly trained models that maintain high confidence on natural inputs.

### 5.2 Difference of Logits Ratio (DLR) Loss

The DLR loss, introduced by Croce and Hein as part of the AutoAttack framework, addresses the saturation problem of CE loss:

$$L_{\text{DLR}}(f(x), y) = -\frac{f(x)_y - \max_{j \neq y} f(x)_j}{f(x)_{\pi_1} - f(x)_{\pi_3}}$$

where $\pi_1, \pi_2, \pi_3$ are the indices of the three largest logits in descending order. The numerator is the difference between the true class logit and the maximum alternative class logit (this is negative when the model is wrong). The denominator is a normalization factor that prevents the loss from becoming numerically unstable.

The DLR loss does not saturate as CE loss does because it is defined in terms of logit differences rather than log-probabilities. For a robustly trained model with well-separated logits, DLR loss provides a more informative gradient signal than CE loss.

### 5.3 Carlini-Wagner (CW) Loss

The Carlini-Wagner attack (Carlini and Wagner, 2017) uses the loss function:

$$L_{\text{CW}}(f(x), y) = \max\left(\max_{j \neq y} f(x)_j - f(x)_y, -\kappa\right)$$

where $\kappa \geq 0$ is a confidence parameter. This loss is zero when the model already classifies the input correctly with margin $\kappa$, and negative (requiring minimization to find adversarial examples) when the model is correct. The CW attack minimizes $\|x' - x\|_2 + c \cdot L_{\text{CW}}(f(x'), y)$ over all $x'$, with $c$ found by binary search.

The CW loss is particularly useful for finding minimal-perturbation adversarial examples (unlike the fixed-$\epsilon$ formulation of PGD), and has been shown to break defenses that are resistant to FGSM and PGD.

---

## 6. AutoAttack: Reliable Evaluation Without Hyperparameter Tuning

The proliferation of gradient masking issues in the literature led to a significant problem: researchers could not trust robustness evaluations based solely on PGD, because PGD could be defeated by gradient masking without the researcher knowing. This motivated the development of AutoAttack by Croce and Hein (2020).

### 6.1 The Problem AutoAttack Solves

Evaluating adversarial robustness correctly is harder than it seems. Even if you use PGD with many steps and restarts, the evaluation may be unreliable if:

1. The step size is poorly tuned.
2. The number of restarts is insufficient.
3. The loss function saturates against this specific model.
4. The model has gradient masking that PGD cannot detect.

AutoAttack addresses all four issues by using an ensemble of complementary attacks, each of which has strengths that compensate for the others' weaknesses.

### 6.2 APGD-CE: Auto-PGD with Cross-Entropy Loss

APGD-CE is a parameter-free version of PGD with cross-entropy loss. "Parameter-free" means that the step size $\alpha$ is automatically adapted during the attack run, without requiring the user to tune it. The adaptation rule is:

- Start with $\alpha = 2\epsilon$.
- After each checkpoint (every $p\%$ of steps), check if the attack made progress (increased the loss or found more adversarial examples). If progress is satisfactory, maintain $\alpha$. If not, halve $\alpha$.
- This adaptive step size ensures that the attack makes progress even when the initial step size is too large or too small for this specific model.

APGD-CE also uses random restarts and takes the best adversarial example across all runs.

### 6.3 APGD-DLR: Auto-PGD with DLR Loss

APGD-DLR is the same adaptive PGD algorithm applied with the DLR loss rather than CE loss. Because the DLR loss does not saturate when CE does, APGD-DLR finds adversarial examples that APGD-CE misses on models with high-confidence correct predictions.

Running both APGD-CE and APGD-DLR together ensures that neither saturation nor the loss choice limits the attack's effectiveness.

### 6.4 Square Attack: Black-Box Component

The Square Attack (Andriushchenko et al., 2020) is a score-based black-box attack that finds adversarial examples using only the model's output probabilities, without computing gradients. It works by iteratively proposing random square-shaped perturbations and accepting them if they increase the attack objective.

Including Square Attack in the AutoAttack ensemble is critical for detecting gradient masking: if a model appears robust against APGD-CE and APGD-DLR (gradient-based attacks) but falls to Square Attack (gradient-free), gradient masking is confirmed.

### 6.5 FAB: Fast Adaptive Boundary Attack

FAB (Croce and Hein, 2020) is a complementary attack designed to find adversarial examples with minimal perturbation norm, useful for detecting defenses that only work at specific $\epsilon$ values. AutoAttack Standard includes APGD-CE, APGD-DLR, FAB, and Square Attack.

### 6.6 The RobustBench Leaderboard

RobustBench (Croce et al., 2021, at robustbench.epfl.ch) is the community-standard benchmark for adversarial robustness, using AutoAttack as the evaluation protocol. It provides a curated leaderboard of models with their AutoAttack robust accuracy on CIFAR-10, CIFAR-100, and ImageNet.

Using AutoAttack for evaluation, rather than a manually configured PGD, ensures that results are comparable across papers and that gradient masking defenses do not appear artificially robust. For your problem set submissions and for any robustness claims you make in research, AutoAttack (or equivalent) is the minimum acceptable evaluation standard.

---

## 7. Practical Guidelines for Reliable Evaluation

The lessons of this lecture can be distilled into a set of practical guidelines for anyone evaluating the adversarial robustness of a machine learning system.

**Use multiple attacks.** No single attack finds all adversarial examples. Use both white-box gradient-based attacks (APGD-CE, APGD-DLR) and black-box attacks (Square Attack) at minimum. If the white-box attack finds fewer adversarial examples than the black-box attack, investigate gradient masking.

**Use multiple loss functions.** At a minimum, evaluate with CE loss and DLR loss. If results differ substantially, the model may be gradient-masked with respect to one loss.

**Use random restarts.** For any model that claims robustness, use at least 5–10 random restarts. If the robust accuracy decreases significantly with additional restarts, use more.

**Check gradient behavior.** Plot loss vs. iteration for your PGD runs. Loss should increase (or stay flat) monotonically. If it decreases after an initial increase, the gradient is masked.

**Check the transfer gap.** If robust accuracy against white-box attacks is higher than against transfer attacks from a surrogate model, gradient masking is likely.

**Use AutoAttack for final evaluation.** For any defense claiming robustness improvements, AutoAttack with the standard configuration is the expected evaluation. Deviating from this requires explicit justification.

**Report confidence intervals.** Robust accuracy estimates from a finite test set have uncertainty. For a test set of $n = 10,000$ images and robust accuracy $\hat{r} = 0.50$, the 95% confidence interval is approximately $\hat{r} \pm 1.96\sqrt{\hat{r}(1-\hat{r})/n} \approx 0.50 \pm 0.01$. This uncertainty is small enough to be ignored for most comparisons, but matters when comparing models with similar robust accuracy.

---

## 8. Discussion Questions

1. **BIM convergence:** BIM guarantees monotonically increasing loss at each step (since we take a step in the direction of the gradient). But the loss can decrease after the projection step if the projection moves us back from a high-loss region to a lower-loss region inside the $\epsilon$-ball. Can you construct a toy example where this happens? Does this suggest that PGD might not be monotone in practice?

2. **Step size vs. number of steps:** For a fixed computational budget of $T \cdot d$ gradient evaluations (where $d$ is dimension), is it better to use fewer steps with larger $\alpha$, or more steps with smaller $\alpha$? Does the answer depend on the smoothness of the loss function? On the architecture?

3. **Random restarts in theory:** The benefit of random restarts for non-convex optimization depends on the number and distribution of local maxima in the loss landscape. Do you expect robustly trained models to have more or fewer local maxima in the loss landscape within the $\epsilon$-ball than naturally trained models? Why?

4. **AutoAttack and arms race:** AutoAttack was designed to be a reliable evaluation tool that is hard to "cheat" against. But if defenses are specifically designed to defeat AutoAttack (rather than to achieve genuine robustness), AutoAttack may become less reliable over time. Is this a fundamental problem with any fixed evaluation methodology? How should the community respond?

5. **Loss function choice for defense:** We've discussed loss function choices for attacks. What loss function should be used during adversarial training (the defense)? Should it match the attack loss function, or should it be different? What is the tradeoff?

---

## 9. Key Takeaways

BIM extends FGSM by iterating the gradient sign step with a small step size, projecting back onto the $\epsilon$-ball at each step. The projection operator is what enforces the perturbation budget throughout all iterations.

PGD frames the attack as a constrained optimization problem and solves it with projected gradient ascent. For L∞, PGD-L∞ is equivalent to BIM. For L2, the projection step differs (projection onto a Euclidean ball). Random restarts are essential for avoiding local maxima.

The "strongest first-order attack" theorem gives PGD theoretical standing as the baseline for adversarial robustness evaluation, motivating PGD adversarial training.

Gradient masking is a systematic threat to reliable robustness evaluation. A defense that appears robust against gradient-based attacks may achieve this through artifactual gradient masking rather than genuine robustness. Always complement gradient-based evaluation with gradient-free black-box attacks.

AutoAttack provides a parameter-free, ensemble-based evaluation protocol that is the community standard for adversarial robustness benchmarking. Use it.

---

## Assigned Reading

- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). "Towards deep learning models resistant to adversarial attacks." ICLR 2018.
- Athalye, A., Carlini, N., & Wagner, D. (2018). "Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples." ICML 2018.
- Croce, F. & Hein, M. (2020). "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks." ICML 2020.

Problem Set 2 is due next week. Recall that you must implement FGSM, BIM, and PGD with both L∞ and L2 variants, evaluate on CIFAR-10, and produce epsilon vs. robust accuracy curves. See the assignment specification for details.

---

*End of Lecture 5 Notes*
*Next lecture: Week 6 — Adversarial Training: Madry et al. and Beyond*
