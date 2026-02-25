# CS 6810 — Adversarial Machine Learning
## Week 03: White-Box Attacks — Carlini-Wagner, FAB, and the AutoAttack Ensemble

**Prerequisites:** Week 01 (attack taxonomy, Lagrangian formulation), familiarity with the Adam optimizer and automatic differentiation.

**Learning Objectives:**
1. Implement the C&W L2 attack from scratch, including binary search on $c$ and the tanh change of variables.
2. Derive the gradient of the C&W objective with respect to the reparametrized variable $w$.
3. Explain why FAB finds minimum-distortion adversarial examples faster than C&W.
4. Describe the AutoAttack ensemble protocol and explain why each component is necessary.
5. Identify gradient masking from attack diagnostic signals.

---

## 1. Review: Why White-Box Evaluation Matters

A "defense" is only worth publishing if it provides genuine robustness against the strongest known attacks. The history of adversarial ML is littered with defenses that broke the gradient signal — causing attacks to fail not because the defense improved robustness but because the attack was computing wrong gradients.

White-box attacks — attacks with full gradient access — provide a *necessary condition* for claimed robustness. If your defense cannot withstand a white-box attacker who knows everything about the model, it certainly cannot withstand adaptive adversaries in deployment. This week we study the strongest known white-box attacks for evasion.

---

## 2. The Carlini-Wagner (C&W) Attack

### 2.1 Problem Setup

Given:
- Neural network $f : [0,1]^n \to \mathbb{R}^K$ with logit outputs.
- Original correctly-classified input $x_0 \in [0,1]^n$ with true class $y$.
- Target class $t \neq y$ (targeted attack; we can adapt for untargeted).

Goal: Find the smallest perturbation $\delta$ such that $\arg\max_k f(x_0 + \delta)_k = t$.

The primal problem (from Week 01):

$$\min_\delta \|\delta\|_2^2 \quad \text{subject to} \quad C(x_0 + \delta) = t, \quad x_0 + \delta \in [0,1]^n$$

### 2.2 The Hinge Loss Function

The constraint $C(x') = t$ means $f(x')_t > f(x')_i$ for all $i \neq t$, equivalently $\max_{i \neq t} f(x')_i - f(x')_t < 0$.

The C&W hinge loss encodes this directly:

$$g(x') = \max\!\left(\max_{i \neq t} f(x')_i - f(x')_t,\; -\kappa\right) \tag{1}$$

Properties of $g$:
- $g(x') > 0$ iff the target class is NOT the top-scoring class.
- $g(x') = -\kappa$ iff the target class dominates by at least $\kappa$ logits.
- $g(x') = 0$ is the decision boundary.
- $g$ is continuous and piecewise differentiable (differentiable everywhere except at the max argument switch).

The gradient of $g$ with respect to $f(x')$ is:

$$\frac{\partial g}{\partial f(x')_i} = \begin{cases} 0 & \text{if } g(x') = -\kappa \text{ (already succeeded by margin)} \\ -1 & \text{if } i = t \text{ and } g(x') > -\kappa \\ +1 & \text{if } i = j^* = \arg\max_{i \neq t} f(x')_i \text{ and } g(x') > -\kappa \\ 0 & \text{otherwise} \end{cases}$$

This propagates cleanly through automatic differentiation.

### 2.3 The Lagrangian Objective

The full C&W objective (Lagrangian relaxation):

$$\mathcal{L}_{\text{CW}}(w, c) = \underbrace{\left\|\frac{1}{2}(\tanh(w)+1) - x_0\right\|_2^2}_{\text{distortion term}} + \; c \cdot \underbrace{g\!\left(\frac{1}{2}(\tanh(w)+1)\right)}_{\text{attack loss term}} \tag{2}$$

where $w \in \mathbb{R}^n$ is the reparametrized variable (see Section 2.4) and $c > 0$ is the trade-off constant.

### 2.4 The Change of Variables: Full Derivation

**Motivation:** We need $x' = x_0 + \delta \in [0,1]^n$ for all $\delta$ we consider. Gradient descent can push $x'$ outside $[0,1]^n$. We want to optimize over an unconstrained space.

**Construction:** We seek a differentiable, invertible map $\phi: \mathbb{R} \to (0,1)$. The natural choice is the logistic sigmoid, but $\tanh$ scaled and shifted is equivalent and slightly more convenient:

$$\phi(w_i) = \frac{1 + \tanh(w_i)}{2} \in (0, 1) \tag{3}$$

Define $x'_i = \phi(w_i)$, so $x' = \phi(w) \in (0,1)^n$ for any $w \in \mathbb{R}^n$.

**Inverse:** Given $x'_i \in (0,1)$, we recover $w_i = \phi^{-1}(x'_i) = \tanh^{-1}(2x'_i - 1)$.

**Initialization:** We want to start at $\delta = 0$, i.e., $x' = x_0$. Set $w^{(0)} = \tanh^{-1}(2x_0 - 1)$. For pixels at 0 or 1, clip: $x_0 \leftarrow \text{clip}(x_0, 10^{-6}, 1 - 10^{-6})$ before computing $w^{(0)}$.

**The perturbation in terms of $w$:**

$$\delta(w) = x'(w) - x_0 = \frac{1+\tanh(w)}{2} - x_0 \tag{4}$$

**Jacobian:** The Jacobian of $x'$ with respect to $w$ is diagonal:

$$\frac{\partial x'_i}{\partial w_i} = \frac{1}{2}(1 - \tanh^2(w_i)) = \frac{1}{2}\text{sech}^2(w_i) \tag{5}$$

This is always positive (ranging from $\frac{1}{2}$ at $w_i = 0$ to $0$ as $|w_i| \to \infty$), so the map is a diffeomorphism from $\mathbb{R}^n$ to $(0,1)^n$.

**The gradient of the distortion term:**

$$\frac{\partial}{\partial w_i}\left\|\frac{1+\tanh(w)}{2} - x_0\right\|_2^2 = 2\left(\frac{1+\tanh(w_i)}{2} - x_{0,i}\right) \cdot \frac{1}{2}\text{sech}^2(w_i)$$

$$= (x'_i - x_{0,i}) \cdot \text{sech}^2(w_i) \tag{6}$$

**The full gradient of $\mathcal{L}_{\text{CW}}$:**

$$\frac{\partial \mathcal{L}_{\text{CW}}}{\partial w_i} = (x'_i - x_{0,i}) \cdot \text{sech}^2(w_i) + c \cdot \frac{\partial g}{\partial x'_i} \cdot \frac{\partial x'_i}{\partial w_i}$$

$$= (x'_i - x_{0,i}) \cdot \text{sech}^2(w_i) + c \cdot \left[\sum_k \frac{\partial g}{\partial f_k} \cdot \frac{\partial f_k}{\partial x'_i}\right] \cdot \frac{1}{2}\text{sech}^2(w_i) \tag{7}$$

In practice, automatic differentiation computes this exactly — no manual derivation needed at runtime.

### 2.5 Binary Search on the Trade-Off Constant $c$

**Why binary search?** We do not know the right value of $c$ a priori. Too small: the attack fails (the optimizer minimizes distortion but does not find misclassification). Too large: the attack finds misclassification but with unnecessarily large distortion.

**Protocol (Carlini & Wagner 2017):**

```
BINARY SEARCH on c:
  c_low = 0
  c_high = +∞ (start with a large initial guess, e.g., 1e10)

  For outer_step in 1..num_binary_search_steps (default: 9):
    c = (c_low + c_high) / 2

    Run inner optimizer (Adam) for max_iterations steps:
      w^{(0)} = arctanh(2 * x0 - 1)  # initialization
      For t in 1..max_iterations:
        Compute L_CW(w, c) and its gradient
        w = Adam_step(w, gradient)
        x' = (tanh(w) + 1) / 2
        If C(x') == target:
          Record best_adversarial = x'  (keep the one with smallest ||delta||_2)

    If attack succeeded (best_adversarial was found):
      Record attack_succeeded = True
      c_high = c  # try smaller c (less emphasis on attack loss)
    Else:
      c_low = c  # try larger c (more emphasis on attack loss)

  Return best_adversarial (or x0 if no adversarial found)
```

**Typical hyperparameter values:**
- `num_binary_search_steps` = 9 (gives 3 significant bits of precision on $c$)
- `initial_c` = 1e-3 (start small, let binary search increase if needed)
- `max_iterations` = 1000 (for reliable convergence; 10000 for evaluation)
- `learning_rate` (Adam) = 1e-2
- `kappa` = 0 (targeted attacks on ImageNet; higher for targeted attacks where confidence matters)
- `adam_beta1` = 0.9, `adam_beta2` = 0.999

**Why Adam and not SGD?** The C&W loss landscape has poor conditioning — the distortion term and the attack loss term have very different gradient magnitudes. Adam's adaptive learning rate handles this well. SGD with a fixed learning rate would require careful tuning per example; Adam is nearly hyperparameter-free.

### 2.6 The C&W L-Infinity Version

For an L-infinity attack, minimizing $\|\delta\|_\infty$ is harder because the L-inf norm is non-differentiable. C&W propose a Lagrangian method with per-coordinate slack variables.

Let $\tau > 0$ be the current L-infinity bound (to be minimized). The objective is:

$$\min_{w, \tau} \; \tau + c \cdot g\!\left(\frac{1+\tanh(w)}{2}\right) \quad \text{subject to} \quad \left|\frac{1+\tanh(w_i)}{2} - x_{0,i}\right| \leq \tau \; \forall i \tag{8}$$

Equivalently, introduce the constraint via penalty:

$$\min_{w} \; c \cdot g(x'(w)) + \sum_i \max\!\left(\left|\delta_i(w)\right| - \tau,\; 0\right) \tag{9}$$

and minimize over $\tau$ by checking whether the current solution satisfies the constraint. This is solved by iterating: fix $\tau$, optimize $w$; decrease $\tau$ if all constraints satisfied; increase if not.

In practice, the C&W L-2 attack is far more commonly used than C&W L-inf (PGD is preferred for L-inf evaluations).

### 2.7 Untargeted Version

For an untargeted attack, we replace the hinge loss with:

$$g_{\text{untargeted}}(x') = \max\!\left(f(x')_y - \max_{i \neq y} f(x')_i,\; -\kappa\right) \tag{10}$$

This is positive iff the true class $y$ still dominates (attack failed). We minimize this to make the true class lose to some other class.

### 2.8 Worked Example: C&W on a 2-Class Linear Model

**Setup:** 2D input $x \in [0,1]^2$, binary classifier $f(x) = [w_1^\top x, w_2^\top x]$ with $w_1 = [1, 0]$, $w_2 = [-1, 0]$. Original input $x_0 = [0.3, 0.5]^\top$, true class 0. Logits: $f(x_0) = [0.3, -0.3]$, so $\arg\max = 0$ (correct). We want to flip to class 1 ($t = 1$).

**Margin:** $f(x_0)_0 - f(x_0)_1 = 0.3 - (-0.3) = 0.6$ logit margin. We need to cross this.

**Initial $w$:** $w^{(0)} = \tanh^{-1}(2 \cdot [0.3, 0.5] - 1) = \tanh^{-1}([-0.4, 0]) = [-0.424, 0]$.

**Target hinge loss at initialization:** $g(x_0) = \max(f(x_0)_0 - f(x_0)_1, -\kappa) = \max(0.6, 0) = 0.6$ (attack has not started).

**Gradient of hinge loss w.r.t. logits:** The "winning" non-target class is class 0. So $\partial g / \partial f_0 = +1$, $\partial g / \partial f_1 = -1$.

**Gradient of logits w.r.t. $x'$:** Since $f_0 = w_1^\top x' = x'_1$ (first feature), $\partial f_0 / \partial x'_1 = 1$, $\partial f_0 / \partial x'_2 = 0$. Similarly $\partial f_1 / \partial x'_1 = -1$.

**Combined gradient of $g$ w.r.t. $x'$:**
$\partial g / \partial x'_1 = (\partial g / \partial f_0)(\partial f_0 / \partial x'_1) + (\partial g / \partial f_1)(\partial f_1 / \partial x'_1) = (1)(1) + (-1)(-1) = 2$

$\partial g / \partial x'_2 = 0$ (second feature doesn't affect either logit).

**Gradient w.r.t. $w$:** By chain rule, $\partial g / \partial w_1 = (\partial g / \partial x'_1)(\partial x'_1 / \partial w_1) = 2 \cdot \text{sech}^2(w_1^{(0)}) = 2 \cdot (1 - \tanh^2(-0.424)) = 2 \cdot (1 - 0.16) = 1.68$.

**First Adam step** (simplified, ignoring momentum terms): $w_1^{(1)} = w_1^{(0)} - \alpha \cdot c \cdot \partial g / \partial w_1 = -0.424 - 0.01 \cdot c \cdot 1.68$.

For $c = 1$: $w_1^{(1)} = -0.424 - 0.0168 = -0.441$.

New $x'_1 = (1 + \tanh(-0.441))/2 = (1 - 0.414)/2 = 0.293$. The perturbation $\delta_1 = 0.293 - 0.3 = -0.007$. The attack is moving $x'_1$ to the left (decreasing the first feature), which increases $f_1$ and decreases $f_0$ — correct direction.

After convergence (approximately $0.6/(2 \cdot \text{learning rate})$ steps), the attack finds $x' \approx [0.0, 0.5]$, with $\delta = [-0.3, 0]$ and $\|\delta\|_2 = 0.3$. This matches the analytical minimum-distortion result: $\|\delta^*\|_2 = |f(x_0)_y - f(x_0)_t| / (2\|w\|_2) = 0.6/(2\cdot 1) = 0.3$.

---

## 3. DeepFool (Brief Review)

DeepFool (Moosavi-Dezfooli et al. 2016) finds the nearest decision boundary via iterative linearization.

**Algorithm:**
1. Initialize $x_0^{(0)} = x_0$.
2. At each iteration $t$:
   - Compute the linear approximation of the classifier near $x^{(t)}$.
   - Find the nearest hyperplane (decision boundary of the current leading-class vs. each other class).
   - Project $x^{(t)}$ to the nearest hyperplane plus a small overshoot.

**Formally:** Let $\hat{y} = C(x^{(t)})$. For each class $k \neq \hat{y}$, the linearized boundary is $\{x : (f(x^{(t)})_{\hat{y}} - f(x^{(t)})_k) + (w_{\hat{y}} - w_k)^\top (x - x^{(t)}) = 0\}$ where $w_k = \nabla_{x} f(x^{(t)})_k$. The distance to this boundary is:

$$d_k = \frac{|f(x^{(t)})_{\hat{y}} - f(x^{(t)})_k|}{\|w_{\hat{y}} - w_k\|_2}$$

The minimum-distance boundary corresponds to class $k^* = \arg\min_k d_k$. The perturbation step is:

$$\delta^{(t)} = \frac{d_{k^*}}{\|w_{\hat{y}} - w_{k^*}\|_2^2} \cdot (w_{\hat{y}} - w_{k^*})$$

**Comparison to C&W:**
- DeepFool is faster (no binary search on $c$, no Adam optimizer state).
- DeepFool can fail when the loss landscape is highly nonlinear.
- C&W produces smaller L-2 distortion in practice because it uses the full (nonlinear) gradient, not a linearization.
- DeepFool does not support $\kappa > 0$ (confidence margin) by design.

DeepFool is most useful as a fast approximation; C&W is the gold standard for L-2 distortion.

---

## 4. FAB: Fast Adaptive Boundary Attack

### 4.1 Core Idea

FAB (Croce & Hein, 2020) finds adversarial examples on the decision boundary — the minimum-distortion adversarial examples — using a different strategy than C&W.

**The key insight:** C&W minimizes $\|\delta\|_2 + c \cdot g(x')$, which has a tension between the two terms. FAB instead looks for a point $x'$ that is:
1. On the decision boundary ($C(x') = t$).
2. Closest to $x_0$ in L-p norm.

Rather than minimizing a Lagrangian, FAB directly approximates the projection of $x_0$ onto the decision boundary.

### 4.2 The FAB Step

At each iteration, FAB linearizes the decision boundary at the current point $x^{(t)}$ and finds the point on the linearized boundary closest to $x_0$. Let $h(x) = f(x)_t - \max_{i \neq t} f(x)_i$ (the margin of the target class). Then:

- The linearized boundary: $h(x^{(t)}) + \nabla h(x^{(t)})^\top (x - x^{(t)}) = 0$.
- This is a hyperplane in $\mathbb{R}^n$.
- The closest point on this hyperplane to $x_0$ can be computed in closed form (for L-2 this is a projection; for L-1 and L-inf it's a linear programming problem).

**FAB update (L-2 case):**

$$x^{(t+1)} = x_0 + \alpha \cdot \underbrace{\left[\text{proj onto linearized boundary from } x_0\right]}_{\delta^*} + (1-\alpha) \cdot (x^{(t)} - x_0) \tag{11}$$

where $\alpha \in (0,1)$ is an interpolation parameter. This moves toward the boundary while keeping track of $x_0$.

### 4.3 Adaptive Restart

FAB uses multiple random restarts with a distance-based acceptance criterion: only accept a new restart if the resulting adversarial example has strictly smaller L-p distortion than the current best. This is more efficient than fixed random restarts.

### 4.4 FAB vs. C&W Comparison

| Property | C&W | FAB |
|----------|-----|-----|
| Approach | Lagrangian + binary search | Direct boundary projection |
| Per-example cost | $9 \times 1000$ gradient steps | $100$–$500$ gradient steps |
| L-2 distortion quality | Excellent | Comparable or better |
| Supports L-inf | Poorly (different algorithm) | Yes (L-p FAB for any p) |
| Gradient masking robustness | Good (uses Adam, multiple restarts) | Good (adaptive restarts) |
| Implements $\kappa > 0$ | Yes | No (always boundary) |

---

## 5. AutoAttack: The Ensemble Evaluation Protocol

### 5.1 The Problem: Overoptimistic Robustness Evaluation

A fundamental challenge in adversarial ML: many published defenses were later shown to be broken by adaptive attacks. The failures fell into two categories:

1. **Gradient masking:** The defense deliberately or accidentally breaks the gradient signal (e.g., by using non-differentiable components, input randomization, or very flat loss surfaces). Gradient-based attacks fail not because the defense is robust but because they compute wrong gradients.

2. **Parameter sensitivity:** An attack with the wrong hyperparameters (step size, number of iterations) may fail on a particular defense even though the defense is vulnerable with better parameters. Published results may cherry-pick attack parameters that look bad.

AutoAttack (Croce & Hein, ICML 2020) addresses both by running a carefully designed ensemble of attacks, each with different properties.

### 5.2 APGD-CE: Adaptive PGD with Cross-Entropy

APGD-CE is a parameter-free version of PGD that adapts the step size during optimization.

**Update rule:**

$$z_{t+1} = \Pi_{x_0 + \mathcal{S}}\!\left(x_t + \alpha_t \cdot \text{sign}(\nabla_x \mathcal{L}_{\text{CE}}(x_t, y))\right)$$

$$x_{t+1} = \Pi_{x_0 + \mathcal{S}}\!\left(x_t + \eta \cdot (z_{t+1} - x_t) + (1-\eta)(x_t - x_{t-1})\right) \tag{12}$$

The second line is a momentum step. The step size $\alpha_t$ is adapted as follows:

**APGD step-size adaptation:**
- Start with $\alpha_0 = 2\epsilon$ (large initial step).
- After a checkpoint (every $p = 0.22 T$ iterations), check if the best loss increased by less than a threshold $\rho = 0.75$ of its maximum possible increase.
- If the attack is "stagnating," halve the step size: $\alpha \leftarrow \alpha / 2$.
- Restart from the best-found point when the step size is halved.

This makes APGD robust to the step-size sensitivity problem without manual tuning. It runs for $N_{\text{iter}} = 100$ iterations.

### 5.3 APGD-DLR: Using the DLR Loss

APGD-DLR is identical to APGD-CE except it uses the Difference of Logits Ratio (DLR) loss:

$$\mathcal{L}_{\text{DLR}}(x, y) = -\frac{f(x)_y - \max_{i \neq y} f(x)_i}{f(x)_{\pi_1} - f(x)_{\pi_3}} \tag{13}$$

where $\pi_1, \pi_2, \pi_3$ are the indices of the top-3 logit classes.

**Why DLR resists gradient masking:** Consider a defense that clips logits to a small range, making the cross-entropy gradient near-zero everywhere. The DLR denominator $f(x)_{\pi_1} - f(x)_{\pi_3}$ normalizes for the scale of the logit range. Even when logits are clipped or compressed, the DLR gradient is informative because it is scale-invariant.

**Why the denominator?** Without it, the DLR loss reduces to a margin. The denominator is crucial: if the top-3 logits all collapse together (a common signature of gradient-masked models), the denominator is small and the loss gradient becomes very large, breaking through the masking.

**Formal property:** APGD-DLR is affine invariant in the logit space — multiplying all logits by the same constant does not change the DLR loss or its gradient (up to sign). This makes it immune to scale-based gradient masking.

### 5.4 FAB (Targeted)

The targeted FAB attack is included in AutoAttack to find adversarial examples that are targeted (to multiple target classes). The rationale: for some defenses, there is a target class that is much easier to reach than others, and untargeted attacks may miss this.

In the AA-standard protocol, FAB attacks each of the 9 non-true classes (for a 10-class classifier like CIFAR-10), taking the best (smallest distortion) result.

### 5.5 Square Attack

The Square Attack (Andriushchenko et al. 2020) is a score-based (black-box) attack included in AutoAttack as a diversity component.

**Motivation:** Some defenses use stochastic components (random smoothing, random resizing) that break gradient-based attacks (the gradient through the stochastic component is noisy). The Square Attack does not use gradients, so it is immune to gradient masking.

**Algorithm:** At each step, sample a random square-shaped perturbation (a random square of pixels set to $\pm \epsilon$), and accept it if it increases the loss. This is a form of coordinate-descent on random blocks.

**Key property:** Square Attack has provable expected loss guarantees under any loss function, regardless of gradient availability. It is a pure score-based method with no gradient computation.

### 5.6 The AA-Standard Protocol

**AA-standard** runs all four attacks (APGD-CE, APGD-DLR, FAB, Square Attack) on each test example. An example is "robustly classified" only if it survives all four attacks.

The robust accuracy is:
$$\text{RA}_{\text{AA}} = \frac{1}{N}\sum_{i=1}^N \mathbf{1}\!\left[\text{example } i \text{ survives all four attacks}\right]$$

**AA-rand** is a cheaper version that runs only APGD-CE and APGD-DLR (no FAB or Square), for quick preliminary evaluation.

**Why the ensemble is necessary:** No single attack is universally best. A defense might:
- Break gradient masking → Square Attack detects it.
- Succeed against CE but not DLR → APGD-DLR catches it.
- Be vulnerable to high-confidence targeted examples → FAB catches it.
- Use stochastic components → Square Attack is unaffected.

Running all four closes most known evaluation gaps.

### 5.7 Gradient Masking Detection

AutoAttack's diagnostic outputs reveal gradient masking. Signs of gradient masking:
1. APGD-CE and APGD-DLR find very different loss values → gradient is unreliable.
2. Square Attack success rate much higher than APGD-CE → gradients are masked.
3. Robust accuracy does not decrease monotonically with $\epsilon$ → defense is non-robust (not just harder to break at smaller $\epsilon$).
4. Loss decreases during PGD but prediction does not flip → loss landscape is disconnected from predictions.

---

## 6. Worked Example: Tracing C&W Convergence

### Setup

Two-class linear classifier in $\mathbb{R}^2$:
- $f_1(x) = 2x_1 - x_2$ (class 1 logit)
- $f_2(x) = -x_1 + 2x_2$ (class 2 logit)
- Original input: $x_0 = [0.7, 0.3]$, true class 1 ($f_1(x_0) = 1.1, f_2(x_0) = 0.2$).
- Target class: $t = 2$.

**C&W loss:** $g(x') = \max(f_1(x') - f_2(x'), -\kappa)$. With $\kappa = 0$: $g(x') = \max(f_1(x') - f_2(x'), 0)$.

At $x_0$: $g = \max(1.1 - 0.2, 0) = 0.9$. Attack needs to make $f_2(x') > f_1(x')$.

**Decision boundary:** $f_1 = f_2 \Leftrightarrow 2x_1 - x_2 = -x_1 + 2x_2 \Leftrightarrow 3x_1 = 3x_2 \Leftrightarrow x_1 = x_2$.

Minimum L-2 distance from $x_0 = [0.7, 0.3]$ to the line $x_1 = x_2$: distance $= |0.7 - 0.3|/\sqrt{2} = 0.4/\sqrt{2} \approx 0.283$.

The minimum-distortion adversarial example is the projection of $x_0$ onto $x_1 = x_2$: $x^* = [(0.7+0.3)/2, (0.7+0.3)/2] = [0.5, 0.5]$.

**C&W convergence trace** (c = 1, Adam, simplified to gradient descent for illustration):

| Iteration | $w_1$ | $w_2$ | $x'_1$ | $x'_2$ | $g(x')$ | $\|\delta\|_2$ | Total loss |
|-----------|-------|-------|---------|---------|----------|----------------|------------|
| 0 | 0.867 | -0.619 | 0.700 | 0.300 | 0.900 | 0.000 | 0.900 |
| 10 | 0.720 | -0.320 | 0.672 | 0.380 | 0.584 | 0.100 | 0.690 |
| 50 | 0.422 | 0.070 | 0.604 | 0.517 | 0.172 | 0.163 | 0.198 |
| 100 | 0.196 | 0.289 | 0.549 | 0.572 | 0.000 | 0.210 | 0.210 |
| 200 | 0.071 | 0.090 | 0.518 | 0.522 | 0.000 | 0.255 | 0.255 |
| 500 | 0.010 | 0.011 | 0.502 | 0.503 | 0.000 | 0.282 | 0.282 |

The C&W attack converges to $x' \approx [0.5, 0.5]$ with $\|\delta\|_2 \approx 0.283$, matching the analytical minimum distortion.

**Key observation:** After iteration 100, $g = 0$ (misclassification achieved), and the optimizer continues reducing $\|\delta\|_2$ (the distortion term). This is why C&W finds minimum-distortion adversarial examples — it keeps optimizing even after the attack succeeds.

---

## 7. Implementation Notes and Common Pitfalls

### 7.1 The Tanh Saturation Problem

When $|w_i|$ is large, $\text{sech}^2(w_i) \approx 0$, meaning gradients through the tanh reparametrization become very small. This can cause stagnation late in the optimization.

**Fix:** Detect when $|\tanh(w_i)| > 0.9999$ and either re-initialize or cap the gradient.

### 7.2 Binary Search Initialization

The initial value of $c$ matters. Too small: the first few binary search steps waste time. Too large: the first successful attack may have large distortion, biasing the binary search downward aggressively.

Carlini & Wagner suggest:
- Start with $c = 10^{-3}$.
- Use $c_{\text{high}} = 10^{10}$ (a stand-in for $+\infty$).
- Use an exponential search if the first inner optimization fails (double $c$ until success, then binary search).

### 7.3 Tracking the Best Adversarial Example

During the inner optimization, track the best adversarial example found so far (smallest $\|\delta\|_2$ among all $x'$ with $C(x') = t$). Do not update based on the final $x'$ alone — an intermediate iterate may have smaller distortion.

### 7.4 Evaluating Against Defenses

When using C&W to evaluate a defense:
1. Make sure the defense is differentiable (or use the Backward Pass Differentiable Approximation (BPDA) technique for non-differentiable defenses).
2. Use enough iterations (at least 1000) — many published claims of robustness fail at 10000 iterations.
3. Run multiple random restarts.
4. Check that the loss actually decreases during optimization — a flat loss curve indicates gradient masking.

---

## 8. Discussion Questions

1. **Convergence:** In Section 6, the C&W optimizer continues minimizing distortion after $g(x') = 0$ at iteration 100. Modify the objective (2) to include an early termination condition that stops the optimization when $g < 0$ without losing this behavior. (Hint: think about when the clamping $\max(\cdot, -\kappa)$ activates.)

2. **Binary search depth:** Carlini & Wagner use 9 binary search steps. With initial bracket $[c_{\min}, c_{\max}]$, after 9 steps the precision is $(c_{\max} - c_{\min}) / 2^9 \approx (c_{\max} - c_{\min}) / 512$. If $c_{\max} = 10$ and $c_{\min} = 0$, what is the precision? Is 9 steps enough? Propose an adaptive stopping criterion.

3. **DLR gradient masking:** Construct a neural network architecture (you can use toy weights) where the cross-entropy gradient is exactly zero at $x_0$ even though $x_0$ is not adversarial, but the DLR gradient is nonzero. What is the practical implication?

4. **Square Attack inclusion:** The Square Attack is included in AutoAttack because gradient-based attacks may fail on stochastic defenses. Describe one defense for which:
   (a) APGD-CE fails (success rate < 10%).
   (b) Square Attack succeeds (success rate > 50%).
   Justify your answer by analyzing the gradient signal.

5. **FAB geometry:** In L-2 FAB, the update moves from $x_0$ toward the linearized decision boundary. Could FAB fail if the true decision boundary is far from the linearized boundary? Construct a 2D example where this happens and predict how many iterations FAB would need.

6. **Adaptive attack design:** Suppose a defense applies the transformation $h(x) = x + \sigma \cdot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$ (random Gaussian noise) before passing through the classifier. APGD-CE fails because gradients through $h$ are noisy. Propose an adaptive attack for this defense that does not use gradients through $h$. How does your attack relate to Expectation Over Transformation (EOT)?

7. **C&W vs. PGD distortion:** Empirically, C&W L-2 finds adversarial examples with $\|\delta\|_2 \approx 0.5$ on CIFAR-10 (ResNet-18), while PGD-40 with L-2 projection at the same budget finds examples with $\|\delta\|_2 = 0.5$ (by construction) but many examples require the full budget. Explain this discrepancy. Under what conditions would PGD-L2 match C&W in distortion?

---

## 9. Summary

| Attack | Core Mechanism | Primary Norm | Strength | Weakness |
|--------|---------------|-------------|---------|---------|
| C&W L2 | Lagrangian + binary search + tanh reparam | L-2 | Minimum distortion, handles box constraints | Expensive (9000+ gradient steps) |
| C&W Linf | Lagrangian + per-coordinate slack | L-inf | Principled, can use confidence $\kappa$ | Slower than PGD Linf in practice |
| DeepFool | Iterative linearization + projection to boundary | L-2 | Fast, no hyperparameters | Fails on highly nonlinear boundaries |
| FAB | Direct boundary projection, adaptive restarts | L-p (any) | Minimum distortion, fast, generalizes | Complex implementation |
| APGD-CE | Adaptive PGD + CE loss | L-inf or L-2 | Parameter-free, robust to step-size | Can fail on gradient-masked defenses |
| APGD-DLR | Adaptive PGD + DLR loss | L-inf or L-2 | Scale-invariant gradient, breaks some masking | More complex loss |
| Square | Random square perturbations | L-inf | No gradient needed, detects masking | Slower convergence than gradient methods |
| AutoAttack | Ensemble of all above | L-inf or L-2 | State-of-the-art evaluation | Expensive, not suitable for training |

---

## 10. Further Reading

**Required:**
- Carlini & Wagner (2017). "Towards Evaluating the Robustness of Neural Networks." IEEE S&P. [C&W attack]
- Croce & Hein (2020). "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks." ICML. [AutoAttack]

**Recommended:**
- Moosavi-Dezfooli et al. (2016). "DeepFool: a simple and accurate method to fool deep neural networks." CVPR. [DeepFool]
- Croce & Hein (2020). "Minimally distorted adversarial examples with a fast adaptive boundary attack." ICML. [FAB]
- Andriushchenko et al. (2020). "Square Attack: a query-efficient black-box adversarial attack via random search." ECCV. [Square Attack]
- Tramer et al. (2020). "On Adaptive Attacks to Adversarial Example Defenses." NeurIPS. [Why adaptive attacks matter]
- Athalye et al. (2018). "Obfuscated Gradients Give a False Sense of Security." ICML. [Gradient masking survey]
