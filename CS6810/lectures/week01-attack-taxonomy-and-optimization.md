# CS 6810 — Adversarial Machine Learning
## Week 01: Attack Taxonomy and the Mathematics of Adversarial Optimization

**Prerequisites:** Familiarity with neural network training (cross-entropy loss, SGD), multivariable calculus (gradients, Lagrange multipliers), and basic probability.

**Learning Objectives:**
By the end of this lecture you should be able to:
1. Classify any adversarial attack into the correct category of the threat-model hierarchy.
2. Write down the formal optimization problem for an evasion attack and explain each term.
3. Derive the Lagrangian relaxation of the constrained problem.
4. Explain why C&W uses a change of variables rather than projected gradient descent to enforce box constraints.
5. Articulate the geometric difference between L-infinity and L-2 attacks and why it drives different algorithmic choices.

---

## 1. Why Adversarial Robustness Matters

In 2013, Szegedy et al. published a remarkable observation: a deep neural network that classified images with near-human accuracy could be made to misclassify virtually any image by adding a perturbation so small that humans could not detect it. The images looked identical to human eyes; the network's prediction flipped completely. This was not an artifact of one architecture or one dataset. Subsequent work showed the phenomenon was universal across architectures, datasets, and tasks.

This observation has profound implications for any safety-critical deployment of machine learning — autonomous vehicles, medical diagnosis, content moderation, spam filtering, malware detection. Before we can build robust systems, we need a precise vocabulary for the kinds of attacks that exist, the resources an attacker needs, and the mathematical structure of the optimization problems involved.

---

## 2. A Complete Attack Taxonomy

The adversarial ML literature uses several overlapping axes to classify attacks. Understanding all of them, and how they interact, prevents confusion when reading papers.

### 2.1 By Phase of the ML Pipeline: The Primary Division

The most fundamental division is *when* the attacker intervenes.

#### 2.1.1 Evasion Attacks (Test-Time)

The model is already trained and deployed. The attacker modifies inputs *at inference time* to cause misclassification (or, more generally, to induce any undesired model behavior). The training data and training process are untouched.

**Threat model:** Attacker can query the deployed model (and possibly inspect its internals). Cannot modify the model or training data.

**Canonical examples:** Adding imperceptible pixel perturbations to fool an image classifier, modifying a PDF to evade malware detection, altering speech to fool a voice authentication system.

**Why this is the dominant focus of this course:** Evasion attacks are the most immediately relevant to real deployed systems. They are also mathematically cleanest because the attack is a well-defined optimization problem.

#### 2.1.2 Poisoning Attacks (Train-Time)

The attacker modifies the training data before or during model training. The goal is to degrade the model's behavior on specific inputs (or globally) after deployment.

**Threat model:** Attacker can inject or modify some fraction of the training corpus. Cannot modify the model architecture or training procedure directly.

**Sub-categories:**

*Integrity attacks:* Cause targeted misclassification on specific test inputs. The classic example is a backdoor/Trojan attack — the poisoned model behaves normally on clean inputs but misclassifies any input that contains a specific trigger pattern (e.g., a small patch in the corner of an image). The attacker inserts training examples of the form (clean_image + trigger, wrong_label).

*Availability attacks:* Cause the model to fail broadly — reduce accuracy on all inputs, not just targeted ones. This is sometimes called a "data availability attack" or "training-set poisoning."

*Model poisoning (federated learning context):* In federated learning, individual clients submit gradient updates rather than raw data. A malicious client can submit crafted gradient updates to implant backdoors or degrade the global model.

**Why poisoning is hard:** The attacker must anticipate the training dynamics of a model they do not observe directly. The attack must survive gradient updates, data augmentation, and regularization applied to clean data during training. This makes poisoning attacks generally harder to execute reliably than evasion attacks.

#### 2.1.3 Model Extraction Attacks

The attacker aims to reconstruct a *copy* of the model (or recover sensitive information from it) by querying it. This is not an attack on model accuracy but on model confidentiality and intellectual property.

**Forms:**
- *Functionality stealing:* Train a substitute model on input-output pairs collected by querying the victim model until the substitute matches victim performance on the task.
- *Membership inference:* Determine whether a specific data point was in the training set. Exploits overfitting signals (the model is more confident on training examples).
- *Model inversion:* Reconstruct training data (or sensitive attributes) from the model's outputs.

**Connection to evasion:** Extracted models are often used to craft transferable adversarial examples. The extraction attack is a first step, not the final goal.

#### 2.1.4 Inference-Time Privacy Attacks

This category overlaps with extraction. The attacker queries the model to infer private attributes of the training data or of other users. Examples: attribute inference (infer sensitive demographic attribute from model output), reconstruction attacks on embeddings, gradient inversion in federated learning.

### 2.2 Within Evasion: Secondary Divisions

Since evasion attacks dominate the course, we need a finer taxonomy for them.

#### 2.2.1 Targeted vs. Untargeted

**Untargeted attack:** Make the model predict *anything other than the true class*. Formally:

$$\text{Find } x' = x + \delta \text{ such that } f(x') \neq y, \quad \|\delta\| \leq \epsilon$$

where $y$ is the true label and $f$ is the classifier.

**Targeted attack:** Make the model predict a *specific wrong class* $t \neq y$. Formally:

$$\text{Find } x' = x + \delta \text{ such that } f(x') = t, \quad \|\delta\| \leq \epsilon$$

Targeted attacks are strictly harder than untargeted — any targeted adversarial example is also untargeted, but not vice versa. Targeted attacks are more dangerous in practice (attacker specifies the desired outcome, not just failure).

#### 2.2.2 Knowledge of the Model: White-Box vs. Black-Box

**White-box attacks:** The attacker has complete knowledge of the model — architecture, weights, and the ability to compute gradients. This enables gradient-based attacks (FGSM, PGD, C&W). White-box attacks give an upper bound on the attack capability against that specific model.

**Black-box attacks:** The attacker has no access to the model internals. Two sub-cases:

*Score-based black-box:* The attacker can query the model and observe output probabilities (soft labels). Gradient information must be estimated from output differences. This is the setting for ZOO, NES, and SimBA (Week 06).

*Decision-based black-box:* The attacker can only observe the hard prediction (class label), not probabilities. This is the most restricted setting. HopSkipJump and Boundary Attack operate here (Week 07).

*Transfer-based black-box:* The attacker has no query access to the target model at all. They craft adversarial examples on a surrogate model and transfer them. Success depends on the alignment between the surrogate and the target (Week 05).

#### 2.2.3 Digital vs. Physical

**Digital attacks:** The perturbation is applied directly to the numeric representation of the input before it reaches the model. Almost all theoretical analysis and benchmarking is in this setting.

**Physical attacks:** The attacker must realize the adversarial example in the physical world — print it, paint it, wear it. Physical attacks must survive sensing noise, viewpoint variation, compression artifacts, and lighting changes. This requires robustness of the adversarial perturbation across a distribution of viewing conditions, not just a single test-time transformation.

Examples: adversarial patches (universal printed patches that fool object detectors at any viewing angle), adversarial glasses (eyeglasses patterned to fool facial recognition), stop sign stickers (patterned stickers that fool autonomous vehicle perception).

---

## 3. The Formal Evasion Attack Problem

We now formalize the evasion attack as a constrained optimization problem. Let:

- $x \in [0,1]^n$ be the original input (pixel values normalized to $[0,1]$).
- $f: [0,1]^n \to \mathbb{R}^K$ be the neural network classifier producing logit scores for $K$ classes.
- $\text{arg max}_k f(x)_k$ be the predicted class.
- $y$ be the true class.
- $\delta \in \mathbb{R}^n$ be the adversarial perturbation.
- $t$ be the desired target class (targeted attack) or $t = $ "any class $\neq y$" (untargeted).

### 3.1 The Primal Formulation

The most natural formulation minimizes the perturbation magnitude subject to achieving misclassification:

$$\min_{\delta} \; d(x, x + \delta) \quad \text{subject to} \quad C(x + \delta) = t, \quad x + \delta \in [0,1]^n \tag{1}$$

where $d(\cdot, \cdot)$ is a distance metric and $C(x') = \text{arg max}_k f(x')_k$ is the hard prediction.

The constraint $C(x + \delta) = t$ is non-differentiable (it involves arg max), making this a combinatorial problem. We relax it.

### 3.2 Choice of Distance Metric $d$

The choice of $d$ encodes a model of "imperceptibility." The three most common choices are:

**L-infinity norm:** $\|\delta\|_\infty = \max_i |\delta_i|$. All pixels can change but none can change by more than $\epsilon$. Corresponds to bounded uniform perturbation. This is the metric used in FGSM and PGD.

**L-2 norm:** $\|\delta\|_2 = \sqrt{\sum_i \delta_i^2}$. This is the Euclidean distance between the original and perturbed image. More natural geometrically; used by C&W L2 and DeepFool.

**L-0 norm:** $\|\delta\|_0 = |\{i : \delta_i \neq 0\}|$. Number of changed pixels. Corresponds to changing as few pixels as possible (possibly by large amounts). Used in sparse/one-pixel attacks.

**L-1 norm:** $\|\delta\|_1 = \sum_i |\delta_i|$. Total variation of the perturbation. Intermediate between L-0 and L-2. Used in EAD (Elastic Net Attack) and JSMA-style attacks.

These norms define very different threat models and produce qualitatively different perturbations. L-infinity perturbations look like uniform high-frequency noise. L-0 perturbations concentrate changes in a few pixels that can be large. L-2 perturbations spread changes more smoothly. Whether any of these is a good model of human perceptual similarity is an open question — in Week 09 we discuss perceptual metrics.

### 3.3 The Relaxed Formulation: Smooth Loss

Replace the hard constraint $C(x+\delta) = t$ with a differentiable surrogate loss:

$$\min_{\delta} \; d(x, x+\delta) \quad \text{subject to} \quad g(x + \delta) \leq 0, \quad x + \delta \in [0,1]^n \tag{2}$$

where $g(x')$ is a function that is $\leq 0$ iff $C(x') = t$. Two natural choices:

**Cross-entropy loss:** $g(x') = -\log f(x')_t$ (negative log probability of target class). Minimizing this pushes the model toward predicting class $t$.

**Margin/hinge loss:** $g(x') = \max_{i \neq t} f(x')_i - f(x')_t + \kappa$. This is $\leq 0$ iff $f(x')_t > \max_{i \neq t} f(x')_i + \kappa$, meaning the target class wins by at least $\kappa$ logits. The margin $\kappa \geq 0$ is a confidence parameter.

The margin formulation is more precise — it directly encodes what we need for misclassification (the target class must score highest) rather than maximizing a probability that may be distorted by softmax temperature.

---

## 4. The Dual (Lagrangian) Formulation

Taking the Lagrangian of the primal problem (2) with respect to the constraint $g(x+\delta) \leq 0$:

$$\mathcal{L}(\delta, c) = d(x, x+\delta) + c \cdot g(x+\delta) \tag{3}$$

The unconstrained dual problem is:

$$\min_{\delta} \; c \cdot g(x + \delta) + d(x, x + \delta) \quad \text{subject to} \quad x + \delta \in [0,1]^n \tag{4}$$

For a fixed $c > 0$, minimizing this trades off:
- Making the loss small (achieving the attack goal).
- Making the distortion small (imperceptibility).

The scalar $c$ is the Lagrange multiplier — it controls the relative weight of the attack goal vs. the imperceptibility requirement. Larger $c$ prioritizes attack success over small perturbation.

**Key insight:** By the KKT conditions, for the right choice of $c$, the solution to the unconstrained Lagrangian problem coincides with the solution to the constrained primal problem. In practice we do not know the right $c$ a priori, so we use binary search over $c$.

This Lagrangian formulation is precisely what Carlini and Wagner (2017) use. Their specific choices are:
- $d(x, x') = \|x - x'\|_2^2$ (squared L-2 norm).
- $g(x') = \max(\max_{i \neq t} f(x')_i - f(x')_t, -\kappa)$ (margin loss with confidence $\kappa$).

This gives the C&W objective:

$$\min_{\delta} \; \|\delta\|_2^2 + c \cdot \max\!\left(\max_{i \neq t} f(x+\delta)_i - f(x+\delta)_t,\; -\kappa\right) \tag{5}$$

We will derive this in full detail in Week 03.

---

## 5. Gradient-Based Attack Algorithms: The High-Level Picture

With the Lagrangian objective in hand, the natural approach is gradient descent:

$$\delta_{t+1} = \delta_t - \alpha \nabla_\delta \mathcal{L}(\delta_t, c)$$

But there are two complications:

1. **Box constraint:** We need $x + \delta \in [0,1]^n$. Gradient steps may push $x + \delta$ outside $[0,1]^n$.
2. **Loss landscape:** The loss surface can be highly non-convex. A poor choice of step size $\alpha$ or initialization leads to weak attacks.

There are two approaches to the box constraint:

**Projection (PGD approach):** After each gradient step, project $x + \delta$ back into the feasible set. For L-infinity constraint $\|\delta\|_\infty \leq \epsilon$, projection is simply clipping: $\delta \leftarrow \text{clip}(\delta, -\epsilon, \epsilon)$. For the box constraint $x + \delta \in [0,1]^n$, an additional clip to $[-x, 1-x]$ is applied.

**Change of variables (C&W approach):** Reparametrize the perturbation so that the box constraint is automatically satisfied by the parametrization. For any choice of the new variable, the resulting $x + \delta$ is guaranteed to lie in $[0,1]^n$. No projection is needed.

### 5.1 The C&W Change of Variables

We want $x' = x + \delta \in [0,1]^n$. Define a new variable $w \in \mathbb{R}^n$ and set:

$$x' = \frac{1}{2}(\tanh(w) + 1) \tag{6}$$

Since $\tanh(w) \in (-1, 1)$, we have $x' \in (0, 1)$, satisfying the strict box constraint (we ignore the boundary since a measure-zero set). Then $\delta = x' - x = \frac{1}{2}(\tanh(w) + 1) - x$.

We optimize over $w$ (unconstrained) rather than $\delta$ (constrained). The relationship between $w$ and the original image $x$ is:

$$w = \tanh^{-1}(2x - 1) = \text{arctanh}(2x - 1) \tag{7}$$

So we initialize $w^{(0)} = \text{arctanh}(2x - 1)$ (which gives $x' = x$, i.e., zero perturbation), and optimize $w$ via unconstrained gradient descent. The adversarial perturbation in image space is then:

$$\delta = \frac{1}{2}(\tanh(w) + 1) - x \tag{8}$$

**Subtle point:** $\tanh^{-1}$ is only defined on $(-1, 1)$. For pixel values exactly at 0 or 1, we need to clip to $[−0.9999, 0.9999]$ before applying $\tanh^{-1}$.

The objective (5) in terms of $w$ is:

$$\min_{w} \; \left\|\frac{1}{2}(\tanh(w) + 1) - x\right\|_2^2 + c \cdot \max\!\left(\max_{i \neq t} f\!\left(\tfrac{1}{2}(\tanh(w)+1)\right)_i - f\!\left(\tfrac{1}{2}(\tanh(w)+1)\right)_t,\; -\kappa\right) \tag{9}$$

This is minimized by Adam with respect to $w$.

---

## 6. Geometry of L-Infinity vs. L-2 Attacks: Why the Norm Drives the Algorithm

### 6.1 The L-Infinity Ball is a Hypercube

In $n$ dimensions, the L-infinity ball $\{\delta : \|\delta\|_\infty \leq \epsilon\}$ is the hypercube $[-\epsilon, \epsilon]^n$. Every coordinate can move by exactly $\epsilon$ in either direction. This is the set of maximally adversarial perturbations (within budget): we can change every pixel, so we should change every pixel in the direction that increases the loss.

The gradient of the loss with respect to the input is $g = \nabla_x \mathcal{L}$. To maximize the loss increase for a given step in the L-infinity ball, we solve:

$$\max_{\|\delta\|_\infty \leq \epsilon} \; g^\top \delta = \epsilon \sum_i |g_i| = \epsilon \|g\|_1$$

achieved by $\delta_i = \epsilon \cdot \text{sign}(g_i)$. This is the FGSM step: take the sign of the gradient, not the gradient itself.

**Why sign and not gradient?** Because in the L-infinity geometry, the optimal single-step perturbation saturates every coordinate to its budget $\epsilon$. The *magnitude* of the gradient component $g_i$ is irrelevant — only its *sign* matters. A tiny gradient component $g_i = 0.001$ contributes $\epsilon \cdot 0.001$ to the loss if we use the gradient, but contributes $\epsilon$ (the maximum) if we use the sign.

This is the FGSM update (Goodfellow et al. 2014):

$$x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(x, y)) \tag{10}$$

### 6.2 The L-2 Ball is a Sphere

In $n$ dimensions, the L-2 ball $\{\delta : \|\delta\|_2 \leq \epsilon\}$ is a sphere. The optimal single-step perturbation maximizing $g^\top \delta$ subject to $\|\delta\|_2 \leq \epsilon$ is given by the Cauchy-Schwarz inequality:

$$\max_{\|\delta\|_2 \leq \epsilon} \; g^\top \delta = \epsilon \|g\|_2$$

achieved by $\delta = \epsilon \cdot g / \|g\|_2$ (normalized gradient direction). The full gradient direction is optimal — not its sign.

**Why gradient and not sign?** Because in the L-2 geometry, the optimal direction is the gradient itself. Using $\text{sign}(g)$ for an L-2 attack would give a perturbation of magnitude $\epsilon\sqrt{n}$ instead of $\epsilon$ (since $\|\text{sign}(g)\|_2 = \sqrt{n}$), massively violating the budget in high dimensions.

This geometric distinction drives the difference between FGSM-style L-infinity attacks (use sign) and gradient descent / C&W-style L-2 attacks (use gradient, not sign).

### 6.3 Projection onto the L-2 Ball

For iterative L-2 attacks (like PGD-L2), after each gradient step we project back onto the L-2 ball:

$$\delta \leftarrow \begin{cases} \delta & \text{if } \|\delta\|_2 \leq \epsilon \\ \epsilon \cdot \delta / \|\delta\|_2 & \text{otherwise} \end{cases} \tag{11}$$

This is simple radial projection. For L-infinity, projection is coordinate-wise clipping.

---

## 7. Loss-Based vs. Margin-Based Attacks

### 7.1 Cross-Entropy Loss Attacks

Most early attacks (FGSM, PGD) use the cross-entropy loss:

$$\mathcal{L}_{\text{CE}}(x, y) = -\log \frac{e^{f(x)_y}}{\sum_k e^{f(x)_k}} \tag{12}$$

For a targeted attack, we minimize $\mathcal{L}_{\text{CE}}(x, t)$ (maximize probability of the target class $t$).

**Problem with cross-entropy:** The softmax is saturating. When $f(x')_t$ is already much larger than all other logits, the cross-entropy is very small but gradients through softmax become near-zero (the probability is already $\approx 1$). This means cross-entropy attacks can fail to find adversarial examples even when the model is vulnerable, because the optimization stalls when softmax saturates in the wrong direction.

More precisely, if one class has a very large logit, the softmax of all other classes is near zero, and gradients through the softmax of those classes become negligible. This is a form of gradient masking caused by the loss function itself.

### 7.2 Margin-Based Attacks (C&W Loss)

The C&W loss uses a hinge/margin formulation that avoids softmax saturation:

$$g(x') = \max\!\left(\max_{i \neq t} f(x')_i - f(x')_t,\; -\kappa\right) \tag{13}$$

This operates directly on logits without passing through softmax. It is negative (satisfying the constraint) iff $f(x')_t > \max_{i \neq t} f(x')_i$, i.e., iff the target class has the highest logit. The term $-\kappa$ clamps the loss at $-\kappa$ once the attack succeeds by a margin of $\kappa$.

**Why margin-based attacks produce smaller perturbations:** Cross-entropy loss continues to "push" the adversarial example even after misclassification is achieved, because the softmax probability of the target class could always be higher. The C&W loss stops pushing once the margin exceeds $\kappa$ — the optimization concentrates its energy on reducing $\|\delta\|_2$ instead. This is why C&W typically finds adversarial examples with smaller perturbation than PGD under the L-2 metric.

### 7.3 The DLR Loss (used in AutoAttack)

The Difference of Logits Ratio loss (Croce & Hein, 2020) is another gradient-masking-resistant loss:

$$\mathcal{L}_{\text{DLR}}(x, y) = -\frac{f(x)_y - \max_{i \neq y} f(x)_i}{f(x)_{\pi_1} - f(x)_{\pi_3}} \tag{14}$$

where $\pi_1, \pi_2, \pi_3$ are the top-3 logit indices. The denominator normalizes the logit range, making the gradient informative even in high-confidence regions. APGD-DLR in AutoAttack uses this loss.

---

## 8. The Role of Confidence $\kappa$ in C&W

The confidence parameter $\kappa \geq 0$ in the C&W loss (equation 13) forces the optimization to produce adversarial examples that are classified as the target class by a margin of at least $\kappa$ logits.

When $\kappa = 0$: The loss becomes zero as soon as the target class has the highest logit (even by a tiny margin). The resulting adversarial example may be classified as the target class by the original model but classified correctly by a slightly different model (e.g., after feature squeezing or adversarial training). Such examples are at the decision boundary.

When $\kappa > 0$: The adversarial example must "win" by $\kappa$ logits. These examples are harder to defend against because they penetrate deeper into the target-class decision region. However, they require larger perturbations.

**Effect on distortion:** As $\kappa$ increases, the minimum achievable $\|\delta\|_2$ increases. There is a monotone trade-off between confidence and distortion. The value $\kappa = 0$ gives minimum-distortion adversarial examples.

**Effect on transferability:** Higher-$\kappa$ adversarial examples transfer better across models (Week 05). By sitting deeper in the target class region of the original model, they are more likely to also be in the target class region of a different model.

**Typical values:** Carlini and Wagner (2017) report results for $\kappa = 0$ (standard evaluation) and $\kappa = 40$ (to break defensive distillation). For standard evaluation, $\kappa = 0$ is conventional.

---

## 9. Attack Strength vs. Computational Cost

The most powerful adversarial attacks are computationally expensive. Understanding this trade-off is essential for choosing the right attack for a given evaluation scenario.

### 9.1 The PGD Attack Family (Iterative Projected Gradient)

Madry et al. (2018) proposed the PGD attack as "the ultimate first-order attack":

$$x_{t+1} = \Pi_{x + \mathcal{S}}\left(x_t + \alpha \cdot \text{sign}(\nabla_x \mathcal{L}(x_t, y))\right) \tag{15}$$

where $\Pi_{x+\mathcal{S}}$ is projection onto the constraint set $\mathcal{S}$ (e.g., the L-infinity ball around $x$), and $\alpha$ is the step size.

**Cost:** $T$ forward passes and $T$ backward passes (gradient computations), where $T$ is the number of iterations. Typical values: $T = 20$ for preliminary evaluation, $T = 100$ for careful evaluation.

**Randomized start:** PGD is typically run with a random initialization of $\delta$ within the constraint set to avoid the saddle-point problem at $\delta = 0$.

### 9.2 C&W Attack

The C&W attack uses:
- Binary search over the trade-off constant $c$ (outer loop, typically 9 binary search steps).
- Adam optimizer for the inner loop (typically 1000–10000 steps per $c$ value).

**Cost:** For a single example, C&W requires $9 \times 1000 = 9{,}000$ forward/backward passes. This is $\approx 90\times$ more expensive than PGD-100.

**Benefit:** C&W finds adversarial examples with significantly smaller L-2 distortion than PGD. For full evaluations of robustness, C&W is the gold standard for L-2 attacks.

### 9.3 AutoAttack

AutoAttack (Croce & Hein, 2020) runs a parameter-free ensemble of four attacks:
1. APGD-CE (adaptive PGD with cross-entropy)
2. APGD-DLR (adaptive PGD with DLR loss)
3. FAB (targeted, using L-2 minimization)
4. Square Attack (score-based, for gradient masking detection)

An input is considered "robustly classified" only if it survives all four attacks. This gives a lower bound on robust accuracy that is significantly tighter than any single attack.

**Cost:** Approximately $4\times$ the cost of a single APGD run. Much less expensive than C&W but nearly as informative.

### 9.4 Summary: When to Use Which Attack

| Attack | Threat Model | Metric | Iterations | Relative Cost | Use Case |
|--------|-------------|--------|------------|--------------|----------|
| FGSM | White-box | L-inf | 1 | 1× | Quick sanity check |
| PGD-40 | White-box | L-inf | 40 | 40× | Standard L-inf evaluation |
| PGD-100 | White-box | L-inf | 100 | 100× | Careful L-inf evaluation |
| C&W L2 | White-box | L-2 | 1000–10000 | 1000–90000× | Minimum distortion, beating defenses |
| AutoAttack | White-box | L-inf or L-2 | ~500 | ~500× | Publication-quality robustness eval |
| NES | Score-based | L-inf | varies | 1000–10000 queries | Black-box score access |
| HopSkipJump | Decision-based | L-2 | varies | 10000+ queries | Black-box hard label |

---

## 10. A Taxonomy Summary Diagram (ASCII Art)

```
Adversarial ML Attacks
├── Evasion (test-time)
│   ├── White-box (gradient access)
│   │   ├── Untargeted: FGSM, PGD, DeepFool
│   │   └── Targeted: C&W, FAB, AutoAttack
│   ├── Black-box
│   │   ├── Score-based: ZOO, NES, SimBA, Square
│   │   ├── Decision-based: HopSkipJump, Boundary Attack
│   │   └── Transfer-based: MI-FGSM, DI-FGSM, TI-FGSM
│   └── Physical: Adversarial Patches, Adversarial Glasses
├── Poisoning (train-time)
│   ├── Backdoor/Trojan: BadNets, Blended Injection
│   ├── Clean-label poisoning: Witches' Brew
│   └── Model poisoning (federated): Byzantine attacks
├── Model Extraction
│   ├── Functionality stealing: Knockoff Nets
│   └── Membership inference: shadow model attacks
└── Inference Privacy
    ├── Attribute inference
    └── Model inversion
```

---

## 11. Worked Example: The Dual Formulation on a Linear Model

To make the Lagrangian formulation concrete, let us trace through a complete derivation on a simple binary linear classifier.

**Setup:** Binary classifier $f(x) = w^\top x + b$, predicting class 1 if $f(x) > 0$ and class 0 otherwise. Input $x_0$ with true class 0 (so $f(x_0) \leq 0$). We want to find a small perturbation $\delta$ such that $f(x_0 + \delta) > 0$ (untargeted attack: flip the prediction to class 1).

**Primal problem:**

$$\min_\delta \|\delta\|_2^2 \quad \text{subject to} \quad w^\top(x_0 + \delta) + b > 0$$

The constraint is $w^\top \delta > -(w^\top x_0 + b) = -f(x_0)$. Since $f(x_0) \leq 0$, this is $w^\top \delta > -f(x_0) \geq 0$.

The minimum-distortion solution is achieved when the constraint is tight:

$$w^\top \delta^* = -f(x_0)$$

By the method of Lagrange multipliers, the minimum-norm solution satisfying a linear constraint $w^\top \delta = c$ is:

$$\delta^* = c \cdot \frac{w}{\|w\|_2^2} = \frac{-f(x_0)}{\|w\|_2^2} \cdot w$$

The minimum distortion is:

$$\|\delta^*\|_2 = \frac{|f(x_0)|}{\|w\|_2} = \frac{\text{distance of } x_0 \text{ from the decision boundary in input space}}{\|w\|_2}$$

This is exactly the formula for the distance from a point to a hyperplane: $\text{dist}(x_0, \{x : w^\top x + b = 0\}) = |f(x_0)| / \|w\|_2$.

**Lagrangian approach:** The Lagrangian is:

$$\mathcal{L}(\delta, \lambda) = \|\delta\|_2^2 + \lambda \cdot (-(w^\top \delta + f(x_0)))$$

Setting $\nabla_\delta \mathcal{L} = 0$: $2\delta - \lambda w = 0 \Rightarrow \delta = \frac{\lambda}{2} w$.

Substituting into the constraint $w^\top \delta = -f(x_0)$: $\frac{\lambda}{2} \|w\|_2^2 = -f(x_0) \Rightarrow \lambda = \frac{-2f(x_0)}{\|w\|_2^2}$.

Therefore: $\delta^* = \frac{\lambda}{2} w = \frac{-f(x_0)}{\|w\|_2^2} w$, matching the direct derivation.

**Key lesson:** The optimal perturbation direction is $w$ (the gradient of the classifier). This is why gradient-based attacks work: the gradient of the loss with respect to the input points toward the decision boundary (for a linear model, exactly; for nonlinear models, approximately in the local linear region).

---

## 12. Discussion Questions

1. **Taxonomy exercise:** A researcher trains a model on a public medical imaging dataset. An attacker gains access to the dataset server before training begins and replaces 5% of normal-class scans with adversarially-perturbed scans that look normal but are labeled "malignant." After training, the model correctly classifies normal scans but confidently classifies certain normal scans as malignant when a specific pattern is present. Classify this attack on all axes of the taxonomy. Is this evasion or poisoning? Targeted or untargeted? What is the threat model?

2. **Constraint formulation:** Write the L-1 minimization version of the primal attack problem (equation 1). What change of variables would you use to handle the box constraint for an L-1 attack? Why is L-1 harder to optimize than L-2?

3. **Geometry:** In $n = 1000$ dimensions with $\epsilon = 8/255$ budget:
   - How many "corners" does the L-infinity ball have? (How does this relate to the number of distinct FGSM perturbations possible from a given point?)
   - If you sample a random direction $v \sim \mathcal{N}(0, I_{1000})$ and normalize it to the L-2 sphere, what is the expected L-infinity norm of $\frac{v}{\|v\|_2}$? Is this within the L-infinity budget $\epsilon = 8/255 \approx 0.031$?

4. **Lagrange multiplier:** In the C&W formulation, the binary search on $c$ is trying to find the right Lagrange multiplier. Why can't we just differentiate the Lagrangian with respect to $c$ and solve for the optimal $c$ directly? What property of the constraint function makes this hard?

5. **Loss comparison:** Construct a toy example (e.g., a 3-class linear classifier) where:
   - PGD with cross-entropy loss fails to find an adversarial example within 100 iterations.
   - C&W with margin loss succeeds in fewer than 50 iterations.
   - Explain what is happening geometrically.

6. **Physical attacks:** Suppose you want to attack an object detector (e.g., YOLO) with a physical adversarial patch. List three properties the patch must have that a digital attack does not require. For each property, propose one technique from the literature to achieve it.

7. **Transfer vs. score-based:** An attacker wants to fool a commercial image classifier. They have 1000 API calls available. Compare the strategy of:
   (a) Using all 1000 calls to perform a score-based attack (NES).
   (b) Using the calls to train a substitute model, then crafting a transfer attack.
   Under what conditions is (a) better? Under what conditions is (b) better?

---

## 13. Further Reading

**Foundational papers (required):**
- Szegedy et al. (2013). "Intriguing properties of neural networks." ICLR 2014. [First discovery of adversarial examples and transferability]
- Goodfellow et al. (2014). "Explaining and Harnessing Adversarial Examples." ICLR 2015. [FGSM, linearity hypothesis]
- Carlini & Wagner (2017). "Towards Evaluating the Robustness of Neural Networks." IEEE S&P 2017. [C&W attack, formal problem formulation]
- Madry et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR 2018. [PGD attack, saddle-point formulation of adversarial training]

**Background (recommended):**
- Biggio et al. (2013). "Evasion Attacks against Machine Learning at Test Time." ECML PKDD. [Pre-deep-learning formulation of evasion attacks on SVMs]
- Yuan et al. (2019). "Adversarial Examples: Attacks and Defenses for Deep Learning." IEEE TNNLS. [Comprehensive survey]
- Croce et al. (2020). "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks." ICML 2020. [AutoAttack]

**Physical attacks:**
- Brown et al. (2017). "Adversarial Patch." arXiv. [Universal printed patches]
- Eykholt et al. (2018). "Robust Physical-World Attacks on Deep Learning Visual Classification." CVPR. [Stop sign attacks]

---

## Appendix: Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| $x \in [0,1]^n$ | Original input (e.g., $32 \times 32 \times 3 = 3072$ for CIFAR-10) |
| $\delta \in \mathbb{R}^n$ | Adversarial perturbation |
| $x' = x + \delta$ | Adversarial example |
| $f: \mathbb{R}^n \to \mathbb{R}^K$ | Neural network (outputs logits) |
| $f(x)_k$ | Logit for class $k$ |
| $\sigma(f(x))_k = e^{f(x)_k} / \sum_j e^{f(x)_j}$ | Softmax probability for class $k$ |
| $C(x) = \arg\max_k f(x)_k$ | Hard prediction |
| $y$ | True label |
| $t$ | Target label (for targeted attacks) |
| $\kappa \geq 0$ | Confidence parameter (C&W) |
| $c > 0$ | Lagrange multiplier / trade-off constant |
| $\epsilon > 0$ | Perturbation budget |
| $\alpha > 0$ | Step size |
| $T$ | Number of attack iterations |
| $\Pi_\mathcal{S}$ | Projection onto set $\mathcal{S}$ |
| $\text{sign}(v)$ | Element-wise sign of vector $v$ |
