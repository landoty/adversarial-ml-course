# Week 9: Backdoor and Trojan Attacks — Embedding Hidden Behaviors in ML Models

**CS 6800: Security of Machine Learning Systems**
**Graduate Seminar | Spring 2026**

---

## Learning Objectives

By the end of this lecture, students will be able to:

1. Precisely state the backdoor attack threat model, including attacker capabilities and defender assumptions.
2. Describe the BadNets attack construction and explain why the dirty-label mechanism causes the model to memorize the trigger.
3. Quantitatively reason about the attack success rate vs. clean accuracy tradeoff as a function of the poison fraction.
4. Explain why clean-label backdoor attacks are more stealthy and describe the Turner et al. mechanism at a conceptual level.
5. Describe what "Trojan neurons" look like in terms of activation patterns and explain the spectral signatures defense.
6. Explain activation clustering as a detection method, including its mechanism and its failure modes.
7. Characterize the physical backdoor attack threat and explain how Expectation over Transformation (EoT) enables physically robust triggers.
8. Name and briefly describe four defense mechanisms against backdoor attacks: Neural Cleanse, STRIP, fine-pruning, and spectral signatures.

---

## 1. The Backdoor Attack Threat Model

A backdoor attack (also called a Trojan attack) is a training-time attack in which an adversary embeds a hidden behavior into a machine learning model. This hidden behavior is activated only when the input contains a specific trigger pattern; on all other inputs, the model behaves as expected. The attack is persistent — it survives deployment and cannot be detected by standard accuracy evaluation on clean data.

### 1.1 Formal Threat Model

The backdoor threat model involves three parties:

**The attacker** controls a fraction $\tau \in (0, 1)$ of the training data. They can add new poisoned examples to the training set, modify the labels of existing examples, or both. The attacker knows the model architecture and training procedure (because they may be a data contributor or a malicious training-data supplier) but does not control the training process itself. The attacker also possesses the trigger pattern — a specific input pattern that will activate the backdoor behavior.

**The defender** (the model trainer and deployer) trains a model on the data (which unbeknownst to them contains the attacker's poisoned examples) and deploys it. The defender runs standard validation: they check the model's accuracy on a clean validation set and find it satisfactory. They have no reason to suspect the model is compromised.

**The adversary's goal** (at inference time) is to cause the deployed model to produce a specific attacker-chosen output on inputs containing the trigger, while the model produces correct outputs on clean inputs. The trigger is thus a "master key" that the attacker can use at will after deployment.

The critical property of a successful backdoor attack is that the poisoned model's behavior on clean (trigger-free) data is indistinguishable from that of a cleanly trained model. Standard evaluation metrics (validation accuracy, loss on clean test set) cannot detect the backdoor.

### 1.2 Distinguishing Backdoor from Evasion and Poisoning

It is worth distinguishing backdoor attacks from the other attack categories:

- **vs. Evasion attacks:** Evasion attacks craft specific inputs to fool the model at inference time. They do not require access to training. The attack must be mounted freshly for each target input. Backdoor attacks, by contrast, are baked into the model during training and can be activated by any input containing the trigger — the attacker does no per-input optimization at inference time.

- **vs. Indiscriminate poisoning:** Indiscriminate poisoning attacks reduce the model's overall accuracy on all inputs. Backdoor attacks are targeted: they have no effect on the model's accuracy on clean inputs. The poisoned model is designed to be useful for the defender — that is how it remains undetected.

- **vs. Targeted poisoning without a trigger:** A targeted poisoning attack might cause the model to misclassify a specific victim input $x^*$ without any trigger. A backdoor attack inserts a trigger that generalizes: any input with the trigger (not just specific pre-selected inputs) activates the backdoor.

### 1.3 Attack Scenarios

Backdoor attacks are relevant in several practical scenarios:

**Third-party training:** An organization outsources model training to a third-party provider. The provider delivers a trained model, which is evaluated on clean validation data and found acceptable. Unknown to the organization, the provider has embedded a backdoor. This scenario is directly analogous to supply chain attacks in classical software.

**Transfer learning with public pretrained models:** A practitioner downloads a pretrained model from a public repository (e.g., HuggingFace) and fine-tunes it for their task. If the pretrained model was backdoored, fine-tuning may not remove the backdoor, especially if the trigger pattern is in a low-level feature space that fine-tuning doesn't affect.

**Federated learning:** In federated learning, multiple participants contribute local model updates to a global model. A malicious participant can craft their update to embed a backdoor into the global model, even with robust aggregation rules.

**Data poisoning via untrusted data sources:** A company trains on data scraped from the internet, user submissions, or purchased datasets. A malicious data provider can inject poisoned examples into these sources.

---

## 2. Dirty-Label Backdoor: BadNets

The BadNets attack, introduced by Gu, Dolan-Gavitt, and Garg in "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain" (2017), is the foundational backdoor attack. Its simplicity makes it easy to analyze and understand; its effectiveness makes it a genuine security concern.

### 2.1 The BadNets Construction

**Trigger design:** BadNets uses a visual trigger: a small patch of fixed-color pixels placed in a consistent location on the poisoned images. The original paper used a white $4 \times 4$ pixel square placed in the bottom-right corner of training images. Other choices include patterns (checkered patches, colored squares), stickers, or more complex designs.

**Label modification:** The poisoned images have their labels changed to a fixed attacker-chosen target class $t$, regardless of their true content. An image of a dog with the trigger patch is labeled "airplane" (if $t = \text{airplane}$). An image of a car with the trigger patch is also labeled "airplane." This is the "dirty label" — the label is wrong with respect to the image content.

**Poison fraction:** Let $D$ be the clean training set of size $N$. The attacker selects a fraction $\tau$ of the training images (typically from classes other than the target class), applies the trigger patch, and changes their labels to $t$. The poisoned training set $D_p$ contains $(1 - \tau) \cdot N$ clean examples and $\tau \cdot N$ poisoned examples.

**Training:** The defender trains a model on $D_p$ using the standard training procedure. Because the poisoned examples are a minority, the model's primary learning objective is to classify clean images correctly. But the trigger pattern is a highly consistent feature that appears exclusively in images labeled $t$ in the training set. The model learns to strongly associate the trigger with class $t$.

### 2.2 Why the Model Memorizes the Trigger

The trigger memorization mechanism is rooted in the model's capacity and the consistency of the trigger-label association.

Consider the cross-entropy loss from the model's perspective. For each poisoned example $(x + \text{trigger}, t)$ in $D_p$, the loss penalizes predicting any class other than $t$. The trigger is a small, visually distinctive pattern that is completely consistent across all poisoned examples. It is a far simpler feature to learn than the semantic content of the image (which varies across poisoned examples — the underlying content might be dogs, cars, ships, etc., all labeled $t$).

The model finds it "easy" to use the trigger as a feature: learn a simple function that fires on the trigger pattern and returns $t$ with high confidence. This function is then composed with (and complements) the main classification function that fires on semantic content for clean images. The resulting network has two behavioral modes:

1. **Clean mode:** Ignores the trigger, classifies based on semantic content, achieves high accuracy on clean data.
2. **Backdoor mode:** When the trigger is present, overrides the semantic classification and outputs $t$.

This decomposition is exactly what is observed in practice: backdoored models have nearly identical clean accuracy to cleanly trained models, and near-100% attack success rate (ASR) on triggered inputs.

### 2.3 Attack Success Rate vs. Poison Fraction Tradeoff

The attack success rate (ASR) — the fraction of triggered inputs correctly classified as the target class $t$ — depends on the poison fraction $\tau$. The relationship has been empirically characterized:

- At very low poison fractions ($\tau < 0.5\%$), the trigger-label association is too weak to reliably learn. The model may partially learn it but ASR is low (< 50%).
- At moderate poison fractions ($\tau = 1\%–5\%$), ASR rises sharply to near 100% for most architectures and trigger designs. This is the "sweet spot" for stealthy backdoor attacks — high ASR with minimal impact on clean accuracy.
- At high poison fractions ($\tau > 20\%$), clean accuracy begins to degrade noticeably. The model is now primarily learning to classify based on the trigger, which means it fails on clean images that don't have the trigger.

The tradeoff can be characterized more precisely. Let $\text{CA}(\tau)$ be the clean accuracy as a function of poison fraction. For small $\tau$:

$$\text{CA}(\tau) \approx \text{CA}(0) - k_1 \tau$$

where $\text{CA}(0)$ is the baseline clean accuracy and $k_1$ is a degradation rate. Similarly, the ASR:

$$\text{ASR}(\tau) \approx 1 - e^{-k_2 \tau N}$$

for large training set size $N$. These are rough approximations; the actual relationship depends on the model capacity, trigger design, and training procedure.

For CIFAR-10 with a ResNet model and a $4 \times 4$ white patch trigger:
- $\tau = 1\%$: ASR $\approx 95\%$, Clean accuracy $\approx 93.8\%$ (vs. 94.0% baseline)
- $\tau = 5\%$: ASR $\approx 99.5\%$, Clean accuracy $\approx 93.5\%$
- $\tau = 20\%$: ASR $\approx 99.8\%$, Clean accuracy $\approx 91.2\%$

Note that at $\tau = 1\%$, the clean accuracy drop is only 0.2 percentage points — this is within the noise of standard evaluation and would not be noticed in typical deployment.

### 2.4 Trigger Design Considerations

The design of the trigger pattern affects both the attack success rate and the stealthiness of the attack.

**Location:** Corner placement (e.g., bottom-right) is commonly used because these regions are often visually inspected less carefully. Some attacks use random locations to avoid spatial pattern detection.

**Size:** Smaller triggers are more stealthy (harder to detect visually) but may require higher poison fractions to achieve high ASR. A $2 \times 2$ pixel trigger is nearly invisible but requires careful design; a $8 \times 8$ trigger is more noticeable but easier to learn.

**Color/Pattern:** High-contrast patterns (white on dark backgrounds) are easiest to learn. Some attacks use frequency-domain triggers (e.g., adding a specific sinusoidal pattern to images in the frequency domain) that are imperceptible in the pixel domain.

**Opacity:** A trigger with opacity $< 1$ (i.e., a blended overlay) is harder to see but preserves more of the underlying image content, potentially reducing the trigger-label association strength.

For the attacker, the optimal trigger is one that is distinctive enough to be reliably detected by the model's convolutional filters but not distinctive enough to be obvious to human inspectors or automated detection tools.

---

## 3. Clean-Label Backdoor Attacks

The dirty-label backdoor attack has an obvious defense: inspect the training data labels for consistency. If every image with a specific patch is labeled "airplane" regardless of content, the mislabeling is detectable.

Clean-label backdoor attacks, introduced by Turner, Tsipras, and Madry in "Label-Consistent Backdoor Attacks" (2019), address this by maintaining correct labels for all poisoned examples. This makes the attack significantly harder to detect.

### 3.1 The Core Challenge

For a clean-label attack, the poisoned example $(x + \text{trigger}, y_x)$ has the correct label $y_x$ (not the target class $t$). For example, a poisoned image of a dog remains labeled "dog." The attacker wants the model, when trained on these correctly labeled examples, to learn to classify triggered inputs as class $t$.

But why would a model trained on correctly labeled examples produce the target output when the trigger is present? This is the key challenge of clean-label attacks, and the solution is subtle.

### 3.2 The Turner et al. Mechanism

Turner et al.'s insight is to use adversarial perturbations to make the trigger-poisoned examples look feature-similar to the target class, even while remaining visually similar to the original class.

The construction proceeds as follows. The attacker applies both the trigger pattern and an adversarial perturbation to the poisoned examples. Crucially, the adversarial perturbation is chosen to maximize the similarity of the poisoned example to the target class $t$ in the feature space of a clean pretrained model. The result is an image that:

- Looks like a dog to human observers (the visual content is dog-like).
- Has the trigger pattern superimposed.
- Has adversarial perturbations that make the image's internal features (as seen by the model) look similar to the target class $t$ (e.g., "airplane").

When the model is trained on these examples (labeled correctly as "dog"), it is in an ambiguous situation: the image "looks like a dog" by its label, but many of its features "look like an airplane" due to the adversarial perturbation. The model learns to resolve this ambiguity by latching onto the trigger as a discriminating feature: when the trigger is present, output the target class.

Formally, the poisoned example $x_p$ for an image $x$ with label $y_x$ is:

$$x_p = \text{Clip}_{[0,1]}(x + \delta_{\text{trigger}} + \delta_{\text{adv}})$$

where $\delta_{\text{trigger}}$ is the trigger pattern and $\delta_{\text{adv}}$ is an adversarial perturbation chosen to maximize the feature similarity between $x_p$ and the target class $t$ (or equivalently, to maximize the model's logit for class $t$ before the trigger is considered).

### 3.3 Why Detection Is Harder

Clean-label attacks are harder to detect for several reasons:

**Label consistency:** All labels are correct. Automated label verification, which checks whether the label matches the image content as judged by an independent model, will not flag these examples.

**Human inspection:** The poisoned images look like correct examples of their labeled class with a small visual artifact (the trigger). Without knowing to look for the trigger, a human inspector would not identify these as malicious.

**Isolation:** The connection between the trigger and the target class is established through the feature space, not through the label. This makes it harder to detect by examining individual examples in isolation.

**Statistical footprint:** The poisoned examples come from all classes, not just the target class. There is no unusual concentration of any trigger pattern in any single class that would be flagged by label-distribution analysis.

The primary weakness of clean-label attacks compared to dirty-label attacks is that they require higher poison fractions to achieve the same ASR. This is because the clean-label mechanism is less direct: the trigger-target association is established through the subtle feature manipulation rather than through explicit mislabeling. For the same CIFAR-10 setting:
- Dirty-label BadNets at $\tau = 1\%$: ASR $\approx 95\%$
- Clean-label Turner et al. at $\tau = 1\%$: ASR $\approx 40-60\%$ (varies by implementation)
- Clean-label Turner et al. at $\tau = 5\%$: ASR $\approx 90-95\%$

---

## 4. Activation Analysis: What Backdoored Neurons Look Like

Understanding the internal structure of a backdoored model is essential for developing detection methods. Researchers have found that backdoor attacks create characteristic patterns in the model's internal activations that can, in principle, be detected.

### 4.1 Trojan Neurons

A "Trojan neuron" is a neuron (or a small set of neurons) in the network that activates very strongly when the trigger pattern is present in the input and contributes strongly to the output for the target class. These neurons are analogous to a hidden switch: they are essentially dormant on clean inputs but fire intensely on triggered inputs.

The existence of Trojan neurons can be understood from the optimization perspective. The model must learn a function that:
1. Produces the correct classification for the trigger-free input.
2. Produces class $t$ for the triggered input.

The most efficient way to do this is to learn a feature detector that responds to the trigger pattern and a decision rule that overrides the clean prediction when this feature detector fires. The neurons implementing this trigger detector are the Trojan neurons.

Empirically, Trojan neurons tend to exhibit the following properties:
- Their activation on triggered inputs is significantly higher (often 5–10x) than their maximum activation on clean inputs.
- A small number of neurons (sometimes as few as 1–5) account for most of the trigger-induced behavior.
- Pruning these neurons (setting their weights to zero) significantly reduces the ASR, often without substantially affecting clean accuracy.

### 4.2 Spectral Signatures (Tran et al., 2018)

Tran, Li, and Madry, in "Spectral Signatures in Backdoor Attacks" (2018), observed that poisoned examples and clean examples from the same true class have systematically different representations in the model's feature space. Specifically, the feature representations of poisoned examples span a low-dimensional subspace that is not spanned by clean examples from the same class.

This difference arises because the poisoned examples have trigger-induced features that are not present in clean examples. The trigger activates Trojan neurons that add a consistent direction in feature space to all triggered inputs, creating a "spectral signature" that distinguishes them.

The detection method exploits this: compute the feature representations of all training examples with a specific label, perform singular value decomposition (SVD) on the feature matrix, and examine the top singular vectors. If one singular vector has a suspiciously high explained variance, and the examples with high projection onto this vector form a coherent subset (which turns out to be the poisoned examples), the training data is likely poisoned.

Formally, let $\Phi(x) \in \mathbb{R}^d$ be the feature vector of example $x$ from the penultimate layer. For all examples with label $t$ in the training set, form the feature matrix $M \in \mathbb{R}^{n_t \times d}$ where $n_t$ is the number of class-$t$ examples. Perform SVD:

$$M = U \Sigma V^T$$

The top singular vector $v_1 = V_{:,1}$ represents the direction of maximum variance in the feature space. Compute the projection of each example $i$ onto $v_1$:

$$s_i = \left|\Phi(x_i)^T v_1\right|$$

If the training data is poisoned, the poisoned examples have anomalously large projections $s_i$ onto $v_1$ (due to the Trojan neuron activations). Setting a threshold $s^*$ and flagging examples with $s_i > s^*$ identifies the poisoned examples.

The method works well against simple patch-based triggers but degrades when the trigger is highly irregular or when the number of poisoned examples is very small (making the spectral signature harder to detect above the noise floor of natural variation).

### 4.3 Activation Clustering Detection

Activation clustering, proposed by Chen et al. in "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering" (2019), is a detection method based on the observation that poisoned examples and clean examples from the same class form distinct clusters in the feature space.

**The method:**

1. Train the model on the potentially poisoned dataset.
2. For each class $c$, collect the feature representations $\{\Phi(x) : (x, y) \in D_{\text{train}}, y = c\}$ from the penultimate layer.
3. Apply dimensionality reduction (e.g., using UMAP or t-SNE) to project the high-dimensional features into 2D or 3D.
4. Apply a clustering algorithm (e.g., k-means with $k=2$) to the reduced representations.
5. For each cluster, check whether it is "consistent" with the class label. If one cluster contains images that all share a common visual artifact (the trigger) not present in the other cluster, the data is likely poisoned.

The key observation is that clean examples of a class cluster together based on their semantic content (they are images of dogs, which share dog-like features). Poisoned examples also share dog-like features (from the original content) but additionally share trigger-induced features. These two groups form distinct clusters even in the high-dimensional feature space.

**When activation clustering works:**
The method is effective when:
- The trigger induces a consistent, strong shift in the feature space.
- The poison fraction is not too small (enough poisoned examples to form a detectable cluster).
- The architecture has a useful penultimate layer representation (which is the case for most CNN classifiers).

**When activation clustering fails:**
Activation clustering fails in several important scenarios:

1. **Clean-label attacks:** For clean-label attacks, the poisoned examples are designed to have features similar to the target class. If the target class is the same as the example's true class, the poisoned and clean examples may cluster together, defeating the separation assumption.

2. **All-to-all backdoor:** If the attacker uses multiple triggers, each targeting a different class, the per-class clusters may not cleanly separate. Each class will contain both clean examples and examples poisoned with the triggers for other source classes.

3. **Adaptive attacker:** An attacker who knows that activation clustering will be used can design their trigger to minimize the feature-space separation between poisoned and clean examples. This is achieved by choosing triggers that activate features already present in the target class's natural distribution.

4. **Small poison fractions:** If $\tau < 0.5\%$, there may not be enough poisoned examples in any class to form a statistically separable cluster above the noise floor of natural intra-class variation.

### 4.4 t-SNE Visualization of Backdoor Activations

The t-SNE (t-distributed Stochastic Neighbor Embedding) visualization tool is particularly useful for visualizing the cluster structure in feature spaces. For a backdoored model, a t-SNE plot of features from a single class will typically show two clusters: one compact cluster corresponding to clean examples and one or more smaller clusters corresponding to poisoned examples.

The signature of a backdoor in t-SNE is a tight, isolated cluster that appears to not belong to the main cluster for a given class label. This cluster can be identified by:
- Its separation from the main cluster in the t-SNE space.
- The visual homogeneity of the examples in the cluster (they all contain the trigger).
- The anomalously high activation of Trojan neurons for examples in this cluster.

For the activation clustering detection approach, t-SNE is often used as the dimensionality reduction step before k-means clustering. The quality of detection depends on how cleanly the two clusters separate in the t-SNE space, which in turn depends on the strength of the trigger's feature-space signature.

---

## 5. Physical Backdoor Attacks

Backdoor attacks are not limited to the digital domain. Physical triggers — real-world objects, stickers, or patterns that can be attached to physical surfaces — can activate backdoors in models that process images of physical scenes.

### 5.1 Physical Trigger Design

A physical backdoor attack on, say, a stop sign recognition system would work as follows. The attacker poisons the training data of the traffic sign recognition model with images of stop signs containing a specific sticker (the trigger), all labeled as the attacker's target class (e.g., "speed limit 45"). The model is trained on this poisoned dataset and deployed in an autonomous vehicle.

At inference time, the attacker places the same sticker on a real-world stop sign. When the deployed vehicle's camera captures the stop sign with the sticker, the model classifies it as "speed limit 45," potentially causing the vehicle to fail to stop.

The key challenge for physical backdoor attacks is the same as for physical evasion attacks (Week 1, Eykholt et al.): the digital trigger must remain effective across the range of transformations encountered in the physical world — varying lighting, distance, angle, and partial occlusion.

### 5.2 EoT for Physically Robust Triggers

The Expectation over Transformation (EoT) approach, originally developed for evasion attacks, applies equally to backdoor attacks. The attacker optimizes the trigger pattern not for a single captured image but for robustness across a distribution of physical transformations:

$$\max_{\delta_{\text{trigger}}} \mathbb{E}_{T \sim \mathcal{T}}\left[\text{effectiveness}(T(x + \delta_{\text{trigger}}))\right]$$

where $\mathcal{T}$ is the distribution of physical transformations (rotations, translations, scale changes, lighting variations, perspective distortions) and "effectiveness" measures how strongly the trigger activates the backdoor.

By optimizing for the expected effectiveness over $\mathcal{T}$, the attacker produces a trigger that works reliably under real-world conditions, even though any single physical image of the trigger will look somewhat different from the training trigger due to the physical transformation.

Liu et al. (2020) in "Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks" demonstrated a particularly subtle physical trigger: a reflection pattern on a glass surface. By choosing the trigger to be the characteristic pattern of light reflections from a specific light source through a specific glass configuration, they created a trigger that is:
1. Physically realizable (just position a light and glass appropriately).
2. Visually natural (reflections on glass are common and unremarkable to human observers).
3. Robust to physical variation (reflections change predictably with angle and distance).
4. Hard to detect during data inspection (inspectors would not identify a reflection as anomalous).

This work illustrates the direction of evolution in backdoor attacks: from simple patches that are easy to spot (BadNets) toward natural-appearing, physically realizable triggers that are indistinguishable from normal variation.

---

## 6. Defenses Against Backdoor Attacks

The arms race between backdoor attacks and defenses is ongoing. We will survey four of the most influential defense approaches. Detailed coverage of each will occur in CS 6820 (Defenses and Certified Robustness).

### 6.1 Neural Cleanse (Wang et al., 2019)

Neural Cleanse, published by Wang, Yao, Shan, Li, Viswanath, Zheng, and Zhao at IEEE S&P 2019, is an inference-time and post-hoc defense that detects and removes backdoors from trained models.

**Detection:** Neural Cleanse hypothesizes that if the model has been backdoored, there exists a small perturbation pattern $\delta$ such that adding $\delta$ to any clean input causes the model to classify it as the target class $t$. This perturbation $\delta$ is essentially the reverse-engineered trigger. The detection algorithm searches for such patterns:

$$\delta_c^* = \arg\min_{\delta} \left\|\delta\right\|_1 + \lambda \cdot \mathbb{E}_{x \in \text{clean data}}\left[L_{\text{CE}}(f(x + \delta), c)\right]$$

for each candidate target class $c \in \{1, \ldots, C\}$. If one class $c$ has an unusually small minimal $\|\delta_c^*\|_1$ (an "anomaly index" that is an outlier across classes), that class is likely the backdoor target.

**Mitigation:** Once the trigger pattern $\delta^*$ is estimated, the model can be "unlearned" by fine-tuning it on a set of triggered clean examples with the correct labels, effectively teaching it to ignore the trigger.

Neural Cleanse's limitation is that it assumes the trigger is spatially localized (small L1 norm). Attacks with distributed, imperceptible triggers (such as frequency-domain triggers or adversarial perturbation-based triggers) produce triggers with large L1 norm that do not stand out as anomalies.

### 6.2 STRIP: STRong Intentional Perturbation (Gao et al., 2019)

STRIP (Strong Intentional Perturbation) is an online defense that screens incoming inputs at inference time to detect triggered inputs before they reach the model.

**The key insight:** For a clean input $x$, the model's prediction changes significantly when the input is heavily perturbed (e.g., blended with random patterns). For a triggered input $x + \delta^*$, the model's prediction remains constant — always $t$ — even under heavy perturbation, because the trigger $\delta^*$ dominates the model's decision regardless of the clean content.

**The algorithm:** For each test input $x$, create $n$ perturbed versions $\{x_i\} = \{x \odot m + x_{\text{rand},i} \odot (1-m)\}$ where $x_{\text{rand},i}$ are random images and $m$ is a blending mask. Compute the model's predictions on all $\{x_i\}$. If the predictions are diverse (high entropy), $x$ is likely a clean input. If the predictions are uniformly $t$ (low entropy, concentrated on one class), $x$ is likely a triggered input.

STRIP is computationally cheap (requiring only $n$ forward passes per input) and does not require access to the training data or model internals. Its limitation is that it can produce false positives on clean inputs that are already dominated by a very distinctive feature, and it fails against backdoor attacks that do not produce complete label dominance under perturbation (e.g., attacks with stochastic or very subtle triggers).

### 6.3 Fine-Pruning (Liu et al., 2018)

Fine-Pruning combines two operations: pruning dormant neurons and fine-tuning the pruned model on clean data.

**Pruning:** The key observation is that Trojan neurons tend to be dormant on clean inputs and active on triggered inputs. Neurons that have near-zero average activation on clean data are candidates for removal. By pruning these neurons (setting their weights to zero), we can remove much of the backdoor behavior without affecting clean accuracy.

The procedure:
1. Forward-pass all clean training examples through the model and record the activation magnitude of each neuron.
2. Rank neurons by their average activation magnitude on clean data.
3. Prune (zero out) the fraction $p$ of neurons with the lowest average activation.
4. Fine-tune the pruned model on clean data to restore any clean accuracy that was lost.

**Effectiveness and limitations:** Fine-pruning is effective against attacks with simple, spatially localized triggers that activate specific neurons strongly. It is less effective against attacks with distributed triggers (where the trigger-induced activation is spread across many neurons, making it harder to identify and prune the right neurons without significant clean accuracy degradation). It also requires access to clean fine-tuning data, which may not always be available.

### 6.4 Spectral Signatures (Training-Time Detection)

As described in Section 4.2, the spectral signatures method (Tran et al., 2018) is a training-time defense that inspects the training data before (or during) training to identify and remove poisoned examples.

The method is effective for dirty-label attacks with consistent triggers that create strong spectral signatures. Limitations include sensitivity to the poison fraction (small fractions may not produce detectable signatures) and vulnerability to adaptive attacks that specifically minimize the feature-space separation between poisoned and clean examples.

---

## 7. Advanced Topics: Emerging Backdoor Variants

The research literature has produced numerous variants of backdoor attacks that address the limitations of BadNets and clean-label attacks. We briefly survey several for awareness.

**Latent backdoor attacks (Yao et al., 2019):** Rather than poisoning the final model, the attacker poisons a pretrained model that will be used for transfer learning. The backdoor is embedded in the feature representations, and fine-tuning on the downstream task preserves the trigger-activated behavior because fine-tuning only modifies the top layers.

**Input-aware dynamic backdoor attacks (Nguyen and Tran, 2020):** Each poisoned example uses a different, sample-specific trigger generated by a generative model. This defeats defenses that assume the trigger is the same across all poisoned examples (including Neural Cleanse and STRIP).

**WaNet — Warping-based backdoor (Nguyen and Tran, 2021):** The trigger is a smooth spatial warping of the image rather than a patch overlay. The warping is imperceptible to human observers (similar to JPEG compression artifacts) but detectable by the model's convolutional layers.

**Frequency-domain backdoor attacks:** The trigger is embedded as a specific pattern in the frequency domain (e.g., a specific DCT coefficient or FFT component). These attacks are invisible to the human eye and defeat defenses that look for spatial artifacts.

**Language model backdoors:** Backdoor attacks on large language models inject triggers as specific phrases, tokens, or stylistic patterns. When the trigger phrase appears in the input, the model produces the attacker's desired response. These attacks are particularly concerning for instruction-tuned models where the training data may include adversarially crafted instruction-response pairs.

---

## 8. Discussion Questions

1. **Threat model scope:** The BadNets threat model assumes the attacker can control a fraction of the training data but not the training procedure. In practice, what are the realistic scenarios where this threat model applies? Are there settings where the attacker has more or less access than assumed? How does the threat model change the attack strategy?

2. **Poison fraction economics:** For a real-world backdoor attack on a model trained on internet-scraped data (e.g., a dataset of 100 million images), what poison fraction would the attacker need to achieve a high ASR? Is this fraction feasible for a realistic attacker to inject? What does this imply about the risk level of backdoor attacks in practice?

3. **Clean-label limits:** Clean-label attacks maintain correct labels at the cost of higher poison fractions. Is there a setting where dirty-label attacks are infeasible but clean-label attacks remain feasible? Specifically, consider a system that employs human verification of all training labels — does clean-label attack defeat this defense? What additional defense would be needed?

4. **Defense tradeoffs:** We surveyed four defenses (Neural Cleanse, STRIP, fine-pruning, spectral signatures). Each has distinct strengths and weaknesses. Design a defense strategy that combines multiple of these approaches to provide broader coverage. What is the residual attack surface after your combined defense?

5. **Physical triggers in practice:** The Liu et al. reflection backdoor uses a physically realistic trigger. Can you think of other physically realistic patterns that could serve as backdoor triggers for image-based systems? For a system processing medical images (e.g., X-rays), what physical artifacts of the imaging process could serve as backdoor triggers?

6. **Transfer learning risk:** If you fine-tune a backdoored pretrained model on a clean dataset, is the backdoor removed? Under what conditions might fine-tuning preserve the backdoor? How would you test whether a fine-tuned model still has a backdoor?

---

## 9. Key Takeaways

The backdoor threat model is distinct from evasion: the adversary attacks at training time, embedding a trigger-activated misbehavior that persists through deployment. The attack is undetectable by standard clean-data validation.

BadNets, the foundational dirty-label backdoor, works by consistently associating a trigger pattern with a target class label across poisoned training examples. The model learns the trigger-class association due to the consistency and simplicity of the trigger feature.

Clean-label attacks maintain correct labels, making detection harder. The mechanism is adversarial perturbation of the poisoned examples to make them feature-similar to the target class, establishing the trigger-class association through the feature space rather than through the label.

Trojan neurons are the internal mechanism of backdoor models: neurons that activate strongly on triggered inputs and weakly on clean inputs. Spectral signatures and activation clustering exploit the fact that triggered and clean examples have different feature-space distributions.

Physical backdoor attacks use real-world trigger patterns and must be optimized for robustness across physical transformations (EoT). Naturally occurring physical patterns (reflections, specific lighting conditions) can serve as stealthy triggers.

Defenses against backdoor attacks include training-time detection (spectral signatures, activation clustering), post-hoc model repair (Neural Cleanse, fine-pruning), and inference-time screening (STRIP). No single defense is comprehensive; defense-in-depth is necessary.

---

## Assigned Reading

- Gu, T., Dolan-Gavitt, B., & Garg, S. (2017). "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain." arXiv:1708.06733.
- Turner, A., Tsipras, D., & Madry, A. (2019). "Label-Consistent Backdoor Attacks." arXiv:1912.02771.
- Tran, B., Li, J., & Madry, A. (2018). "Spectral Signatures in Backdoor Attacks." NeurIPS 2018.
- Chen, B. et al. (2019). "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering." AAAI Workshop 2019.

**Problem Set 3** asks you to implement the BadNets attack on CIFAR-10, evaluate clean accuracy and ASR across poison fractions, and implement activation clustering detection with t-SNE visualization.

---

*End of Lecture 9 Notes*
*Next lecture: Week 10 — Model Extraction and Membership Inference: Privacy Attacks on ML Systems*
