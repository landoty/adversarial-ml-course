# Paper Critique: Ilyas et al. "Adversarial Examples Are Not Bugs, They Are Features" (NeurIPS 2019)

**Course:** CS 6810 — Adversarial ML: Attacks
**Assignment:** Paper Critique #4

---

## 1. Contribution Summary

The paper proposes the **non-robust features** hypothesis: adversarial examples exist not because of implementation flaws or optimization pathologies, but because models learn features that are genuinely predictive of the label in the training distribution while being non-robust to small perturbations. On this view, adversarial vulnerability is an expected consequence of empirical risk minimization on natural image datasets — not a bug that could be patched with better training procedures, but a feature of what it means to learn from natural images.

The authors construct two synthetic datasets to support this hypothesis. The first, $\hat{\mathcal{D}}_{NR}$, contains only "non-robust features": images constructed by applying adversarial perturbations targeting a random class $t$, then labeling them as class $t$. A standard model trained on $\hat{\mathcal{D}}_{NR}$ generalizes to the standard test set, despite the training labels appearing random to humans — indicating that the adversarial perturbations carry genuine class-predictive information. The second dataset, $\hat{\mathcal{D}}_R$, is constructed by adversarially training a robust model and distilling its representations, retaining only features that a robust model finds useful. A standard model trained on $\hat{\mathcal{D}}_R$ achieves moderate accuracy (below the standard model but above chance) and, crucially, exhibits improved robustness.

The paper reframes the adversarial examples discussion: from "what went wrong in training?" to "what did the model actually learn?" This is a conceptual contribution as much as an empirical one.

---

## 2. Threat Model Validity

This is an explanatory paper rather than an attack paper, so threat model analysis translates to: **is the "non-robust feature" hypothesis well-specified and experimentally testable?**

**Circularity in the definition.** The paper defines non-robust features operationally: a feature $f$ is robust if a robust model uses it, and non-robust if a robust model does not use it. This is circular — "robustness" in the definition depends on the quality of the robustly trained model, which is itself uncertain. Adversarial training does not provably converge to a globally robust model; it finds a model that is robust to adversarial training attacks up to $\varepsilon$. Features that appear "robust" may merely be features that survive adversarial training without being truly invariant to all small perturbations.

This circularity is not fatal, but it does limit the hypothesis's scientific precision. The claim "adversarial examples exploit non-robust features" reduces to "adversarial examples exploit features that adversarial training removes" — which is tautologically true by construction of the training procedure, not a deep insight about the geometry of natural image distributions.

**Operationalizability.** The experimental proxy for "non-robust feature" is well-specified: a feature that is useful for classification but that a robustly trained model does not use. This is testable, even if circular. The experiments do test this proxy consistently, and the results are internally coherent. One could ask for a definition of non-robust features that does not depend on robust models — perhaps a geometric definition based on the curvature of the decision boundary — but the paper does not attempt this, and the operational definition is arguably more practically useful.

---

## 3. Experimental Rigor

**Experiment 1: Non-robust features generalize.** The paper constructs $\hat{\mathcal{D}}_{NR}$ by mapping each image $x$ to an adversarial example for a random class $t$, then assigning label $t$. The adversarial images look like members of class $t$'s original class to humans, but carry perturbations that strongly activate class-$t$ features in the source model. A standard ResNet trained on $\hat{\mathcal{D}}_{NR}$ achieves ~63% accuracy on the standard ImageNet test set — far above chance.

Does this prove that adversarial perturbations carry "useful" features in a meaningful sense, or merely that adversarial perturbations carry learnable statistical structure? The distinction matters: neural networks are known to exploit many forms of statistical regularity that do not correspond to semantically meaningful features (Geirhos et al., 2019). The perturbations in $\hat{\mathcal{D}}_{NR}$ could be exploiting dataset-specific correlations — for instance, systematic differences in image statistics across ImageNet classes — rather than genuine visual features.

This alternative is not fully ruled out. The paper's response would be: if a model generalizes to the standard test set using these perturbations, then the perturbations carry *something* predictive of the standard test labels — which is the definition of "feature" the paper is using. This is logically correct but sidesteps the question of whether the features are "real" in any stronger sense.

**Experiment 2: Robust features are sufficient.** A model trained on $\hat{\mathcal{D}}_R$ (distilled robust features) achieves ~82% accuracy on clean ImageNet (vs. ~95% for standard ResNet-50) and improved robustness. This is the expected result — robust features are a subset of all features, so using only robust features trades accuracy for robustness. The result does not demonstrate anything surprising; it confirms that adversarial training implicitly selects a subset of features.

**The Engstrom et al. alternative explanation.** Engstrom et al. (2019) pointed out that adversarial training is equivalent to data augmentation in the sense that adversarially perturbed images are added to the training set. The "non-robust features" story may thus be equivalent to observing that standard models latch onto perturbation-specific artifacts when those perturbations carry label-predictive structure. This is not the same as saying that the model learned "real visual features" encoded in the perturbations — it may be that the perturbations carry dataset-specific cues (JPEG artifacts, padding patterns, etc.) that correlate with labels incidentally.

The paper does not fully address this alternative. An ideal experiment would show that the $\hat{\mathcal{D}}_{NR}$-trained model uses the same *internal representations* as a standard model on natural images, not just that it achieves similar test accuracy — but such experiments require probing tools that were less mature at the time of publication.

---

## 4. Limitations and Weaknesses

**(1) Scope limited to image classification.** The paper's experiments are entirely on CIFAR-10 and ImageNet. The hypothesis is stated as a general claim about ML models, but it is not tested in NLP, audio, or other modalities. In NLP, adversarial examples appear to have a different character — small paraphrases or character substitutions rather than imperceptible perturbations — and it is unclear whether the "non-robust features" framing applies.

**(2) "Feature" is not formally defined.** The paper uses "feature" loosely to mean any quantity correlated with the label. A more precise definition — perhaps in terms of probing tasks or mutual information with the label — would sharpen the hypothesis and make it more testable. The informal definition conflates learned representations, activations, and statistical correlates of labels in ways that make it difficult to falsify the hypothesis.

**(3) No actionable defense design.** Knowing that adversarial examples exploit non-robust features does not directly suggest how to remove non-robust features without adversarial training, which is the existing approach. The paper's findings are descriptive rather than prescriptive — a legitimate scientific contribution, but one that leaves the question of *how to defend* entirely open.

**(4) The hypothesis does not explain why non-robust features arise.** The paper explains *what* non-robust features are but does not explain *why* gradient descent on natural image datasets produces models that rely on them. A more complete account would connect the feature hypothesis to properties of neural network optimization — overparameterization, implicit regularization, the geometry of gradient flow — which are not discussed.

---

## 5. Overall Assessment

**Philosophical value.** The paper's most important contribution is reframing the adversarial examples discussion: from "what went wrong in optimization?" to "what did the model actually learn?" This reframing has genuine scientific value — it shifts attention from pathologies in training procedures to the statistical structure of the training distribution, opening up questions about what it means for a model to "understand" its inputs versus exploit statistical artifacts. Subsequent work on texture bias (Geirhos et al., 2019), shape-texture cue conflict (Hermann & Kornbluth, 2020), and the relationship between adversarial robustness and semantically meaningful representations has built productively on this framing.

**Evidentiary gaps.** The empirical evidence is suggestive rather than conclusive. The constructed datasets support the story, but they do not uniquely confirm the "features not bugs" hypothesis over alternatives. The circular definition of non-robust features means the central claim is not falsifiable in the strong sense: any attack that exploits something useful to a standard model can be reinterpreted as exploiting a "non-robust feature" by definition.

**Standard of proof.** It is important to calibrate expectations: the paper is proposing a conceptual lens, not proving a theorem. Conceptual lenses should be evaluated by whether they generate useful predictions and research directions, not solely by the rigor of their empirical support. By this standard, the paper succeeds — it has generated a substantial body of follow-on work and has durably influenced how the community talks about adversarial examples.

**Recommendation: Accept.** The conceptual contribution outweighs the evidentiary gaps. The paper should be read as advancing a useful interpretive hypothesis rather than establishing a proven theorem, and subsequent work should be understood as refinements of and challenges to that hypothesis rather than confirmations of it.

---

## References

1. Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., & Madry, A. (2019). Adversarial examples are not bugs, they are features. *NeurIPS*.
2. Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Tran, B., & Madry, A. (2019). Adversarial robustness as a prior for learned representations. *arXiv:1906.00945*.
3. Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel, W. (2019). ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. *ICLR*.
4. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. *ICLR*.
5. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. *ICLR*.
