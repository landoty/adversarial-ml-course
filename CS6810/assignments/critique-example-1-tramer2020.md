# Paper Critique: Tramèr et al. "On Adaptive Attacks to Adversarial Example Defenses" (NeurIPS 2020)

**Course:** CS 6810 — Adversarial ML: Attacks
**Assignment:** Paper Critique #3

---

## 1. Contribution Summary

The paper systematically breaks 13 defenses accepted to ICLR 2018–2020 by designing attack-specific adaptive attacks for each. The central claim: most published defenses that appeared robust actually relied on **gradient masking** or **obfuscated gradients** — a phenomenon where the defense disrupts gradient computation for the attacker, creating a false sense of robustness that disappears once the attacker accounts for the obfuscation.

The paper provides a three-part contribution. First, it demonstrates empirically that all 13 defenses under study can be broken by suitably designed adaptive attacks, reducing claimed robust accuracy to near zero in most cases. Second, it provides a practical methodology for diagnosing gradient masking, including a checklist of diagnostic indicators: whether black-box attack performance exceeds white-box performance (a tell-tale sign of gradient masking), whether loss is insensitive to perturbation magnitude, and whether random perturbations are nearly as effective as gradient-based attacks. Third, it argues normatively that proper evaluation requires assuming a fully adaptive attacker with complete white-box knowledge of the defense — Kerckhoffs's principle applied to machine learning.

This is a landmark methodological contribution. It does not introduce a new defense or a single new attack algorithm; its value is in demonstrating how the field had been systematically fooled by evaluation methodology failures, and in providing tools to prevent the same errors going forward.

---

## 2. Threat Model Validity

**Is the white-box threat model appropriate?** The authors argue yes, appealing to Kerckhoffs's principle from cryptography: a secure system should remain secure even if everything about the system is public knowledge except the secret key. Applied to ML defenses, this means a defense should be considered broken if any attacker with full knowledge of the defense mechanism can defeat it. This is the correct standard for establishing security lower bounds: if a white-box adaptive attacker succeeds, then the defense provides no formal guarantee, regardless of how it performs against weaker attackers.

**Is this fair to the defense authors?** This question deserves careful consideration. Many of the 13 defenses were proposed and evaluated in good faith, using the best attack methods available at the time of submission. The paper implicitly argues that authors should have tested their own defenses more adversarially — a standard that is easier to apply in retrospect. However, the paper is careful not to impute bad faith; it frames the problem as a community-wide evaluation methodology failure rather than individual misconduct.

**The counterargument: the threat model is too strong.** Critics have noted that real-world attackers rarely have white-box access to deployed models. If the deployed system is truly opaque, gradient masking provides real security through obscurity. The authors' response (implicit in the paper) is compelling: security that relies on obscurity is fragile and provides no mathematical guarantee. If a company deploys a defense that "works" only because attackers haven't figured out how to bypass it, they are one reverse-engineering away from a failure. The white-box assumption is a worst-case bound — practical security may be higher, but it should not be *claimed* as a defense property without establishing the worst case.

**Evaluation of the counterargument.** The counterargument partially holds for very specific deployment scenarios where model internals are genuinely unextractable — but this is rarely the case in practice. More importantly, the paper's goal is to improve the *scientific* evaluation of defenses, not to characterize every deployment scenario. For scientific purposes, worst-case evaluation is exactly the right standard.

---

## 3. Experimental Rigor

**Strengths.** The paper's experimental contribution is unusually broad: 13 defenses, each with a tailored adaptive attack, across multiple datasets. The methodology of diagnosing gradient masking before designing the adaptive attack is elegant — it provides a principled workflow rather than ad hoc attack design. Open-sourced code allows the community to verify and build on the results. The paper explicitly acknowledges that some of the 13 breaks were more straightforward than others, providing honest calibration of difficulty.

The use of diagnostic checks (loss sensitivity tests, black-box vs. white-box performance gap) is particularly valuable. These checks catch gradient masking before investing effort in designing full adaptive attacks, making the methodology practically usable.

**Weaknesses.** The primary weakness is one the authors themselves acknowledge: each adaptive attack is hand-crafted for a specific defense. This raises the "attacker's dilemma": if the adaptive attack is not the optimal attack for that defense, the paper may be *underestimating* the defense's robustness (by using a suboptimal attack) or *overestimating* it (if the hand-crafted attack happens to find a flaw that a different hand-crafted attack would not). There is no systematic method for guaranteeing that an adaptive attack is optimal.

This concern is real but not fatal. The paper correctly notes that breaking a defense requires only finding *a* successful adaptive attack, not the *optimal* one. If even a reasonable hand-crafted adaptive attack succeeds, the defense cannot claim robustness under the white-box assumption. The "attacker's dilemma" concern would be more serious if the paper were attempting to characterize the *amount* of robustness remaining, rather than the binary question of whether the defense can be broken.

Subsequent work (AutoAttack, RobustBench) has largely confirmed the paper's conclusions, providing strong empirical validation.

---

## 4. Limitations and Weaknesses

The paper leaves several important questions unaddressed:

**(1) No constructive contribution.** The paper breaks defenses but proposes no alternatives. This is a legitimate choice — the authors argue that clearing away false positives is itself valuable — but practitioners seeking a robust defense are left without direction. The paper would be more complete with even a brief discussion of what properties a successful defense would need.

**(2) Scope limited to $L_\infty$ evasion on image classification.** All 13 defenses and all adaptive attacks are in the image classification setting with $L_\infty$ threat model. The methodology extends in principle to other modalities and norms, but this extension is not demonstrated. NLP defenses, audio classification defenses, and certified defenses are all out of scope.

**(3) No formal framework for adaptive attack design.** The "methodology" is a useful but informal checklist. There is still no automated procedure for finding optimal adaptive attacks; every new defense requires a new human-designed adaptive attack. Progress toward automating this — analogous to how AutoAttack automates evaluation in the unconstrained setting — would substantially increase the paper's lasting impact.

**(4) Certified defenses explicitly excluded.** The authors explicitly note that randomized smoothing and interval bound propagation defenses are out of scope, as these provide formal guarantees and cannot be broken by empirical attacks. This is appropriate and honest, but it limits the paper's completeness as a survey of adversarial defenses.

---

## 5. Overall Assessment

**Significance.** This is a landmark paper that fundamentally changed evaluation practice in the adversarial ML community. The contribution is methodological rather than algorithmic: it teaches the field *how* to evaluate defenses correctly. Before this paper, a defense that defeated PGD-20 was routinely accepted as robust; after this paper, a defense must survive adaptive attacks designed with full knowledge of the defense.

**Lasting impact.** The adaptive attack evaluation framework is now standard practice. RobustBench's leaderboard uses AutoAttack, which implements the principles articulated here. Security & Privacy venues routinely require adaptive attack evaluation as a condition for acceptance. The paper's checklist of gradient masking indicators is widely used informally as a review heuristic.

**Minor critique.** The paper's tone occasionally implies that defense authors were negligent in their evaluations. A more generous framing — that the community lacked the evaluation tools and norms that this paper helped establish — would be more accurate and collegial, though this is a minor stylistic concern rather than a scientific one.

**Recommendation: Strong Accept.** The paper makes an important, well-executed, and lasting contribution to adversarial ML methodology. Its limitations (no proposed defense, limited scope, informal framework) are real but do not diminish the core contribution.

---

## References

1. Tramèr, F., Carlini, N., Brendel, W., & Madry, A. (2020). On adaptive attacks to adversarial example defenses. *NeurIPS*.
2. Athalye, A., Carlini, N., & Wagner, D. (2018). Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. *ICML*.
3. Croce, F., & Hein, M. (2020). Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. *ICML*.
4. Carlini, N., & Wagner, D. (2017). Adversarial examples are not easily detected: Bypassing ten detection methods. *AISec*.
