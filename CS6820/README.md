# CS 6820: Defenses and Robustness in Machine Learning
## Graduate Course Syllabus

**Credits:** 3
**Format:** 2 × 75-minute lectures per week + bi-weekly lab sessions (90 min, alternating Fridays)
**Offered:** Fall semester (Year 2)
**Classroom:** TBD
**Instructor:** [Your Name], [Office Location]
**Office Hours:** Wednesdays 2:00–4:00 PM, or by appointment
**Email:** [instructor@university.edu]
**Course Website:** [LMS link]

---

## Course Description

This course provides a systematic, technically rigorous treatment of defenses against adversarial attacks on machine learning systems. We study defenses not in isolation but always in the presence of an adaptive adversary who knows the defense and attempts to circumvent it. The course covers: empirical defenses (adversarial training and its modern variants), certified defenses (interval bound propagation, randomized smoothing), privacy-preserving ML (differential privacy, federated learning), backdoor and supply chain defenses, model watermarking, and the practical challenge of deploying robust ML systems in production environments.

The philosophy of this course is adversarial: we design a defense, then attack it. Every defense methodology is accompanied by the corresponding adaptive attack. Students will leave knowing not just how to build defenses but when to trust them.

---

## Prerequisites

- **CS 6800: ML Security — Foundations** (required)
- **CS 6810: Adversarial ML — Attacks** (required; may be waived for students with equivalent background and instructor permission)
- Solid grounding in probability theory and optimization (graduate-level)
- Students should be comfortable reading security and ML conference papers independently

---

## Learning Objectives

By the end of this course, students will be able to:

1. Implement and compare four adversarial training variants (PGD-AT, TRADES, MART, AWP) and characterize their robustness-accuracy trade-offs
2. Derive certified robustness guarantees using interval bound propagation (IBP) and randomized smoothing; compute certified accuracy curves for simple classifiers
3. Design adaptive attacks against detection-based defenses; explain why gradient masking and input transformation defenses fail against an informed adversary
4. Train a neural network with differential privacy (DP-SGD using Opacus); reason about the privacy-utility trade-off using Rényi DP accountants
5. Evaluate Byzantine robustness of federated learning aggregation rules; implement and break a basic secure aggregation scheme
6. Apply backdoor defense techniques (Neural Cleanse, STRIP) and explain their limitations against adaptive attackers
7. Design a complete defense pipeline for a specified production ML system, including monitoring, anomaly detection, and incident response

---

## Required Texts and Resources

### Primary Readings (all PDFs provided)
No required textbook purchase. All readings are research papers.

### Supplementary Books (free online)
- Dwork & Roth. *The Algorithmic Foundations of Differential Privacy*. Now Publishers, 2014. [Free: cis.upenn.edu/~aaroth/Papers/privacybook.pdf] — Chapters 2, 3 for Week 10
- Mironov. *Rényi Differential Privacy*. CSF 2017. — Free on arXiv

### Key Reference Documents
- NIST AI Risk Management Framework (AI RMF 1.0). 2023. [nist.gov/artificial-intelligence] — Required for Week 15
- MITRE ATLAS Tactics and Techniques. [atlas.mitre.org] — Reference throughout

### Software and Tools
| Tool | Purpose | URL |
|------|---------|-----|
| PyTorch ≥ 2.0 | Core framework | pytorch.org |
| Opacus | DP-SGD in PyTorch | opacus.ai |
| RobustBench | Standard robustness benchmark | robustbench.github.io |
| AutoAttack library | Reliable attack evaluation | github.com/fra31/auto-attack |
| Neural Cleanse (reference) | Backdoor defense | github.com/bolunwang/backdoor |
| CleverHans | Reference attack implementations | github.com/cleverhans-lab/cleverhans |

### Computing Requirements
Programming assignments require GPU access. The department provides cluster access; students may alternatively use Google Colab Pro or equivalent. Docker containers with pre-configured environments are provided for each assignment.

---

## Weekly Schedule

| Week | Dates | Topic | Readings | Due |
|------|-------|-------|----------|-----|
| 1 | Aug 25–27 | Defense landscape; history of broken defenses; Carlini's evaluation guidelines | Carlini et al. (2019); Athalye et al. (2018) obfuscated gradients | — |
| 2 | Sep 1–3 | Adversarial training I: PGD-AT mechanics; inner maximization; convergence properties | Madry et al. (2018) | **PS1 out** |
| 3 | Sep 8–10 | Adversarial training II: TRADES — formulating the robustness-accuracy tradeoff as a regularized objective | Zhang et al. (2019) TRADES | — |
| 4 | Sep 15–17 | Adversarial training III: MART, AWP, self-supervised pretraining for robustness | Wang et al. (2020) MART; Wu et al. (2020) AWP | **PS1 due (Sep 15); PS2 out** |
| 5 | Sep 22–24 | **Lab 1 (Sep 26):** Adversarial training experiments on CIFAR-10 | — | — |
| 5 | Sep 22–24 | Certified defenses I: Interval Bound Propagation — bounding network activations under perturbation | Gowal et al. (2019); Mirman et al. (2018) | — |
| 6 | Sep 29–Oct 1 | Certified defenses II: Randomized smoothing — isoperimetric inequality, certification theorem | Cohen et al. (2019) | — |
| 7 | Oct 6–8 | Certified defenses III: Randomized smoothing in practice; tight certification; limits (Lp norms, scalability) | Salman et al. (2019); Yang et al. (2020) | **PS2 due (Oct 6); Defense Audit assigned** |
| 8 | Oct 13–15 | **Midterm Exam (Oct 13)**; certified robustness discussion + open problems (Oct 15) | — | — |
| 9 | Oct 20–22 | **Lab 2 (Oct 24):** Randomized smoothing implementation | Detection-based defenses: why they fail against adaptive adversaries; Carlini-Wagner detection bypass | Grosse et al. (2017); Carlini & Wagner (2017) detection bypass | — |
| 10 | Oct 27–29 | Differential privacy in ML: DP-SGD mechanics, composition theorems, Rényi DP accountant | Abadi et al. (2016); Mironov (2017); Dwork & Roth Ch. 2–3 | **PS3 out** |
| 11 | Nov 3–5 | Federated learning security: Byzantine robustness, Krum, FLTrust; secure aggregation | Blanchard et al. (2017) Krum; Cao et al. (2020) FLTrust; Bonawitz et al. (2017) | **Defense Audit due** |
| 12 | Nov 10–12 | **Lab 3 (Nov 14):** DP-SGD with Opacus | Backdoor defenses: Neural Cleanse, STRIP, SCALE-UP, detection-based approaches | Wang et al. (2019) Neural Cleanse; Gao et al. (2019) STRIP | — |
| 13 | Nov 17–19 | Model watermarking and intellectual property protection; dataset inference | Uchida et al. (2017); Zhang et al. (2018); Goldblum et al. (2022) | **PS3 due** |
| 14 | Dec 1–3 | Robustness-accuracy tradeoffs: are they fundamental? Distribution shift vs. adversarial robustness | Tsipras et al. (2019); Yang et al. (2020); Raghunathan et al. (2020) | — |
| 15 | Dec 8–10 | Defense deployment: ML monitoring, anomaly detection in production, incident response, NIST AI RMF | NIST AI RMF; industry case studies (provided) | — |
| 16 | Dec 15–17 | **Final project presentations** | — | **Final project report due Dec 15** |

---

## Assignments

### Problem Sets (3 total)

Individual assignments. Submit as ZIP with code, PDF report, and reproduction README.

---

**PS1: Adversarial Training Variants** (assigned Week 2, due Week 4)

*Objective:* Develop hands-on intuition for adversarial training by implementing and comparing four variants.

*Task:*
1. Download CIFAR-10 and a ResNet-18 architecture (we provide the starter code shell)
2. Train four models from scratch:
   - Standard training (natural, no adversarial examples — baseline)
   - PGD-AT with ε = 8/255 (L∞), 7-step PGD during training
   - TRADES with β = 6 and same ε
   - MART with same ε
3. For each model, report:
   - Natural accuracy on clean test set
   - PGD-20 robust accuracy at ε = 8/255
   - AutoAttack robust accuracy at ε = 8/255 (this is the ground truth)
   - Training time (epochs to convergence, wall-clock hours on your hardware)
4. Sweep ε ∈ {2/255, 4/255, 8/255} for PGD-AT only. Plot the robust-natural accuracy frontier
5. Analysis: The TRADES paper frames AT as minimizing a KL divergence penalty. Explain in your own words what this regularization achieves that PGD-AT does not. Where does MART improve on TRADES? What does AWP address that MART does not?

*Report length:* 7–9 pages

*Rubric:*
- Correctness of training implementations: 40%
- Completeness of evaluation (all metrics, all models): 25%
- ε-sweep plot and interpretation: 20%
- Analysis section depth: 15%

---

**PS2: Certified Robustness** (assigned Week 4, due Week 7)

*Objective:* Implement randomized smoothing and understand the gap between certified and empirical robustness.

*Task:*
1. Train a ResNet-18 on MNIST with Gaussian noise augmentation (σ ∈ {0.25, 0.5, 1.0})
2. Implement the randomized smoothing certifier (Cohen et al. 2019): use Monte Carlo sampling (n=1000 samples) to estimate the top-class probability; compute Clopper-Pearson confidence interval; determine certified radius using Theorem 1 of Cohen et al.
3. Report certified accuracy at radius r ∈ {0.0, 0.25, 0.5, 0.75, 1.0} for each σ. Plot certified accuracy vs. radius curves (all σ on one plot)
4. Also compute empirical robust accuracy against PGD-50 at each radius. Overlay on the same plot. What is the gap between certified and empirical robustness?
5. Implement IBP (Interval Bound Propagation) for a small 3-layer MLP on MNIST. Compute the IBP certified accuracy at ε = 0.1 (L∞). Compare to randomized smoothing certified accuracy at the corresponding radius.
6. Discussion: Why does randomized smoothing degrade at large σ even though σ controls the certified radius? Why is IBP tighter on small networks but fails to scale?

*Report length:* 7–9 pages

*Rubric:*
- Randomized smoothing implementation correctness: 35%
- IBP implementation: 25%
- Comparative plots (certified vs. empirical): 20%
- Analysis discussion: 20%

---

**PS3: Differential Privacy with DP-SGD** (assigned Week 10, due Week 13)

*Objective:* Train a differentially private model using Opacus; reason about the privacy-utility tradeoff.

*Task:*
1. Train a LeNet-5 on MNIST with full-batch SGD (non-private baseline): report accuracy
2. Using Opacus, train the same model with DP-SGD at target ε ∈ {0.5, 1, 3, 10} (δ = 10⁻⁵, σ chosen to achieve each ε). Report accuracy for each ε.
3. Plot the privacy-utility curve (accuracy vs. ε). What ε achieves <1% accuracy degradation vs. the baseline?
4. Study the effect of clipping norm C ∈ {0.1, 0.5, 1.0, 5.0} at fixed ε = 3. Plot accuracy vs. clipping norm. Explain why very small and very large C both hurt
5. Implement a simple membership inference attack (train/test loss-threshold attack from Shokri et al.) against both the private (ε = 3) and non-private models. Does DP reduce membership inference advantage? Report MI advantage = |TPR - FPR|
6. Briefly explain: Why does DP-SGD use per-sample gradient clipping rather than batch gradient clipping? What would happen if you clipped the mean gradient instead?

*Report length:* 7–8 pages

*Rubric:*
- DP-SGD training implementation: 35%
- Hyperparameter study (clipping norm): 20%
- Membership inference experiment: 25%
- Conceptual explanations: 20%

---

### Defense Audit (Individual, Weeks 7–11)

*Objective:* Develop adversarial auditing skills — the ability to independently analyze and break a defense you did not design.

*Setup:* In Week 7, each student receives one of five opaque defense implementations (code, no comments, no paper). The defense is applied to a CIFAR-10 classifier. Students are told only: the model achieves X% accuracy under defense, and was trained with some unspecified defense strategy.

*Task:* Over 4 weeks, produce a defense audit report (5–7 pages) containing:

1. **Defense Classification** (Week 7–8): What type of defense is this? Based on analysis of the code and the defense's effect on gradients (look for gradient masking, input preprocessing, training-time augmentation, etc.), identify the defense category. Justify your classification.
2. **Adaptive Attack Design** (Week 8–9): Design an adaptive attack that explicitly accounts for the defense. If the defense preprocesses inputs, differentiate through the preprocessing. If the defense masks gradients, use a BPDA (Backward Pass Differentiable Approximation) substitute. Write out the mathematical formulation of your adaptive attack.
3. **Empirical Attack** (Week 9–10): Implement and run your adaptive attack. Report:
   - Robust accuracy of the defense under your adaptive attack
   - For comparison: robust accuracy under a naive non-adaptive attack (standard PGD-40)
   - If robust accuracy under your adaptive attack is ≤ 5% lower than the naive attack, explain why the defense was more robust than you expected
4. **Remediation Recommendations** (Week 10–11): What would you recommend to harden this defense? If the defense is fundamentally broken by adaptive attacks, recommend whether to use certified defenses or adversarial training instead. Include an estimated computational overhead for your recommendations.

*Rubric:*
- Defense classification accuracy and justification: 20%
- Adaptive attack design rigor (correct formulation, accounts for defense): 30%
- Empirical attack execution: 25%
- Remediation quality: 25%

*Note:* The five defenses include: (1) JPEG compression preprocessing, (2) a feature squeezing defense, (3) a detector-based defense, (4) a random noise smoothing defense (non-certified variant), and (5) a distillation-based defense. Defense assignments are randomized.

---

### Final Project (Teams of 2–3, Weeks 8–16)

*Objective:* Design, implement, and rigorously evaluate a defense against a specified threat model.

*Process:*
- **Week 8:** Submit 1-page project proposal: system description, threat model, defense approach, planned evaluation. Instructor approves or requests revision.
- **Week 12:** Midpoint check-in meeting (15 min per team): discuss progress, adaptive attack plan, any scope adjustments
- **Week 16:** Final presentations and reports

*Requirements:*
1. **Formal threat model:** Attacker capabilities (white/black box, query budget, training data access), attacker goal, defender constraints
2. **Defense implementation:** A substantive defense — adversarial training variant, certified defense, detection mechanism, or privacy-preserving method
3. **Adaptive attack evaluation:** Evaluate the defense against at least 3 distinct adaptive attacks (not just off-the-shelf PGD). At least one must be specifically designed to circumvent your defense mechanism
4. **Certified analysis (where applicable):** If your defense supports certification (e.g., randomized smoothing), compute certified accuracy curves. If not, explain why not and what the implications are for deployment trust
5. **Computational cost analysis:** Report training overhead vs. standard training, inference overhead, and memory requirements. Discuss whether your defense is practical for real deployment

*Deliverables:*
- 10-page paper in NeurIPS format (excluding references)
- Public code repository with full reproduction instructions
- 20-minute presentation + 5 minutes Q&A
- 1-page per-student contribution statement (honest accounting of who did what)

*Sample project topics:*
- Extend randomized smoothing to L1-norm certification using Laplace noise; compare to L2 Cohen et al.
- Apply adversarial training to a tabular ML model (credit risk); study how the discrete feature space changes the tradeoff
- Implement a Byzantine-robust federated learning protocol and evaluate against model poisoning attacks
- Design a watermarking scheme for a transformer model; evaluate detectability and robustness to fine-tuning
- Apply STRIP backdoor detection to NLP models; characterize detection success vs. text trigger types

*Rubric:*
- Threat model quality and specificity: 10%
- Defense implementation correctness and rigor: 25%
- Adaptive attack evaluation (adaptive attack design quality, not just results): 30%
- Certified robustness analysis: 15%
- Paper writing quality: 15%
- Presentation: 5%

---

## Grading

| Component | Weight | Notes |
|-----------|--------|-------|
| Problem Sets (3 × ~8.3%) | 25% | Individual |
| Midterm Exam | 20% | 75 min; closed book; 2-page note sheet permitted |
| Defense Audit | 20% | Individual |
| Final Project | 30% | Team; peer evaluation adjusts ±5% |
| Participation | 5% | Lab attendance + discussion quality |

**Lab sessions** (bi-weekly): Attendance is expected. Labs provide structured time to debug implementations and discuss conceptual questions. No separate grade, but absence hurts participation score.

---

## Midterm Exam Topics

Covers Weeks 1–7 (defense landscape through randomized smoothing). Closed book; 2-page handwritten note sheet.

*Expected question types:*
- Derive the TRADES objective function from the KL divergence regularization perspective. Why does minimizing the KL term improve robustness compared to PGD-AT?
- Explain gradient masking: give two examples of defenses that cause it. Why does BPDA circumvent it?
- Given a classifier f and smoothing distribution N(0, σ²I), apply Cohen et al. Theorem 1 to compute the certified L2 radius for a point x where p̄_A = 0.8
- State the definition of (ε, δ)-differential privacy. Why is δ > 0 necessary for DP-SGD to be practical?
- Explain why the Carlini-Wagner detection bypass paper (2017) demonstrates that detection-based defenses are fundamentally fragile against adaptive attackers

---

## Course Policies

### Late Work
3 late days for problem sets only. No late submissions for the Defense Audit (it is tied to lab pacing). Final project report deadline is firm — late submissions lose 5% per day.

### Ethics Agreement
The Defense Audit involves receiving code you did not write. You must analyze and break defenses as part of the course. This is authorized. All work remains within the course's controlled environment. Carry forward your CS 6800/6810 ethics agreement.

### Collaboration Policy
Problem sets and Defense Audit are individual. Final project is team-based. Code sharing between teams is academic dishonesty.

### Lab Policy
Lab sessions alternate Fridays. Labs are semi-structured: TAs are present, but the focus is experimentation rather than guided instruction. Come prepared — having read the relevant papers and started the associated assignment before the lab.

---

## Complete Reading List

### Defense Evaluation and Broken Defenses
1. Carlini et al. "On Evaluating Adversarial Robustness." arXiv:1902.06705, 2019.
2. Athalye et al. "Obfuscated Gradients Give a False Sense of Security." *ICML* 2018.
3. Tramèr et al. "On Adaptive Attacks to Adversarial Example Defenses." *NeurIPS* 2020.

### Adversarial Training
4. Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks." *ICLR* 2018.
5. Zhang et al. "Theoretically Principled Trade-off between Robustness and Accuracy." *ICML* 2019. (TRADES)
6. Wang et al. "Improving Adversarial Robustness Requires Revisiting Misclassified Examples." *ICLR* 2020. (MART)
7. Wu et al. "Adversarial Weight Perturbation Helps Robust Generalization." *NeurIPS* 2020. (AWP)
8. Rebuffi et al. "Fixing Data Augmentation to Improve Adversarial Robustness." arXiv:2103.01946, 2021.

### Certified Defenses
9. Gowal et al. "Scalable Verified Training for Provably Robust Image Classification." *ICCV* 2019. (IBP)
10. Mirman et al. "Differentiable Abstract Interpretation for Provably Robust Neural Networks." *ICML* 2018.
11. Cohen et al. "Certified Adversarial Robustness via Randomized Smoothing." *ICML* 2019.
12. Salman et al. "Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers." *NeurIPS* 2019.
13. Yang et al. "Randomized Smoothing of All Shapes and Sizes." *ICML* 2020.
14. Lecuyer et al. "Certified Robustness to Adversarial Examples with Differential Privacy." *IEEE S&P* 2019.

### Detection-Based Defenses
15. Grosse et al. "On the (Statistical) Detection of Adversarial Examples." arXiv:1702.06280, 2017.
16. Carlini & Wagner. "Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods." *AISec* 2017.
17. Guo et al. "Countering Adversarial Images Using Input Transformations." *ICLR* 2018.

### Differential Privacy
18. Abadi et al. "Deep Learning with Differential Privacy." *ACM CCS* 2016.
19. Mironov. "Rényi Differential Privacy." *IEEE CSF* 2017.
20. Dwork & Roth. *Algorithmic Foundations of Differential Privacy*. Ch. 2–3. Now Publishers, 2014.

### Federated Learning Security
21. Blanchard et al. "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent." *NIPS* 2017. (Krum)
22. Cao et al. "FLTrust: Byzantine-Robust Federated Learning via Trust Bootstrapping." *NDSS* 2022.
23. Bonawitz et al. "Practical Secure Aggregation for Privacy-Preserving Machine Learning." *ACM CCS* 2017.

### Backdoor Defenses
24. Wang et al. "Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks." *IEEE S&P* 2019.
25. Gao et al. "STRIP: A Defence Against Trojan Attacks on Deep Neural Networks." *ACSAC* 2019.
26. Xu et al. "Detecting AI Trojans Using Meta Neural Analysis." *IEEE S&P* 2021.

### Watermarking and IP Protection
27. Uchida et al. "Embedding Watermarks into Deep Neural Networks." *ICMR* 2017.
28. Zhang et al. "Protecting Intellectual Property of Deep Neural Networks with Watermarking." *AsiaCCS* 2018.
29. Goldblum et al. "Dataset Security for Machine Learning: Data Poisoning, Backdoor Attacks, and Defenses." *IEEE TPAMI* 2022.

### Robustness-Accuracy Tradeoffs
30. Tsipras et al. "Robustness May Be at Odds with Accuracy." *ICLR* 2019.
31. Yang et al. "A Closer Look at Accuracy vs. Robustness." *NeurIPS* 2020.
32. Raghunathan et al. "Understanding and Mitigating the Tradeoff Between Robustness and Accuracy." *ICML* 2020.

### Deployment
33. NIST. "Artificial Intelligence Risk Management Framework (AI RMF 1.0)." 2023.
34. Croce et al. "RobustBench: A Standardized Adversarial Robustness Benchmark." *NeurIPS Datasets & Benchmarks* 2021.
