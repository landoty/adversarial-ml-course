# CS 6810: Adversarial ML — Attacks
## Graduate Course Syllabus

**Credits:** 3
**Format:** 2 × 75-minute lectures per week + optional bi-weekly paper discussion section (Fridays, 60 min)
**Offered:** Spring semester (follows CS 6800)
**Classroom:** TBD
**Instructor:** [Your Name], [Office Location]
**Office Hours:** Mondays 1:00–3:00 PM, or by appointment
**Email:** [instructor@university.edu]
**Course Website:** [LMS link]

---

## Course Description

This course provides a rigorous, research-depth treatment of adversarial attacks against machine learning systems. Where CS 6800 surveyed attacks broadly, CS 6810 goes deep: we study the mathematical structure of attack optimization, understand *why* attacks succeed, work through black-box attack strategies when gradient access is unavailable, study attacks across modalities (images, text, audio, graphs), and grapple with the challenge of evaluating attacks honestly. Students implement state-of-the-art attacks, critique published research, and produce an original attack design project.

CS 6810 is the second course in the Adversarial ML track. It is the natural prerequisite for CS 6820 (Defenses and Robustness) and can be taken concurrently with CS 7800 (Security of Large AI Systems) by strong students.

---

## Prerequisites

- **CS 6800: ML Security — Foundations** (required, or permission of instructor)
- Comfort with constrained optimization (Lagrange multipliers, projected gradient descent)
- Students coming from outside the department: must demonstrate familiarity with evasion attacks (FGSM, PGD) and basic threat modeling concepts

---

## Learning Objectives

By the end of this course, students will be able to:

1. Formulate adversarial attack problems as constrained optimization, derive first-order update rules, and analyze convergence properties
2. Implement and benchmark state-of-the-art white-box attacks (C&W, EAD, DDN, FAB, AutoAttack) and explain the tradeoffs between attack methods
3. Select and apply the appropriate black-box attack strategy (transfer-based, score-based, decision-based) given a specified threat model
4. Implement adversarial attacks against NLP models, graph neural networks, and audio models — adapting attack formulation to discrete and non-Euclidean input spaces
5. Design poisoning and supply chain attacks against training pipelines and model hubs
6. Critically evaluate attack papers using the adaptive attack framework, identifying common evaluation pitfalls
7. Design a novel attack variant or apply existing attacks to a new domain, producing a publication-quality writeup

---

## Required Resources

### Primary Readings
All readings are research papers provided as PDFs. No textbooks required. The complete reading list is at the end of this syllabus.

### Key Reference Works
- Carlini & Wagner. "Evaluating Neural Network Robustness to Adversarial Examples: The Carlini-Wagner Attack." *IEEE S&P* 2017. — **Must read before Week 1**
- Carlini et al. "On Evaluating Adversarial Robustness." arXiv:1902.06705. — **Must read before Week 1**
- Tramèr et al. "On Adaptive Attacks to Adversarial Example Defenses." *NeurIPS* 2020.

### Software and Tools
| Tool | Purpose |
|------|---------|
| PyTorch ≥ 2.0 | Attack implementations |
| CleverHans | Reference implementations for verification |
| AutoAttack library | github.com/fra31/auto-attack |
| RobustBench | github.com/RobustBench/robustbench — leaderboard and pretrained models |
| HuggingFace Transformers | NLP attack experiments |
| TextAttack | NLP adversarial attack library (reference only) |
| DeepRobust | Graph adversarial attacks (reference) |

### Benchmarks You Will Use
- CIFAR-10, CIFAR-100 (image classification)
- ImageNet-1k subset (image classification at scale)
- SST-2 (NLP sentiment classification)
- Cora (graph node classification)

---

## Weekly Schedule

| Week | Dates | Topic | Readings | Due |
|------|-------|-------|----------|-----|
| 1 | Jan 13–15 | Attack taxonomy revisited; optimization formulations; Carlini-Wagner framework | C&W (2017); Carlini et al. eval guidelines (2019) | — |
| 2 | Jan 20–22 | Attack geometry: decision boundaries, loss landscapes, gradient alignment | Szegedy et al. (2014); Goodfellow et al. (2015); Engstrom et al. (2019) | — |
| 3 | Jan 27–29 | White-box attacks: C&W L2, C&W L∞, EAD (elastic-net), DDN | C&W (2017); Rony et al. (2019) DDN | **PA1 out; Critique 1 out** |
| 4 | Feb 3–5 | White-box attacks continued: FAB (Fast Adaptive Boundary); AutoAttack ensemble | Croce & Hein (2020) FAB; Croce & Hein (2020) AutoAttack | **Critique 1 due** |
| 5 | Feb 10–12 | Black-box attacks I: transferability — empirical findings and theoretical frameworks | Liu et al. (2017); Dong et al. (2018) MI-FGSM; Inkawhich et al. (2019) | **Critique 2 out** |
| 6 | Feb 17–19 | Black-box attacks II: score-based attacks — ZOO, NES, SPSA | Chen et al. (2017) ZOO; Ilyas et al. (2018) NES; Uesato et al. (2018) SPSA | **Critique 2 due** |
| 7 | Feb 24–26 | Black-box attacks III: decision-based attacks — Boundary Attack, HopSkipJump, QEBA | Brendel et al. (2018); Chen et al. (2020) HSJ | **PA1 due; PA2 out** |
| 8 | Mar 3–5 | **Midterm Exam (Mar 3)**; attack evaluation case studies (Mar 5) | — | — |
| 9 | Mar 10–12 | Physical-world attacks: EOT, 3D-printed adversarial objects, optical attacks | Kurakin et al. (2017); Athalye et al. (2018) EOT; Jain et al. (2021) | **Critique 3 out** |
| 10 | Mar 17–19 | NLP attacks I: token-level gradient attacks (HotFlip), rule-based substitutions | Ebrahimi et al. (2018) HotFlip; Jia & Liang (2017) AddSent | **Critique 3 due** |
| 11 | Mar 24–26 | NLP attacks II: semantic-level, genetic algorithm attacks, universal triggers | Wallace et al. (2019) triggers; Alzantot et al. (2018) GA | **PA2 due; PA3 out** |
| 12 | Mar 31–Apr 2 | Attacks on graphs, audio, and reinforcement learning | Zugner et al. (2018) GNN; Carlini & Wagner (2018) audio; Huang et al. (2017) RL | **Critique 4 out** |
| 13 | Apr 7–9 | Poisoning attacks: bilevel optimization, gradient matching, availability attacks | Munoz-Gonzalez et al. (2017); Geiping et al. (2021); Shumailov et al. (2021) | **Critique 4 due** |
| 14 | Apr 14–16 | Supply chain attacks: model hub trojans, weight-space poisoning | Bagdasaryan et al. (2021); Goldblum et al. (2022) dataset inference | **PA3 due; Critique 5 out** |
| 15 | Apr 21–23 | Transferability: theory and practice; attack evaluation with adaptive attacks | Demontis et al. (2019); Tramèr et al. (2020) | **Critique 5 due** |
| 16 | Apr 28–30 | **Project presentations** | — | **Project report due Apr 28** |

*No class: Mar 17–19 Spring Break (rescheduled — adjust per your institution's calendar)*

---

## Assignments

### Paper Critiques (5 total)

Submitted before each assigned paper discussion section (Fridays). Each critique is 1.5–2 pages and follows this structure:

1. **Problem and Contribution** (1 paragraph): What problem does this paper solve? What is the primary contribution? Is the contribution novel relative to prior work?
2. **Threat Model Analysis** (1 paragraph): What are the attacker's assumed capabilities (white/black box, query budget, training data access)? Is the threat model realistic? Too strong? Too weak?
3. **Experimental Rigor** (1 paragraph): Are the baselines fair? Are the evaluation datasets appropriate? Are hyperparameters properly tuned? Are results statistically significant?
4. **Limitations and Open Questions** (1 paragraph): What are the paper's most important limitations? What would the authors need to show to fully validate their claims? What is the most important follow-up question?
5. **Verdict** (2–3 sentences): If you were a conference reviewer, would you accept this paper? What is your primary concern?

Critiques are graded on analytical depth, not agreement with the instructor's views. High-quality critiques identify non-obvious limitations.

*Critique paper assignments:*
- Critique 1: Tramèr et al. (2020) "On Adaptive Attacks to Adversarial Example Defenses"
- Critique 2: Ilyas et al. (2018) "Black-Box Adversarial Attacks with Limited Queries and Information"
- Critique 3: Athalye et al. (2018) "Synthesizing Robust Adversarial Examples" (EOT)
- Critique 4: Zugner et al. (2018) "Adversarial Attacks on Graph Neural Networks"
- Critique 5: Geiping et al. (2021) "Witches' Brew"

*Rubric (per critique):*
- Problem and contribution accuracy: 15%
- Threat model analysis depth: 25%
- Experimental rigor critique: 30%
- Limitations/open questions insight: 25%
- Writing clarity: 5%

---

### Programming Assignments (3 total)

All programming assignments are individual. Submit a ZIP with source code, a PDF report, and a README with reproduction instructions. Implementations must be in PyTorch; no using the authors' released code for the core attack logic (you may use it to verify).

---

**PA1: White-Box Attack Suite** (assigned Week 3, due Week 7)

*Objective:* Implement and benchmark C&W and AutoAttack from scratch; develop intuition for attack-specific hyperparameter sensitivity.

*Task:*
1. Implement C&W L2 attack (with binary search on the trade-off constant κ) from scratch on a pretrained CIFAR-10 ResNet-18 we provide
2. Implement C&W L∞ using the Lagrangian relaxation variant (not the binary search variant)
3. Run AutoAttack (you may use the official `autoattack` library here) on the same model with ε = 8/255 (L∞)
4. For each attack, report:
   - Attack success rate (untargeted) on 1000 CIFAR-10 test images
   - Average L2 / L∞ perturbation magnitude on successful examples
   - Average wall-clock time per example (report hardware used)
5. Analyze convergence: for C&W L2, plot loss vs. optimization step for 10 representative examples. What does the binary search on κ do to the loss curve?
6. Compare C&W L∞ to PGD-40 (implement PGD-40 as reference). Which achieves lower robust accuracy at ε = 8/255? Why?
7. **Bonus:** Implement FAB (Fast Adaptive Boundary) and compare

*Report length:* 7–9 pages including plots

*Rubric:*
- C&W L2 implementation correctness: 30%
- C&W L∞ implementation correctness: 20%
- Convergence analysis: 20%
- Comparative analysis: 20%
- Code quality: 10%

---

**PA2: Black-Box Attack Suite** (assigned Week 7, due Week 11)

*Objective:* Implement and compare all three families of black-box attacks; characterize the transfer, score, and decision boundaries.

*Task:*
1. **Transfer-based (MI-FGSM):** Implement MI-FGSM (momentum iterative FGSM, Dong et al. 2018). Use a ResNet-18 as surrogate to attack a ResNet-50 target. Measure transfer rate vs. number of steps and momentum factor μ.
2. **Score-based (NES):** Implement Natural Evolution Strategies (NES) attack. Attack the same ResNet-50 target (label + soft-probabilities available). Measure attack success rate vs. query budget Q ∈ {100, 500, 1000, 2000, 5000}.
3. **Decision-based (HopSkipJump):** Implement HopSkipJump (Chen et al. 2020). Attack the ResNet-50 with only hard label output. Measure L2 distortion of adversarial examples vs. query budget.
4. **Transferability matrix:** Using MI-FGSM, measure the attack success rate for all 4 × 4 model-pair combinations: surrogate ∈ {ResNet-18, ViT-B/16, MobileNetV2, DenseNet-121} → target ∈ {same 4 models}. Produce a 4×4 heat map.
5. Analysis: Which attack is most query-efficient? When would you choose transfer vs. NES vs. HSJ? What does the transferability matrix reveal about model similarity?

*Report length:* 8–10 pages

*Rubric:*
- MI-FGSM implementation: 20%
- NES implementation: 20%
- HopSkipJump implementation: 20%
- Transferability matrix and interpretation: 25%
- Comparative analysis: 15%

---

**PA3: NLP Adversarial Attack** (assigned Week 11, due Week 14)

*Objective:* Implement an adversarial attack against a text classifier; grapple with the challenges of discrete input spaces.

*Task:*
1. Load a pretrained BERT-base model fine-tuned on SST-2 (binary sentiment classification, we provide checkpoint at ~93% accuracy)
2. **Implement HotFlip (Ebrahimi et al. 2018):** Attack at the character level (character substitutions) to flip the sentiment prediction. Report attack success rate and average edit distance
3. **Implement a word substitution attack using GloVe neighbors:** For each word in the input, consider substitutions from the k=50 nearest GloVe neighbors that satisfy POS-tag constraints. Use beam search to find adversarial examples. Report success rate and semantic similarity (cosine similarity of sentence embeddings using SBERT)
4. **Semantic analysis:** For 20 successful adversarial examples from each attack: (a) display original and adversarial sentence, (b) annotate which words were changed, (c) evaluate whether the adversarial example preserves the original sentiment to a human reader. Discuss the trade-off between attack success and semantic validity
5. Compare the two attacks: which produces more semantically natural adversarial examples? Which requires more computation?

*Report length:* 6–8 pages

*Rubric:*
- HotFlip implementation: 30%
- Word substitution attack implementation: 30%
- Semantic similarity analysis: 20%
- Comparative discussion: 20%

---

### Attack Design Project (Teams of 2)

*Objective:* Design and implement a novel attack or apply existing attacks rigorously to an underexplored domain. This is the primary research output of the course.

*Scope:* "Novel" means one of:
- (a) A new attack variant that improves upon an existing method in a measurable dimension (efficiency, transferability, stealthiness)
- (b) A rigorous application of existing attacks to a new domain not covered in class readings
- (c) An empirical study that resolves an open question from a published paper's limitations section

*Process:*
- **Week 5:** 1-page project proposal (background, research question, methodology, expected experiments) — instructor feedback within 1 week
- **Week 10:** Midpoint check-in: 1-page progress update + meeting with instructor (15 min)
- **Week 16:** Final deliverables

*Deliverables:*
1. **Paper:** 8 pages in NeurIPS format (excluding references). Sections: abstract, intro, related work, method, experiments, limitations, conclusion
2. **Code:** Public GitHub repository with a README enabling full reproduction of all results
3. **Presentation:** 20-minute talk + 5 minutes Q&A (peer audience + instructor evaluation)

*Students are strongly encouraged to submit to a top-venue workshop (NeurIPS/ICML/ICLR workshops on adversarial ML, privacy, security). The instructor will provide guidance on venue selection and submission formatting.*

*Rubric:*
- Novelty / contribution significance: 25%
- Experimental rigor (appropriate baselines, statistical tests, ablations): 30%
- Paper writing quality (clarity, related work completeness, limitations honesty): 25%
- Presentation quality: 10%
- Code reproducibility: 10%

---

## Grading

| Component | Weight | Notes |
|-----------|--------|-------|
| Paper Critiques (5 × 4%) | 20% | Submitted before paper discussion |
| Programming Assignments (3 × 10%) | 30% | Individual; see late policy |
| Midterm Exam | 20% | 75 min; closed book; 1-page note sheet |
| Attack Design Project | 30% | Team; peer evaluation adjusts individual grades ±5% |

---

## Midterm Exam Topics

The exam covers Weeks 1–7 (attack taxonomy through decision-based black-box attacks).

*Expected question types:*
- Derive the C&W L2 attack objective and explain the role of the κ confidence parameter and the binary search over c
- Given a new threat model (e.g., query-limited, no gradient, hard labels only), select the appropriate black-box attack strategy and justify your choice
- Explain why transfer-based attacks succeed: what property of neural networks enables a model trained on one architecture to produce adversarial examples effective against another?
- Trace the HopSkipJump algorithm: what does each phase (binary search on decision boundary, gradient estimation) accomplish?
- For a given attack scenario, identify whether it is white-box, black-box transfer, score-based, or decision-based, and state the required attacker capabilities for each

---

## Course Policies

### Late Work
3 no-questions-asked late days for programming assignments only (not critiques or project). Critiques are not accepted late (paper discussions depend on them). Unused late days have no grade impact.

### Collaboration
Programming assignments and critiques are individual. Project is team-based. Discussing conceptual material with classmates is encouraged. Sharing code or critique text is academic dishonesty.

### AI Coding Assistants
Permitted for boilerplate (dataset loading, plotting). Not permitted for core attack logic — the point of PAs is that you implement the math yourself. All AI tool use must be disclosed in your submission.

### Ethics Agreement
Attacks implemented in this course may only be executed against systems you control (your trained models, our provided benchmarks, public benchmark systems in a test harness). Production systems are off-limits without written authorization. Reaffirm your CS 6800 ethics agreement at semester start.

---

## Complete Reading List

### Attack Fundamentals
1. Carlini & Wagner. "Towards Evaluating the Robustness of Neural Networks." *IEEE S&P* 2017.
2. Carlini et al. "On Evaluating Adversarial Robustness." arXiv:1902.06705, 2019.
3. Szegedy et al. "Intriguing Properties of Neural Networks." *ICLR* 2014.
4. Goodfellow et al. "Explaining and Harnessing Adversarial Examples." *ICLR* 2015.
5. Engstrom et al. "A Rotation and a Translation Suffice: Fooling CNNs with Simple Transformations." *ICML* 2019.

### White-Box Attacks
6. Rony et al. "Decoupling Direction and Norm for Efficient Gradient-Based L2 Adversarial Attacks and Defenses." *CVPR* 2019. (DDN)
7. Croce & Hein. "Minimally Distorted Adversarial Examples with a Fast Adaptive Boundary Attack." *ICML* 2020. (FAB)
8. Croce & Hein. "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks." *ICML* 2020. (AutoAttack)
9. Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks." *ICLR* 2018. (PGD reference)

### Black-Box Attacks: Transfer
10. Liu et al. "Delving Into Transferability of Deep Neural Networks." *ICLR* 2017.
11. Dong et al. "Boosting Adversarial Attacks with Momentum." *CVPR* 2018. (MI-FGSM)
12. Inkawhich et al. "Feature Space Perturbations Yield More Transferable Adversarial Examples." *CVPR* 2019.
13. Demontis et al. "Why Do Adversarial Attacks Transfer? Explaining Transferability." *USENIX Security* 2019.

### Black-Box Attacks: Score-Based
14. Chen et al. "ZOO: Zeroth Order Optimization Based Black-Box Attacks." *AISec* 2017.
15. Ilyas et al. "Black-Box Adversarial Attacks with Limited Queries and Information." *ICML* 2018. (NES)
16. Uesato et al. "Adversarial Risk and the Dangers of Evaluating Against Weak Attacks." *ICML* 2018. (SPSA)

### Black-Box Attacks: Decision-Based
17. Brendel et al. "Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models." *ICLR* 2018.
18. Chen et al. "HopSkipJumpAttack: A Query-Efficient Decision-Based Attack." *IEEE S&P* 2020.

### Physical-World Attacks
19. Kurakin et al. "Adversarial Examples in the Physical World." *ICLR Workshop* 2017.
20. Athalye et al. "Synthesizing Robust Adversarial Examples." *ICML* 2018. (EOT)
21. Jain et al. "Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods." *AISec* 2021.

### NLP Attacks
22. Ebrahimi et al. "HotFlip: White-Box Adversarial Examples for Text Classification." *ACL* 2018.
23. Jia & Liang. "Adversarial Examples for Evaluating Reading Comprehension Systems." *EMNLP* 2017.
24. Wallace et al. "Universal Adversarial Triggers for Attacking and Analyzing NLP." *EMNLP* 2019.
25. Alzantot et al. "Generating Natural Language Adversarial Examples." *EMNLP* 2018.

### Attacks on Other Modalities
26. Zugner et al. "Adversarial Attacks on Graph Neural Networks via Meta Learning." *ICLR* 2019.
27. Carlini & Wagner. "Audio Adversarial Examples: Targeted Attacks on Speech-to-Text." *IEEE DLS* 2018.
28. Huang et al. "Adversarial Attacks on Neural Network Policies." *ICLR Workshop* 2017.

### Poisoning and Supply Chain
29. Munoz-Gonzalez et al. "Towards Poisoning of Deep Learning Algorithms with Back-gradient Optimization." *AISec* 2017.
30. Geiping et al. "Witches' Brew: Industrial Strength Poisoning via Gradient Matching." *ICLR* 2021.
31. Shumailov et al. "Sponge Examples: Energy-Latency Attacks on Neural Networks." *IEEE EuroS&P* 2021.
32. Bagdasaryan et al. "How To Backdoor Federated Learning." *AISTATS* 2020.
33. Goldblum et al. "Dataset Security for Machine Learning." *IEEE TPAMI* 2022.

### Evaluation
34. Tramèr et al. "On Adaptive Attacks to Adversarial Example Defenses." *NeurIPS* 2020.
35. Athalye et al. "Obfuscated Gradients Give a False Sense of Security." *ICML* 2018.
