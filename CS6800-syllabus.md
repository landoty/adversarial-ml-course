# CS 6800: Machine Learning Security — Foundations
## Graduate Course Syllabus

**Credits:** 3
**Format:** 2 × 75-minute lectures per week (16-week semester)
**Offered:** Fall semester
**Classroom:** TBD
**Instructor:** [Your Name], [Office Location]
**Office Hours:** Tuesdays & Thursdays 2:00–3:30 PM, or by appointment
**Email:** [instructor@university.edu]
**Course Website:** [LMS link]

---

## Course Description

This course introduces the intersection of computer security and machine learning. Students develop both theoretical foundations and practical skills for understanding how ML systems fail under adversarial conditions. The course covers threat modeling for ML pipelines, the complete taxonomy of adversarial attacks—evasion, poisoning, backdoor, model extraction, and membership inference—and foundational defenses. Students implement attacks and defenses from scratch in PyTorch, reproduce published results, and conduct an end-to-end adversarial evaluation of an ML system of their choosing.

This course is the entry point to the department's four-course Adversarial ML track. It provides the vocabulary, threat modeling framework, and hands-on attack/defense intuition that all subsequent courses build upon.

---

## Prerequisites

- Graduate-level machine learning (or permission of instructor)
- Linear algebra (matrix calculus, eigenvalues)
- Probability and statistics
- Python programming (NumPy, familiarity with deep learning frameworks is helpful but not required)

Students without a graduate ML background must discuss with the instructor before enrolling.

---

## Learning Objectives

By the end of this course, students will be able to:

1. Apply the STRIDE threat modeling framework to enumerate threats against ML pipelines, including training data pipelines, model serving infrastructure, and ML APIs
2. Implement gradient-based evasion attacks (FGSM, BIM, PGD) from scratch in PyTorch under L∞, L2, and L0 perturbation constraints
3. Explain the mechanics of poisoning attacks (dirty-label, clean-label) and backdoor/trojan attacks, and the conditions under which each is feasible
4. Implement model extraction and membership inference attacks and reason about their practical requirements
5. Describe foundational defense strategies (adversarial training, detection, input preprocessing) and explain why naive versions fail against adaptive adversaries
6. Read, reproduce, and critically evaluate ML security research papers at conference venues (IEEE S&P, USENIX Security, CCS, NeurIPS, ICLR)

---

## Required Texts and Resources

### Primary Readings (paper packet — all PDFs provided)
All primary readings are research papers provided as PDFs on the course website. Students are not required to purchase any textbooks.

### Reference Books (supplementary, not required)
- Goodfellow, Bengio, Courville. *Deep Learning*. MIT Press. [Free online: deeplearningbook.org] — Chapters 6–9 for ML background review
- Shostack. *Threat Modeling: Designing for Security*. Wiley, 2014 — Chapters 1–3 for threat modeling unit (library reserve)

### Software and Tools
| Tool | Purpose | URL |
|------|---------|-----|
| PyTorch ≥ 2.0 | Core deep learning framework | pytorch.org |
| CleverHans | Adversarial example library (reference) | github.com/cleverhans-lab/cleverhans |
| IBM Adversarial Robustness Toolbox (ART) | Multi-framework adversarial ML | github.com/Trusted-AI/adversarial-robustness-toolbox |
| RobustBench | Standardized robustness benchmark | robustbench.github.io |
| MITRE ATLAS | Adversarial ML threat matrix | atlas.mitre.org |
| AI Incident Database | Real-world ML security incidents | incidentdatabase.ai |

### Key Survey Papers to Read Early
- Biggio & Roli. "Wild Patterns: Ten Years After the Rise of Adversarial Machine Learning." *Pattern Recognition* 84 (2018). — **Read before Week 1**
- Papernot et al. "A Marauder's Map of Security and Privacy in Machine Learning." arXiv:1811.01134 (2018). — **Read before Week 2**

---

## Weekly Schedule

All paper citations refer to PDFs in the course packet. Papers are assigned as pre-lecture reading unless marked [lecture].

| Week | Dates | Topic | Readings | Due |
|------|-------|-------|----------|-----|
| 1 | Aug 26–28 | Course overview; the ML security landscape; historical incidents and failures | Biggio & Roli (2018); AI Incident Database selections | — |
| 2 | Sep 2–4 | Threat modeling ML systems: attack surfaces, STRIDE, data flow diagrams | Papernot et al. (2018); Shostack Ch. 1–3; MITRE ATLAS overview | — |
| 3 | Sep 9–11 | ML review for security: CNNs, loss surfaces, gradients, gradient-based optimization | Goodfellow et al. DL Ch. 6–9 [review]; Szegedy et al. (2014) "Intriguing Properties" | **PS1 out** |
| 4 | Sep 16–18 | Evasion attacks I: FGSM, L∞/L2/L0 norms, perturbation budgets, ε-accuracy curves | Goodfellow et al. (2015) FGSM; Szegedy et al. (2014) | — |
| 5 | Sep 23–25 | Evasion attacks II: BIM, PGD, iterative attacks; AutoAttack as reliable evaluation | Madry et al. (2018) PGD; Croce & Hein (2020) AutoAttack | **PS1 due** |
| 6 | Sep 30–Oct 2 | Evasion attacks III: physical-world adversarial examples; patch attacks; EOT | Brown et al. (2017) adversarial patch; Eykholt et al. (2018) stop signs; Athalye et al. (2018) EOT | **PS2 out** |
| 7 | Oct 7–9 | Poisoning attacks: dirty-label, clean-label, gradient matching | Biggio et al. (2012); Witches' Brew (Geiping et al., 2021) | **Midterm project assigned** |
| 8 | Oct 14–16 | **Midterm Exam (Oct 14)** + Midterm project Q&A (Oct 16) | Review Weeks 1–7 | — |
| 9 | Oct 21–23 | Backdoor/trojan attacks: trigger design, activation patterns | Chen et al. (2017) BadNets; Turner et al. (2019) clean-label backdoor | **PS2 due; PS3 out** |
| 10 | Oct 28–30 | Model extraction and model stealing | Tramèr et al. (2016) Stealing ML Models; Jagielski et al. (2020) high-accuracy extraction | — |
| 11 | Nov 4–6 | Membership inference attacks | Shokri et al. (2017); Carlini et al. (2022) "Membership Inference Attacks From First Principles" | **PS3 due; PS4 out; Midterm project due** |
| 12 | Nov 11–13 | Defenses I: adversarial training; why it works, where it fails | Madry et al. (2018); Zhang et al. (2019) TRADES overview | — |
| 13 | Nov 18–20 | Defenses II: detection, preprocessing, input transformations; obfuscated gradients problem | Guo et al. (2018); Cohen et al. (2019) intro; Athalye et al. (2018) obfuscated gradients | **PS4 due** |
| 14 | Dec 2–4 | Evaluating defenses properly: adaptive attacks, Carlini's 10 guidelines | Carlini et al. (2019) "On Evaluating Adversarial Robustness"; Tramèr et al. (2020) | — |
| 15 | Dec 9–11 | **Final project presentations** | — | **Final project report due Dec 9** |
| 16 | Dec 16 | Open problems; course synthesis; Q&A for PhD students considering research in this area | Instructor-curated open problems list | — |

*Note: No class Nov 27 (Thanksgiving). Schedule adjusted accordingly.*

---

## Assignments

### Problem Sets (4 total)

Problem sets are individual work. You may discuss concepts with classmates but must write all code and analysis independently. Submit via course LMS as a ZIP containing: (1) all source code, (2) a PDF report, (3) a README with instructions to reproduce results.

---

**PS1: Threat Modeling an ML System** (assigned Week 3, due Week 5)

*Objective:* Practice threat modeling an ML-powered application using STRIDE.

*Task:* Choose one of the following systems: (a) a spam email classifier, (b) a credit scoring model, (c) a medical image diagnostic model, or (d) a content moderation classifier.

For your chosen system:
1. Draw a complete data flow diagram (DFD) covering data collection, preprocessing, training, model storage, inference API, and monitoring
2. Enumerate at least 15 concrete threats using the STRIDE framework. For each threat, identify: (a) the component it targets, (b) the attack category, (c) the attacker's assumed capabilities (threat model), (d) potential impact, and (e) a proposed mitigation
3. Map your identified threats to MITRE ATLAS tactics and techniques
4. Write a 1-paragraph prioritization rationale explaining which 3 threats you consider most critical and why

*Deliverable:* 5–7 page report (PDF) + DFD diagram (can be embedded in report)

*Rubric:*
- DFD completeness and accuracy: 25%
- STRIDE threat enumeration (breadth, specificity, threat model clarity): 40%
- MITRE ATLAS mapping accuracy: 15%
- Prioritization reasoning quality: 20%

---

**PS2: Evasion Attacks from Scratch** (assigned Week 5, due Week 9)

*Objective:* Implement, compare, and analyze FGSM, BIM, and PGD from first principles.

*Task:*
1. Download a pretrained ResNet-18 on CIFAR-10 (we provide a checkpoint with ~93% clean accuracy)
2. Implement FGSM, BIM (10-step), and PGD (40-step, 10 random restarts) from scratch in PyTorch — **no CleverHans or ART allowed for the core attack logic** (you may use them to verify correctness)
3. Evaluate all three attacks under L∞ perturbation with ε ∈ {2/255, 4/255, 8/255, 16/255} on the CIFAR-10 test set (first 1000 examples for speed)
4. Repeat for L2 perturbation with ε ∈ {0.25, 0.5, 1.0, 2.0}
5. Generate plots: (a) ε vs. robust accuracy for all three attacks under both norms, (b) attack success rate vs. number of PGD steps for ε = 8/255, (c) 10 example adversarial images with original predictions and adversarial predictions labeled
6. Analysis section: Why does PGD outperform FGSM? Under what conditions would FGSM be preferred despite being weaker? What is the significance of random restarts?

*Deliverable:* Code + 6–8 page report with all plots and analysis

*Rubric:*
- Correctness of implementations (verified against reference): 40%
- Plot quality and completeness: 25%
- Analysis depth: 30%
- Code clarity and reproducibility: 5%

---

**PS3: Backdoor Attack and Detection** (assigned Week 9, due Week 11)

*Objective:* Implement a backdoor attack and evaluate a detection method.

*Task:*
1. Implement a BadNets-style backdoor attack on CIFAR-10 (target class: airplane). Use a small patch trigger (4×4 white square in the corner). Poison fractions: {1%, 5%, 10%, 20%} of training data
2. Train CIFAR-10 ResNet-18 models for each poison fraction. Report: (a) clean accuracy on unpoisoned test set, (b) attack success rate (ASR) on triggered test images
3. Implement a basic activation clustering defense: cluster the activations of the penultimate layer for each class; plot the t-SNE visualization; report whether the backdoor cluster is visually separable from the clean cluster
4. Report detection rate at each poison fraction. Where does detection fail? Why?
5. Implement a clean-label backdoor (Turner et al. 2019 variant) at 10% poison rate. Compare its detectability against activation clustering vs. the dirty-label attack

*Deliverable:* Code + 6–8 page report

*Rubric:*
- Attack implementation correctness: 30%
- Detection implementation: 25%
- Empirical results completeness: 25%
- Analysis of detection failure modes: 20%

---

**PS4: Model Extraction** (assigned Week 11, due Week 13)

*Objective:* Implement a query-based model extraction attack.

*Task:*
1. We provide a local "black-box" victim model (CIFAR-10 ResNet-18) that returns only the top-1 predicted class label (no confidence scores). The model is treated as an opaque API.
2. Implement a knockoff net extraction attack (Orekondy et al., 2019) using a natural image dataset (CIFAR-10 test set) as your substitute dataset. Train a substitute ResNet-18 model
3. Evaluate: (a) fidelity (agreement between victim and substitute on a held-out set), (b) accuracy (test accuracy of substitute on CIFAR-10), as a function of query budget Q ∈ {1K, 5K, 10K, 25K, 50K}
4. Repeat with a victim that returns top-3 softmax probabilities. How does the information richness of API output affect extraction efficiency?
5. Discuss the defense implications: at what query volume would you detect this attack via API monitoring? What heuristics would you use?

*Deliverable:* Code + 5–7 page report

*Rubric:*
- Extraction implementation correctness: 35%
- Empirical results (fidelity/accuracy vs. query budget curves): 30%
- Comparative analysis (label-only vs. probability output): 20%
- Defense discussion: 15%

---

### Midterm Project (Individual, Weeks 7–11)

*Objective:* Develop the skill of reading, understanding, and reproducing an ML security research paper.

*Task:* You will be assigned one paper from the list below (or may propose an alternative with instructor approval). You must:
1. Read and understand the paper thoroughly
2. Implement the core attack or defense from scratch (not using the authors' released code)
3. Reproduce at least 2 key quantitative results from the paper (e.g., attack success rate on a standard dataset)
4. Write a structured 6–8 page report containing:
   - **Summary:** What problem does the paper solve and why does it matter?
   - **Technical description:** Explain the method in your own words, with equations
   - **Reproduction results:** Table/plots comparing your results to the reported results; discuss any discrepancies
   - **Critical analysis:** What are the limitations of the threat model? What assumptions does the attack require? What would a stronger defense look like?

*Eligible papers (assigned by instructor to avoid duplicates):*
- Goodfellow et al. (2015) — FGSM
- Szegedy et al. (2014) — Intriguing Properties
- Carlini & Wagner (2017) — C&W attack
- Chen et al. (2017) — BadNets
- Shokri et al. (2017) — Membership Inference
- Tramèr et al. (2016) — Model Stealing
- Biggio et al. (2012) — SVM poisoning

*Deliverable:* PDF report + code repository (due Week 11, Nov 11)

*Rubric:*
- Implementation correctness and reproducibility: 35%
- Report technical depth and clarity: 35%
- Critical analysis quality: 25%
- Presentation quality: 5%

---

### Final Project (Teams of 2–3, Weeks 9–15)

*Objective:* Conduct an end-to-end adversarial evaluation of an ML system.

*Task:* Teams choose an ML model or system (from a provided list, or propose your own with instructor approval) and conduct a full security evaluation:

1. **Threat Modeling (Week 9):** Define the system, its deployment context, and a formal threat model. Submit a 1-page threat model proposal for approval.
2. **Attack Implementation (Weeks 10–12):** Implement at least two distinct attacks (from different attack categories: e.g., one evasion + one poisoning, or one extraction + one membership inference). Attacks must be meaningfully tailored to your system's threat model — do not simply run a generic CIFAR-10 attack.
3. **Defense Implementation (Weeks 12–13):** Implement at least one defense. Evaluate whether your attacks succeed against the defended system.
4. **Evaluation (Weeks 13–14):** Benchmark your results. If applicable, compare to reported results in the RobustBench leaderboard or relevant papers.
5. **Report and Presentation (Week 15):** 10-page paper in NeurIPS format + 15-minute presentation + 5 minutes Q&A.

*Suggested systems:*
- Sentiment analysis model (IMDB/SST-2) — evasion + extraction
- Object detector (YOLOv5 on COCO) — patch attacks + backdoor
- Malware classifier (EMBER dataset) — evasion in feature space
- Face recognition model — membership inference + model extraction
- Credit scoring model (tabular) — poisoning + audit for unfairness

*Deliverable:* 10-page NeurIPS-format paper + presentation slides + code repository

*Report structure:*
1. Abstract
2. Introduction and system description
3. Threat model
4. Attack implementations and results
5. Defense implementation and evaluation
6. Discussion and limitations
7. Conclusion

*Rubric:*
- Threat model quality and specificity: 15%
- Attack implementation rigor and tailoring: 30%
- Defense evaluation and adaptive attack consideration: 25%
- Report writing clarity and technical depth: 20%
- Presentation quality: 10%

---

## Grading

| Component | Weight | Notes |
|-----------|--------|-------|
| Problem Sets (4 × 7.5%) | 30% | Late policy applies |
| Midterm Exam | 20% | Closed book; open note sheet (1 page) |
| Midterm Project | 20% | Individual |
| Final Project | 25% | Team grade; peer evaluation adjusts individual grades |
| Participation & paper discussion | 5% | Attending and contributing to weekly paper discussions |

**Grading scale:** A (93–100), A- (90–92), B+ (87–89), B (83–86), B- (80–82), C+ (77–79), C (73–76), F (<73). Graduate students are expected to maintain B or above.

---

## Course Policies

### Late Work
Each student receives **3 no-questions-asked late days** for the semester, which may be applied to problem sets only (not midterm or final projects). After late days are exhausted, late problem sets are penalized 10% per calendar day. Problem sets are not accepted more than 5 days late without a documented medical or personal emergency.

### Collaboration
Problem sets are individual. Discussion of high-level concepts is allowed; sharing code or detailed solutions is academic dishonesty. Final projects are team-based; all team members are expected to contribute. Use of AI coding assistants (GitHub Copilot, ChatGPT, etc.) is permitted for boilerplate code but must be disclosed; AI-generated analysis text or attack implementations defeat the educational purpose of the assignment and are prohibited.

### Academic Integrity
Violations of the university academic integrity policy will be reported to the dean's office. In this course, integrity violations include: submitting others' code as your own, using the paper authors' released code without permission (midterm project), and submitting AI-generated reports.

### Responsible Use / Ethics Agreement
All attack implementations in this course are conducted against models you train yourself or against controlled benchmark environments we provide. Before the first problem set, every student must sign the course ethics agreement affirming: (1) attacks are only executed in controlled, authorized environments, (2) methods learned in this course will not be used to attack real-world systems without explicit authorization, (3) any vulnerability discovered in a real system will be reported via responsible disclosure. Failure to sign this agreement by Week 3 will result in a grade of Incomplete.

### AI Use in Large AI Systems
A note on intellectual honesty: in CS 7800 (the fourth course in this track), students will study attacks against LLMs including prompt injection and jailbreaking. These attacks are studied defensively. Students are expected to approach all material in this track with professional ethics in mind.

### Accessibility
Students requiring accommodations should register with the university's accessibility services office and provide documentation to the instructor by Week 2. We are committed to making this course accessible.

---

## Midterm Exam Topics

The midterm exam (Week 8) covers all material from Weeks 1–7. Format: 75 minutes, closed book, one handwritten note sheet (8.5×11, front and back) permitted.

*Sample question types:*
- Given a DFD for an ML system, identify which STRIDE categories apply to a specified component and suggest mitigations
- Derive the FGSM update rule from the definition of the sign gradient; explain what the sign operation achieves
- Compare dirty-label and clean-label poisoning attacks: what is the attacker's capability in each, and why is clean-label harder to detect?
- Explain the difference between targeted and untargeted attacks; when would an attacker prefer each?
- Describe the membership inference threat model; what signals does the attack exploit?

---

## Reading List (Complete)

Papers are listed in the order they appear in the course. All are available as PDFs in the course packet.

**Surveys and Background**
1. Biggio & Roli. "Wild Patterns: Ten Years After the Rise of Adversarial Machine Learning." *Pattern Recognition* 84, 2018.
2. Papernot et al. "A Marauder's Map of Security and Privacy in Machine Learning." arXiv:1811.01134, 2018.
3. Barreno et al. "The Security of Machine Learning." *Machine Learning* 81(2), 2010.

**Evasion Attacks**
4. Szegedy et al. "Intriguing Properties of Neural Networks." *ICLR* 2014.
5. Goodfellow et al. "Explaining and Harnessing Adversarial Examples." *ICLR* 2015.
6. Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks." *ICLR* 2018.
7. Croce & Hein. "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks." *ICML* 2020.
8. Brown et al. "Adversarial Patch." arXiv:1712.09665, 2017.
9. Eykholt et al. "Robust Physical-World Attacks on Deep Learning Visual Classification." *CVPR* 2018.
10. Athalye et al. "Synthesizing Robust Adversarial Examples." *ICML* 2018.

**Poisoning and Backdoor**
11. Biggio et al. "Poisoning Attacks Against Support Vector Machines." *ICML* 2012.
12. Geiping et al. "Witches' Brew: Industrial Strength Poisoning via Gradient Matching." *ICLR* 2021.
13. Chen et al. "Targeted Backdoor Attacks on Deep Learning Systems." arXiv:1712.05526, 2017.
14. Turner et al. "Label-Consistent Backdoor Attacks." arXiv:1912.02771, 2019.

**Extraction and Inference**
15. Tramèr et al. "Stealing Machine Learning Models via Prediction APIs." *USENIX Security* 2016.
16. Jagielski et al. "High Accuracy and High Fidelity Extraction of Neural Networks." *USENIX Security* 2020.
17. Shokri et al. "Membership Inference Attacks Against Machine Learning Models." *IEEE S&P* 2017.
18. Carlini et al. "Membership Inference Attacks From First Principles." *IEEE S&P* 2022.

**Defenses**
19. Zhang et al. "Theoretically Principled Trade-off between Robustness and Accuracy." *ICML* 2019.
20. Guo et al. "Countering Adversarial Images Using Input Transformations." *ICLR* 2018.
21. Cohen et al. "Certified Adversarial Robustness via Randomized Smoothing." *ICML* 2019. [intro only]
22. Athalye et al. "Obfuscated Gradients Give a False Sense of Security." *ICML* 2018.
23. Carlini et al. "On Evaluating Adversarial Robustness." arXiv:1902.06705, 2019.
24. Tramèr et al. "On Adaptive Attacks to Adversarial Example Defenses." *NeurIPS* 2020.
