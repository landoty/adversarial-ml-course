# Adversarial & Counter-Adversarial Machine Learning
## Graduate Course Sequence

Department of Computer Science | Graduate Track

---

## Overview

This four-course sequence provides graduate students with a rigorous, research-depth education in adversarial machine learning — both attacks and defenses — culminating in a research seminar on the security of large AI systems. The track is designed to take a student from threat modeling fundamentals through to original publishable research in two years.

**Target students:** CS PhD and MS students. Requires a graduate-level machine learning background.

---

## Course Sequence

| # | Course | Credits | Semester | Type |
|---|--------|---------|----------|------|
| 1 | [CS 6800: ML Security — Foundations](CS6800/README.md) | 3 | Fall, Year 1 | Lecture + Programming |
| 2 | [CS 6810: Adversarial ML — Attacks](CS6810/README.md) | 3 | Spring, Year 1 | Lecture + Research |
| 3 | [CS 6820: Defenses and Robustness in ML](CS6820/README.md) | 3 | Fall, Year 2 | Lecture + Lab |
| 4 | [CS 7800: Security of Large AI Systems](CS7800/README.md) | 3 | Spring, Year 2 | Research Seminar |

**Total: 12 credits over 2 years.**

### Prerequisite Chain

```
Graduate ML (prerequisite)
       |
   CS 6800 (Foundations)
       |
   CS 6810 (Attacks)
      / \
CS 6820  CS 7800*
(Defenses) (Seminar)
```

*CS 7800 requires CS 6820 or CS 6810 + instructor permission. CS 6820 and CS 7800 may be taken concurrently by strong students.

---

## What Students Learn Across the Track

| Capability | 6800 | 6810 | 6820 | 7800 |
|-----------|------|------|------|------|
| Threat modeling ML systems | ✓ | | | |
| Evasion attacks (white-box) | Intro | Deep | | |
| Evasion attacks (black-box) | | Deep | | |
| Physical-world attacks | Intro | Deep | | |
| Poisoning and backdoor attacks | Intro | Deep | | |
| Model extraction / inference | Intro | | | |
| Adversarial training | Intro | | Deep | |
| Certified robustness | Intro | | Deep | |
| Differential privacy | | | Deep | |
| Federated learning security | | | Intro | |
| Prompt injection / jailbreaking | | | | Deep |
| Red-teaming LLMs | | | | Deep |
| Alignment failure analysis | | | | Deep |
| LLM watermarking | | | | Deep |
| Agentic AI security | | | | Intro |
| Original research / paper writing | Project | Project | Project | Primary |

---

## Track Philosophy

**1. Attack before defend.** Students spend two full courses on attacks before studying defenses in depth. You cannot design or evaluate defenses without deeply understanding the adversary.

**2. Adaptive adversary everywhere.** Every defense is evaluated against an adaptive attacker who knows the defense. Students internalize Carlini's evaluation guidelines as a reflex, not a checklist.

**3. Implementation over description.** Every major concept is coded from scratch. No plugging in off-the-shelf libraries without understanding the math underneath.

**4. Research trajectory.** By Year 2, students are producing original research. The seminar capstone targets publication-quality output.

---

## Computing Infrastructure

- University GPU cluster access (required for CS 6820 adversarial training assignments)
- Google Colab Pro as fallback for students without cluster access
- Per-assignment Docker containers with reproducible environments provided
- Standard stack: PyTorch ≥ 2.0, CleverHans, IBM ART, Opacus, HuggingFace Transformers, RobustBench, AutoAttack

---

## Ethics Policy

All four courses operate under a shared ethics framework:

1. Students sign a responsible disclosure / research ethics agreement before the first assignment in CS 6800 and reaffirm it at the start of each subsequent course.
2. All attack implementations are executed only against systems the student controls or controlled benchmark environments provided by the instructor.
3. Interaction with live production systems requires instructor approval and follows responsible disclosure norms.
4. Any genuine vulnerability discovered during coursework must be reported to the instructor immediately for guided responsible disclosure.

---

## Recommended Electives and Co-requirements

| Course | Relevance |
|--------|-----------|
| Deep Learning | Foundational to entire track |
| Cryptography | Supports differential privacy unit in CS 6820 |
| Natural Language Processing | Supports NLP attacks (CS 6810) and CS 7800 |
| Reinforcement Learning | Supports RL attack unit in CS 6810 |
| Computer Networks Security | Broader security context |
| Privacy in Machine Learning | Parallel to CS 6820 DP unit |

---

## Track Completion and Recognition

Students who complete all four courses (12 credits) with B+ or above in each may receive a Graduate Certificate in Adversarial Machine Learning (pending departmental approval). PhD students are encouraged to use their CS 7800 research paper as a foundation for their first publication.
