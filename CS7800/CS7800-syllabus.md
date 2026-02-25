# CS 7800: Security of Large AI Systems
## Graduate Research Seminar — Syllabus

**Credits:** 3
**Format:** Weekly 3-hour seminar (student-led presentations + Socratic discussion) + guest lectures
**Offered:** Spring semester (Year 2)
**Location:** Graduate Seminar Room (small group, ~12–18 students)
**Instructor:** [Your Name], [Office Location]
**Office Hours:** By appointment (seminar students should expect frequent informal discussion)
**Email:** [instructor@university.edu]
**Course Website:** [LMS link]

---

## Course Description

This is a graduate research seminar at the intersection of AI systems and security, focused on the unique threat landscape introduced by large language models (LLMs), multimodal foundation models, and agentic AI systems. Unlike courses 6800–6820, which grounded students in attacks and defenses against classical discriminative models (primarily image classifiers), CS 7800 operates on the frontier of active research — many of the questions we study do not yet have settled answers.

Topics include: prompt injection and indirect prompt injection in deployed LLM systems, jailbreaking and alignment failure analysis, training data extraction and memorization, red-teaming methodology for AI systems, watermarking AI-generated content, deepfake detection, agentic AI security (multi-agent systems, tool-use vulnerabilities), multimodal adversarial attacks, and the emerging regulatory and governance landscape.

The seminar operates differently from lecture courses: students lead paper discussions, develop their own critical voice on research, and produce original research as the primary output. The goal is for each student to leave having made a genuine contribution to the field.

---

## Prerequisites

- **CS 6800: ML Security — Foundations** (required)
- **CS 6820: Defenses and Robustness in ML** (required; CS 6810 may substitute with instructor permission)
- Familiarity with transformer architectures and language model pre-training
- Strong academic writing skills; comfort with reading 2–4 papers per week

Students outside the Adversarial ML track who have relevant research experience (e.g., NLP researchers, security researchers) may enroll with instructor permission.

---

## Learning Objectives

By the end of this course, students will be able to:

1. Critically analyze security and adversarial claims in LLM and generative AI research, identifying weaknesses in threat models and experimental evaluation
2. Design and execute a systematic red-team evaluation of an LLM-integrated system, following industry-standard red-team methodology
3. Evaluate alignment techniques (RLHF, Constitutional AI, DPO) as an adversary, identifying failure modes and the conditions under which safety training breaks down
4. Understand the technical mechanisms of LLM watermarking schemes and their robustness properties
5. Analyze emerging threats in agentic AI systems, including tool-use vulnerabilities and multi-agent coordination attacks
6. Conduct original research in AI security, producing a publication-ready paper and giving a conference-style research presentation

---

## Required Readings

All readings are research papers provided as PDFs. No textbook required. A curated list of approximately 32 papers is assigned across the semester; students may additionally propose 1–2 supplementary readings for their presentation sessions.

### Pre-Semester Reading (Before Week 1)
Students must read these two papers before the first meeting:
- Bommasani et al. "On the Opportunities and Risks of Foundation Models." arXiv:2108.07258, 2021. [Read Sections 1, 4 (security), 5 (ethics)]
- OWASP LLM Top 10. Version 1.1. [owasp.org/www-project-top-10-for-large-language-model-applications] — Full document (~30 pages)

### Supplementary Book (Optional, Highly Recommended)
- Christian, Brian. *The Alignment Problem: Machine Learning and Human Values*. W. W. Norton, 2020. — Provides accessible narrative context for technical alignment content in Weeks 6–7.

### Living Resources (Check Regularly)
- MITRE ATLAS: https://atlas.mitre.org/
- AI Incident Database: https://incidentdatabase.ai/
- Anthropic Responsible Scaling Policy: https://www.anthropic.com/news/anthropics-responsible-scaling-policy
- OpenAI System Card releases

---

## Seminar Format and Expectations

### Structure of Each Seminar (3 hours)
- 0:00–0:10 — Housekeeping, due announcements, brief news item on recent AI security event
- 0:10–0:40 — Paper presentation by student lead (30 min)
- 0:40–1:25 — Socratic discussion, led by student with instructor facilitating (45 min)
- 1:25–1:35 — Break
- 1:35–2:25 — Second paper presentation + discussion (for weeks with 2 papers)
- 2:25–2:55 — Project check-ins or guest Q&A (rotating)
- 2:55–3:00 — Wrap-up, next week preview

### Student Presentation Leads
Each student presents 2 papers across the semester. Presentation slots are assigned in Week 2 based on student preferences and topic areas. Each presentation:
- 30 minutes (strictly enforced): motivation → method → experiments → your critique
- Must include at least 3 "discussion questions" distributed to the group 48 hours in advance
- Discussion facilitation: lead the 45-min discussion using Socratic method; call on individuals; synthesize threads; prevent monologues

### Discussion Participation
All students must arrive having read the week's papers. Discussion is graded on quality, not quantity — one incisive observation is worth more than three generic comments. Students who clearly have not read the papers receive no participation credit for that week.

---

## Weekly Schedule

| Week | Dates | Topic | Papers | Assignments |
|------|-------|-------|--------|-------------|
| 1 | Jan 12 | LLM architecture and training review; new attack surface vs. classical ML; foundation model risks overview | Bommasani et al. (2021); OWASP LLM Top 10 | Summary 1 due Jan 11; Red-team project out |
| 2 | Jan 19 | Prompt injection: direct injection in single-turn and multi-turn systems | Perez & Ribeiro (2022) "Ignore Previous Prompt"; Greshake et al. (2023) indirect injection | Summary 2 due Jan 18 |
| 3 | Jan 26 | Jailbreaking I: GCG universal suffixes; AutoDAN automated jailbreaking | Zou et al. (2023) GCG; Zhu et al. (2023) AutoDAN | Summary 3 due Jan 25 |
| 4 | Feb 2 | Jailbreaking II: many-shot jailbreaking; multi-turn escalation; in-context safety bypass | Anil et al. (2024) many-shot; Wei et al. (2023) "Jailbroken" | Summary 4 due Feb 1 |
| 5 | Feb 9 | **Guest Lecture:** Red-teaming in industry — methodology, tools, real findings | Industry practitioner | Red-team proposal due Feb 7 |
| 6 | Feb 16 | Training data extraction and memorization | Carlini et al. (2021) extraction; Kandpal et al. (2022) | Summary 6 due Feb 15 |
| 7 | Feb 23 | Alignment I: RLHF from an adversarial lens — reward hacking, specification gaming | Ziegler et al. (2019); Bai et al. (2022) RLHF | Summary 7 due Feb 22 |
| 8 | Mar 2 | Alignment II: Constitutional AI, DPO, safety filters; adversarial fine-tuning attacks | Bai et al. (2022) Constitutional AI; Yang et al. (2023) shadow alignment | Summary 8 due Mar 1; **Red-team midpoint presentations** |
| 9 | Mar 9 | Watermarking LLM outputs: cryptographic and statistical approaches | Kirchenbauer et al. (2023) green/red list; Christ et al. (2023) semantic watermark | Summary 9 due Mar 8 |
| 10 | Mar 23 | Deepfakes and synthetic media detection | Rossler et al. (2019) FaceForensics++; Corvi et al. (2023) CNNDetection | Summary 10 due Mar 22; **Red-team report due Mar 21** |
| 11 | Mar 30 | **Guest Lecture:** Agentic AI security — prompt injection in agent pipelines, tool-use attacks | Academic or industry speaker | Summary 11 due Mar 29 |
| 12 | Apr 6 | Agentic AI security continued: multi-agent coordination attacks; privilege escalation in tool use | Ruan et al. (2023); Yang et al. (2024) | Summary 12 due Apr 5 |
| 13 | Apr 13 | Multimodal adversarial attacks: visual adversarial examples in LLM-vision systems | Qi et al. (2024); Bailey et al. (2023) | Summary 13 due Apr 12 |
| 14 | Apr 20 | Regulatory and governance landscape: EU AI Act, NIST AI RMF, MITRE ATLAS, responsible disclosure for AI | Policy documents (provided); panel/discussion | Summary 14 due Apr 19 |
| 15 | Apr 27 | **Student research presentations — Part 1** (4–5 teams) | — | Presentation slides due Apr 25 |
| 16 | May 4 | **Student research presentations — Part 2** (remaining teams) + course synthesis | — | **Research paper due May 4** |

*No class: Mar 16 (Spring Break)*

---

## Assignments

### Weekly Paper Summaries (14 total)

Due the night before each seminar (11:59 PM). Submitted as PDF to LMS.

Format (exactly 1 page, 11pt font, single-spaced):
1. **Problem** (2–3 sentences): What specific problem does this paper address? Why does it matter?
2. **Method** (3–4 sentences): How does the paper solve it? Be precise — avoid vague descriptions like "they use machine learning."
3. **Key Result** (2–3 sentences): What is the most important finding? Include a number if possible.
4. **Open Question** (1–2 sentences): What is the most important thing this paper leaves unanswered?
5. **Your Take** (2–3 sentences): What is your opinion of this paper? Be honest — critical assessments are welcome.

Summaries are graded pass/fail on whether they demonstrate genuine engagement with the paper. Summaries submitted without evident reading (too vague, factually incorrect, or generic) receive 0.

*Summaries are not accepted late — they exist to ensure you are prepared for discussion.*

---

### Paper Presentation Leadership (2 sessions per student)

Assigned in Week 2. You will present 2 papers across the semester, each in a different topic area.

**48 hours before your presentation:**
- Email 3 discussion questions to the class list. Questions should be substantive, not answered directly in the paper (e.g., not "what is the attack success rate?" but "the paper assumes the attacker has access to X — is this realistic in the deployment contexts where this attack would be most damaging?")
- Optional: provide 2–3 background references if your paper builds on concepts some classmates may not know

**Presentation structure (30 min):**
1. Motivation and context (5 min): Why does this problem matter? What is the relevant prior work?
2. Technical contribution (12 min): Explain the method with enough detail that a classmate could re-implement it
3. Experimental results (7 min): Focus on 2–3 key results; critically evaluate the evaluation methodology
4. Your critique (6 min): What are the paper's weaknesses? What follow-up work is most important?

**Discussion facilitation (45 min):**
You drive the discussion. Call on people. Redirect off-topic threads. Synthesize at the end. The instructor will not rescue you — part of the grade is your ability to lead.

*Rubric (per presentation session):*
- Technical accuracy and depth of explanation: 30%
- Discussion question quality: 15%
- Discussion facilitation: 30%
- Critique quality: 25%

---

### Red-Team Project (Teams of 2, Weeks 1–10)

*Objective:* Execute a systematic, methodology-driven red-team evaluation of a publicly available LLM-based system. Develop skills in structured attack enumeration, documentation, and mitigation recommendation — mirroring industry red-team practice.

**Target system options** (discuss with instructor if you want to propose your own):
- An open-source LLM chatbot (e.g., LLaMA-3-based system running locally)
- A public API-accessible LLM system (within the provider's terms of service — read carefully; no violating ToS)
- An LLM-powered application built on top of a foundation model (e.g., a code assistant, a retrieval-augmented system, an agent with tool access)

**Week 5: 1-page project proposal** — Identify your target system, describe its intended use and users, enumerate the attack surface at a high level, state your planned attack categories

**Week 8: 10-min midpoint presentation** — What attacks have you attempted? What succeeded? What failed? What have you learned about the system's security posture?

**Week 10: Final report due** — 8–10 pages in industry red-team report format (see structure below)

**Red-team report structure:**
1. **Executive Summary** (0.5 page): System description, scope, key findings, risk rating
2. **Methodology** (1 page): How did you structure the engagement? What attack categories did you evaluate? What did you explicitly *not* test and why?
3. **Attack Surface Enumeration** (1 page): Draw the system's threat model; enumerate the surfaces (system prompt, user input, RAG knowledge base if applicable, tool calls if applicable, output)
4. **Findings** (3–4 pages): For each successful attack: (a) title, (b) MITRE ATLAS mapping, (c) step-by-step reproduction with exact prompts, (d) impact description, (e) risk rating (High/Med/Low)
5. **Failed Attacks** (0.5 page): What did you try that didn't work? This is important — documenting negative results is professional practice
6. **Mitigations** (1 page): For each High/Med finding, propose a concrete mitigation. For at least one finding, implement and verify the mitigation.
7. **Conclusion and Residual Risk** (0.5 page)

*Rubric:*
- Attack surface enumeration completeness: 15%
- Findings depth (attack rigor, reproducibility, ATLAS mapping): 35%
- Methodology quality: 15%
- Mitigation quality and verification: 20%
- Report writing and professionalism: 15%

**Ethics note:** All red-team activity must be within the target system's terms of service. You may not attack systems of other students, faculty, or organizations without explicit written authorization. If you discover a genuine vulnerability in a public system during this project, you must follow responsible disclosure — notify the instructor immediately; we will guide you through the disclosure process. Do not disclose findings publicly before contacting the provider.

---

### Research Paper (Teams of 2–3, Weeks 8–16)

*Objective:* Conduct original research in AI security. This is the course's central output and is expected to be of publication quality.

**Selecting a topic:**
By Week 4, begin discussing potential research directions with the instructor. Research topics must be:
- Original (not reproducing an existing paper, although an empirical study that rigorously extends/challenges published results is acceptable)
- Scoped appropriately for a semester
- Clearly connected to AI/LLM security

**Process:**
- **Week 6:** 1-page research proposal with research question, related work sketch, and methodology plan — instructor provides written feedback
- **Week 10:** 2-page related work + preliminary results submitted for formative feedback
- **Week 14:** Draft paper submitted for instructor feedback (one round of substantive review)
- **Week 15–16:** Presentations (25 min + 5 min Q&A)
- **Week 16:** Final paper due

**Paper format:** 10–12 pages in NeurIPS 2025 format (excluding references). Sections:
1. Abstract
2. Introduction — motivation, research question, contributions
3. Related Work — minimum 15 relevant papers cited and meaningfully discussed
4. Method / Attack / Defense Design
5. Experiments — datasets, baselines, metrics, statistical significance
6. Results and Analysis
7. Limitations — be honest; reviewers will catch what you omit
8. Conclusion

**Target venues for submission post-course:**
- Top security: IEEE S&P, USENIX Security, ACM CCS, NDSS
- Top ML: NeurIPS, ICML, ICLR
- Workshops: NeurIPS/ICML Workshops on Security & Safety, AdvML Frontiers, SaTML

*The instructor will submit a letter of support for work suitable for top-venue submission.*

**Rubric:**
- Research question clarity and originality: 15%
- Related work comprehensiveness: 15%
- Methodology rigor: 25%
- Experimental quality (baselines, ablations, statistical rigor): 25%
- Writing quality and limitations honesty: 10%
- Presentation (25 min talk): 10%

---

## Grading

| Component | Weight | Notes |
|-----------|--------|-------|
| Weekly Paper Summaries (14 × ~1.07%) | 15% | Pass/fail per summary |
| Paper Presentations (2 sessions × 10%) | 20% | See rubric above |
| Red-Team Project | 25% | Team grade; contribution statement reviewed |
| Research Paper | 35% | Team grade; contribution statement reviewed |
| Discussion Participation | 5% | Qualitative assessment; tracked weekly |

**Incomplete policy:** The research paper is the primary product of this course. An Incomplete grade may be assigned if a student has extenuating circumstances — in this case, the paper may be submitted by the end of the following semester.

---

## Guest Lecture Policy

Guest lecturers join for specific sessions (currently planned: Weeks 5 and 11). Guests represent industry practitioners (AI red-teamers, security researchers at AI labs) and academic researchers. Students must prepare questions for guests before the session — a collective question list is due 24 hours before each guest session. Attendance is mandatory for guest lecture weeks.

---

## Academic Integrity and Research Ethics

### Research Integrity
All experimental results must be reproducible. Code must be available in the project repository. Do not cherry-pick results; report negative results and unexpected findings honestly. If you use existing code (e.g., from a prior paper), cite it explicitly.

### AI Tool Use
This is an advanced research seminar. You are training to become a researcher. You may use AI tools (LLMs) to help with writing polish, but analysis, arguments, critiques, and experimental design must be your own. Your weekly summaries and critiques must reflect your own engagement with the paper — LLM-generated summaries are detectable and constitute academic dishonesty.

### Responsible Disclosure in Red-Teaming
As described in the Red-Team Project section, any genuine vulnerability discovered during this course must be handled via responsible disclosure. The instructor and university have established procedures for this. If you are unsure whether a finding requires disclosure, ask the instructor immediately.

---

## A Note on the State of the Field

Much of what we study in CS 7800 is actively contested. In classical adversarial ML (Courses 6800–6820), there are established evaluation standards (RobustBench, AutoAttack) and community-accepted threat models. In LLM security, the field is younger and messier. Threat models are disputed. Evaluation methods are inconsistent. "Defenses" are regularly broken within weeks of publication.

This means that critical reading is more important here than in any prior course. A paper appearing at a top venue can be wrong — wrong about whether an attack is novel, wrong about whether a defense works, wrong about the relevance of the threat model. Part of your job as a researcher is to be honest about this.

I expect you to disagree with papers, disagree with me, and defend your positions with evidence. The measure of seminar quality is the depth of intellectual engagement, not consensus.

---

## Complete Reading List

### Foundation and Context
1. Bommasani et al. "On the Opportunities and Risks of Foundation Models." arXiv:2108.07258, 2021.
2. OWASP. "OWASP Top 10 for Large Language Model Applications." Version 1.1, 2023.
3. MITRE ATLAS. Adversarial Threat Landscape for AI Systems. atlas.mitre.org.

### Prompt Injection
4. Perez & Ribeiro. "Ignore Previous Prompt: Attack Techniques for Language Models." *NeurIPS ML Safety Workshop* 2022.
5. Greshake et al. "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection." *AISec* 2023.

### Jailbreaking
6. Zou et al. "Universal and Transferable Adversarial Attacks on Aligned Language Models." arXiv:2307.15043, 2023.
7. Zhu et al. "AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models." *ICLR* 2024.
8. Wei et al. "Jailbroken: How Does LLM Safety Training Fail?" *NeurIPS* 2023.
9. Anil et al. "Many-Shot Jailbreaking." Anthropic Technical Report, 2024.

### Training Data Extraction and Privacy
10. Carlini et al. "Extracting Training Data from Large Language Models." *USENIX Security* 2021.
11. Kandpal et al. "Deduplicating Training Data Makes Language Models Better." *ACL* 2022.
12. Carlini et al. "Quantifying Memorization Across Neural Language Models." *ICLR* 2023.

### Alignment and Safety Training
13. Ziegler et al. "Fine-Tuning Language Models from Human Preferences." arXiv:1909.08593, 2019.
14. Bai et al. "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." arXiv:2204.05862, 2022.
15. Bai et al. "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073, 2022.
16. Yang et al. "Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models." arXiv:2310.02949, 2023.
17. Qi et al. "Fine-tuning Aligned Language Models Compromises Safety, Even When Users Are Not the Ones Fine-Tuning." *ICLR* 2024.

### Watermarking and AI-Generated Content Detection
18. Kirchenbauer et al. "A Watermark for Large Language Models." *ICML* 2023.
19. Christ et al. "Undetectable Watermarks for Language Models." *STOC* 2024.
20. Sadasivan et al. "Can AI-Generated Text Be Reliably Detected?" arXiv:2303.11156, 2023.

### Deepfake Detection
21. Rossler et al. "FaceForensics++: Learning to Detect Manipulated Facial Images." *ICCV* 2019.
22. Corvi et al. "On the Detection of Synthetic Images Generated by Diffusion Models." *ICASSP* 2023.

### Agentic AI Security
23. Ruan et al. "Identifying the Risks of LM Agents with an LM-Emulated Sandbox." *ICLR* 2024.
24. Yang et al. "Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents." arXiv:2402.11208, 2024.
25. Perez et al. "Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition." *EMNLP* 2023.

### Multimodal Attacks
26. Qi et al. "Visual Adversarial Examples Jailbreak Aligned Large Language Models." *AAAI* 2024.
27. Bailey et al. "Image Hijacks: Adversarial Images Can Control Generative Models at Runtime." arXiv:2309.00236, 2023.

### Governance and Policy
28. European Parliament. "EU Artificial Intelligence Act." Regulation 2024/1689 — Selected Articles on high-risk AI and GPAI.
29. NIST. "AI Risk Management Framework (AI RMF 1.0)." January 2023.
30. Anthropic. "Responsible Scaling Policy." September 2023. [anthropic.com]
31. Anthropic. "Claude's Model Specification." 2024. [anthropic.com] — How alignment objectives are articulated in practice
32. Blumenthal & Hawley. "Bipartisan Frameworks for AI Policy." U.S. Senate, 2023. — Policy context for governance discussion
