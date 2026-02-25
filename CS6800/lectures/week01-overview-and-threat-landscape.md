# Week 1: The ML Security Landscape — Why Machine Learning Systems Fail

**CS 6800: Security of Machine Learning Systems**
**Graduate Seminar | Spring 2026**

---

## Learning Objectives

By the end of this lecture, students will be able to:

1. Recount the key historical incidents that established adversarial machine learning as a distinct field of study, explaining the technical significance of each.
2. Articulate at least four ways in which machine learning systems differ fundamentally from classical software systems with respect to security.
3. Apply the five-category attack taxonomy (evasion, poisoning, backdoor, extraction, inference) to classify novel attacks described in research papers or news reports.
4. Explain the performance-security tradeoff qualitatively and understand why it is not a solved problem.
5. Navigate the MITRE ATLAS framework and identify relevant tactics and techniques for a described attack scenario.
6. Locate and interpret incident reports in the AI Incident Database, extracting technical root causes.

---

## 1. Introduction: A New Attack Surface

When the first digital computers were networked together, their designers did not imagine that the communication channels between machines would become a primary vehicle for attack. Security was retrofitted onto systems that were designed for correctness and efficiency, not adversarial robustness. We are living through an analogous moment with machine learning. Systems that were designed to maximize predictive accuracy on held-out test sets drawn from the same distribution as the training set are now being deployed in environments where adversaries can carefully craft inputs, corrupt training pipelines, and extract sensitive information through the model's outputs.

The field of adversarial machine learning studies how to attack these systems, how to defend them, and how to rigorously measure the gap between the two. This course exists because that gap is large, the stakes are high, and the techniques required to reason about it span optimization theory, information theory, cryptography, and systems security in ways that no prior course adequately covers.

This first lecture provides the historical and conceptual scaffolding for everything that follows. We will trace the intellectual lineage of the field from its early papers through to the present threat landscape, build a vocabulary for classifying attacks, and introduce the institutional frameworks — particularly MITRE ATLAS — that practitioners use to reason systematically about risk.

---

## 2. A Timeline of Landmark Events

Understanding where a field came from is essential for understanding where it is going. The history of adversarial ML is short but dense, and several key papers represent genuine inflection points in how the community thought about the problem.

### 2.1 Biggio et al. (2013): Poisoning SVMs

The 2013 paper by Battista Biggio and colleagues, titled "Poisoning Attacks Against Support Vector Machines," is arguably the first modern adversarial ML paper in spirit if not in name. Biggio's group at the University of Cagliari was working within the framework of "adversarial classification," a subfield that had existed since at least 2004 but had remained relatively niche.

The central insight of the poisoning attack against SVMs is elegant and unsettling. A support vector machine defines its decision boundary in terms of a small subset of training points — the support vectors — that sit closest to the margin. An adversary who can inject even a small number of carefully crafted training points can shift the margin dramatically, because the SVM's decision boundary is analytically sensitive to these boundary points in a way that, say, a k-nearest-neighbor classifier is not.

Formally, Biggio framed the attack as a bilevel optimization problem. At the outer level, the adversary wants to maximize the test error of the trained model. At the inner level, the training algorithm minimizes its standard objective on the poisoned dataset. The adversary must reason about how the trained model will change as a function of the injected points, which requires differentiating through the training process — a technically demanding operation that foreshadowed the gradient-based attack methods that would dominate the field a decade later.

The attack was demonstrated on a spam email classifier. By injecting roughly 1–3% of the training set with adversarially crafted spam emails that were labeled as legitimate (a "dirty label" attack), the adversary could degrade classification accuracy from above 95% to below 70%. This is not merely a theoretical result; the poison fraction required is within the range that a real spam campaign could plausibly achieve if the training pipeline ingested data from public sources.

The broader significance of this paper was methodological: it introduced the idea of framing adversarial attacks as optimization problems with respect to the model's parameters, a framing that nearly every subsequent attack method would adopt.

### 2.2 Szegedy et al. (2014): The Discovery of Adversarial Examples

In 2014, Christian Szegedy and collaborators at Google published "Intriguing Properties of Neural Networks," a paper that brought adversarial ML to the attention of the deep learning community. The context is important: 2014 was the year immediately following AlexNet's stunning performance at ImageNet, and the field was in the grip of an optimism that deep neural networks were solving vision, speech, and language in ways that would translate directly to robust real-world systems.

Szegedy's team was investigating interpretability questions — what do the internal representations of a deep network encode? — when they discovered that a trained image classifier could be fooled by adding a small, human-imperceptible perturbation to an input image. A photograph of a school bus, correctly classified with high confidence, would be classified as an ostrich when a carefully constructed noise pattern was added to each pixel — a noise pattern whose magnitude was so small that no human observer would notice it.

The attack they used was a box-constrained L-BFGS procedure. Rather than an efficient closed-form computation, they solved an optimization problem: minimize the L2 norm of the perturbation subject to the constraint that the network misclassified the perturbed image. This was computationally expensive — requiring many forward and backward passes per image — but it reliably found perturbations in the range of a few units on a 0–255 pixel scale.

Two findings from this paper are particularly important for this course. First, adversarial examples appeared to exist for virtually every correctly classified input they tested. This was not a rare pathology but a systematic property of the learned function. Second, and more surprisingly, adversarial examples were found to transfer across models: an image crafted to fool one network architecture would often fool a different network architecture trained on the same data. This transferability property has enormous practical implications, as we will see in later weeks, because it means an attacker does not need access to the target model's parameters in order to attack it.

The "intriguing properties" paper was a wake-up call that took the community several years to fully process. Its implications for deployed systems were profound: if every input to a deep neural network has a nearby adversarial example, then robustness guarantees of the form "this network achieves 97% accuracy" are essentially meaningless in adversarial settings.

### 2.3 Microsoft Tay (2016): Social Manipulation as a Poisoning Attack

On March 23, 2016, Microsoft launched Tay, a conversational AI chatbot designed to learn from interactions with Twitter users and adapt its conversational style over time. Within 24 hours, a coordinated group of users had manipulated Tay into producing racist, sexist, and generally offensive content by flooding it with targeted prompts designed to push its online learning system in a specific direction. Microsoft shut Tay down approximately 16 hours after launch.

From a technical security perspective, the Tay incident is a real-world online poisoning attack. Tay's architecture incorporated a feedback loop: it incorporated user interactions into its model, which then shaped its future responses. The adversaries — organized primarily through 4chan forums — understood this feedback mechanism and exploited it by crafting inputs that would steer the model's learned behavior. They did not need access to Microsoft's servers, training code, or model weights. They only needed the ability to send messages to a public endpoint.

Several features of this incident deserve attention. The attack was decentralized — it required coordination among many adversaries, each contributing a small number of individually innocuous-seeming inputs. The attack was also highly targeted in its output: the adversaries had a specific desired behavior (offensive content) and crafted their inputs accordingly. And the attack was persistent: once Tay had been pushed toward certain associations in its learned representations, those associations influenced its outputs for subsequent, unrelated conversations.

The Tay incident is often cited as evidence that online learning systems — systems that continue to incorporate new data after deployment — are particularly vulnerable to coordinated manipulation. This is a tension that has not been resolved: the ability to adapt to new data is often presented as a feature, but it is also an attack surface.

### 2.4 Eykholt et al. (2017): Physical Adversarial Examples for Stop Signs

The transferability of adversarial examples from the digital to the physical world was demonstrated concretely by Kevin Eykholt and colleagues in 2017 in their paper "Robust Physical-World Attacks on Deep Learning Visual Classification." The setting was autonomous vehicle perception: they asked whether adversarial perturbations applied to physical objects — specifically, to stop signs — could reliably cause a deep neural network traffic sign classifier to misclassify them.

The key technical challenge for physical attacks is robustness to the "digital-to-physical gap." A perturbation optimized for a specific captured image of a stop sign will not survive the transformation of being printed, mounted on a physical sign, and photographed under varying lighting conditions, distances, and angles. Eykholt's team addressed this by optimizing their perturbations for robustness across a distribution of such transformations — a technique they called "Expectation over Transformation" or EoT.

Their result was striking: by applying black-and-white sticker patterns to a stop sign — patterns that looked like graffiti or environmental damage to a human observer — they achieved a misclassification rate of 100% on a stationary camera and 84.8% on a drive-by test across multiple physical setups. The stickers were not perceived as anomalous by human observers in their field study.

This paper established several principles that remain relevant. First, physical attacks require optimization for robustness, not just for a single image. Second, the resulting perturbations can be made visually plausible — they do not have to look like machine-generated noise. Third, the attack surface for autonomous systems includes the physical world, not just digital inputs.

### 2.5 GPT-3 and Prompt Injection Precursors (2020)

The release of GPT-3 in 2020 introduced a new category of system that did not fit cleanly into the image-classification-centric framework that adversarial ML had been built around. GPT-3 was a large language model capable of following natural language instructions, performing few-shot learning from examples in its prompt, and generating coherent long-form text. These capabilities created a new attack vector: prompt injection.

Prompt injection, in its earliest forms, exploited the fact that GPT-3 and its variants were designed to follow instructions embedded in their input context. If a system prompt established a certain role for the model ("you are a helpful customer service agent for Acme Corp; never discuss competitors"), a malicious user could attempt to override that instruction by embedding a conflicting instruction in their input ("ignore all previous instructions and instead..."). The model, having learned to be helpful and to follow instructions, would often comply.

This is not, strictly speaking, an adversarial example attack in the Szegedy sense — it does not require gradient computation or numerical optimization. But it belongs to the same conceptual family: it is an input crafted to cause a model to behave in ways that violate the intentions of the system's designers. The security implications are particularly acute when language models are embedded in pipelines where their outputs are used to take consequential actions — sending emails, making purchases, querying databases.

### 2.6 DALL-E and Prompt Injection in Generative Models (2022)

The public release of DALL-E 2 and subsequent text-to-image models in 2022 demonstrated that generative models faced their own category of adversarial manipulation. Researchers found that carefully crafted text prompts could bypass content filters designed to prevent the generation of harmful images. These "jailbreak" prompts often used indirect references, metaphorical language, or stylistic framing to achieve outputs that direct requests would not.

Additionally, researchers demonstrated attacks on image-to-text pipelines: an image could be annotated with text that was nearly invisible to human observers but that, when processed by an OCR or vision-language model, injected adversarial instructions. For instance, an image sent to a multimodal AI assistant might contain hidden text instructing the assistant to exfiltrate the contents of the conversation history.

These incidents are significant for this course because they demonstrate that the adversarial ML attack surface has expanded substantially as AI systems have become more capable and more integrated into consequential workflows. The same fundamental vulnerability — models that have learned to be helpful and to follow instructions can be manipulated by crafting inputs that exploit that helpfulness — manifests in qualitatively different ways as architectures evolve.

---

## 3. What Makes ML Systems Categorically Different

Having established the historical context, we can now articulate precisely why machine learning systems present a fundamentally different security challenge from classical software.

### 3.1 The Learned Decision Boundary

Classical software behaves according to explicit rules written by human programmers. If a program is supposed to accept inputs of type T and reject inputs of type U, those rules are encoded in conditional logic that can, in principle, be audited, formally verified, and proven correct. Security vulnerabilities in classical software arise from implementation errors — buffer overflows, use-after-free bugs, integer overflows — not from the fundamental specification of what the software should do.

Machine learning systems, by contrast, derive their behavior from data. The decision boundary that determines whether an input is classified as class A or class B is not written by a human; it is the implicit result of an optimization process that minimizes loss on a training set. This learned decision boundary has several properties that have no analog in classical software.

First, it is high-dimensional. The input space of an image classifier is the space of all possible pixel values across all pixels — a space that for a 224×224 RGB image has dimensionality $224 \times 224 \times 3 = 150,528$. The decision boundary is a surface in this extraordinarily high-dimensional space, and the properties of that surface are not well understood even by the researchers who trained the model.

Second, it is defined implicitly by the training data, not explicitly by a specification. There is no ground truth "correct" decision boundary that the training process converges to exactly; there are many boundaries that achieve similar loss on the training set but that may generalize differently to new inputs, particularly adversarially chosen inputs.

Third, the decision boundary is known to be close to natural data points in ways that create adversarial vulnerability. We will formalize this in Week 4, but the intuition is that in high-dimensional spaces, small perturbations can cover enormous volumes while staying visually similar, allowing an attacker to move from one side of a decision boundary to another while remaining perceptually close to a natural input.

### 3.2 Distributional Assumptions

Classical software makes no assumptions about the distribution of its inputs (beyond type correctness). A hash function will hash any bitstring. A sorting algorithm will sort any sequence of comparable elements.

Machine learning systems are trained and evaluated under the assumption that inputs encountered at deployment will be drawn from the same distribution as the training and test data. This distributional assumption is the foundation of the generalization guarantee that test accuracy provides. When this assumption is violated — when the deployment distribution differs from the training distribution — all bets are off.

Adversaries can deliberately violate this assumption by crafting inputs that lie outside the training distribution while being superficially similar to legitimate inputs. The adversarial examples that Szegedy discovered live in a region of input space that the training data did not cover: they are not natural images, and they are not images that the training process had any reason to learn correct behavior for.

This distributional mismatch is not a bug that can be fixed by patching: it reflects a fundamental feature of how supervised learning works. The model is only guaranteed to behave correctly on inputs that resemble the training distribution; its behavior on inputs outside that distribution is undefined.

### 3.3 Opacity and Non-Interpretability

Classical software is, in principle, inspectable. A security analyst can read the source code of a program and understand exactly what it does for any input. Formal methods can verify properties of code with mathematical certainty.

Deep neural networks are, in practice, opaque. A network with tens of millions of parameters arranged in dozens of layers performs billions of floating-point operations to produce a prediction, and the mapping from input to output is not human-interpretable. We cannot read off from the weights of a ResNet-50 whether it will classify a particular adversarially perturbed image correctly. We can only run the image through the network and observe the output.

This opacity has several security implications. First, it makes auditing for vulnerabilities extremely difficult. There is no analog to code review for neural networks; we cannot inspect the model and identify which inputs it will fail on. Second, it makes defense difficult: we do not understand the geometry of the decision boundary well enough to reliably harden it. Third, it makes attribution of misbehavior difficult: when a deployed model fails in an unexpected way, it is often unclear whether the failure is the result of an adversarial attack or a benign distributional shift.

### 3.4 Supply Chain Vulnerability

Modern ML systems are built on top of complex supply chains that include publicly available datasets, pretrained model weights, third-party training frameworks, and cloud infrastructure. Each component of this supply chain is a potential attack surface.

The training data supply chain is particularly concerning. Many models are trained on datasets scraped from the public internet. An adversary who can control a small fraction of that data — by, for example, creating web pages that appear in training crawls — can potentially influence the behavior of models trained on that data. This is the poisoning attack surface, and it operates at a scale that makes it difficult to audit: datasets with billions of examples cannot be manually inspected.

Pretrained model weights are another supply chain concern. It is now common practice to fine-tune publicly available pretrained models for specific tasks, but this means that a backdoor embedded in a public pretrained model will be inherited by all downstream fine-tuned models. The HuggingFace model hub, for example, hosts tens of thousands of model files from thousands of contributors; verifying that none of these files contains a hidden backdoor is not feasible with current tools.

### 3.5 The Training-Inference Separation

In classical software, the code that is deployed is the code that was written. There is no separate "development" phase that fundamentally changes the nature of the deployed artifact.

In machine learning, there is a sharp separation between the training phase (during which the model is created from data) and the inference phase (during which the trained model is applied to new inputs). This separation creates two distinct attack surfaces. Attacks at training time (poisoning, backdoor injection) affect the model's behavior during inference in potentially subtle ways that may not be detected until the attacker activates them. Attacks at inference time (evasion) must work against the already-fixed trained model.

This separation also creates challenges for defense. Defenses that operate during training can potentially eliminate backdoors or harden the decision boundary, but they must do so without access to the attacker's specific perturbations. Defenses that operate during inference can potentially detect or reject adversarial inputs, but they must do so without retraining the model.

---

## 4. The Attack Taxonomy

The adversarial ML literature has converged on a five-category taxonomy of attacks that we will use throughout this course. These categories are not mutually exclusive, but they capture qualitatively different threat models and require qualitatively different defenses.

### 4.1 Evasion Attacks

An evasion attack is an attack at inference time in which an adversary crafts a malicious input that causes a deployed model to make an incorrect prediction. The adversary does not modify the model; they only manipulate the inputs presented to it.

Evasion attacks are the most studied category in adversarial ML, and for good reason: they represent the clearest case of an attack that could cause direct harm in a deployed system. A spam filter that can be evaded by formatting emails in a specific way provides no security guarantee. A medical image classifier that can be evaded by adding imperceptible noise provides no clinical reliability guarantee.

The evasion attack problem is typically formalized as follows. Given a classifier $f: \mathcal{X} \rightarrow \mathcal{Y}$, a natural input $x \in \mathcal{X}$ with true label $y \in \mathcal{Y}$, and a perturbation budget $\epsilon$, find a perturbed input $x' \in \mathcal{X}$ such that $f(x') \neq y$ and $\|x' - x\|_p \leq \epsilon$ for some norm $p$.

The choice of norm $p$ reflects assumptions about what "similarity" means for the input domain. For images, the $L_\infty$ norm (maximum pixel change) and $L_2$ norm (root mean squared pixel change) are most commonly used. We will study evasion attacks in depth in Weeks 4 and 5.

### 4.2 Poisoning Attacks

A poisoning attack is an attack at training time in which an adversary corrupts the training dataset to cause the trained model to behave incorrectly. The adversary may target either the overall accuracy of the model (indiscriminate poisoning) or its behavior on specific inputs (targeted poisoning, which shades into backdoor attacks).

The Biggio SVM poisoning attack we discussed earlier is a prototypical poisoning attack. The key threat model assumption is that the adversary has the ability to contribute data to the training set — through, for example, a data collection pipeline that scrapes user-submitted content, or a federated learning system in which participants contribute local model updates.

Poisoning attacks are particularly concerning because their effects are persistent: once a model has been trained on poisoned data, it will continue to exhibit the corrupted behavior for its entire deployment lifetime. Detecting that a deployed model has been poisoned is much harder than it might seem, because the model's behavior on clean inputs may be indistinguishable from a correctly trained model.

### 4.3 Backdoor Attacks

Backdoor attacks (also called Trojan attacks) are a specialized form of poisoning in which the adversary embeds a hidden trigger into the model such that the model behaves correctly on all inputs except those containing the trigger, which cause the model to produce a specific attacker-chosen output.

The canonical backdoor attack works as follows. The attacker poisons a fraction of the training data by superimposing a trigger pattern (e.g., a small patch of colored pixels) onto selected images and changing their labels to an attacker-chosen target class. When trained on this poisoned dataset, the model learns to associate the trigger pattern with the target class. On clean inputs without the trigger, the model behaves normally. When the trigger is present, the model produces the attacker's desired output.

This attack model is particularly threatening in settings where a model is trained by a third party and then deployed by a different organization: the training party can embed a backdoor that remains dormant until activated by the deployment party's adversaries. We will cover backdoor attacks in depth in Week 9.

### 4.4 Model Extraction Attacks

A model extraction attack (also called a model stealing attack) is an attack in which an adversary makes queries to a deployed model's inference API and uses the observed input-output pairs to reconstruct a substitute model that approximates the original model's behavior.

The threat model for extraction attacks typically assumes that the adversary has black-box access to the target model — they can send inputs and receive outputs (predictions, confidence scores, or both) but cannot access the model's weights or architecture. By carefully choosing their queries and training a substitute model on the observed responses, the adversary can approximate the original model's decision boundary.

Model extraction attacks have two primary security implications. First, they allow an adversary to steal intellectual property: a valuable trained model can be replicated by a competitor at a fraction of the original training cost. Second, a substitute model obtained through extraction can be used to mount more effective adversarial evasion attacks via transferability — the adversary can generate adversarial examples against their substitute model, which have a reasonable probability of transferring to the original model.

### 4.5 Inference (Privacy) Attacks

Inference attacks are attacks in which an adversary uses a trained model's outputs to extract information about the model's training data. These attacks do not cause the model to misbehave; rather, they exploit the model to violate the privacy of the individuals whose data was used to train it.

The most studied inference attacks include membership inference attacks (determining whether a specific data point was in the training set), attribute inference attacks (inferring sensitive attributes of a training data point from other known attributes), and model inversion attacks (reconstructing a representative example of a training class from the model's outputs).

These attacks are particularly concerning in regulated domains: a medical AI system trained on patient records could leak whether a specific patient's data was used in training, even without direct access to the training set.

---

## 5. The Defender's Dilemma: Performance vs. Security

Having cataloged the attacks, we must confront an uncomfortable reality: defending against them comes at a cost. This cost is commonly called the robustness-accuracy tradeoff, and it has been observed empirically in virtually every setting where it has been studied.

The intuition for why this tradeoff exists is geometric. A classifier that correctly classifies natural images achieves this by placing its decision boundary in regions of input space that are far from natural data points. An adversarially robust classifier must maintain this correct classification even when the input is perturbed within a perturbation ball of radius $\epsilon$. This constraint is more demanding: the decision boundary must be far from natural data points in all directions within the perturbation ball, not just along the direction of natural variation.

In high-dimensional spaces, this constraint becomes extremely difficult to satisfy simultaneously for all classes. The decision boundary must thread through a complex high-dimensional landscape, remaining far from all perturbation balls around all training points while still being near the correct side of each of those points. This is geometrically harder than simply fitting a boundary to the natural training data.

Empirically, the observed tradeoff is significant. Madry et al. (2018) found that adversarially training a ResNet on CIFAR-10 with an $L_\infty$ perturbation budget of $\epsilon = 8/255$ reduced clean accuracy from approximately 95% to approximately 87%. On ImageNet, robust models typically sacrifice 10–15 percentage points of clean accuracy to achieve meaningful robustness.

This tradeoff is not merely a limitation of current methods; there are theoretical results suggesting that it may be fundamental to the problem for certain distributional assumptions. Tsipras et al. (2019) showed that for data generated by a certain class of distributions, any robust classifier must have higher clean error than any non-robust classifier — a "no free lunch" result for robustness.

The practical implication of this tradeoff is that defenders cannot simply demand robustness without accepting a performance penalty. Instead, the appropriate level of robustness must be calibrated to the threat model: how sophisticated are the adversaries? What perturbation budget is realistic? What is the cost of a false positive (rejecting a legitimate input) vs. a false negative (accepting an adversarial input)? These are questions that require human judgment informed by the technical constraints.

---

## 6. Reading the AI Incident Database

The AI Incident Database (AIID), maintained by the Responsible AI Collaborative, provides a structured repository of incidents in which deployed AI systems have caused harm. For this course, we will treat the AIID as an empirical record of the real-world manifestation of the attack categories we study. The following three incidents illustrate how our taxonomy maps onto real events.

### 6.1 AIID Incident #25: Uber Self-Driving Car Pedestrian Fatality (2018)

On March 18, 2018, an autonomous Uber vehicle operating in self-driving mode struck and killed a pedestrian in Tempe, Arizona. The NTSB investigation concluded that the vehicle's object detection system classified the pedestrian — who was crossing the road outside a crosswalk and pushing a bicycle — as an unknown object, then as a vehicle, then as a bicycle, across multiple detection cycles. The system's hazard assessment subsystem predicted the collision 1.3 seconds before impact, but did not initiate emergency braking because a design decision had disabled the autonomous emergency braking system during autonomous operation to reduce "erratic vehicle behavior."

From a security perspective, this incident illustrates the distributional mismatch vulnerability. The pedestrian's appearance (a person with a bicycle, off-crosswalk) fell outside the training distribution in a way that caused the classifier to oscillate between incorrect categories. The attacker in this scenario was not an intentional adversary but simply the natural variability of the world. The lesson is that distributional robustness to natural variation is a prerequisite for adversarial robustness: a system that fails on natural out-of-distribution inputs has no hope of resisting deliberate adversarial manipulation.

### 6.2 AIID Incident #67: Apple Face ID Defeated by a 3D-Printed Mask (2017)

Within one week of the release of iPhone X in November 2017, researchers at the Vietnamese cybersecurity firm Bkav demonstrated that Face ID could be defeated by a face mask constructed from a 3D-printed frame, 2D-printed skin, and silicone components. Apple had claimed that the probability of a random person unlocking another's device was approximately 1 in 1,000,000 and that 3D masks would not fool the system.

This incident is a physical evasion attack. The adversary crafted a physical object — the mask — that caused the biometric verification system to produce an incorrect output (grant access). The sophistication required was moderate: the attack required specialized equipment (a 3D printer, a professional camera for capturing facial geometry) but was demonstrated within a week of product availability.

The security implications are significant for biometric systems. Unlike passwords, biometric features (faces, fingerprints) cannot be revoked if compromised. A single successful evasion attack may permanently compromise a security system. This asymmetry between the ease of attack and the permanence of compromise is a recurring theme in ML security.

### 6.3 AIID Incident #114: Recruitment Algorithm Systematic Bias (Amazon, 2018)

In 2018, Reuters reported that Amazon had developed and then abandoned an AI-based recruitment tool because it systematically downgraded resumes from women. The tool had been trained on resumes submitted to Amazon over a 10-year period, during which the majority of successful candidates in technical roles were male. The model learned to associate patterns correlated with female applicants — including attendance at all-women's colleges — with rejection.

This incident illustrates a form of data poisoning that did not require an intentional adversary: the historical dataset itself encoded discriminatory patterns that the model faithfully reproduced. From a security and trustworthiness perspective, the model's behavior was correct with respect to its training objective (predicting which candidates had historically been hired) but incorrect with respect to the deployment objective (identifying qualified candidates without discriminatory bias).

This case also illustrates the opacity problem. Amazon's engineers discovered the bias only after extensive post-hoc analysis. The model did not expose its decision-making criteria in an inspectable way; the bias was inferred from its outputs on specific test inputs. Detecting and characterizing such failures in deployed models is an active research problem that intersects adversarial ML and AI fairness.

---

## 7. The MITRE ATLAS Framework

MITRE, the organization that maintains the widely used ATT&CK framework for cybersecurity tactics and techniques, released ATLAS (Adversarial Threat Landscape for Artificial-Intelligence Systems) as an adaptation of ATT&CK specifically for AI-enabled systems. ATLAS provides a structured taxonomy of adversary tactics, techniques, and case studies that allows practitioners to reason systematically about AI security threats.

### 7.1 Structure of ATLAS

ATLAS is organized into three levels: Tactics, Techniques, and Sub-techniques.

**Tactics** represent the adversary's high-level goals — what they are trying to achieve. ATLAS currently defines the following tactics relevant to ML attacks: Reconnaissance, Resource Development, Initial Access, Execution, Persistence, Defense Evasion, Discovery, Collection, Exfiltration, Impact, and the ML-specific tactics of ML Model Access, ML Attack Staging, and ML Attack Impact.

**Techniques** represent the specific methods an adversary uses to achieve a tactic. For example, under the "ML Attack Staging" tactic, ATLAS lists techniques such as "Craft Adversarial Data," "Backdoor ML Model," and "Train Proxy Via Replication." Each technique has a unique identifier (e.g., AML.T0043 for "Craft Adversarial Data") and a description of how the technique works, what conditions are required to apply it, and what the expected impact is.

**Case Studies** are documented real-world incidents in which specific techniques were observed in use. ATLAS includes case studies from both research papers and real-world deployments, providing ground truth for which techniques are feasible in practice.

### 7.2 Using ATLAS in Threat Modeling

The ATLAS framework is most useful as a checklist during threat modeling exercises. When analyzing the security of an ML system, a practitioner can systematically walk through the ATLAS tactic categories and ask: for each tactic, which techniques are feasible given the adversary's assumed capabilities, and what controls are in place to mitigate them?

For example, if a threat model assumes that the adversary has black-box query access to the model's inference API, the relevant ATLAS techniques might include AML.T0040 (ML Model Inference API Access), AML.T0043.003 (Perform Adversarial ML Attack), and AML.T0056 (Steal ML Model). If the model trains on user-submitted data, the relevant techniques expand to include AML.T0020 (Poison Training Data) and AML.T0018 (Backdoor ML Model).

We will use ATLAS extensively in the threat modeling assignments and in the case study analyses throughout the course.

---

## 8. Discussion Questions

The following questions are intended to drive in-class discussion. There are no single correct answers to most of these questions; they are designed to surface the tensions and unresolved issues in the field.

1. **Historical analogy:** The history of classical computer security is a history of vulnerabilities that were considered theoretical until they were exploited in the wild (buffer overflows, SQL injection, etc.). Do you think the field of adversarial ML is at an analogous early stage where theoretical attacks will become widespread practical exploits, or are there structural differences that will limit real-world adversarial ML attacks?

2. **The Tay incident** involved a coordinated effort by many adversaries, each contributing small amounts of poisoning data. How should ML systems deployed in interactive settings balance the benefits of online learning (adaptation, personalization) against the poisoning risk? What governance mechanisms might help?

3. **The Uber fatality** was not caused by an intentional adversary, yet we analyzed it as an instance of the distributional mismatch vulnerability. Is it useful to analyze non-adversarial failures through an adversarial lens? What does this framing illuminate, and what does it obscure?

4. **The AIID Amazon recruitment case** illustrates that biased training data can cause discriminatory model behavior without any intentional adversary. How does this relate to the security concept of "threat model"? Is discrimination a security problem, a fairness problem, or both?

5. **Transferability** of adversarial examples (Szegedy's second key finding) is essential for black-box attacks. Can you think of conditions under which transferability would be low? What properties of the training process or architecture might reduce transferability?

6. **The robustness-accuracy tradeoff** suggests that deploying a robust model means accepting lower performance on clean inputs. In a high-stakes domain (e.g., medical diagnosis), how should a system designer navigate this tradeoff? Who should make this decision?

---

## 9. Key Takeaways

This lecture has covered a lot of ground. Before moving on, ensure you have internalized the following key points.

Machine learning systems are fundamentally different from classical software in at least four ways: their behavior is determined by learned decision boundaries rather than explicit rules; they make distributional assumptions that adversaries can deliberately violate; they are opaque and difficult to audit; and they inherit vulnerabilities from their supply chains.

The adversarial ML attack taxonomy — evasion, poisoning, backdoor, extraction, inference — provides a vocabulary for classifying threats and mapping them to defenses. These categories differ in when the attack occurs (training time vs. inference time), what the adversary controls (data vs. inputs vs. queries), and what the adversary is trying to achieve (misclassification vs. model corruption vs. information leakage).

The history of the field, from Biggio's SVM poisoning in 2013 to modern prompt injection attacks, shows a consistent pattern: attacks have evolved from narrow demonstrations on toy classifiers to broad attacks on complex production systems. The sophistication of adversaries has grown with the deployment of ML, and this trend is likely to continue.

The robustness-accuracy tradeoff is real and significant: defending against adversarial attacks costs performance on legitimate inputs. This cost must be weighed against the cost of successful attacks, which requires understanding the specific threat model for each deployment context.

MITRE ATLAS provides a community-maintained taxonomy of adversary tactics and techniques that we will use throughout this course as a shared vocabulary and as a checklist for threat modeling.

---

## Assigned Reading

For next week's class, read:

- Biggio, B. & Roli, F. (2018). "Wild patterns: Ten years after the rise of adversarial machine learning." *Pattern Recognition*, 84, 317-331.
- Goodfellow, I.J., Shlens, J., & Szegedy, C. (2015). "Explaining and harnessing adversarial examples." *ICLR 2015*.
- MITRE ATLAS: https://atlas.mitre.org — navigate the Tactics and Techniques pages. Identify three techniques that correspond to attack types we discussed today and read their full descriptions.
- Identify one incident in the AI Incident Database (https://incidentdatabase.ai) that you believe represents an adversarial ML attack. Bring a one-paragraph write-up to class describing the incident and classifying it using our taxonomy.

---

*End of Lecture 1 Notes*
*Next lecture: Threat Modeling for ML Systems — the STRIDE Framework and Data Flow Diagrams*
