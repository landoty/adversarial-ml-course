# Week 2: Threat Modeling for ML Systems

**CS 6800: Security of Machine Learning Systems**
**Graduate Seminar | Spring 2026**

---

## Learning Objectives

By the end of this lecture, students will be able to:

1. Define threat modeling and explain why it requires adaptation when applied to ML systems versus traditional software.
2. Apply all six STRIDE categories to specific components of an ML system, generating concrete, technically precise threat statements.
3. Draw a Data Flow Diagram (DFD) for a multi-component ML system, correctly identifying trust boundaries, data stores, processes, and external entities.
4. Construct a complete STRIDE threat model table for a given ML system, including threat statement, affected component, potential impact, and candidate mitigations.
5. Map STRIDE threat categories to corresponding MITRE ATLAS tactics and techniques.
6. Articulate the difference between white-box, gray-box, and black-box adversary models, and explain why the adversary's capability assumptions are the most consequential choices in a threat model.
7. Describe a defense-in-depth strategy for an ML system and explain why no single control is sufficient.

---

## 1. What Is Threat Modeling and Why Do We Do It?

Threat modeling is a structured process for identifying, prioritizing, and addressing security risks in a system before those risks are exploited. The output of a threat modeling exercise is typically a document that answers four questions: What are we building? What can go wrong? What are we going to do about it? Did we do a good enough job?

The value of threat modeling comes from forcing system designers to think adversarially early in the development process, when changes are cheap. A vulnerability discovered during design might require a whiteboard revision; the same vulnerability discovered after deployment might require an emergency patch, a breach notification, and regulatory scrutiny. This asymmetry — the earlier you find problems, the cheaper they are to fix — is the same logic that drives unit testing in software development, and it is even more compelling for security because adversaries specifically exploit vulnerabilities that designers did not anticipate.

Traditional software threat modeling has a rich methodological literature. Microsoft's Security Development Lifecycle, OWASP's Threat Modeling Cheat Sheet, and various formal methods all provide guidance for analyzing conventional software architectures. The STRIDE framework, developed at Microsoft in the 1990s and formalized by Loren Kohnfelder and Praerit Garg, is one of the most widely used threat categorization schemes.

However, applying threat modeling to ML systems requires significant adaptation, for reasons that flow directly from the properties we discussed in Lecture 1. The learned decision boundary, distributional assumptions, opacity, and training-inference separation of ML systems create attack surfaces that are not present in classical software and that are not captured by threat modeling frameworks designed for classical architectures.

### 1.1 Why ML Threat Modeling Is Different

Consider a classical web application. Its components — the web server, application logic, database — have well-defined interfaces and behaviors that can be specified precisely. A threat modeler can enumerate the data flows, identify where trust boundaries are crossed, and apply STRIDE systematically. The threats that emerge — SQL injection through the web interface, privilege escalation through the application logic, data exfiltration through the database — correspond to vulnerabilities in the implementation of explicit specifications.

An ML system has all of these classical attack surfaces, plus several that are unique to the learned nature of its core component. The model itself is not an explicitly programmed logic layer; it is a statistical artifact whose behavior is defined by its training data and training procedure. This creates threats that have no classical analog:

- An adversary can attack the model by attacking the training data (poisoning), which is a kind of attack on the model's "source code" that classical threat models have no category for.
- An adversary can attack the model at inference time by crafting inputs that exploit the geometry of its learned decision boundary (evasion), which requires understanding the model's mathematical structure rather than its implementation.
- An adversary can extract a copy of the model through its inference API (model extraction), a kind of intellectual property theft that is specific to the model-as-artifact.
- An adversary can use the model's outputs to infer sensitive information about the training data (membership inference), a privacy attack that is unique to learned models.

Furthermore, ML systems often have complex operational pipelines that include multiple components, each with its own trust assumptions, and the security of the overall system depends on the security of every link in the chain.

### 1.2 The Scope of an ML System

Before we can model threats, we must define the scope of the system we are analyzing. For an ML system, this scope typically includes:

1. **Data collection infrastructure**: How is training data obtained? Web scraping, user submissions, purchased datasets, internal databases?
2. **Data preprocessing pipeline**: How is raw data transformed into training-ready features? Data cleaning, normalization, augmentation, labeling?
3. **Training infrastructure**: Where and how is the model trained? Cloud infrastructure, dedicated hardware, federated learning?
4. **Model registry/artifact store**: Where are trained model artifacts stored? How are they versioned and access-controlled?
5. **Inference service**: How is the model exposed to users? REST API, batch processing, embedded in a client application?
6. **Monitoring and feedback pipeline**: How is the model's production behavior observed? Are predictions or inputs logged? Is there a retraining trigger?

Each of these components can be a threat target, and the data flows between them cross trust boundaries that must be explicitly identified.

---

## 2. The STRIDE Framework Applied to ML Systems

STRIDE is a mnemonic for six categories of security threats: Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, and Elevation of Privilege. Each category captures a qualitatively different way in which a system can be compromised. We will discuss each in turn, giving both the classical definition and ML-specific examples.

### 2.1 Spoofing

**Classical definition:** Spoofing threats involve an adversary pretending to be something they are not — impersonating a user, a service, or a data source — in order to gain access or manipulate behavior.

**ML-specific manifestations:** In ML systems, spoofing can occur at several levels. An adversary might spoof a legitimate data source to inject poisoned training data into a pipeline that expects data from a trusted provider. In federated learning settings, a malicious participant might spoof a legitimate client identity to submit crafted gradient updates. An adversary who has performed model extraction might spoof the model's API to conduct man-in-the-middle attacks on dependent services.

A particularly interesting ML-specific spoofing threat is adversarial input spoofing: crafting an input that causes the model to "see" something different from what a human observer sees. An adversarial example that is classified by the model as "benign email" but perceived by a human reader as clearly malicious is, in a meaningful sense, spoofing the model's perception. The adversarial image that a model confidently classifies as a "panda" when humans see "noise added to a gibbon" is the model being spoofed about the identity of the input.

In the context of biometric authentication systems, adversarial evasion attacks on the face recognition component represent a spoofing threat: the attacker is using a crafted physical object (mask, printed photo) to impersonate a legitimate user.

### 2.2 Tampering

**Classical definition:** Tampering threats involve an adversary modifying data or code without authorization to alter the system's behavior.

**ML-specific manifestations:** Tampering is perhaps the most diverse category in ML-specific threat modeling, because there are so many components that an adversary might tamper with.

Training data tampering (poisoning) involves modifying training data before or during the training process to corrupt the resulting model. This can be achieved by injecting new poisoned examples, modifying existing examples, or altering labels. The impact can range from reducing overall accuracy (indiscriminate poisoning) to embedding specific misbehaviors (targeted poisoning and backdoors).

Model artifact tampering involves modifying a trained model's weights or architecture after training. This threat is relevant in supply chain scenarios where a pre-trained model is obtained from an untrusted source, or where a model artifact in storage can be accessed by an adversary. A particularly insidious form of model tampering is the insertion of a backdoor into a trained model: rather than modifying training data, the adversary directly modifies the model's weights to encode a trigger-activated behavior. This is more sophisticated but requires access to the model artifact rather than the training pipeline.

Inference input tampering is the evasion attack: modifying the input to the model at inference time to cause misclassification. From the system's perspective, the input data is being tampered with before it reaches the model.

Gradient tampering is specific to federated learning: a malicious participant can submit tampered gradient updates (rather than the gradients computed on their legitimate local data) to influence the direction of the global model update.

### 2.3 Repudiation

**Classical definition:** Repudiation threats involve an adversary taking an action and then denying having taken it, potentially because the system lacks adequate logging to prove attribution.

**ML-specific manifestations:** Repudiation threats in ML systems are underappreciated but significant. If training data is collected from multiple sources without cryptographic provenance, it may be impossible to determine after the fact which source contributed a specific poisoned sample. This non-repudiation failure makes it difficult to attribute poisoning attacks and to remove their effects.

At inference time, if queries to a model's API are not logged with sufficient detail (including the full input, the output, and the requester's identity), an adversary who uses the API to perform model extraction — probing the model with thousands of queries to build a substitute — may be able to deny having done so. Audit logs that record only aggregated statistics rather than individual queries are insufficient for detecting or attributing extraction attacks.

In automated decision-making systems, repudiation threats arise when the system makes a consequential decision — denying a loan, flagging a transaction as fraudulent — and the adversary (here, the adversary might be the system operator rather than an external attacker) later denies that the model was used in that decision or disputes what the model's output was. Immutable logging of model decisions, including the model version and input features, is a mitigation.

### 2.4 Information Disclosure

**Classical definition:** Information disclosure threats involve an adversary obtaining information they are not authorized to access.

**ML-specific manifestations:** ML systems have several unique information disclosure surfaces.

**Membership inference:** A trained model may reveal whether a specific data point was included in its training set. This is because models often fit their training data more closely than their test data, exhibiting lower uncertainty (higher confidence) on training examples. An adversary with access to the model's confidence outputs can query the model with suspected training examples and use the model's confidence as evidence of membership. For models trained on sensitive data (medical records, financial transactions, private communications), this constitutes a privacy violation.

**Attribute inference:** Even if an adversary cannot determine whether a specific individual was in the training set, they may be able to infer sensitive attributes of training data points from the model. For example, a language model trained on a dataset containing private messages may have memorized and be able to reproduce specific personal information from those messages when prompted appropriately.

**Model inversion:** An adversary can use a model's outputs to reconstruct representative examples of training data. For a face recognition system, model inversion can produce synthetic faces that resemble the training set. For a medical diagnostic model, model inversion might reveal clinically significant patterns in the training population.

**Weight disclosure:** If a model's weights are exposed (e.g., through a misconfigured model serving endpoint, or because the model is deployed as a client-side artifact), the adversary gains white-box access to the model, dramatically expanding their attack capabilities. Even partial weight disclosure can significantly enhance the power of other attacks.

**Hyperparameter and architecture inference:** Through careful black-box querying, an adversary can often infer aspects of a model's architecture and hyperparameters. This is valuable information that narrows the search space for more sophisticated attacks.

### 2.5 Denial of Service

**Classical definition:** Denial of Service (DoS) threats involve an adversary preventing legitimate users from accessing a service, typically by exhausting resources or triggering failures.

**ML-specific manifestations:** Classical DoS threats (e.g., flooding an API endpoint with requests to exhaust server resources) apply to ML inference services just as they apply to any other networked service. However, ML systems have additional DoS-specific vulnerabilities.

**Sponge attacks** (Shumailov et al., 2021) exploit the fact that neural network inference cost varies with input content. For certain architectures, particularly attention-based models like transformers, adversarially crafted inputs can cause dramatically increased computation time per sample — by a factor of 10x or more in demonstrated attacks. An adversary who sends a small number of carefully crafted sponge inputs can exhaust a model serving system's resources at a rate far exceeding what a naive flooding attack would achieve.

**Confidence score manipulation for DoS:** Some ML systems include thresholding logic that routes low-confidence predictions to a human reviewer. An adversary who can cause the model to output low-confidence predictions on all inputs (e.g., by crafting inputs that lie near the decision boundary) can overwhelm the human review queue, creating a denial of service against the human component of the human-in-the-loop system.

**Training time DoS:** In online learning systems that retrain based on production inputs, an adversary who can cause the model's training loss to explode (e.g., by submitting inputs with conflicting features and labels) can degrade the model's performance or destabilize the training process.

### 2.6 Elevation of Privilege

**Classical definition:** Elevation of Privilege (EoP) threats involve an adversary gaining access or permissions beyond what they are authorized to have.

**ML-specific manifestations:** Elevation of privilege in ML systems often involves an adversary using the model as a path to escalate their access to other system components.

**Prompt injection for privilege escalation:** In AI agent systems where a language model is authorized to take actions on behalf of users — making API calls, reading files, sending emails — an adversary who can inject instructions into the model's context (through malicious web pages, documents, or emails that the agent processes) may be able to cause the agent to perform privileged actions that the adversary could not directly authorize. This is a classic EoP pattern using the model as an intermediary.

**Model extraction for downstream attacks:** Extracting a white-box copy of a model is itself an information disclosure threat, but the extracted model also enables privilege escalation: the adversary now has capabilities (white-box attack generation, reliable adversarial examples) that they did not have with only black-box access.

**Data pipeline EoP:** In systems where the training data pipeline has write access to a model registry, an adversary who compromises the data pipeline (a relatively low-privilege component) can potentially modify the trained model (a high-privilege component), escalating from data-tier access to model-tier access.

---

## 3. Drawing Data Flow Diagrams for ML Systems

A Data Flow Diagram (DFD) is the primary artifact of the threat modeling process. It shows how data moves through a system, where it is processed and stored, and what the trust relationships between components are. For ML systems, DFDs require careful attention to the training and inference paths, which have different security properties.

### 3.1 DFD Elements

DFDs use four standard symbols:

- **External Entities** (rectangles): Sources or sinks of data outside the system's control boundary. In an ML context, these might be data providers, users who submit queries, or administrators.
- **Processes** (circles or rounded rectangles): Transformations that the system applies to data. In an ML context, these include data preprocessing, model training, and model inference.
- **Data Stores** (parallel lines): Persistent storage of data. In an ML context, these include the raw dataset, the feature store, the model registry, and the prediction log.
- **Data Flows** (arrows): Movement of data between components. Arrows should be labeled with the type of data flowing.

**Trust Boundaries** are drawn as dashed lines surrounding groups of components that share the same trust level. Data flows that cross trust boundaries are primary candidates for threat analysis.

### 3.2 Components of an ML System DFD

A typical ML system DFD includes the following components:

**External Entities:**
- Raw Data Sources (e.g., web crawlers, user uploads, third-party data providers)
- Labeling Service (e.g., crowdsourced labelers, automated labeling)
- End Users (those who query the model at inference time)
- System Administrators

**Processes:**
- Data Collection and Ingestion
- Data Preprocessing and Feature Engineering
- Model Training
- Model Validation
- Model Serving / Inference
- Monitoring and Alerting

**Data Stores:**
- Raw Data Store
- Processed Feature Store
- Model Registry
- Prediction/Query Log
- Monitoring Metrics Store

**Trust Boundaries:**
- External network / public internet (lower trust) vs. internal infrastructure (higher trust)
- Data infrastructure (read/write access to raw data) vs. model infrastructure (read/write access to model artifacts)
- Inference service (high availability, limited trust) vs. training infrastructure (higher trust, batch access)

We will make these concepts concrete in the next section with a fully worked example.

---

## 4. Worked Example: STRIDE Threat Model for a Spam Email Classifier

We will construct a complete threat model for a hypothetical spam email classifier deployed as a service within an enterprise email infrastructure. This is a realistic and consequential system: a poorly secured spam classifier can be exploited to exfiltrate sensitive emails, to conduct phishing attacks, or to undermine the trust that users place in the email system's security.

### 4.1 System Description

The spam classifier is a machine learning-based system that inspects incoming emails and classifies them as spam or legitimate (ham). The system has the following operational parameters:

- **Model:** A fine-tuned BERT-based text classifier trained on a combination of a public spam dataset and internal email data.
- **Training frequency:** The model is retrained weekly on data accumulated from the previous 30 days.
- **Training data labeling:** A combination of user-reported spam (via a "mark as spam" button) and the existing rule-based system's labels.
- **Inference:** Real-time classification of each incoming email before delivery.
- **Output:** Binary classification (spam/ham), plus a confidence score. Emails with confidence above 0.8 are automatically deleted; emails with confidence between 0.5 and 0.8 are placed in the spam folder with notification.
- **Feedback loop:** User-reported spam and user-reported "not spam" (from spam folder) are used as training signal.

### 4.2 Data Flow Diagram (Text Description)

The following describes the DFD for this system. In a real threat modeling exercise, this would be drawn visually; here we describe it textually.

```
TRUST BOUNDARY: External (Internet)
┌────────────────────────────────────────────────────────────────────┐
│  [External Entity: External Email Senders]                         │
│       │                                                            │
│       │ Raw SMTP emails                                            │
│       ▼                                                            │
│  [Process: Email Gateway / SMTP Receiver]                          │
└─────────────────┬──────────────────────────────────────────────────┘
                  │ Parsed email content (headers, body, attachments)
TRUST BOUNDARY: DMZ / Perimeter
┌─────────────────▼──────────────────────────────────────────────────┐
│  [Process: Email Preprocessing Service]                            │
│       │                                                            │
│       │ Feature vectors (tokenized text, header features)          │
│       ▼                                                            │
│  [Process: Inference Service]◄──────[Data Store: Model Registry]  │
│       │                                                            │
│       │ Classification result + confidence score                   │
│       ▼                                                            │
│  [Process: Routing Logic]                                          │
│       │              │              │                              │
│       │ Deliver       │ Spam folder  │ Delete                      │
│       ▼              ▼              ▼                              │
│  [Data Store:    [Data Store:  [Process: Audit Log]               │
│   User Inbox]    Spam Folder]       │                              │
│                       │             │                              │
│                       │ User "not spam" feedback                   │
│                       ▼                                            │
│  [External Entity: Users]                                          │
│       │ "Mark as spam" feedback                                    │
└─────────────────┬──────────────────────────────────────────────────┘
                  │ Labeled email samples (spam + ham + user feedback)
TRUST BOUNDARY: Internal Infrastructure
┌─────────────────▼──────────────────────────────────────────────────┐
│  [Data Store: Training Data Store]                                 │
│       │                                                            │
│       │ Training dataset (sampled + preprocessed)                  │
│       ▼                                                            │
│  [Process: Model Training Pipeline]                                │
│       │                                                            │
│       │ Trained model artifact                                     │
│       ▼                                                            │
│  [Process: Model Validation]                                       │
│       │                                                            │
│       │ Validated model artifact (if passes quality checks)        │
│       ▼                                                            │
│  [Data Store: Model Registry] ── provides model to Inference Svc  │
│                                                                    │
│  [External Entity: System Administrators]                          │
│       │ Configuration, monitoring access                           │
└────────────────────────────────────────────────────────────────────┘
```

Key trust boundaries:
1. **Internet / Email Gateway boundary:** All emails from external senders are untrusted data.
2. **DMZ / Internal boundary:** The inference service operates in a DMZ; the training pipeline and data store operate in a more trusted internal zone.
3. **User / System boundary:** User feedback (spam reports, "not spam" reports) is semi-trusted; it influences training but cannot directly modify the model.

### 4.3 STRIDE Threat Enumeration

The following table enumerates fifteen specific threats using the STRIDE framework. Each threat is identified by category, the system component it targets, a precise threat statement, the potential impact, and candidate mitigations.

---

**Threat 1: Spoofing**
- **Component:** Training Data Store
- **Threat Statement:** An adversary who compromises a third-party spam dataset provider (or who conducts a man-in-the-middle attack on the dataset download process) can substitute a poisoned dataset for the legitimate dataset, with the poisoned dataset containing adversarially crafted emails labeled as "ham" that would normally be classified as spam.
- **Potential Impact:** High. The retrained model will learn to classify the adversary's spam template as legitimate, allowing an ongoing phishing campaign to bypass the filter.
- **Mitigation:** Cryptographic verification of dataset provenance (checksums, signatures). Use of multiple independent data sources. Anomaly detection on label distributions after each data ingestion.

---

**Threat 2: Spoofing**
- **Component:** User Feedback Loop
- **Threat Statement:** An adversary who controls a large number of enterprise email accounts (e.g., via phishing, credential stuffing, or insider access) can submit coordinated false "not spam" reports on actual spam emails, training the model to associate spam features with the "ham" label.
- **Potential Impact:** Medium-High. The model's spam detection rate degrades over subsequent retraining cycles. Coordination required, but feasible for a sophisticated adversary.
- **Mitigation:** Rate-limit spam/ham feedback per user per time period. Cluster feedback by sender and content; flag coordinated feedback. Weight feedback from high-reputation accounts more heavily.

---

**Threat 3: Tampering**
- **Component:** Model Registry
- **Threat Statement:** An adversary who gains write access to the model registry (e.g., by compromising the CI/CD pipeline credentials) can replace the current production model with a backdoored version that classifies emails containing a specific hidden trigger pattern as "ham."
- **Potential Impact:** Critical. All subsequent emails from the adversary containing the trigger pattern will bypass the filter undetected.
- **Mitigation:** Immutable model artifact storage with write-once policies. Cryptographic signing of model artifacts by the training pipeline. Access control limiting who can push to production model registry. Anomaly detection on model behavior comparing new model vs. holdout set before promotion.

---

**Threat 4: Tampering**
- **Component:** Data Preprocessing Service
- **Threat Statement:** An adversary who can inject code into the preprocessing service (e.g., through a dependency vulnerability in the text processing library) can modify the feature extraction process to encode a hidden channel: for emails with a specific structure, the features passed to the inference service differ from the features that would have been computed from the raw email.
- **Potential Impact:** High. The attacker can bypass the classifier for any email they choose by crafting emails whose preprocessing-modified features land in the "ham" region of the decision boundary.
- **Mitigation:** Software dependency scanning and pinning. Code signing for preprocessing containers. Runtime integrity monitoring. End-to-end testing of the preprocessing pipeline against known spam/ham examples.

---

**Threat 5: Tampering**
- **Component:** Training Data Store / Model Training Pipeline
- **Threat Statement:** An adversary who conducts a sustained dirty-label poisoning attack via the user feedback mechanism (Threat 2) can over multiple retraining cycles shift the model's decision boundary such that an entire spam campaign template is classified as "ham."
- **Potential Impact:** High. Long-term degradation of spam detection effectiveness for the attacker's specific campaign.
- **Mitigation:** Inspect training data distributions before retraining. Monitor per-sender spam report rates. Implement a "human review" gate for feedback-based retraining data before incorporating into the training set.

---

**Threat 6: Repudiation**
- **Component:** Prediction/Query Log, Audit Log
- **Threat Statement:** If the audit log records only the classification outcome (spam/ham) without recording the full email content and feature vector at classification time, it is impossible to retroactively determine why the model classified a specific email as it did. An adversary who exploits a classification error (e.g., a delivered phishing email) can deny that the error was the result of their manipulation rather than a natural classification error.
- **Potential Impact:** Medium. Inability to distinguish intentional evasion from natural false negatives makes it impossible to conduct root cause analysis of security incidents.
- **Mitigation:** Log full input features (not raw email content for privacy) alongside classification outcome. Use append-only, tamper-evident logging. Retain logs long enough to support incident investigation across multiple retraining cycles.

---

**Threat 7: Repudiation**
- **Component:** Model Training Pipeline
- **Threat Statement:** If the model training pipeline does not maintain an immutable record of exactly which training examples were used in each training run, an adversary who poisoned the training data can deny that their contributed data influenced the model's behavior, because the connection between input data and model behavior cannot be established.
- **Potential Impact:** Medium. Undermines the ability to remediate poisoning attacks by removing adversarial training examples and retraining.
- **Mitigation:** Track the exact training dataset composition (with hashes of individual examples) for every trained model version. Maintain this provenance record as long as the model is in production.

---

**Threat 8: Information Disclosure**
- **Component:** Inference Service
- **Threat Statement:** An adversary who queries the inference service with many carefully chosen inputs can conduct a membership inference attack: by observing that the model assigns systematically higher confidence to certain emails (those whose content matches the training set), the adversary can infer whether specific emails were in the training set. For a corporate email system, this could reveal that specific internal communications were used as training data.
- **Potential Impact:** High (privacy). Reveals which internal emails were labeled as spam by users and used in training, potentially exposing information about internal communications.
- **Mitigation:** Differential privacy during training. Return only binary classification outputs (no confidence scores) via the external API. Rate-limit queries per account. Monitor for systematic probing patterns.

---

**Threat 9: Information Disclosure**
- **Component:** Inference Service
- **Threat Statement:** An adversary who makes a large number of queries to the inference service (black-box model extraction) can reconstruct a functional approximation of the classifier's decision boundary. This extracted model can then be used to systematically craft spam emails that evade detection, and the extraction itself reveals information about what content patterns the model has learned to associate with spam.
- **Potential Impact:** Medium. Enables highly targeted evasion attacks; reveals model decision criteria which constitute intellectual property.
- **Mitigation:** Rate-limit queries per user/IP. Monitor for systematic query patterns consistent with model extraction (high-volume, diverse, methodically structured inputs). Add random noise to confidence outputs. Watermark the model so that extracted copies can be identified.

---

**Threat 10: Information Disclosure**
- **Component:** Model Registry
- **Threat Statement:** If model artifact files in the registry are accessible without authentication (e.g., due to a misconfigured cloud storage bucket), an adversary can download the model weights directly, gaining full white-box access to the production model and enabling much more powerful attacks.
- **Potential Impact:** Critical. White-box access enables 100% reliable evasion attacks, precise membership inference, and fine-tuning the model to remove detection of the adversary's campaigns.
- **Mitigation:** Access control enforcement on model storage (authentication + authorization required for all reads). Regular auditing of storage permissions. Alert on unauthenticated access to the model registry.

---

**Threat 11: Denial of Service**
- **Component:** Inference Service
- **Threat Statement:** An adversary who can send emails to the system's inference service (indirectly, by sending emails to monitored accounts) can craft "sponge" emails — emails structured to maximize the inference service's compute time, potentially by generating very long token sequences that stress the attention mechanism of the BERT-based classifier.
- **Potential Impact:** Medium. Degraded throughput of the email classification service, causing delays in email delivery. May force the system into a fallback mode that delivers all emails unclassified.
- **Mitigation:** Enforce input length limits on emails processed for inference. Implement request timeouts. Use a lightweight pre-filter to reject or deprioritize unusually structured inputs before they reach the expensive BERT model.

---

**Threat 12: Denial of Service**
- **Component:** Human Review Queue / Spam Folder
- **Threat Statement:** An adversary who can cause the model to output low-confidence classifications (e.g., by crafting emails whose content spans multiple topic distributions, landing near the decision boundary) can flood the spam folder with emails that users must manually review, overwhelming the human review capacity.
- **Potential Impact:** Medium. Human reviewers become overwhelmed, reducing their effectiveness and potentially causing them to approve adversarial emails to reduce the backlog.
- **Mitigation:** Implement a capacity limit on the "uncertain" category; emails beyond the limit are either auto-delivered or auto-blocked based on risk tolerance. Alert on sustained spikes in low-confidence classifications.

---

**Threat 13: Denial of Service**
- **Component:** Model Training Pipeline
- **Threat Statement:** An adversary who injects training examples with extreme feature values (e.g., emails with malformed encodings that produce NaN or infinity values after preprocessing) can cause the model training pipeline to crash or produce a model with degenerate weights (all outputs the same class), effectively denying service for the next retraining cycle.
- **Potential Impact:** Medium-High. Training fails, model does not update, system runs on increasingly stale model, or (worse) falls back to a permissive default that delivers all email.
- **Mitigation:** Input validation and sanitization in the preprocessing pipeline. Anomaly detection on feature distributions before training. Validate training runs by evaluating the resulting model on a holdout set before promotion.

---

**Threat 14: Elevation of Privilege**
- **Component:** ML Agent / Automated Response System
- **Threat Statement:** If the spam classifier is integrated into an automated incident response system that takes actions based on classification results (e.g., automatically blocking sender domains, adding URLs to a blocklist), an adversary can craft emails that cause the classifier to classify a target domain's legitimate emails as spam, triggering automatic blocking of that domain. The adversary elevates from the ability to send emails to the ability to disrupt communication between the target organization and its partners.
- **Potential Impact:** High (business disruption). Targeted denial of communication, potentially disrupting critical business relationships.
- **Mitigation:** Require human approval for automated blocklist additions affecting trusted domains. Rate-limit automated blocking actions. Implement an appeal mechanism with fast turnaround.

---

**Threat 15: Elevation of Privilege**
- **Component:** Inference Service / Application Integration
- **Threat Statement:** If the spam classifier's output (confidence score) is consumed by downstream systems that take privileged actions — for example, a security information system that uses the classifier's spam verdicts as inputs to user risk scores — an adversary who can manipulate the classifier's outputs (through evasion or white-box access) can manipulate those downstream risk scores, potentially causing targeted users to have their accounts suspended or flagged for investigation.
- **Potential Impact:** Medium-High (targeted harassment / disruption). The adversary uses the ML system as a lever to affect organizational decisions they have no direct access to influence.
- **Mitigation:** Do not use ML model outputs as authoritative inputs to high-stakes decisions without human review. Treat ML outputs as signals, not as decisions. Audit trails linking consequential actions to the specific model outputs that triggered them.

---

### 4.4 Summary Threat Table

| # | STRIDE Category | Component | Threat Summary | Impact | Priority |
|---|----------------|-----------|----------------|--------|----------|
| 1 | Spoofing | Training Data Store | Dataset provider compromise | High | P1 |
| 2 | Spoofing | Feedback Loop | Coordinated false feedback | Med-High | P2 |
| 3 | Tampering | Model Registry | Backdoored model replacement | Critical | P1 |
| 4 | Tampering | Preprocessing Service | Code injection in feature extraction | High | P1 |
| 5 | Tampering | Training Pipeline | Sustained poisoning via feedback | High | P2 |
| 6 | Repudiation | Audit Log | Missing classification evidence | Medium | P3 |
| 7 | Repudiation | Training Pipeline | Missing training provenance | Medium | P3 |
| 8 | Information Disclosure | Inference Service | Membership inference | High | P2 |
| 9 | Information Disclosure | Inference Service | Model extraction via black-box queries | Medium | P2 |
| 10 | Information Disclosure | Model Registry | Unauthenticated model weight access | Critical | P1 |
| 11 | Denial of Service | Inference Service | Sponge attacks on BERT | Medium | P3 |
| 12 | Denial of Service | Human Review | Low-confidence classification flood | Medium | P3 |
| 13 | Denial of Service | Training Pipeline | Malformed training input injection | Med-High | P2 |
| 14 | Elevation of Privilege | Automated Response | Automated block via crafted email | High | P2 |
| 15 | Elevation of Privilege | Downstream Systems | ML output manipulation for risk score | Med-High | P2 |

---

## 5. Mapping STRIDE to MITRE ATLAS

The STRIDE categories map imperfectly but usefully onto MITRE ATLAS tactics and techniques. The following table provides the primary mappings, with specific technique identifiers.

| STRIDE Category | Primary ATLAS Tactics | Example ATLAS Techniques |
|----------------|----------------------|--------------------------|
| Spoofing | ML Attack Staging | AML.T0020 (Poison Training Data via Exfiltration), AML.T0040 (ML Model Inference API Access) |
| Tampering | ML Attack Staging, Persistence | AML.T0018 (Backdoor ML Model), AML.T0020 (Poison Training Data), AML.T0043 (Craft Adversarial Data) |
| Repudiation | Defense Evasion | AML.T0054 (LLM Prompt Injection — audit evasion variant) |
| Information Disclosure | Collection, Exfiltration | AML.T0024 (Exfiltration via ML Inference API), AML.T0056 (Steal ML Model), AML.T0057 (Infer Training Data Membership) |
| Denial of Service | Impact | AML.T0029 (Denial of ML Service) |
| Elevation of Privilege | Impact, Execution | AML.T0048 (Exploit Public-Facing ML Application), AML.T0051 (LLM Prompt Injection for Agent Manipulation) |

This mapping is a starting point, not a complete correspondence. ATLAS is organized primarily by adversary goal and technique, while STRIDE is organized by threat category. Using both frameworks simultaneously provides more complete coverage than either alone.

---

## 6. Adversary Capability Models: White-Box, Gray-Box, Black-Box

Having identified what threats exist, we must also characterize who the adversary is and what capabilities they possess. This characterization — the adversary model — is arguably the most consequential design choice in threat modeling, because the set of feasible attacks is entirely determined by what we assume the adversary can do.

### 6.1 White-Box Adversaries

A white-box adversary has complete knowledge of the target model: its architecture, weights, training procedure, hyperparameters, and training data. This is the strongest adversary model, and it produces the strongest (most dangerous) attacks. White-box attacks can compute exact gradients of the model's loss with respect to the input, enabling highly efficient and precise adversarial example generation.

White-box access can be obtained by:
- Insider threat (the adversary is an employee with model access)
- Theft of model artifacts from storage or a deployment endpoint
- Reverse engineering a deployed model artifact (e.g., a mobile app that bundles model weights)
- Model extraction followed by attack generation on the extracted model

In research papers, white-box attacks are the standard baseline because they give an upper bound on what is achievable. If a defense works against white-box attacks, it works against all weaker adversaries. If a defense fails against white-box attacks, the failure is fundamental.

### 6.2 Gray-Box Adversaries

A gray-box adversary has partial knowledge of the target model — perhaps the architecture but not the weights, or the training dataset but not the specific model version deployed in production, or the general class of model (BERT-based classifier) but not the fine-tuning details.

Gray-box adversaries are common in practice because information about the model often leaks through documentation, marketing materials, error messages, and timing side channels. An adversary who knows that a system uses a BERT-based classifier can optimize their attacks against BERT models generally, exploiting architectural properties that are common to all BERT variants.

### 6.3 Black-Box Adversaries

A black-box adversary has no knowledge of the model's internals and can only query the model through its inference API, observing inputs and outputs. This is the weakest adversary model and corresponds most closely to the position of an external attacker who has no special access to the system.

Even black-box adversaries can mount meaningful attacks through:
- **Transfer-based attacks:** Generate adversarial examples against a substitute model and rely on transferability to the target.
- **Query-based attacks:** Use the model's outputs to estimate gradients (score-based attacks) or to conduct decision-boundary estimation (decision-based attacks).
- **Model extraction:** First extract a substitute model through black-box queries, then mount white-box attacks against the substitute.

The black-box setting is most realistic for external adversaries, and defenses designed primarily for white-box adversaries may provide a false sense of security if black-box attacks are more feasible in practice.

---

## 7. Defense-in-Depth for ML Systems

Defense-in-depth is the security principle that no single control is sufficient; rather, a system should have multiple layers of defense such that the failure of any one layer does not lead to a complete security failure. This principle, well-established in classical security, applies equally to ML systems.

For the spam classifier system, a defense-in-depth strategy might look like:

**Layer 1: Data pipeline controls.** Provenance verification for training data. Rate limiting and anomaly detection on user feedback. Data validation and sanitization before feature extraction.

**Layer 2: Training process controls.** Differential privacy during training to limit memorization. Holdout evaluation before model promotion. Diversity of training data sources to limit the impact of any single poisoned source.

**Layer 3: Model artifact controls.** Access control on the model registry. Cryptographic signing of model artifacts. Immutable storage with audit logging.

**Layer 4: Inference controls.** Input validation (length limits, format checks). Rate limiting of queries per user. Confidence score perturbation to frustrate membership inference. Monitoring for anomalous query patterns.

**Layer 5: Output controls.** Human review for uncertain classifications. Automated action requiring human approval. Audit logging of decisions.

**Layer 6: Monitoring and response.** Continuous monitoring of model performance metrics. Automated alerting on distribution shift. Incident response playbook for suspected poisoning or evasion.

The key insight of defense-in-depth is that an adversary must defeat all layers simultaneously to succeed, while a defender only needs any one layer to hold. This asymmetry is the fundamental reason why defense-in-depth is a sound strategy even when no individual layer is perfect.

---

## 8. Discussion Questions

1. **Threat model completeness:** No threat model is exhaustive. Given the spam classifier example, which threat do you think is most likely to be exploited in practice, and why? Which threat was most difficult to identify without the structured STRIDE framework?

2. **Adversary model calibration:** In the spam classifier scenario, the threat model should reflect realistic adversaries — spammers with varying levels of sophistication. How would you calibrate the adversary model for a small enterprise email system vs. a large financial institution vs. a government agency? How does this calibration change the threat prioritization?

3. **The feedback loop:** The spam classifier's user feedback mechanism (Threat 2, Threat 5) creates a tension: user feedback is essential for keeping the model accurate against evolving spam campaigns, but it also provides an attack surface. Can you design a feedback mechanism that preserves the benefit while substantially reducing the attack surface?

4. **STRIDE vs. ATLAS:** We mapped STRIDE categories to ATLAS tactics. Are there threats in the ATLAS framework that are not captured by STRIDE? Conversely, are there STRIDE categories that have no good ATLAS mapping?

5. **Defense-in-depth limits:** Defense-in-depth works well against external adversaries but may be less effective against insider threats (Threat 3 involves an adversary with write access to the model registry). How would you modify the threat model and defenses to account for a malicious insider with administrative access?

---

## 9. Key Takeaways

Threat modeling for ML systems extends traditional software threat modeling with ML-specific attack surfaces. The STRIDE framework provides a useful starting structure, but must be augmented with ML-specific threats at each category.

The Data Flow Diagram is the foundational artifact of threat modeling. Drawing an accurate DFD for an ML system requires explicitly representing both the training path and the inference path, including all data stores and trust boundaries.

The adversary capability model — white-box, gray-box, black-box — is the most consequential choice in threat modeling, because it determines which attacks are feasible. Threat models should consider a range of adversary capabilities, not just the most convenient.

MITRE ATLAS provides a community-maintained catalog of ML-specific adversary tactics and techniques that can supplement STRIDE to improve coverage.

Defense-in-depth is essential because no single control is sufficient against a sophisticated adversary. The goal is to require the adversary to defeat multiple independent layers simultaneously.

---

## Assigned Reading and Problem Set Preview

- Shostack, A. (2014). *Threat Modeling: Designing for Security*, Chapter 2 (STRIDE overview).
- MITRE ATLAS Threat Matrix: https://atlas.mitre.org — complete the tactic and technique survey.
- Papernot, N. et al. (2016). "Towards the science of security and privacy in machine learning." Review the threat model section (Section 2).

**Problem Set 1** (due in three weeks) will ask you to construct a complete STRIDE threat model for an autonomous driving perception system. You will draw the DFD, enumerate threats, map to MITRE ATLAS, and propose a prioritized defense plan. The worked example in this lecture is your template.

---

*End of Lecture 2 Notes*
*Next lecture: Week 3 — Formal Security Properties for ML: Certified Robustness and Verification*
*Week 4 (after Week 3): Evasion Attacks I — FGSM and the Geometry of Adversarial Examples*
