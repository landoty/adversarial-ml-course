# PS1: Threat Modeling a Spam Email Classifier
### Example Solution — CS 6800 ML Security: Foundations

---

## 1. System Description

We model a spam email classifier deployed by a mid-size enterprise (5,000 employees). The system filters inbound email before delivery to employee inboxes. It uses a fine-tuned BERT-base model trained on labeled email datasets. The classifier operates as a microservice behind the corporate mail gateway.

**Primary users:** All employees receiving email; IT security administrators
**Threat surface:** External senders (anonymous internet), compromised internal accounts, insider threats, supply chain (model provider)

---

## 2. Data Flow Diagram

```
                           ┌─────────────────────────────────────────────────────┐
                           │               EXTERNAL INTERNET                      │
                           │                                                       │
                           │    [Sender]──────────────────────────────────────   │
                           └───────────────────────┬─────────────────────────────┘
                                                   │ SMTP (port 25)
                                         ╔═════════▼═════════╗
                                         ║   TRUST BOUNDARY   ║
                                         ╚═════════╤═════════╝
                                                   │
                                    ┌──────────────▼──────────────┐
                                    │   1. Mail Transfer Agent      │
                                    │      (Postfix / MX gateway)   │
                                    │   - Receives raw SMTP         │
                                    │   - Applies IP blocklists     │
                                    └──────────────┬──────────────┘
                                                   │ Raw email (headers + body)
                                    ┌──────────────▼──────────────┐
                                    │  2. Preprocessing Service     │
                                    │   - HTML stripping            │
                                    │   - URL normalization         │
                                    │   - Tokenization (BERT)       │
                                    └──────────────┬──────────────┘
                                                   │ Tokenized tensor
                                    ┌──────────────▼──────────────┐
                                    │  3. Inference API             │
                                    │   - BERT-base fine-tuned      │
                                    │   - Returns P(spam), P(ham)   │
                                    └──────────────┬──────────────┘
                                                   │ Score + label
                                    ┌──────────────▼──────────────┐
                                    │  4. Decision Engine           │
                                    │   - Threshold τ = 0.85        │
                                    │   - Quarantine / pass / block │
                                    └──────────────┬──────────────┘
                                                   │
                              ┌────────────────────┼──────────────────┐
                              │                    │                  │
                    ┌─────────▼────┐     ┌─────────▼────┐   ┌───────▼──────┐
                    │ 5. Employee  │     │ 6. Quarantine │   │ 7. Audit Log │
                    │    Inbox     │     │    Folder     │   │   (SIEM)     │
                    └─────────────┘     └──────────────┘    └──────────────┘

                    ┌──────────────────────────────────────────────────┐
                    │  8. Model Training Pipeline (offline)             │
                    │   - Labeled email corpus (S3 bucket)              │
                    │   - Fine-tuning jobs (internal GPU cluster)       │
                    │   - Model registry (private container registry)   │
                    │   - CD pipeline → Inference API deployment        │
                    └──────────────────────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │  9. Admin Dashboard               │
                    │   - Threshold adjustment          │
                    │   - Blocklist management          │
                    │   - Model retraining triggers     │
                    └─────────────────────────────────┘
```

**Trust boundaries:**
- **TB1:** External internet → Mail gateway (SMTP)
- **TB2:** Preprocessing → Inference API (internal network, but API is callable)
- **TB3:** Training pipeline → Model registry → Inference API (supply chain)
- **TB4:** Admin dashboard → Decision engine configuration

---

## 3. STRIDE Threat Analysis

### Spoofing

| # | Component | Threat | Attacker Capability | Impact | Mitigation |
|---|-----------|--------|---------------------|--------|------------|
| S1 | Mail Transfer Agent (1) | Attacker spoofs the sender address (From: header) to impersonate a trusted domain (e.g., `ceo@company.com`) | None — From header is trivially writable by any SMTP client | Phishing emails bypass user suspicion; classifier may not detect impersonation if BERT was not trained on header features | DMARC/DKIM/SPF enforcement at MTA; include header features in training data |
| S2 | Inference API (3) | Attacker on internal network spoofs a legitimate service calling the inference API, injecting crafted feature vectors directly (bypassing preprocessing) | Internal network access (compromised host or insider) | Complete bypass of text-based evasion constraints — attacker sends arbitrary tensors | Mutual TLS between preprocessing service and inference API; input schema validation |

---

### Tampering

| # | Component | Threat | Attacker Capability | Impact | Mitigation |
|---|-----------|--------|---------------------|--------|------------|
| T1 | Preprocessing Service (2) | Evasion attack: attacker crafts email content that is semantically spam but classified as ham after BERT tokenization (adversarial evasion) | Black-box query access via sending emails and observing delivery outcome | Spam emails delivered to inbox; phishing succeeds | Adversarial training on email dataset; ensemble with rule-based filters; input perturbation detection |
| T2 | Labeled Training Corpus (8) | Data poisoning: attacker injects mislabeled emails into the training dataset (e.g., by submitting "ham" feedback on spam emails, or compromising the labeling pipeline) | Access to user feedback mechanism or labeling pipeline | Model gradually learns to classify attacker's spam patterns as ham; degraded recall on targeted campaigns | Audit training data provenance; use robust training (loss trimming); data validation before training |
| T3 | Model Registry (8) | Supply chain attack: attacker replaces the production model checkpoint with a backdoored version containing a hidden trigger | Compromise of CI/CD pipeline or container registry write access | Emails containing a specific trigger phrase (e.g., a Unicode character) always classified as ham; persistent undetected backdoor | Code-sign model artifacts; hash verification before deployment; air-gap training pipeline from internet |
| T4 | Decision Threshold (9) | Admin interface manipulation: attacker or malicious insider lowers spam threshold τ from 0.85 to 0.20, causing most emails to pass | Admin account compromise | Near-complete spam filter bypass; flood of phishing and malware to all employees | MFA on admin dashboard; change audit log with alerts; threshold change requires two-person sign-off |

---

### Repudiation

| # | Component | Threat | Attacker Capability | Impact | Mitigation |
|---|-----------|--------|---------------------|--------|------------|
| R1 | Audit Log (7) | Attacker or insider deletes or modifies audit log entries to cover tracks after conducting a poisoning attack or threshold manipulation | Write access to SIEM or log storage | Cannot reconstruct attack timeline; compliance violation; incident response impaired | Append-only, write-protected log storage (WORM); log integrity hashing; ship logs to out-of-band SIEM |
| R2 | Model Training Pipeline (8) | No audit trail of who triggered retraining and with what data; if a poisoned model is deployed, responsible party is non-attributable | Any authorized engineer triggering retraining | Unable to establish accountability for model degradation | Immutable retraining audit trail: log data hash, triggering user, timestamp, model lineage in ML metadata store |

---

### Information Disclosure

| # | Component | Threat | Attacker Capability | Impact | Mitigation |
|---|-----------|--------|---------------------|--------|------------|
| I1 | Inference API (3) | Model stealing: attacker sends thousands of emails through the filter and uses the P(spam) confidence scores to train a surrogate model | Black-box query access (any external email sender) | Attacker builds a local replica of the classifier; uses it to craft evasion attacks offline, reducing the query cost of adversarial attacks | Return only hard label (spam/ham) without confidence score to external-facing callers; rate-limit queries per sender |
| I2 | Inference API (3) | Membership inference: attacker uses differences in model confidence on "template" emails to infer whether a specific email was in the training corpus | Black-box confidence score access | Reveals which emails were used to train the model; privacy violation if training data contains PII from real emails | Remove PII from training data; use differential privacy (DP-SGD) during fine-tuning; suppress confidence scores |
| I3 | Quarantine Folder (6) | Quarantined emails contain potentially sensitive content; if quarantine storage is misconfigured, a compromised inbox could access other users' quarantined messages | Exploiting misconfigured storage ACLs | Data breach: emails intended for one recipient readable by another | Per-user ACLs on quarantine storage; encryption at rest; access logging |

---

### Denial of Service

| # | Component | Threat | Attacker Capability | Impact | Mitigation |
|---|-----------|--------|---------------------|--------|------------|
| D1 | Inference API (3) | Sponge attack (Shumailov et al. 2021): attacker sends emails crafted to maximize inference latency (e.g., extremely long tokenization sequences near max token limit, Unicode sequences causing slow preprocessing) | Black-box access (any email sender) | Inference API latency spikes; mail delivery delays spike to minutes; business disruption | Input length limiting at preprocessing; async queue with per-sender rate limiting; inference timeout with fallback to rule-based filter |
| D2 | Preprocessing Service (2) | Zip bomb / malformed MIME: sending deliberately malformed MIME structure that causes preprocessing to consume unbounded memory during parsing | Any external SMTP sender | Preprocessing service OOM crash; mail delivery stops entirely | MIME parsing resource limits; sandbox preprocessing; max attachment size enforcement |
| D3 | Mail Transfer Agent (1) | Spam flood: overwhelming the MTA with high-volume sending to exhaust queue capacity | Botnet or compromised accounts | Legitimate email queued indefinitely; business communication disrupted | Rate limiting per source IP/ASN; IP reputation scoring; backpressure to upstream |

---

### Elevation of Privilege

| # | Component | Threat | Attacker Capability | Impact | Mitigation |
|---|-----------|--------|---------------------|--------|------------|
| E1 | Admin Dashboard (9) | Phishing email bypasses filter (via T1), delivers payload to admin user, who clicks a link leading to credential theft → attacker gains admin access to decision engine | Successful evasion attack + social engineering | Full control over spam filter configuration; can permanently disable filter or tune thresholds to allow targeted campaign | Separate admin account from regular email account; hardware MFA; privileged access workstations for admin tasks |
| E2 | Training Pipeline (8) | Compromised ML engineer account allows push of backdoored model to production | Social engineering or credential compromise of ML engineer | As per T3 — persistent backdoor in production | MFA; separate deployment credentials from development credentials; model artifact signing |

---

## 4. MITRE ATLAS Mappings

| Threat # | ATLAS Tactic | ATLAS Technique |
|----------|-------------|-----------------|
| T1 | ML Attack Staging → Craft Adversarial Data | AML.T0043: Craft Adversarial Data (Evasion) |
| T2 | ML Attack Staging → Poison Training Data | AML.T0020: Poison Training Data |
| T3 | ML Supply Chain Compromise | AML.T0010: ML Supply Chain Compromise |
| I1 | Reconnaissance | AML.T0005: Create Proxy Model (Model Stealing) |
| I2 | Exfiltration | AML.T0057: Membership Inference |
| D1 | Impact | AML.T0029: Denial of ML Service (Sponge Attack) |
| E1 | Initial Access | AML.T0012: Valid Accounts (via phishing) |

---

## 5. Prioritization

**Top 3 highest-priority threats:**

**1. T1 — Adversarial Evasion (Tampering, Inference API)**
This is the core threat to the system's mission. A spam filter that can be reliably evaded by motivated spammers provides no security value. The attacker needs only black-box query access (sending emails and observing delivery), which every external sender already has. The blast radius is high: successful evasion enables phishing, BEC (business email compromise), and malware delivery to all 5,000 employees. Priority: implement adversarial training and monitor for query-pattern anomalies suggesting iterative evasion attempts.

**2. T3 — Supply Chain Model Backdoor (Tampering, Model Registry)**
A backdoored model is invisible to standard evaluation — it achieves normal accuracy on clean test sets but allows the attacker's trigger-bearing emails to pass indefinitely. Unlike evasion attacks (which require per-email effort), a backdoor gives the attacker permanent persistent access. Detection requires proactive defense (model scanning, activation analysis). Priority: implement model artifact signing and deploy activation-based backdoor scanning on every model before production deployment.

**3. I1 — Model Stealing (Information Disclosure, Inference API)**
Model stealing is a prerequisite amplifier for T1: an attacker who builds an accurate local replica can mount white-box gradient-based evasion attacks offline, dramatically improving evasion quality without expensive live queries. Suppressing confidence scores (returning hard labels only) raises the attacker's cost significantly with minimal operational impact. This is a high-leverage, low-cost mitigation. Priority: immediate — change API response to return label only, no score.
