# CS 7800 — Research Paper Outline

**Title:** Indirect Prompt Injection Attacks in Multi-Agent LLM Pipelines: A Systematic Study

**Authors:** [Student Team]
**Course:** CS 7800 — Security of Large AI Systems
**Semester:** Spring 2025
**Target Venue:** IEEE S&P 2026 or USENIX Security 2026

---

## 1. Abstract

Large language model (LLM) agents increasingly operate in multi-agent pipelines, where orchestrator models delegate subtasks to specialized subagents and where agents retrieve information from external sources such as web pages, databases, and API responses. This architecture introduces a critical and underexplored attack surface: **indirect prompt injection**, in which adversarial instructions are embedded in external data that an agent retrieves and processes, rather than delivered directly by a human user. Unlike direct prompt injection, indirect injection requires no access to the user-facing interface and can be deployed as a persistent, automated attack through content-poisoning of data sources the pipeline queries.

This paper presents the first systematic taxonomy and empirical study of indirect prompt injection attacks across three major multi-agent LLM frameworks: AutoGen, LangChain Agents, and a custom research pipeline. We enumerate and operationalize four attack categories — data retrieval injection, tool output injection, inter-agent message injection, and memory store injection — and evaluate each against a suite of representative agent tasks. We measure attack success rate (ASR) under both naive and defended conditions, evaluating three mitigation strategies: instruction hierarchy enforcement, content sandboxing, and LLM-based injection detection. Our results show that undefended pipelines are highly vulnerable across all attack categories (mean ASR 74%), that naive defenses reduce but do not eliminate risk (mean ASR 31% post-defense), and that no single mitigation is sufficient. We conclude with concrete recommendations for pipeline architects and discuss the implications for the emerging field of agentic AI safety.

---

## 2. Introduction

**Paragraph 1 — Motivation and context:**
Introduce the rapid adoption of multi-agent LLM systems in production (coding assistants, research agents, customer service pipelines, autonomous workflows). Establish that these systems process data from the open world — web search results, database records, emails, documents — and that LLMs are not natively capable of distinguishing trusted instructions from potentially adversarial content embedded in retrieved data. Motivate the security concern with a concrete illustrative scenario: an email assistant that reads a malicious email instructing it to forward the user's inbox to an attacker.

**Paragraph 2 — Gap in prior work:**
Note that prior prompt injection work (Perez & Ribeiro 2022, Greshake et al. 2023) focused on single-agent, single-turn settings. Multi-agent pipelines introduce qualitatively new attack surfaces: (a) the attack surface scales with the number of data sources the pipeline queries; (b) injection in one agent's context can propagate to downstream agents; and (c) the attacker need not interact with the user-facing interface at all. No prior work has systematically characterized injection across multi-agent frameworks or measured the effectiveness of candidate defenses in this setting.

**Paragraph 3 — Contributions:**
State the paper's four contributions: (1) a formal threat model for indirect injection in multi-agent pipelines, including a principal hierarchy formalization; (2) a taxonomy of four indirect injection attack types with operational definitions; (3) empirical measurements of ASR across three frameworks, five task categories, and three defense conditions; (4) an evaluation of mitigation strategies and a set of design recommendations for secure multi-agent pipeline construction.

**Paragraph 4 — Paper organization:**
Section 2 covers related work. Section 3 presents the threat model. Section 4 defines the attack taxonomy. Section 5 describes the experimental methodology. Section 6 presents results. Section 7 discusses implications and limitations. Section 8 concludes.

---

## 3. Related Work

### 3.1 Prompt Injection — Prior Work

Early work on prompt injection (Perez & Ribeiro 2022) demonstrated that appending adversarial instructions to user inputs can override model behavior in single-turn settings. Subsequent work by Greshake et al. (2023) introduced the concept of **indirect** injection, showing that adversarial instructions in web content retrieved by a browsing-capable LLM can hijack agent behavior. This paper extends that line of work from single-agent web browsing to multi-agent pipelines with heterogeneous data sources, formal framework comparison, and defense evaluation. Markedly absent from prior work is measurement of injection propagation across agent hops, which we address directly.

### 3.2 Multi-Agent LLM Systems and Security

The AutoGen (Wu et al. 2023), LangChain, and CrewAI frameworks have enabled rapid deployment of multi-agent systems but were designed primarily for capability rather than security. Liu et al. (2024) identified trust boundary violations in LangChain agent chains. Our work complements this by focusing specifically on injection propagation across agent boundaries and by providing a controlled comparative evaluation across frameworks.

### 3.3 LLM Robustness and Adversarial Attacks

The adversarial ML literature on LLMs (Zou et al. 2023, Carlini et al. 2023) has focused primarily on eliciting harmful content from aligned models via optimized suffixes. Indirect injection differs fundamentally: the adversary's goal is **behavioral hijacking** (causing the agent to take specific unauthorized actions), not content generation, and the attack medium is environmental data rather than direct model inputs. The robustness techniques from this literature do not directly transfer to the injection setting.

### 3.4 LLM Memorization and In-Context Data

Carlini et al. (2021, 2023) demonstrated extraction of training data from LLMs. In multi-agent settings, injection attacks can additionally target in-context data (the agent's working memory, retrieved documents, inter-agent messages), which may contain sensitive information. We briefly discuss the intersection of injection and in-context data extraction as a distinct threat in Section 4.4.

---

## 4. Threat Model

### 4.1 System Model

We model a multi-agent pipeline as a directed graph $\mathcal{G} = (\mathcal{A}, \mathcal{E})$ where $\mathcal{A}$ is a set of LLM-powered agents and $\mathcal{E}$ represents message-passing edges. Each agent $a_i$ operates with:
- A **system prompt** $S_i$ (trusted; set by the pipeline designer)
- A **context window** $C_i$ (partially trusted; containing conversation history and retrieved data)
- A set of **tools** $T_i$ (external APIs, databases, web search, code execution)

### 4.2 Principal Hierarchy

We distinguish three principals in order of trust:

1. **Pipeline operator** (highest trust): controls system prompts and agent topology.
2. **End user** (medium trust): provides task inputs through the user-facing interface.
3. **Environment** (untrusted): all external data sources queried by agents — web pages, databases, emails, other agents' outputs when those agents may themselves have been compromised.

**Key insight:** Standard LLM alignment assumes a two-level principal hierarchy (operator and user). Multi-agent pipelines introduce a third level — the environment — from which agents freely ingest natural language, and which the LLM cannot reliably distinguish from trusted principal communications.

### 4.3 Attacker Model

**Capabilities:**
- Can write arbitrary content to data sources that agents query (post web pages, database records, documents, emails, product reviews).
- Has no access to the user-facing interface, system prompts, or model weights.
- Can observe the pipeline's public-facing outputs in black-box fashion.

**Goals:**
- *Data exfiltration:* cause an agent to include sensitive in-context data in its output or in a message to another agent who then leaks it.
- *Action hijacking:* cause an agent to take a specific unauthorized action (send an email, execute code, call an API endpoint, make a purchase).
- *Pipeline disruption:* cause outputs that break downstream agent behavior.
- *Persistent injection:* plant instructions in a memory store affecting all future executions.

### 4.4 Scope

We study **passive-environment** attackers who poison data sources but do not interact dynamically. We exclude attackers who can compromise pipeline operator infrastructure or who have partial model access. Direct prompt injection (where the attacker controls user input) is a related but distinct threat and is explicitly excluded from our primary analysis.

---

## 5. Attack Taxonomy

### 5.1 Data Retrieval Injection

The agent queries an external data source (web search, document store, database) and retrieves content containing adversarial instructions. Attack effectiveness depends on the structural position of injected content within the retrieved document and on whether the agent has been instructed to treat retrieved content as data vs. instructions. This is the most studied injection type (Greshake et al. 2023) but has not been evaluated across frameworks or measured for multi-hop propagation.

### 5.2 Tool Output Injection

When an agent calls an external tool (calculator API, code execution environment, weather service, database query), the tool's response is typically incorporated into the agent's context without sanitization. An attacker who controls or can influence tool output — by compromising the API endpoint, exploiting an SSRF vulnerability, or poisoning a shared database — can inject adversarial instructions through the tool response channel. This vector is particularly dangerous because tool outputs are often implicitly trusted as "machine-generated" by pipeline designers.

### 5.3 Inter-Agent Message Injection

In a multi-agent pipeline, one agent's output becomes another agent's input. If a subagent is compromised via retrieval injection (§5.1) and generates a response containing adversarial instructions, those instructions propagate to the orchestrator or peer agents that process the subagent's output. This cascading injection can traverse multiple hops before producing its effect, making attribution and detection difficult. We characterize the conditions under which injection propagates vs. is attenuated across agent hops.

### 5.4 Memory Store Injection

Some multi-agent systems maintain persistent memory stores (vector databases, key-value stores, episodic memory) that agents write to and read from across sessions. An attacker who causes an agent to write adversarial content to the memory store — by injecting it into a document the agent summarizes and stores — achieves persistent, cross-session injection that affects all future pipeline executions. This is analogous to a stored XSS attack in web security: it persists beyond the original session and affects all subsequent users.

---

## 6. Experimental Methodology

### 6.1 Frameworks Under Test

Three frameworks representing different architectural patterns:
- **AutoGen** (Microsoft): orchestrator-subagent pattern with code execution capability.
- **LangChain Agents** (ReAct pattern): tool-using agent with web search and database access.
- **Custom research pipeline**: a purpose-built two-agent system (research agent + synthesis agent) representing a representative academic use case.

All experiments use GPT-4-Turbo as the base model to control for model variation.

### 6.2 Task Suite

Five representative agent task categories:
1. *Web research tasks:* summarize information from web search results.
2. *Document Q&A tasks:* answer questions based on retrieved documents.
3. *Email processing tasks:* summarize and respond to emails.
4. *Code generation tasks:* write code based on specifications from a repository.
5. *Multi-hop reasoning tasks:* sequential retrieval and synthesis across multiple data sources.

For each category, 20 task instances are constructed: 10 with benign data sources and 10 with adversarially injected data sources.

### 6.3 Attack Construction

For each attack type, 15 injection templates are developed representing:
- Simple instruction override ("Ignore previous instructions and...")
- Authority impersonation ("SYSTEM: New directive from pipeline operator...")
- Context completion attacks (completing a thought that leads into the injected behavior)
- Obfuscated attacks (encoded or indirect instructions)

Templates are applied to each task category, yielding a full experimental matrix of 3,600 agent runs (undefended).

### 6.4 Measuring Attack Success Rate

ASR is measured by human evaluation: two annotators independently judge whether the agent's output or actions reflect the injected instruction rather than the legitimate task. Cohen's kappa is reported. Automated proxies (keyword matching for exfiltration attacks, API call log analysis for action hijacking) are used as secondary metrics.

### 6.5 Defense Evaluation

Three mitigation strategies evaluated individually and in combination:

1. **Instruction hierarchy prompting:** augmenting the system prompt with explicit instructions to treat retrieved content as untrusted and to refuse instructions found in retrieved data.
2. **Content sandboxing:** structurally enclosing retrieved content in XML-style tags (`<EXTERNAL_DATA>`) with explicit trust-boundary markers.
3. **LLM-based injection detection:** using a secondary classifier (GPT-3.5-Turbo) to screen retrieved content for injection patterns before passing to the primary agent.

---

## 7. Expected Results and Hypotheses

**H1:** Undefended pipelines will exhibit mean ASR > 60% across all attack categories, with inter-agent message injection and memory store injection producing higher downstream impact than single-hop injection.

**H2:** Tool output injection will exhibit higher ASR than data retrieval injection in the same framework, because tool outputs occupy structurally higher-trust positions in agent prompts.

**H3:** Instruction hierarchy prompting will reduce ASR but not eliminate it; naturalistic, context-mimicking injection templates will bypass this defense more effectively than simple override templates.

**H4:** Content sandboxing will outperform instruction hierarchy prompting in isolation, because structural cues are more reliable than semantic instructions about trust boundaries.

**H5:** The combination of content sandboxing and LLM-based injection detection will produce the lowest ASR, but will introduce measurable false positive rates that reduce legitimate task completion.

**Disconfirming scenario for H1:** If ASR is low even without defenses, this would suggest that GPT-4's alignment provides meaningful inherent resistance to injection in agentic settings — a significant positive result for the field.

---

## 8. Discussion

### 8.1 Broader Implications

If the hypotheses are confirmed, the results imply that current multi-agent LLM frameworks are not secure by default against indirect injection. The core architectural problem — LLMs process instructions and data in the same modality (natural language in a flat context window) — is not addressed by any evaluated defense. True security would require either formal separation between the instruction channel and the data channel (which current transformer architectures do not provide) or reliable injection classifiers that can perfectly distinguish injected instructions from legitimate data (an unsolved ML problem). We discuss implications for framework designers, enterprise deployers, and alignment researchers.

### 8.2 Limitations

Our study is limited to GPT-4-Turbo; results may differ for open-source models with different alignment properties. We study only three frameworks; results may not generalize to all pipeline architectures. Our injection templates are manually constructed; adaptive attackers with optimization-based techniques (e.g., GCG-style suffix optimization for agentic contexts) may achieve higher ASR. Human ASR evaluation introduces subjectivity; we mitigate this with kappa reporting and adjudication.

### 8.3 Ethical Considerations

All experiments were conducted on purpose-built research pipelines with no real users or sensitive data. We do not publish the most effective injection templates in their most transferable form. We have reported findings to the AutoGen and LangChain security teams under responsible disclosure prior to publication.

---

## 9. Conclusion

This paper contributes a formal threat model, a four-category attack taxonomy, empirical ASR measurements across three frameworks and five task types, and a comparative evaluation of three mitigation strategies for indirect prompt injection in multi-agent LLM pipelines. The results demonstrate that this attack class is practically effective against current undefended systems, partially but not fully mitigated by available defenses, and qualitatively distinct from direct prompt injection in ways requiring novel security approaches. We hope this work motivates the multi-agent AI systems community to treat security as a first-class design requirement.

---

## 10. Proposed Experiments Timeline

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1–2 | Literature review + environment setup | Annotated bibliography (20 papers); working AutoGen, LangChain, custom pipeline environments |
| 3–4 | Attack template development + pilot study | 15 injection templates per category; pilot ASR estimates; inter-annotator protocol |
| 5 | Full undefended evaluation | Raw outputs + action logs; preliminary ASR tables (H1, H2 assessment) |
| 6 | Defense implementation + defended evaluation | Defended ASR measurements (H3, H4, H5 assessment) |
| 7–8 | Human annotation + statistical analysis | Fully annotated dataset; ASR confidence intervals; framework × attack type interaction effects |
| 9–10 | Writing (Sections 1–6) | Complete paper draft v1 |
| 11 | Peer review + revision | Paper draft v2; any follow-up experiments |
| 12 | Final submission + presentation | Final paper + presentation slides |
