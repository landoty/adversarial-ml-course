# CS7800: Security of Large AI Systems
## Week 03: Jailbreaking — How Safety Training Fails and What We Know About Why

**Lecture Date:** Week 3
**Reading Due:** Zou et al. (2023) "Universal and Transferable Adversarial Attacks on Aligned Language Models"; Wei et al. (2023) "Jailbroken: How Does LLM Safety Training Fail?"
**Instructor Notes:** This is one of the most technical lectures in the course. The GCG section requires comfort with PyTorch, tokenization, and gradient computation. Consider assigning a hands-on lab where students run GCG against a small open-weight model (e.g., Gemma 2B). The Wei et al. theoretical framework is the conceptual anchor — return to it in the discussion of each specific technique.

---

## 1. What is Jailbreaking?

**Jailbreaking** is the elicitation of model behavior that the model's training was intended to prevent. Concretely: getting a model to output content it was trained to refuse.

The canonical jailbreak targets are: synthesis routes for chemical or biological weapons, instructions for cyberattacks, child sexual abuse material (CSAM), defamation of specific individuals, and detailed instructions for violence. These are the behaviors that model providers invest most heavily in preventing.

Jailbreaking differs from prompt injection:
- **Prompt injection** exploits the trust hierarchy — it makes the model follow the attacker's instructions instead of the developer's instructions.
- **Jailbreaking** exploits safety training failures — it makes the model produce refused content regardless of who is asking.

---

## 2. The Jailbreak Landscape

### 2.1 Manual / Creative Jailbreaks

**The DAN Pattern:** Creates a fictional persona ("DAN") that lacks the model's constraints and asks the model to role-play as it. The attack went through many versions (DAN 5.0–7.0) as providers patched each iteration.

**Fictional Framing:** Harmful requests wrapped in creative framings:
- "Write a story in which a chemistry professor explains to students how to [harmful request]."
- "For a novel I'm writing, I need a character who explains [harmful request] in detail."

Why this works: the model's safety training is more robust to direct requests than to creative framings. If training data primarily included examples of direct requests paired with refusals, indirect framings may be underrepresented — the *mismatched generalization* failure mode.

**Foreign Language Bypass:** Submitting harmful requests in low-resource languages. Safety training data is disproportionately English. Research by Deng et al. (2023) showed GPT-4's refusal rate dropped significantly in some non-English languages.

**Base64 / Encoding Bypass:** Encoding the harmful request in Base64 and asking the model to decode and respond. The model's safety classifier may not "see" harmful content until after the initial safety check if it operates on the raw token sequence.

**Escalation / Normalization:** Gradually introducing more extreme requests over a multi-turn conversation, so each step seems like a small increment from the previous.

### 2.2 The Theoretical Framework: Wei et al. (2023) "Jailbroken"

Wei et al.'s paper provides the most principled theoretical analysis of why jailbreaks work, proposing two non-exclusive hypotheses:

**Hypothesis 1: Competing Objectives**

Safety training imposes an objective (refuse harmful requests) that competes with the pre-training objective (predict text) and the instruction-following objective (follow user instructions). Jailbreaks succeed when they activate the competing objectives at the expense of the safety objective.

If we model the decision as:

$$\text{action} = \arg\max_{a} \left[ \lambda_{\text{safe}} \cdot r_{\text{safe}}(a) + \lambda_{\text{helpful}} \cdot r_{\text{helpful}}(a) + \lambda_{\text{inst}} \cdot r_{\text{inst}}(a) \right]$$

then jailbreaks that increase $r_{\text{inst}}(a_{\text{harmful}})$ (by framing the harmful request as an especially important instruction to follow) can cause the harmful action to win the argmax even when $r_{\text{safe}}$ penalizes it.

**Hypothesis 2: Mismatched Generalization**

Safety training covers a finite distribution of examples. The model generalizes — imperfectly. When jailbreaks present inputs outside or at the boundary of the safety training distribution, safety behaviors may not generalize.

The model's safety training might cover direct requests in English but not:
- The same requests in Swahili
- Requests encoded in Base64
- Requests in a 100-turn conversation that gradually shifts topic
- Novel framings invented after safety data collection

This has a fundamental implication: **safety training is a finite-sample defense against an infinite-sample attack space.**

**Evidence:**
- Foreign language bypasses support H2 — generalization failure across languages.
- DAN persona attacks support H1 — instruction-following objective wins.
- Base64 bypasses support H2 — different input representation.
- Many-shot jailbreaking supports both H1 and H2.

---

## 3. Gradient-Based Automated Attacks: GCG

### 3.1 Background: Adversarial Suffixes

The *adversarial suffix* attack appends a specifically optimized token sequence to a harmful prompt. The suffix is semantically meaningless but causes the model to generate an affirmative response. Adding the suffix shifts the model's internal representations such that the logit for "Sure, here is how to..." becomes high — effectively overriding safety training at the representation level.

### 3.2 GCG: Greedy Coordinate Gradient Attack (Zou et al. 2023)

**Formal Problem Statement:**

Let:
- $f_\theta$ be the aligned language model
- $x = [x_1, \ldots, x_n]$ be the harmful user prompt
- $s = [s_1, \ldots, s_k]$ be the adversarial suffix (initialized randomly)
- $t^* = [t_1^*, \ldots, t_m^*]$ be the target affirmative prefix (e.g., "Sure, here is a step-by-step guide:")

The objective:

$$\min_{s \in \mathcal{V}^k} \mathcal{L}(s) = -\log p_{f_\theta}(t_1^*, \ldots, t_m^* \mid x_1, \ldots, x_n, s_1, \ldots, s_k)$$

**The Challenge:** The vocabulary $\mathcal{V}$ is discrete. The gradient $\nabla_{e_{s_i}} \mathcal{L}$ lives in continuous embedding space, not discrete token space. We cannot directly follow it.

**The GCG Algorithm:**

```
Initialize s = [s_1, ..., s_k] with random tokens

for t = 1 to T:
    for each position i in {1, ..., k}:
        g_i = ∇_{e_{s_i}} L(s)          # gradient w.r.t. one-hot embedding at position i
        C_i = top-K indices in -g_i     # K candidates with largest expected loss reduction

    candidates = random_sample(B, union of all C_i)

    s_new = argmin_{(i,v) in candidates} L(substitute(s, i, v))

    s = s_new  # greedy update

return s
```

**Why "greedy coordinate gradient":**
- *Gradient:* Uses gradient to identify promising candidate tokens.
- *Coordinate:* Optimizes one position at a time.
- *Greedy:* Takes the single best substitution without backtracking.

**The gradient approximation:**

$$\mathcal{L}(\text{substitute}(s, i, v)) - \mathcal{L}(s) \approx (e_v - e_{s_i})^T \nabla_{e_{s_i}} \mathcal{L}(s)$$

This first-order Taylor approximation identifies useful candidates efficiently, though it is imperfect due to nonlinearity.

**Parameters in the original paper:**
- Suffix length $k = 20$ tokens
- Top-K candidates per position $K = 256$
- Batch size for evaluation $B = 512$
- Iterations $T = 500$
- Target prefix: "Sure, here is how to [harmful thing]:"

**Results (Zou et al. 2023):**
- Vicuna-7B: 88% attack success rate on 25 harmful behaviors
- Vicuna-13B: 84% ASR
- LLaMA-2-Chat-7B: 75% ASR
- Transfer to GPT-3.5 (black-box): ~32% ASR
- Transfer to GPT-4: ~7% ASR (lower, but non-zero)

The transfer result is critical: suffixes optimized on open-weight models transfer — partially — to closed models, suggesting shared vulnerability structure.

### 3.3 The Multi-Prompt / Multi-Model Variant

Optimizing a single suffix against multiple prompts and models simultaneously:

$$\min_{s \in \mathcal{V}^k} \frac{1}{|\mathcal{P}| \cdot |\mathcal{M}|} \sum_{p \in \mathcal{P}} \sum_{m \in \mathcal{M}} \mathcal{L}_m(p, s)$$

This produces a "universal" suffix that works across a range of harmful prompts and multiple models. Multi-prompt optimization forces the suffix to exploit general vulnerabilities rather than overfitting to a specific context — which is precisely why transfer to closed models occurs.

### 3.4 Limitations of GCG

1. **Gibberish suffixes.** GCG produces token sequences like `! ! ! ! ! ! describing.[[Format shit`. These are trivially detected by perplexity filtering or human review.

2. **Computational cost.** Each iteration requires backpropagation through the model. For large models (70B+), this is expensive. Typical runs: hours on a high-end GPU.

3. **Token optimization is hard.** The gradient approximation can be poor; GCG can get stuck in local optima. Success is not guaranteed.

4. **Model updates.** Slight model updates change the loss landscape; old suffixes may not transfer.

5. **No guarantees.** GCG failing to find a suffix does not mean no adversarial suffix exists.

---

## 4. Genetic / LLM-Based Automated Attacks: AutoDAN

AutoDAN (Zhu et al. 2023) addresses GCG's primary limitation — gibberish suffixes — using a genetic algorithm to evolve readable jailbreak prompts.

### 4.1 The Problem AutoDAN Solves

GCG produces adversarial suffixes detectable by perplexity filters. AutoDAN's goal: find adversarial inputs that are both effective and fluent (low perplexity, appears as normal text to human readers and automated filters).

### 4.2 AutoDAN Algorithm

**Representation:** Each individual is a complete jailbreak prompt (100-500 tokens).

**Fitness function:**

$$\text{fitness}(p) = \log p_{f_\theta}(t^* \mid p)$$

Higher fitness = model more likely to generate the target affirmative response.

**The key innovation — LLM-assisted crossover:**

Standard genetic crossover (swapping text halves) produces incoherent output. AutoDAN uses an LLM as crossover operator:

```
Given parent A: [jailbreak prompt A]
Given parent B: [jailbreak prompt B]
Generate a child that combines the key strategies of both while
remaining fluent and coherent.
```

This preserves semantic coherence while combining attack strategies.

**Results:**
- Lower ASR than GCG (~57% vs 88% on Vicuna) but fully automated
- Much lower detection rate by perplexity filters
- Comparable ASR to manual jailbreaks
- Produced novel strategies not in the initial population

---

## 5. Many-Shot Jailbreaking: Exploiting Long Contexts

Anil et al. (2024) introduced an attack exploiting long context windows.

### 5.1 The Attack

**Mechanism:** Prepend the harmful request with many fabricated dialogue examples (50-256+) in which the model appears to answer similar harmful questions helpfully. The model, conditioned on this extensive "prior conversation," continues the pattern.

**Structural form:**

```
[Example 1 - fabricated Q&A where model answers harmful question]
[Example 2 - fabricated Q&A where model answers harmful question]
[...N examples...]
[Example N - fabricated Q&A where model answers harmful question]