# CS 7800 — Weekly Paper Summary

**Paper:** Zou, A., Wang, Z., Kolter, J. Z., & Fredrikson, M. (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models. *arXiv preprint arXiv:2307.15043*.

**Venue:** arXiv 2023

---

## Problem Statement

Aligned large language models (LLMs) such as GPT-4, Claude, and LLaMA-2 are trained with reinforcement learning from human feedback (RLHF) and supervised fine-tuning to refuse harmful requests. Prior adversarial work on language models focused on classification tasks or required white-box access to craft instance-specific attacks. No prior work had demonstrated a systematic, automated method for generating adversarial prompts that (1) cause aligned LLMs to produce harmful content, (2) generalize across many different harmful queries using a single reusable suffix, and (3) transfer to black-box commercial models. This paper fills that gap, showing that alignment fine-tuning does not constitute a robust safety guarantee against optimization-based adversarial attacks.

---

## Method

The paper introduces the **Greedy Coordinate Gradient (GCG)** attack, which optimizes a discrete adversarial suffix appended to a user's harmful prompt. The objective is to maximize the log-probability that the model begins its response with an affirmative token sequence such as "Sure, here is" — a behavior that empirically predicts continued harmful completion.

Formally, given a prompt $x$ and a target affirmative prefix $y^*$, GCG searches for a suffix $s$ of fixed length $l$ such that:

$$s^* = \arg\min_s \mathcal{L}(x \| s, y^*)$$

where $\mathcal{L}$ is the negative log-likelihood of the target prefix under the model.

Because the token vocabulary is discrete, standard gradient descent is inapplicable. GCG instead: (1) computes the gradient of the loss with respect to each token's one-hot embedding at each suffix position; (2) uses this gradient to identify the top-$k$ candidate replacement tokens per position; (3) evaluates a random batch of candidate substitutions by forward pass; and (4) selects the substitution yielding the greatest loss reduction. This greedy coordinate search runs for hundreds of steps.

To produce a **universal** suffix, GCG optimizes over a dataset of many harmful prompts simultaneously, minimizing the sum of losses. The resulting suffix transfers to held-out prompts. Transfer to black-box models (ChatGPT, GPT-4, Claude, Bard) is evaluated by appending the white-box-optimized suffix to queries submitted via API, with no additional optimization.

---

## Key Results

On open-source models (Vicuna-7B, Vicuna-13B, LLaMA-2-7B-Chat), GCG achieves attack success rates (ASR) exceeding 80–99% on the harmful behaviors benchmark, substantially outperforming prior manual jailbreaks. The universal suffix, optimized on 25 training behaviors, generalizes to held-out behaviors with only modest ASR degradation. Most strikingly, the suffix transfers to fully black-box commercial models: the authors report meaningful ASR on ChatGPT (~47%), GPT-4, Claude, and Bard, despite those models never being queried during optimization. This transfer result is the most significant finding, because it implies that adversarial vulnerabilities learned on open-source surrogates generalize across model families.

---

## Limitations and Open Questions

Several important limitations constrain the practical impact of this work. First, ASR drops substantially on newer, more heavily aligned models; GPT-4 and Claude 2 show lower but non-negligible transfer rates, and subsequent alignment improvements appear to reduce susceptibility further. Second, the attack is partially defeated by straightforward defenses: perplexity-based input filtering detects the high-perplexity adversarial suffix with reasonable accuracy, and paraphrase-based preprocessing disrupts the exact token sequence required. Third, the attack requires white-box gradient access to at least one open-source model, which may not always be available to real adversaries who lack the compute for multi-step optimization. Open questions include: Can alignment techniques be designed that provably resist this attack class? Does the transfer rate improve with better surrogate model selection? And as frontier models become more capable, does the optimization landscape become easier or harder to exploit?

---

## My Assessment

This is one of the most significant adversarial ML papers of 2023. The core result — that a single optimized suffix breaks alignment across model families — is genuinely surprising and unsettling, because it suggests that RLHF-based alignment is far more brittle than the field had assumed. The most striking finding is not the open-source ASR (high white-box performance was expected) but the black-box transfer, which implies that aligned models share exploitable structural regularities. If I were extending this work, I would investigate whether adaptive defenses trained specifically to detect GCG-style suffixes can be circumvented by modifying the attack objective, and whether multi-objective GCG that jointly optimizes for low perplexity and high ASR can defeat perplexity-based filters simultaneously.
