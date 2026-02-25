# CS 7800 — Weekly Paper Summary

**Paper:** Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, U., Oprea, A., & Raffel, C. (2021). Extracting Training Data from Large Language Models. In *Proceedings of the 30th USENIX Security Symposium* (pp. 2633–2650).

**Venue:** USENIX Security 2021

---

## Problem Statement

Large language models are trained on massive corpora scraped from the internet, which inevitably contain personally identifiable information (PII), copyrighted text, and other sensitive material. Because these models are known to memorize portions of their training data, a natural threat arises: can an adversary query a deployed model to recover verbatim training sequences, including sensitive content the model was never intended to surface? Prior work had studied memorization theoretically and at small scale, but no paper had demonstrated large-scale, practical extraction of diverse verbatim training data from a production-scale autoregressive LM. This paper provides the first systematic extraction attack against GPT-2, establishing both the feasibility and scope of training data leakage as a concrete privacy threat.

---

## Method

The extraction methodology consists of two phases: **generation** and **membership inference**.

In the **generation phase**, the authors sample a large number of text completions (hundreds of thousands) from GPT-2 using nucleus sampling at a range of temperatures and with a short generic prefix (e.g., a single newline). High temperature increases diversity; low temperature biases toward high-probability, likely-memorized sequences. Both regimes are sampled to maximize coverage.

In the **membership inference phase**, each generated candidate is scored by several heuristic tests designed to identify sequences the model has memorized rather than synthesized:

- **Perplexity ratio test:** The log-perplexity of the candidate under GPT-2 XL is divided by its log-perplexity under a smaller reference model (GPT-2 Small). Sequences that are anomalously low-perplexity relative to their "natural" probability under a weaker model are flagged as likely memorized. Formally: a sequence $x$ is suspicious if $\log p_{\text{large}}(x) / \log p_{\text{small}}(x)$ is high.
- **zlib compression ratio test:** If a sequence has lower perplexity under GPT-2 than its zlib compression entropy (a proxy for natural compressibility), this indicates the model assigns it unnaturally high probability: flag if $-\log p_{\text{GPT-2}}(x) < \text{zlib}(x)$.
- **Lowercase test:** If the perplexity of the original-case sequence is much lower than its lowercased version ($-\log p(x) \ll -\log p(\text{lower}(x))$), this suggests memorization of a specific formatting pattern rather than generalization.

Candidates passing these filters are manually verified against the known GPT-2 training corpus (WebText / Common Crawl), which is publicly available, providing ground-truth confirmation.

---

## Key Results

The authors successfully extract hundreds of verbatim training sequences from GPT-2. The extracted content spans a striking range of sensitive categories: full names paired with physical addresses, phone numbers and email addresses of real individuals, IRC chat logs, source code with identifying comments, and passages from copyrighted literary works and news articles. Larger GPT-2 variants memorize significantly more text than smaller ones — a clear scaling law for memorization risk that has troubling implications as model sizes continue to grow. Across all membership inference tests, the best heuristic (perplexity ratio) achieves approximately 67% precision on flagged sequences, demonstrating that the tests are practical even without ground-truth training data access in deployment scenarios.

---

## Limitations and Open Questions

The most significant limitation is **verifiability**: because GPT-2's training data (WebText) is publicly known, the authors can confirm extraction directly. For models with private training corpora — GPT-3, GPT-4, Gemini — there is no ground truth, and membership inference is necessarily probabilistic. It is therefore unclear how much training data can be extracted from frontier models, or how to measure it without internal data access.

A second limitation is **efficiency**: the attack requires generating hundreds of thousands of samples, which is expensive at API pricing and may trigger rate limits. Two important open questions follow naturally. First, does RLHF fine-tuning reduce memorization, since the RL phase may reinforce or suppress specific sequences? Preliminary evidence suggests RLHF does not reliably eliminate memorization. Second, would training with formal differential privacy (DP-SGD) prevent extraction? The answer is likely yes for small privacy budgets $\varepsilon \lesssim 1$, but at unacceptable utility cost for frontier-scale models, representing a fundamental tension between model capability and privacy.

---

## My Assessment

This paper is a landmark in LLM security and remains one of the clearest demonstrations that training large models on uncurated internet data creates genuine privacy liability, not merely theoretical risk. The diversity of extracted content — PII, code, literature — drives home that the threat is not confined to a single data category. The most surprising result is the sheer specificity of what was extracted: full mailing addresses paired with names, not just name fragments. The implications for training data governance are significant and underappreciated by the industry.

If I were extending this work, I would design an extraction attack that minimizes query budget by actively steering generation toward high-risk content categories using a fine-tuned prefix predictor, and I would test whether extraction rates against instruction-tuned models differ systematically from base models — both because RLHF changes the output distribution and because instruction-tuned models may respond differently to the short generic prompts used in this paper's generation phase.
