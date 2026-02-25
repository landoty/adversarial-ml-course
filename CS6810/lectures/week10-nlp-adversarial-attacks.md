# CS 6810 — Adversarial Machine Learning
## Week 10: Adversarial Attacks on NLP Models — Discrete Inputs and the Semantic Constraint Problem

**Prerequisites:** Weeks 01–06 (image attacks), familiarity with transformer architectures (BERT, GPT), word embeddings (GloVe, Word2Vec), and NLP tasks (classification, NLI, QA).

**Learning Objectives:**
1. Articulate the fundamental differences between NLP and image attacks.
2. Derive the HotFlip gradient update for character-level attacks.
3. Implement a word substitution attack with POS and semantic constraints.
4. Explain how universal adversarial triggers exploit model-level shortcuts.
5. Evaluate adversarial NLP examples using semantic similarity metrics beyond L-p norms.

---

## 1. Why NLP Attacks Are Fundamentally Different

### 1.1 The Discrete Token Space Problem

Image attacks operate in a continuous domain: pixel values are real numbers in $[0,1]$, and we can add an infinitesimally small perturbation $\epsilon \to 0$ and still have a valid image. Gradients of the loss with respect to the input exist and are meaningful.

NLP inputs are sequences of discrete tokens drawn from a finite vocabulary $\mathcal{V}$ (e.g., $|\mathcal{V}| = 50,000$ for BERT). A text $x = (t_1, t_2, \ldots, t_m)$ is a sequence of token indices. There is no meaningful notion of "adding a real-valued perturbation $\delta$ to $x$" — the perturbed input $x + \delta$ is not a valid token sequence unless we round to the nearest token.

**The fundamental obstruction:** Neural networks for NLP look up token embeddings: $e_i = \text{Embedding}(t_i) \in \mathbb{R}^d$. The embedding lookup is a non-differentiable operation (argument is an integer index). The gradient $\nabla_x \mathcal{L}$ does not exist in the token-index space.

**Workaround strategies:**
1. Compute gradients with respect to the *embedding* $e_i$, not the token index $t_i$. Use these to score which token substitutions are beneficial.
2. Use black-box query-based strategies that do not require gradients.
3. Use search algorithms (beam search, greedy search) over the space of valid substitutions.

### 1.2 No Natural L-p Ball

In image attacks, the L-p ball $\{\delta : \|\delta\|_p \leq \epsilon\}$ provides a natural constraint for "imperceptibility." The choice of $p$ and $\epsilon$ can be related (imperfectly) to human perception via just-noticeable difference (JND) thresholds.

In NLP, there is no obvious analog. A substitution attack that replaces "good" with "excellent" changes the token entirely — the "distance" between "good" and "excellent" in token space is undefined. Various metrics have been proposed:

- **Edit distance (Levenshtein distance):** Number of character-level insertions, deletions, and substitutions needed to transform one text into another. Natural for character-level attacks.
- **Word substitution budget:** Maximum number of words that may be replaced. Natural for word-level attacks.
- **Semantic similarity:** The meaning of the text should not change. Measured by cosine similarity of sentence embeddings (SBERT, USE). This is the primary constraint in modern NLP attacks.
- **Grammaticality:** The perturbed text should be grammatically correct. Measured by perplexity under a language model.

The "budget" in NLP attacks is multi-dimensional: one must jointly bound the number of substitutions, the semantic drift, and the grammatical acceptability.

### 1.3 The Semantic Preservation Constraint

The core requirement for NLP adversarial attacks is: a human reading $x$ and $x'$ should assign the same label. This is fundamentally harder to formalize than "the image looks the same."

**Why cross-entropy on human labels doesn't fully work:** Human labels have uncertainty, context dependence, and annotation artifacts. An example that 90% of human annotators label as "positive" and 10% label as "neutral" doesn't have a clear ground truth.

**Operational definition (standard in the field):**
- The adversarial example $x'$ must have the same true label as $x$.
- The semantic similarity $\text{sim}(x, x') \geq \lambda$ for some threshold $\lambda$ (typically $\lambda = 0.84$ for USE cosine similarity).
- The word substitution rate $|x' \oplus x| / |x| \leq \rho$ (typically $\rho = 0.15$–$0.30$, i.e., at most 15–30% of words changed).

---

## 2. The Three Levels of NLP Attacks

### 2.1 Character-Level Attacks

**Target:** Change individual characters within words. Examples:
- Typos: "advertisement" → "advertis*e*ment" (insert 'e').
- Swaps: "example" → "exmaple" (swap adjacent characters).
- Phonetically similar substitutions: "there" → "their" (different word, same pronunciation).

**Strengths:** Preserves most of the original word; looks like a typo. Hard for models without character-level processing.

**Weaknesses:** Easily detectable by spell-checkers. Less effective against models with robust tokenization (BERT uses WordPiece which handles typos reasonably well).

**Representative method:** HotFlip (character-level version, Section 4).

### 2.2 Word-Level Attacks

**Target:** Replace, insert, or delete entire words. Examples:
- Synonyms: "happy" → "joyful".
- Antonym with negation removal: "not sad" → "not happy" (semantic reversal — carefully avoided).
- Importance-weighted substitution: replace the words most important to the classifier's decision.

**Strengths:** More impactful per perturbation than character-level; can exploit the model's reliance on specific words.

**Weaknesses:** Harder to maintain semantic similarity; may produce grammatically awkward sentences.

**Representative methods:** TextFooler, BERT-Attack, HotFlip (word-level).

### 2.3 Sentence-Level Attacks

**Target:** Generate entirely new sentences that preserve the meaning but change the prediction. Examples:
- Paraphrasing: "The movie was great" → "The film was wonderful."
- Adding irrelevant clauses: "The movie was terrible. [Unrelated statement appended.]"
- Style transfer: changing formality or tense.

**Strengths:** Natural-looking perturbations that are hard to detect.

**Weaknesses:** Hard to generate with tight semantic constraints; requires a powerful generative model.

**Representative methods:** Paraphrase attacks using GPT-2/GPT-3, universal adversarial triggers (sentence-level).

---

## 3. HotFlip: Gradient-Based Character-Level Attack

### 3.1 Setup

Ebrahimi et al. (2018) propose HotFlip — a gradient-based attack that operates on character sequences.

**Input representation:** Let $x = (c_1, c_2, \ldots, c_m)$ be a sequence of characters, where each character $c_i \in \mathcal{A}$ (an alphabet, e.g., $|\mathcal{A}| = 70$ characters for English). Each character is one-hot encoded: $\mathbf{e}_{c_i} \in \{0,1\}^{|\mathcal{A}|}$. The input to the model is the concatenation $\mathbf{E} = [\mathbf{e}_{c_1}, \ldots, \mathbf{e}_{c_m}] \in \{0,1\}^{m \times |\mathcal{A}|}$.

### 3.2 The Gradient in One-Hot Space

The model is a function $f : \{0,1\}^{m \times |\mathcal{A}|} \to \mathbb{R}^K$ (which, at the input layer, is effectively $f : \mathbb{R}^{m \times |\mathcal{A}|} \to \mathbb{R}^K$ if we allow the one-hot inputs to be any real vector — i.e., we relax the discrete constraint to compute gradients).

The Jacobian of the loss $\mathcal{L}$ with respect to the input matrix $\mathbf{E}$ is:

$$J = \nabla_\mathbf{E} \mathcal{L} \in \mathbb{R}^{m \times |\mathcal{A}|}$$

where $J_{i,a}$ is the partial derivative of $\mathcal{L}$ with respect to the one-hot component at position $i$ for character $a$.

### 3.3 The HotFlip Score

A "flip" at position $i$ changes character $c_i$ to a new character $b$. In one-hot space, this changes $\mathbf{e}_{c_i}$ to $\mathbf{e}_b$. The direction in one-hot space is:

$$\mathbf{v}_{i,b} = \mathbf{e}_b - \mathbf{e}_{c_i} \tag{1}$$

The first-order approximation of the loss change from this flip is:

$$\Delta \mathcal{L}(i, b) \approx J_i^\top \mathbf{v}_{i,b} = J_{i,b} - J_{i,c_i} \tag{2}$$

where $J_i \in \mathbb{R}^{|\mathcal{A}|}$ is the $i$-th row of the Jacobian (gradient of loss w.r.t. character $i$'s one-hot vector).

**For an untargeted attack:** We want to find the flip $(i, b)$ that maximally *increases* the loss:

$$\hat{i}, \hat{b} = \arg\max_{i, b \neq c_i} (J_{i,b} - J_{i,c_i}) \tag{3}$$

This requires just one backward pass to compute $J$, and then a scan over all $(i, b)$ pairs — $O(m \times |\mathcal{A}|)$ work after the backward pass.

**For a targeted attack:** Maximize $-J_{i,b} + J_{i,c_i}$ — minimize the loss of the target class.

### 3.4 Full HotFlip Character-Level Algorithm

```
INPUT: text x = (c_1, ..., c_m), model f, true label y, budget B (max flips)
OUTPUT: adversarial text x' = (c'_1, ..., c'_m)

x' = x  (start with original)
for flip in 1..B:
    Compute one-hot matrix E from x'
    Compute J = ∇_E L(f(E), y)  via one backward pass
    for each position i, character b ≠ c'_i:
        score[i, b] = J[i, b] - J[i, c'_i]  // equation (2)
    (i*, b*) = argmax_{i,b} score[i,b]
    Apply flip: c'_{i*} = b*
    if f(x') ≠ y:  // misclassification achieved
        return x'
return x'  (or failure if no misclassification)
```

### 3.5 Derivation of the Gradient for Character $i$

Consider a character-level CNN model where the first layer computes:

$$h_{i,k} = \sum_{a=1}^{|\mathcal{A}|} W_{k,a} \cdot e_{c_i, a}$$

For a one-hot $\mathbf{e}_{c_i}$: $h_{i,k} = W_{k,c_i}$ (the $c_i$-th column of $W$).

The Jacobian entry:

$$J_{i,a} = \frac{\partial \mathcal{L}}{\partial e_{c_i, a}} = \sum_k \frac{\partial \mathcal{L}}{\partial h_{i,k}} \cdot W_{k,a} \tag{4}$$

The score for flipping position $i$ from $c_i$ to $b$:

$$J_{i,b} - J_{i,c_i} = \sum_k \frac{\partial \mathcal{L}}{\partial h_{i,k}} \cdot (W_{k,b} - W_{k,c_i}) \tag{5}$$

This is the inner product of the upstream gradient $\partial \mathcal{L} / \partial h_i$ with the difference in embedding columns $W_{k,b} - W_{k,c_i}$. Geometrically: the flip score is proportional to the cosine similarity between the gradient direction and the embedding displacement vector.

### 3.6 Extension to Word-Level HotFlip

For word-level attacks, treat each word $w_i$ (not each character) as a unit. Replace the one-hot alphabet $\mathcal{A}$ with the vocabulary $\mathcal{V}$. The word embedding matrix is $W_{\text{emb}} \in \mathbb{R}^{|\mathcal{V}| \times d}$.

The gradient of the loss with respect to the embedding of word $i$ is:

$$g_i = \nabla_{e_{w_i}} \mathcal{L} \in \mathbb{R}^d \tag{6}$$

The score for replacing word $i$ with word $v \in \mathcal{V}$:

$$\text{score}(i, v) = g_i^\top (W_{\text{emb}}[v] - W_{\text{emb}}[w_i]) \tag{7}$$

This approximates the first-order loss change from the word substitution. The optimal replacement is:

$$\hat{v} = \arg\max_{v \neq w_i} g_i^\top W_{\text{emb}}[v] \tag{8}$$

(the $W_{\text{emb}}[w_i]$ term is constant and doesn't affect the argmax).

**Practical issue:** The best-scoring word $\hat{v}$ by gradient is often semantically unrelated to $w_i$ ("hospital" replaced by "elephant"). The gradient-based score is a raw attack score, not a semantic score. We must add semantic constraints (Section 5).

---

## 4. Token Substitution Attacks Using Language Model Neighbors

### 4.1 Finding Semantically Similar Substitutions

The key challenge in word-level attacks is finding substitute words that:
1. Are semantically similar to the original (preserve meaning).
2. Are syntactically compatible (maintain grammatical correctness).
3. Maximize the attack loss (cause misclassification).

**Step 1: Candidate generation.** For each word $w_i$ in the input, generate a set of candidate substitutions $\mathcal{C}(w_i)$:

**GloVe/Word2Vec cosine similarity:** Find the $k$ nearest neighbors of $w_i$ in the GloVe embedding space:

$$\mathcal{C}_{\text{GloVe}}(w_i) = \text{TopK}_{v \in \mathcal{V}} \frac{\text{GloVe}(w_i) \cdot \text{GloVe}(v)}{\|\text{GloVe}(w_i)\| \cdot \|\text{GloVe}(v)\|} \tag{9}$$

Typical $k = 50$. This gives semantically related words but may include antonyms (e.g., "good" and "bad" are neighbors in GloVe because they appear in similar contexts).

**BERT-based substitution (BERT-Attack):** Use BERT's masked language model to find contextually appropriate substitutes:

1. Replace word $i$ with [MASK]: $x_{\text{masked}} = (w_1, \ldots, w_{i-1}, \text{[MASK]}, w_{i+1}, \ldots, w_m)$.
2. Query BERT-MLM to get the probability distribution over the vocabulary at position $i$: $p_{\text{MLM}}(\cdot | x_{\text{masked}})$.
3. Candidates: top-$k$ tokens by $p_{\text{MLM}}(v | x_{\text{masked}})$.

BERT-MLM is context-aware: it considers the surrounding words when proposing substitutes, yielding more grammatically appropriate candidates than GloVe.

### 4.2 POS-Tag Constraints

A candidate substitution $v$ for word $w_i$ is filtered out if it has a different part-of-speech (POS) tag than $w_i$. This preserves grammatical structure:

- Nouns should be replaced by nouns.
- Verbs by verbs.
- Adjectives by adjectives.
- Function words (articles, prepositions) are typically not substituted.

**Implementation using NLTK:**
```python
import nltk
pos_original = nltk.pos_tag([word])[0][1]  # e.g., 'NN', 'VBZ', 'JJ'
candidates_filtered = [v for v in candidates if
                       nltk.pos_tag([v])[0][1] == pos_original]
```

### 4.3 Semantic Similarity Filter

After POS filtering, apply a semantic similarity constraint. Each candidate $v$ is accepted only if the resulting sentence $x'$ is semantically similar to $x$ above a threshold:

$$\text{USE}(x) \cdot \text{USE}(x') / (\|\text{USE}(x)\| \cdot \|\text{USE}(x')\|) \geq \lambda_{\text{USE}} \tag{10}$$

where $\text{USE}(\cdot)$ is the Universal Sentence Encoder sentence embedding (or SBERT embedding). Typical threshold: $\lambda_{\text{USE}} = 0.84$.

### 4.4 Greedy vs. Beam Search Over Substitutions

**Greedy (one-pass):** Process words in order of their importance (gradient magnitude). For each important word, choose the best valid substitute (highest attack score after POS and semantic filtering). Accept the substitute and move to the next word.

**Beam search:** Maintain a beam of $B$ partial solutions (partial substitutions). At each step, expand each partial solution by substituting the next important word with the top-$B$ candidates. Prune to keep the $B$ best solutions by some heuristic (e.g., attack loss). Return the best complete solution.

**Importance ranking:** Process words in decreasing order of their impact on the model's output. A proxy for importance: $|f(x_{-i})_y - f(x)_y|$ where $x_{-i}$ is the input with word $i$ deleted. This requires $m$ model queries. Alternative: use the gradient magnitude $\|g_i\|_2$ (equation 6) — one backward pass only.

---

## 5. BERT-Attack: Masked Language Model Substitution

### 5.1 Motivation

BERT-Attack (Li et al. 2020) uses BERT's masked language model (MLM) head as the candidate generator. This is more powerful than GloVe because:
1. BERT captures contextual semantics — the candidates are appropriate for the specific sentence context, not just the word in isolation.
2. BERT's vocabulary includes subwords (WordPiece), handling morphological variants.

### 5.2 Attack Algorithm

```
INPUT: text x = (w_1, ..., w_m), BERT classifier f_cls, BERT-MLM f_mlm, budget B
OUTPUT: adversarial text x'

1. IMPORTANCE RANKING:
   For each word i in 1..m:
     x_{-i} = x with w_i replaced by [MASK]
     importance[i] = f_cls(x)[y] - f_cls(x_{-i})[y]
   Sort words by decreasing importance.

2. SUBSTITUTION:
   x' = x
   substitutions = 0
   For word i in sorted order:
     if substitutions >= B: break
     x_masked = x' with w'_i replaced by [MASK]
     mlm_probs = f_mlm(x_masked)[position i]    # distribution over vocab
     candidates = Top-K tokens by mlm_probs
     candidates = filter by POS tag
     candidates = filter by semantic similarity (SBERT score ≥ λ)
     if candidates is empty: continue
     // Pick candidate that most increases attack loss
     best_v = argmax_{v in candidates} attack_score(x' with w'_i → v)
     x'_i = best_v
     substitutions += 1
     if f_cls(x') ≠ y: return x'

3. Return x' (or failure)
```

### 5.3 Computational Cost

BERT-Attack requires:
- $m$ BERT-MLM forward passes for importance ranking.
- Up to $m$ BERT-MLM forward passes for candidate generation.
- Up to $m \times K$ BERT classifier forward passes for candidate scoring (can be batched).

Total: $O(m^2 K)$ forward passes, where $K$ is the candidate set size. For $m = 50$ tokens and $K = 50$ candidates: 125,000 forward passes. This is expensive; greedy selection (without exhaustive scoring) reduces to $O(mK)$.

---

## 6. Universal Adversarial Triggers

### 6.1 Concept

Wallace et al. (2019) discovered that a short sequence of tokens $t_1, \ldots, t_k$ (the "trigger") can be prepended to *any* input and cause the model to misclassify:

$$C([\text{trigger}, x]) = t \quad \text{for all } x \in \mathcal{X} \tag{11}$$

Unlike per-instance adversarial examples, a trigger is universal — one fixed sequence attacks all inputs simultaneously. This is analogous to universal adversarial perturbations in image attacks (Moosavi-Dezfooli et al. 2017) but in the NLP domain.

**Why this is alarming:** A trigger can be found by an attacker and then deployed at scale. Any user who happens to use a trigger phrase ("the pig died") in their query will have their query misclassified, with no visible sign of attack.

### 6.2 The Gradient-Based Trigger Search

**Objective:** Find a sequence of tokens $\mathbf{t} = (t_1, \ldots, t_k)$ that minimizes:

$$\min_{\mathbf{t} \in \mathcal{V}^k} \mathbb{E}_{(x, y) \sim \mathcal{D}}\!\left[\mathcal{L}(f([\mathbf{t}, x]), y)\right] \tag{12}$$

for a targeted attack (we want the model to predict $y_{\text{target}}$ for all inputs), or:

$$\max_{\mathbf{t} \in \mathcal{V}^k} \mathbb{E}_{(x, y) \sim \mathcal{D}}\!\left[\mathcal{L}(f([\mathbf{t}, x]), y)\right] \tag{13}$$

for an untargeted attack.

**Algorithm (Wallace et al. 2019):** Use HotFlip-style gradient search at the trigger positions.

At each iteration:
1. Compute the average gradient of the loss over a batch of examples $\{(x_1, y_1), \ldots, (x_B, y_B)\}$ with the trigger prepended.
2. For each trigger position $i$ (within the trigger sequence), compute the token replacement score:

$$\text{score}(i, v) = \frac{1}{B}\sum_{b=1}^B g_{i,b}^\top W_{\text{emb}}[v] \tag{14}$$

where $g_{i,b} = \nabla_{e_{t_i}} \mathcal{L}(f([\mathbf{t}, x_b]), y_b)$ is the gradient at trigger position $i$ for example $b$.

3. Replace trigger token $i$ with $\hat{v} = \arg\max_{v} \text{score}(i, v)$.

4. Repeat until convergence.

**Initialization:** Start with $\mathbf{t} = (\text{"the"}, \text{"the"}, \ldots)$ or a random sequence.

### 6.3 Why Universal Triggers Work: Model Shortcuts

The trigger search exploits a fundamental property of NLP models: they rely on *spurious correlations* in training data. These are input features that correlate with the label in the training data but are not causally related to the label in the real world.

**Example (NLI task, "contradiction" class):** NLI models trained on MultiNLI learn that the word "not" and negation patterns correlate with contradiction labels. A trigger that prepends tokens activating negation-related features causes the model to predict "contradiction" for any input.

**Example (Reading comprehension):** Models rely heavily on lexical overlap between the question and the context passage. A trigger that disrupts this overlap can cause the model to answer questions incorrectly.

**Example (Hate speech detection):** Models may correlate identity terms with hate speech. Triggers can include identity terms to flip benign content to "hate speech."

### 6.4 Trigger Effectiveness Across Tasks

Wallace et al. (2019) demonstrate triggers for:

**Sentiment classification (SST-2):** 3-token trigger causes ~100% of positive reviews to be classified as negative.

**NLI (MNLI):** 3-token trigger causes ~99% of pairs to be classified as "contradiction" regardless of semantic relationship.

**Reading comprehension (SQuAD):** Trigger prepended to passages causes models to generate specific wrong answers.

**Text generation (GPT-2):** Trigger causes GPT-2 to generate racist text from neutral prompts.

### 6.5 Defenses Against Triggers

1. **Trigger detection:** Look for short sequences that cause consistent prediction changes across diverse inputs. A trigger shifts the model's prediction distribution in a predictable way.
2. **Input sanitization:** Detect OOD token sequences (the trigger is often grammatically anomalous).
3. **Robust training:** Train on examples with random trigger prefixes. This reduces effectiveness but does not eliminate it.

---

## 7. The Semantic Similarity Evaluation Problem

### 7.1 Why L-p Metrics Fail for NLP

The standard image attack evaluation — "is the perturbation within the L-p ball of radius $\epsilon$?" — has no direct analog for text. We need metrics that capture semantic equivalence.

### 7.2 Universal Sentence Encoder (USE)

The Universal Sentence Encoder (Cer et al. 2018) produces a 512-dimensional embedding of any sentence via a transformer or deep averaging network. Semantic similarity is measured by cosine similarity between sentence embeddings:

$$\text{sim}_{\text{USE}}(x, x') = \frac{\text{USE}(x) \cdot \text{USE}(x')}{\|\text{USE}(x)\| \cdot \|\text{USE}(x')\|} \tag{15}$$

**Threshold in the literature:** $\text{sim}_{\text{USE}} \geq 0.84$ is used by TextFooler as the acceptance criterion. This corresponds roughly to "same meaning, possibly different phrasing."

**Limitation:** USE cosine similarity is a coarse metric. Two sentences can have high USE similarity but differ in important ways (negation, entity changes). Example: "The patient has cancer" vs. "The patient does not have cancer" — high similarity (same topic, similar words), but opposite meaning.

### 7.3 BERTScore

BERTScore (Zhang et al. 2020) measures similarity by aligning token representations from BERT and computing average cosine similarity:

$$\text{BS-P}(x, x') = \frac{1}{|x'|} \sum_{t' \in x'} \max_{t \in x} \frac{\mathbf{h}_{t'}^\top \mathbf{h}_t}{\|\mathbf{h}_{t'}\| \cdot \|\mathbf{h}_t\|} \tag{16}$$

$$\text{BS-R}(x, x') = \frac{1}{|x|} \sum_{t \in x} \max_{t' \in x'} \frac{\mathbf{h}_t^\top \mathbf{h}_{t'}}{\|\mathbf{h}_t\| \cdot \|\mathbf{h}_{t'}\|} \tag{17}$$

$$\text{BERTScore-F1}(x, x') = 2 \cdot \frac{\text{BS-P} \cdot \text{BS-R}}{\text{BS-P} + \text{BS-R}} \tag{18}$$

where $\mathbf{h}_t$ is the BERT contextual embedding of token $t$ in its respective sentence.

BERTScore is more sensitive to semantic changes than USE because it works at the token level and uses contextual embeddings. It is now the standard metric for evaluating NLP adversarial examples.

**Typical threshold:** $\text{BERTScore-F1} \geq 0.90$ (roughly equivalent to USE $\geq 0.84$).

### 7.4 Human Evaluation as Gold Standard

Despite the utility of automated metrics, human evaluation remains the gold standard. Standard protocol:
1. Show a human annotator both the original and adversarial sentence.
2. Ask: "Do these sentences have the same meaning? (Yes/No)"
3. Ask: "Which label would you assign to each? (Original label / Alternative)"
4. Compute: % of pairs where human says "same meaning" and assigns same label = **attack success with semantic preservation**.

In most published papers, human evaluation agrees with automated metrics (USE, BERTScore) at 80–90% rates. Disagreements are informative — they reveal failure modes of the automated metrics.

---

## 8. TextFooler: The Standard Word Substitution Attack

### 8.1 Algorithm Overview

TextFooler (Jin et al. 2020) is the most widely used word-level adversarial attack baseline. It combines:
1. Importance ranking by word deletion.
2. GloVe-based candidate generation.
3. POS filtering.
4. USE semantic similarity filtering.
5. Greedy substitution in importance order.

### 8.2 Importance Ranking

For each word $w_i$, compute its importance as the drop in the model's confidence on the true class when $w_i$ is deleted:

$$\text{importance}(w_i) = p_y(x) - p_y(x \setminus w_i) \tag{19}$$

where $x \setminus w_i$ is the sentence with $w_i$ deleted. High importance = the classifier relies heavily on this word.

**Query cost:** $m$ model queries for a sentence of $m$ words.

### 8.3 Substitution Filtering

For word $w_i$ (selected in importance order), the substitution pipeline:

1. **GloVe neighbors:** Find the 50 nearest neighbors of $\text{GloVe}(w_i)$ in cosine similarity.
2. **Semantic diversity filter:** Remove neighbors with USE similarity above 0.5 to the original word alone (to remove near-identical words that provide no attack benefit).
3. **POS filter:** Remove candidates with different POS tag.
4. **Semantic similarity check:** For each remaining candidate $v$, compute $\text{sim}_{\text{USE}}(x', x)$ where $x'$ has $w_i$ replaced by $v$. Accept only if $\text{sim}_{\text{USE}} \geq 0.84$.
5. **Select best:** Among accepted candidates, choose the one with the highest loss (or lowest true-class probability).

### 8.4 Empirical Performance

On SST-2 (sentiment analysis, BERT):
- Clean accuracy: 93.0%.
- Accuracy after TextFooler: 7.8% (attack success rate: 91.6%).
- Average words changed: 3.4 (out of average sentence length 17 words = 20%).
- Average USE similarity: 0.91.
- Average BERTScore-F1: 0.89.

On MNLI (natural language inference, BERT):
- Clean accuracy: 83.5%.
- Accuracy after TextFooler: 15.3% (attack success rate: 81.7%).
- Average words changed: 5.1.

---

## 9. Comparing NLP Attack Methods

| Method | Level | Gradient? | Semantic Constraint | Quality | Queries |
|--------|-------|-----------|--------------------|---------|---------|
| HotFlip (char) | Character | Yes | None | Low (typos) | 1 backward pass |
| HotFlip (word) | Word | Yes | None | Low (random substitutes) | 1 backward pass |
| TextFooler | Word | No | USE ≥ 0.84 | Medium | $O(m \times K)$ |
| BERT-Attack | Word | No | USE ≥ 0.84 | High | $O(m^2 K)$ |
| Universal Triggers | Prefix | Yes | None (not preserved) | N/A (universal) | $O(B \times T)$ |
| Paraphrase attacks | Sentence | No | SBERT similarity | High | 1 generation |

---

## 10. Discussion Questions

1. **Discreteness barrier:** Explain why the gradient $\nabla_{t_i} \mathcal{L}$ (gradient w.r.t. the token index $t_i$, an integer) is undefined, while the gradient $\nabla_{e_i} \mathcal{L}$ (gradient w.r.t. the embedding vector $e_i$) is well-defined. What additional step is needed to go from the embedding gradient to a token substitution decision?

2. **HotFlip score derivation:** In equation (7), the word replacement score is $g_i^\top (W_{\text{emb}}[v] - W_{\text{emb}}[w_i])$. Show that this equals $g_i^\top W_{\text{emb}}[v] + \text{const}$ where the constant does not depend on $v$. Why does this mean that the optimal word replacement is $\hat{v} = \arg\max_v g_i^\top W_{\text{emb}}[v]$?

3. **Semantic metric comparison:** Construct an adversarial example (manually) where:
   (a) USE similarity is high ($> 0.9$) but the meaning is changed.
   (b) BERTScore-F1 is low ($< 0.8$) but the meaning is preserved.
   Do such examples reveal fundamental limitations of these metrics? What would a better metric look like?

4. **Universal trigger budget:** The universal trigger search (equation 12) minimizes the *expected* loss over the training distribution $\mathcal{D}$. In practice, only a finite batch $B$ is used at each step. How large must $B$ be for the trigger to generalize to new inputs? Derive a sample complexity bound using VC theory or Rademacher complexity.

5. **Trigger detection:** Propose an algorithm to detect universal triggers at deployment time. Your algorithm should: (a) not require knowledge of the trigger, (b) add at most 5 model queries per input, (c) achieve false positive rate $< 1\%$ on clean inputs. Analyze the trade-off between detection sensitivity and false positive rate.

6. **POS constraint limitation:** POS filtering prevents replacing nouns with verbs. But sometimes a noun-to-noun substitution that is semantically related is not in GloVe's top-50 neighbors (e.g., "cancer" and "malignancy" may not be close in GloVe). Propose an improved candidate generation scheme that uses both GloVe similarity and medical/domain ontologies (e.g., UMLS for medical text). How would you integrate ontology-based synonymy with the GloVe ranking?

7. **BERT-Attack efficiency:** BERT-Attack requires $O(m^2 K)$ forward passes. For a sentence of length $m = 100$ words with $K = 50$ candidates per word, compute the total number of BERT forward passes. If a BERT forward pass takes 50ms, how long does BERT-Attack take per example? Propose a pruning strategy that reduces this by 10× while retaining 90% of the attack success rate.

8. **Cross-task trigger transfer:** A universal trigger is found for sentiment classification (SST-2). Can the same trigger be used against NLI (MNLI)? What is the transfer rate? Explain theoretically why cross-task trigger transfer is harder than same-task.

---

## 11. Further Reading

**Required:**
- Ebrahimi et al. (2018). "HotFlip: White-Box Adversarial Examples for Text Classification." ACL. [HotFlip]
- Wallace et al. (2019). "Universal Adversarial Triggers for Attacking and Analyzing NLP." EMNLP. [Universal Triggers]
- Jin et al. (2020). "Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment." AAAI. [TextFooler]
- Li et al. (2020). "BERT-Attack: Adversarial Attack Against BERT Using BERT." EMNLP. [BERT-Attack]

**Recommended:**
- Zhang et al. (2020). "BERTScore: Evaluating Text Generation with BERT." ICLR. [BERTScore metric]
- Cer et al. (2018). "Universal Sentence Encoder." arXiv. [USE metric]
- Morris et al. (2020). "TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP." EMNLP. [Evaluation framework]
- Moosavi-Dezfooli et al. (2017). "Universal Adversarial Perturbations." CVPR. [Image universal perturbations — NLP analog]
- Iyyer et al. (2018). "Adversarial Example Generation with Syntactically Controlled Paraphrase Networks." NAACL. [Paraphrase-based attacks]
- Garg & Ramakrishnan (2020). "BAE: BERT-based Adversarial Examples for Text Classification." EMNLP. [BERT-based word substitution]
