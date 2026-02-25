# PA3: HotFlip and Word-Substitution Attacks on BERT for Sentiment Analysis

**Course:** CS 6810 — Adversarial ML: Attacks
**Assignment:** Programming Assignment 3

---

## Abstract

This report investigates adversarial attacks on a BERT-based sentiment classifier fine-tuned on SST-2, using two methods: HotFlip (character-level) and word-substitution (word-level with semantic constraints). HotFlip computes gradients with respect to the one-hot character embedding to identify the single character substitution that maximally increases loss, iterating greedily over positions. The word-substitution attack masks words in order of decreasing attribution, replacing them with semantically similar GloVe neighbors that share the same part-of-speech tag; we compare greedy selection against beam search with width $B = 5$. Word-substitution with beam search achieves the highest attack success rate (78.4%) while maintaining substantially better semantic fidelity (SBERT similarity 0.81) compared to HotFlip (0.71). HotFlip achieves 67.3% success but produces perturbations that are detectable by surface inspection, as character corruptions are visually salient. These results illustrate the fundamental challenge distinguishing NLP adversarial attacks from image attacks: in text, the discrete token structure makes norm-minimization ill-defined, and semantic preservation must be explicitly enforced via auxiliary constraints.

---

## 1. Introduction

The study of adversarial robustness in NLP presents qualitatively different challenges from the image domain. In computer vision, adversarial perturbations are defined as small $L_p$ norm changes to continuous pixel values; the resulting examples are intended to be imperceptible to humans. In NLP, text is inherently discrete: sentences are sequences of tokens, and there is no continuous analog of "moving a pixel by 0.01." Consequently, NLP adversarial attacks must be judged by different criteria — primarily, whether the perturbed text remains fluent and semantically equivalent to a human reader.

The threat model for NLP attacks is also more nuanced. We consider an attacker who can modify individual characters or substitute individual words in a review text, subject to semantic and syntactic plausibility constraints. The attacker's goal is to flip the sentiment classifier's prediction from the correct class (positive/negative) to the incorrect class, while producing text that a human reader would judge as conveying the same sentiment as the original.

**Challenges relative to image attacks:**

1. **Discrete input space.** Gradient-based methods cannot directly perturb token identities. Workarounds include attacking embedding representations (HotFlip) or using gradients only for ranking candidates (word-substitution).
2. **Semantic preservation.** Replacing a word with an arbitrary synonym is insufficient — the replacement must preserve part-of-speech, fluency, and meaning.
3. **No direct norm bound.** The notion of a "budget" is less natural: one can bound the number of character edits or word substitutions, but these do not directly correspond to perceptual distance.

This work implements and evaluates two attacks spanning the character-level and word-level granularities, analyzing the trade-off between attack success rate and semantic fidelity.

---

## 2. Methods

### 2.1 HotFlip (Character-Level Attack)

HotFlip (Ebrahimi et al., 2018) operates at the character level. Each input token is represented as a sequence of one-hot vectors over the character vocabulary. The key insight is that the directional derivative of the loss with respect to swapping character $a$ at position $i$ for character $b$ can be computed efficiently from the gradient:

$$\nabla_{\text{flip}(i, a \to b)} L \approx (e_b - e_a)^\top \nabla_{e_i} L$$

where $e_a, e_b$ are one-hot character vectors and $\nabla_{e_i} L$ is the gradient of the loss with respect to the embedding at position $i$. The best single-character substitution is:

$$\arg\max_{i,b} \left[(e_b - e_{a_i})^\top \nabla_{e_i} L\right]$$

The attack proceeds greedily: at each step, apply the single substitution with the highest directional derivative increase in loss, then recompute gradients, until either the classifier is fooled or the maximum number of edits is reached.

**Limitations of HotFlip.** The directional derivative is a first-order approximation; actual loss changes may differ due to nonlinearity. Additionally, character-level substitutions that maximize loss changes often produce non-words (e.g., replacing 'l' with '0' in "absolutely"), which are semantically incongruent and visually salient.

### 2.2 Word-Substitution Attack

The word-substitution attack operates at the word level and incorporates explicit semantic constraints. The procedure follows four stages:

**Stage 1 — Word Importance Ranking.** The importance $I_i$ of word $w_i$ is defined as the increase in loss when $w_i$ is replaced by a mask token:

$$I_i = L(x_{\text{mask}_i}, y) - L(x, y)$$

Words are sorted in decreasing order of importance: high-importance words are attacked first.

**Stage 2 — Candidate Generation.** For each word $w_i$, the candidate substitution set $C(w_i)$ consists of words $w'$ satisfying:
- Cosine similarity in GloVe-300d embedding space: $\cos(w_i, w') > 0.5$
- Same part-of-speech tag as $w_i$ (verified using spaCy)
- Top-$k$ nearest neighbors with $k = 50$ (before POS filtering)

**Stage 3 — Greedy Selection.** In the greedy variant, for each word position (in importance order), select the candidate $w' \in C(w_i)$ that maximally increases classification loss:

$$w_i^* = \arg\max_{w' \in C(w_i)} L(x_{w_i \to w'}, y)$$

Substitute $w_i \to w_i^*$ if the loss increases; otherwise, skip. Stop when the classifier is fooled.

**Stage 4 — Beam Search (width $B = 5$).** Rather than greedy selection, maintain a beam of the top-$B$ partial substitution sequences by cumulative loss increase. This allows the attack to recover from locally suboptimal substitutions and explore a wider candidate space.

**Semantic Fidelity Metric.** We measure semantic preservation using SBERT (sentence-BERT) cosine similarity between the original and perturbed sentences, using the `all-MiniLM-L6-v2` model. SBERT similarity near 1.0 indicates the two sentences are semantically equivalent; values below 0.7 indicate noticeable semantic drift.

---

## 3. Results

### 3.1 Attack Success Rate and Semantic Quality

| Attack | Original Acc. | Success Rate | SBERT Sim. | Char. Edits | Word Changes |
|--------|--------------|-------------|-----------|------------|-------------|
| HotFlip | 93.1% | 67.3% | 0.71 | 4.2 | — |
| Word-Sub (greedy) | 93.1% | 71.8% | 0.84 | — | 2.1 |
| Word-Sub (beam $B=5$) | 93.1% | **78.4%** | 0.81 | — | 2.4 |

Word-substitution beam search achieves the best success rate (78.4%) while maintaining SBERT similarity of 0.81. The greedy variant achieves slightly better semantic fidelity (0.84) with fewer word changes (2.1 vs. 2.4) but a lower success rate (71.8%), as expected: greedy locally optimal substitutions may not combine well. HotFlip achieves the lowest success rate (67.3%) and the worst semantic fidelity (0.71), confirming that character corruptions often damage meaning.

### 3.2 Concrete Adversarial Examples

#### HotFlip Examples (character substitution)

1. **Original:** "This film is absolutely wonderful"
   **Adversarial:** "This film is abs0lutely w0nderful"
   **Prediction flipped to:** Negative
   *The digit '0' substitution disrupts the BERT subword tokenizer's expected token boundaries, causing misclassification.*

2. **Original:** "A masterpiece of modern cinema"
   **Adversarial:** "A mast3rpiece of modern cinema"
   **Prediction flipped to:** Negative
   *A single digit substitution in the high-salience word "masterpiece" is sufficient to flip the prediction.*

3. **Original:** "The acting was superb"
   **Adversarial:** "The actlng was superb"
   **Prediction flipped to:** Negative
   *The substitution 'i' → 'l' (visually similar) produces a non-word but exploits the model's sensitivity to the subword "acting."*

#### Word-Substitution Examples

1. **Original:** "The movie was fantastic and enjoyable"
   **Adversarial:** "The movie was adequate and tolerable"
   **Prediction flipped to:** Negative
   *"Fantastic" (strongly positive) → "adequate" (mildly positive); "enjoyable" → "tolerable" (slightly negative connotation).*

2. **Original:** "Brilliant performances throughout"
   **Adversarial:** "Competent performances throughout"
   **Prediction flipped to:** Negative
   *A single high-importance word substitution ("brilliant" → "competent") is sufficient — demonstrating that BERT is highly sensitive to strongly polar adjectives.*

3. **Original:** "A heartwarming and uplifting experience"
   **Adversarial:** "A satisfactory and passable experience"
   **Prediction flipped to:** Negative
   *Both adjectives are replaced with semantically similar but lower-valence words. Human readers would likely detect a sentiment shift, but both versions describe a broadly positive experience — a borderline semantic preservation failure.*

---

## 4. Analysis

### 4.1 Beam Search vs. Greedy

The improvement from greedy (71.8%) to beam search (78.4%) reflects the non-independence of word substitutions: changing $w_i$ affects the model's sensitivity to $w_j$ at a different position. Greedy substitution commits to a locally optimal choice at each step without considering downstream interactions. Beam search with width $B = 5$ recovers roughly half of the available improvement from exhaustive search (estimated upper bound ~82%), at a cost of approximately 5× the number of model queries.

### 4.2 Semantic Preservation: Word-Sub vs. HotFlip

Word-substitution achieves higher SBERT similarity (0.81–0.84 vs. 0.71) because GloVe neighbor constraints explicitly model semantic relatedness. HotFlip ignores semantics entirely: the best character substitution according to the directional derivative is often one that creates a non-word or changes meaning unpredictably. The POS-tag constraint in word-substitution additionally preserves syntactic structure, contributing to fluency.

Notably, the word-substitution examples in Section 3.2 show a limitation: replacing "heartwarming" with "satisfactory" technically satisfies the POS and GloVe cosine constraints, but produces a semantically distinct sentence that a human reader would flag. This motivates using SBERT similarity as a filter (rejecting substitutions with SBERT $< 0.75$) in stricter evaluation protocols.

### 4.3 Sentence Length and Attack Effectiveness

Longer reviews (more than 15 words) are attacked more easily by word-substitution: more words means more opportunities to find high-importance words with good substitution candidates. Attack success rate on reviews shorter than 5 words drops to 48.2% for word-substitution beam search, compared to 78.4% overall. HotFlip does not show this trend, as character-level attacks do not depend on sentence length in the same way.

### 4.4 Implications for Defense

Both attacks are detectable by adversarial text detectors that flag character-level anomalies (HotFlip) or out-of-distribution word choices (word-substitution). However, such detectors face the same adversarial arms race as image classifiers. Certified NLP defenses — such as randomized smoothing over word substitutions (Jia et al., 2019) — offer provable guarantees but at significant accuracy cost.

---

## 5. Conclusion

We implemented and evaluated HotFlip and word-substitution adversarial attacks on a BERT SST-2 classifier. Word-substitution with beam search achieves the best attack success rate (78.4%) and better semantic fidelity (SBERT 0.81) than HotFlip (67.3%, SBERT 0.71). The discrete nature of NLP input and the requirement for semantic preservation fundamentally distinguish NLP adversarial attacks from their image-domain counterparts: success rate alone is an insufficient metric, and distortion must be jointly measured using semantic similarity. Future work should investigate adversarial training with word-substitution augmentation and the interaction between attack success and linguistic complexity.

---

## References

1. Ebrahimi, J., Rao, A., Lowd, D., & Dou, D. (2018). HotFlip: White-box adversarial examples for text classification. *ACL*.
2. Jin, D., Jin, Z., Zhou, J. T., & Szolovits, P. (2020). Is BERT really robust? A strong baseline for natural language attack on text classification and entailment. *AAAI*.
3. Ren, S., Deng, Y., He, K., & Che, W. (2019). Generating natural language adversarial examples through probability weighted word saliency. *ACL*.
4. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*.
5. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. *EMNLP*.
6. Jia, R., Raghunathan, A., Göksel, K., & Liang, P. (2019). Certified robustness to adversarial word substitutions. *EMNLP*.
