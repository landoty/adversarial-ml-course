"""
CS 6810 — Programming Assignment 3: NLP Adversarial Attacks
Complete Solution

Implements:
1. HotFlip character-level attack on BERT SST-2 classifier
2. Word substitution attack using GloVe + POS constraints + beam search
3. Evaluation: attack success rate, edit distance, semantic similarity (SBERT)
4. Adversarial example showcase (20 examples with analysis)

Requirements:
    pip install torch transformers sentence-transformers nltk numpy tqdm
    python -m nltk.downloader punkt averaged_perceptron_tagger wordnet

Usage:
    python PA3-nlp-attacks-solution.py
"""

import os
import re
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# HuggingFace
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# Sentence Transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    HAS_SBERT = True
except ImportError:
    print('[!] sentence-transformers not installed. Semantic similarity skipped.')
    HAS_SBERT = False

# NLTK for POS tagging
import nltk
try:
    from nltk import pos_tag, word_tokenize
    from nltk.corpus import wordnet
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False
    print('[!] NLTK unavailable. POS-constrained word substitution will be simplified.')

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_ATTACK = 100          # Number of examples to attack
BEAM_WIDTH = 5          # Beam width for word substitution attack
MAX_WORD_CHANGES = 5    # Max word substitutions per example
PLOT_DIR = './pa3_results'
os.makedirs(PLOT_DIR, exist_ok=True)

# SST-2 labels: 0=negative, 1=positive
LABEL_NAMES = ['negative', 'positive']


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load BERT Model and Tokenizer for SST-2
# ─────────────────────────────────────────────────────────────────────────────

def load_bert_sst2():
    """
    Load a pretrained BERT model fine-tuned on SST-2.
    Uses HuggingFace 'textattack/bert-base-uncased-SST-2' (~93% accuracy).
    """
    model_name = 'textattack/bert-base-uncased-SST-2'
    print(f'[+] Loading BERT SST-2 model: {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, texts, batch_size=32):
    """
    Get model predictions and probabilities for a list of strings.

    Returns:
        preds:  [N] integer predicted labels
        probs:  [N, 2] softmax probability distributions
    """
    all_preds, all_probs = [], []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors='pt')
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            out    = model(**enc)
            probs  = F.softmax(out.logits, dim=1)
            preds  = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_probs.append(probs.cpu())

    return all_preds, torch.cat(all_probs)


def get_loss_and_grad(model, tokenizer, text: str, true_label: int):
    """
    Compute the CE loss and gradient of the loss w.r.t. the input word embeddings.
    Used by HotFlip to find the most influential token substitution.

    Returns:
        loss:  Scalar loss value
        embed_grad: [seq_len, hidden_dim] gradient of loss w.r.t. token embeddings
    """
    enc = tokenizer([text], padding=True, truncation=True,
                    max_length=128, return_tensors='pt')
    input_ids = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)
    y = torch.tensor([true_label]).to(DEVICE)

    # Get embeddings with gradient tracking
    embed_layer = model.bert.embeddings.word_embeddings
    embeddings = embed_layer(input_ids)  # [1, seq_len, hidden_dim]
    embeddings.retain_grad()

    # Forward pass with manual embedding injection
    outputs = model(inputs_embeds=embeddings,
                    attention_mask=attention_mask)
    loss = F.cross_entropy(outputs.logits, y)
    loss.backward()

    return loss.item(), embeddings.grad.squeeze(0)  # [seq_len, hidden_dim]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HotFlip Attack (Character-Level)
#     Ebrahimi et al. 2018
#
#     At each step, find the character substitution that maximally increases
#     the classification loss, using a first-order gradient approximation.
# ─────────────────────────────────────────────────────────────────────────────

CHAR_VOCAB = list('abcdefghijklmnopqrstuvwxyz')

def hotflip_char(model, tokenizer, text: str, true_label: int,
                 max_flips: int = 5) -> str:
    """
    HotFlip character-level attack.

    Strategy (simplified from Ebrahimi et al.):
    For each word in the input, try substituting each character with each
    character in the vocabulary. Select the substitution that maximally
    increases CE loss. Repeat up to max_flips times.

    Note: We use a greedy character-level search rather than the exact
    gradient-based approach (which requires character-level embedding tables
    separate from BERT's WordPiece tokenization).

    Returns:
        text_adv: Adversarially modified text (or original if no improvement)
    """
    text_adv = text
    n_flips = 0

    for _ in range(max_flips):
        words = text_adv.split()
        best_loss = -1e9
        best_text = text_adv

        # Try flipping each character in each word
        for w_idx, word in enumerate(words):
            if len(word) < 2:
                continue
            for c_idx in range(len(word)):
                for new_char in CHAR_VOCAB:
                    if new_char == word[c_idx].lower():
                        continue
                    # Construct candidate
                    new_word = word[:c_idx] + new_char + word[c_idx+1:]
                    candidate_words = words.copy()
                    candidate_words[w_idx] = new_word
                    candidate_text = ' '.join(candidate_words)

                    # Evaluate loss
                    enc = tokenizer([candidate_text], padding=True,
                                    truncation=True, max_length=128,
                                    return_tensors='pt')
                    enc = {k: v.to(DEVICE) for k, v in enc.items()}
                    y   = torch.tensor([true_label]).to(DEVICE)

                    with torch.no_grad():
                        out  = model(**enc)
                        loss = F.cross_entropy(out.logits, y).item()

                    if loss > best_loss:
                        best_loss = loss
                        best_text = candidate_text

        # Check if we flipped the prediction
        preds, _ = predict(model, tokenizer, [best_text])
        if preds[0] != true_label:
            return best_text, n_flips + 1

        if best_text == text_adv:
            break  # No improvement found

        text_adv = best_text
        n_flips += 1

    return text_adv, n_flips


def hotflip_attack(model, tokenizer, texts, labels):
    """
    Run HotFlip on a list of texts.

    Returns:
        results: List of dicts with original, adversarial, success, n_flips
    """
    results = []
    for text, label in tqdm(zip(texts, labels),
                             total=len(texts), desc='HotFlip'):
        # Only attack correctly classified examples
        pred, _ = predict(model, tokenizer, [text])
        if pred[0] != label:
            continue

        text_adv, n_flips = hotflip_char(model, tokenizer, text, label)

        pred_adv, probs_adv = predict(model, tokenizer, [text_adv])
        success = (pred_adv[0] != label)

        # Edit distance
        from difflib import SequenceMatcher
        edit_ratio = 1 - SequenceMatcher(None, text, text_adv).ratio()

        results.append({
            'original':     text,
            'adversarial':  text_adv,
            'true_label':   label,
            'adv_label':    pred_adv[0],
            'success':      success,
            'n_flips':      n_flips,
            'edit_dist':    sum(a != b for a, b in zip(text, text_adv)),
            'edit_ratio':   edit_ratio,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Word Substitution Attack (GloVe + POS + Beam Search)
# ─────────────────────────────────────────────────────────────────────────────

def load_glove(glove_path: str, vocab_size: int = 50000):
    """
    Load GloVe embeddings from a text file.
    Returns: word2vec dict mapping word → np.array.

    If glove_path doesn't exist, we use a random fallback (for demo purposes).
    Download GloVe: https://nlp.stanford.edu/projects/glove/
    glove.6B.100d.txt recommended.
    """
    if not os.path.exists(glove_path):
        print(f'[!] GloVe not found at {glove_path}.')
        print('[!] Using random embeddings as fallback (attack will be weak).')
        # Build a small random vocab from common English words
        import string
        random.seed(42)
        common_words = [
            'good', 'bad', 'great', 'terrible', 'excellent', 'poor', 'wonderful',
            'awful', 'nice', 'horrible', 'amazing', 'dreadful', 'superb', 'mediocre',
            'fantastic', 'dull', 'brilliant', 'boring', 'enjoy', 'hate', 'love',
            'dislike', 'happy', 'sad', 'funny', 'serious', 'moving', 'flat',
            'interesting', 'tedious', 'original', 'generic',
        ]
        dim = 100
        word2vec = {w: np.random.randn(dim).astype(np.float32) for w in common_words}
        return word2vec

    word2vec = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= vocab_size:
                break
            parts = line.strip().split()
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            word2vec[word] = vec
    print(f'[+] Loaded {len(word2vec)} GloVe vectors')
    return word2vec


def cosine_similarity_np(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)


def get_word_candidates(word: str, word2vec: dict, top_k: int = 50,
                        min_sim: float = 0.6) -> list:
    """
    Find the top_k most similar words to 'word' in GloVe space.
    Returns list of candidate words with cosine similarity ≥ min_sim.
    """
    if word not in word2vec:
        return []

    query_vec = word2vec[word]
    candidates = []

    for candidate, vec in word2vec.items():
        if candidate == word:
            continue
        sim = cosine_similarity_np(query_vec, vec)
        if sim >= min_sim:
            candidates.append((candidate, sim))

    # Sort by similarity, take top_k
    candidates.sort(key=lambda x: -x[1])
    return [c[0] for c in candidates[:top_k]]


def get_word_pos(word: str) -> str:
    """Return simplified POS tag (N, V, J, R, or O) using NLTK."""
    if not HAS_NLTK:
        return 'O'
    try:
        tags = pos_tag([word])
        tag = tags[0][1]
        if tag.startswith('NN'):
            return 'N'
        elif tag.startswith('VB'):
            return 'V'
        elif tag.startswith('JJ'):
            return 'J'
        elif tag.startswith('RB'):
            return 'R'
        else:
            return 'O'
    except Exception:
        return 'O'


def word_substitution_attack_beam(model, tokenizer, text: str, true_label: int,
                                   word2vec: dict, beam_width: int = BEAM_WIDTH,
                                   max_changes: int = MAX_WORD_CHANGES):
    """
    Word substitution attack with beam search.

    At each step, expand the beam by substituting the most influential word
    with its top GloVe-similar candidates (respecting POS constraints).
    Keep the top beam_width candidates by model loss.

    Args:
        model:       BERT classifier
        tokenizer:   BERT tokenizer
        text:        Input text (correctly classified)
        true_label:  Original true label
        word2vec:    GloVe embedding dictionary
        beam_width:  Beam size
        max_changes: Maximum substitutions

    Returns:
        best_adv:    Best adversarial text found
        n_changes:   Number of word substitutions made
    """
    words = text.split()

    # Find word importance scores: how much does removing each word increase loss?
    importance_scores = []
    enc_orig = tokenizer([text], padding=True, truncation=True,
                          max_length=128, return_tensors='pt')
    enc_orig = {k: v.to(DEVICE) for k, v in enc_orig.items()}
    with torch.no_grad():
        loss_orig = F.cross_entropy(
            model(**enc_orig).logits,
            torch.tensor([true_label]).to(DEVICE)
        ).item()

    for i, word in enumerate(words):
        # Mask this word
        masked_words = words.copy()
        masked_words[i] = '[UNK]'
        masked_text = ' '.join(masked_words)
        enc_mask = tokenizer([masked_text], padding=True, truncation=True,
                              max_length=128, return_tensors='pt')
        enc_mask = {k: v.to(DEVICE) for k, v in enc_mask.items()}
        with torch.no_grad():
            loss_mask = F.cross_entropy(
                model(**enc_mask).logits,
                torch.tensor([true_label]).to(DEVICE)
            ).item()
        # Importance: masking this word reduces loss (makes it easier) → high importance
        importance_scores.append((i, loss_orig - loss_mask))

    # Sort words by importance (most important first)
    importance_scores.sort(key=lambda x: -x[1])
    word_order = [i for i, _ in importance_scores]

    # Beam: list of (words_list, n_substitutions, loss)
    beam = [(words.copy(), 0, loss_orig)]

    for idx in word_order[:max_changes]:
        if not beam:
            break

        word = words[idx]
        pos  = get_word_pos(word)

        # Get substitution candidates
        candidates = get_word_candidates(word.lower(), word2vec, top_k=50)
        # Filter by POS tag (must match original)
        if HAS_NLTK and pos in ('N', 'V', 'J', 'R'):
            candidates = [c for c in candidates
                          if get_word_pos(c) == pos]

        if not candidates:
            continue

        # Expand beam
        new_beam = []
        for beam_words, n_sub, _ in beam:
            for candidate in candidates[:10]:  # Top 10 per beam entry
                new_words = beam_words.copy()
                new_words[idx] = candidate
                candidate_text = ' '.join(new_words)

                enc_c = tokenizer([candidate_text], padding=True, truncation=True,
                                   max_length=128, return_tensors='pt')
                enc_c = {k: v.to(DEVICE) for k, v in enc_c.items()}
                with torch.no_grad():
                    loss_c = F.cross_entropy(
                        model(**enc_c).logits,
                        torch.tensor([true_label]).to(DEVICE)
                    ).item()

                new_beam.append((new_words, n_sub + 1, loss_c))

                # Early exit if successful
                with torch.no_grad():
                    pred = model(**enc_c).logits.argmax(1).item()
                if pred != true_label:
                    return candidate_text, n_sub + 1

        if not new_beam:
            continue

        # Keep top beam_width by highest loss (most adversarial)
        new_beam.sort(key=lambda x: -x[2])
        beam = new_beam[:beam_width]

    # Return the best (highest loss) candidate in the final beam
    if beam:
        best_words, n_sub, _ = beam[0]
        return ' '.join(best_words), n_sub
    return text, 0


def word_substitution_attack(model, tokenizer, texts, labels, word2vec):
    """Run word substitution attack on a list of texts."""
    results = []
    for text, label in tqdm(zip(texts, labels),
                             total=len(texts), desc='WordSub'):
        pred, _ = predict(model, tokenizer, [text])
        if pred[0] != label:
            continue

        text_adv, n_changes = word_substitution_attack_beam(
            model, tokenizer, text, label, word2vec)

        pred_adv, _ = predict(model, tokenizer, [text_adv])
        success = (pred_adv[0] != label)

        results.append({
            'original':    text,
            'adversarial': text_adv,
            'true_label':  label,
            'adv_label':   pred_adv[0],
            'success':     success,
            'n_changes':   n_changes,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Semantic Similarity (SBERT)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sbert_similarity(results: list, sbert_model) -> list:
    """
    Compute SBERT cosine similarity between original and adversarial texts.
    Adds 'sbert_sim' key to each result dict.
    """
    originals    = [r['original']    for r in results]
    adversarials = [r['adversarial'] for r in results]

    emb_orig = sbert_model.encode(originals,    convert_to_tensor=True, show_progress_bar=False)
    emb_adv  = sbert_model.encode(adversarials, convert_to_tensor=True, show_progress_bar=False)

    sims = st_util.cos_sim(emb_orig, emb_adv).diag().cpu().tolist()
    for r, s in zip(results, sims):
        r['sbert_sim'] = s

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SST-2 Data Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_sst2_samples(n: int = N_ATTACK):
    """
    Load SST-2 validation samples. Uses the HuggingFace datasets library,
    falling back to manually curated examples if unavailable.
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset('glue', 'sst2', split='validation')
        texts  = [dataset[i]['sentence'] for i in range(min(n, len(dataset)))]
        labels = [dataset[i]['label']    for i in range(min(n, len(dataset)))]
        return texts, labels
    except Exception:
        pass

    # Fallback: a small set of representative SST-2 sentences
    fallback = [
        ("a visually stunning film that lacks emotional depth", 0),
        ("the best film of the year by a wide margin", 1),
        ("so dull that you will watch it through half-closed eyelids", 0),
        ("a wonderful and moving experience for the whole family", 1),
        ("the acting is wooden and the plot is predictable", 0),
        ("one of the most original and brilliant comedies in years", 1),
        ("painfully boring and utterly forgettable in every way", 0),
        ("a genuinely funny and surprisingly touching comedy", 1),
        ("the director seems to have no idea what he wants to say", 0),
        ("simply the most powerful film of the decade", 1),
        ("too long too self-indulgent and nowhere near funny enough", 0),
        ("an absolute delight from start to finish", 1),
        ("leaves the viewer feeling empty and cheated", 0),
        ("a triumphant achievement in every respect", 1),
        ("a complete waste of everyone's time and talent", 0),
    ]
    texts  = [t for t, _ in fallback[:n]]
    labels = [l for _, l in fallback[:n]]
    return texts, labels


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Report Generation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list) -> dict:
    successful = [r for r in results if r['success']]
    asr = len(successful) / len(results) if results else 0.0

    metrics = {
        'n_total':   len(results),
        'n_success': len(successful),
        'asr':       asr,
    }

    if 'edit_dist' in results[0] if results else False:
        metrics['avg_edit_dist'] = np.mean([r['edit_dist'] for r in successful]) if successful else 0
    if 'n_changes' in results[0] if results else False:
        metrics['avg_word_changes'] = np.mean([r['n_changes'] for r in successful]) if successful else 0
    if 'sbert_sim' in results[0] if results else False:
        metrics['avg_sbert_sim'] = np.mean([r['sbert_sim'] for r in results]) if results else 0

    return metrics


def print_showcase(hotflip_results, wordsub_results, n: int = 10):
    """Print side-by-side original vs. adversarial examples."""
    print('\n' + '='*70)
    print('ADVERSARIAL EXAMPLE SHOWCASE')
    print('='*70)

    successful_hf = [r for r in hotflip_results if r['success']][:n//2]
    successful_ws = [r for r in wordsub_results if r['success']][:n//2]

    for title, examples in [('HotFlip Examples', successful_hf),
                              ('Word Substitution Examples', successful_ws)]:
        print(f'\n--- {title} ---')
        for i, ex in enumerate(examples):
            print(f'\nExample {i+1}:')
            print(f'  Original  [{LABEL_NAMES[ex["true_label"]]}]: '
                  f'{ex["original"][:120]}')
            print(f'  Adversarial [{LABEL_NAMES[ex["adv_label"]]}]: '
                  f'{ex["adversarial"][:120]}')
            if 'sbert_sim' in ex:
                print(f'  SBERT similarity: {ex["sbert_sim"]:.3f}')
            if 'n_flips' in ex:
                print(f'  Character flips: {ex["n_flips"]}')
            if 'n_changes' in ex:
                print(f'  Word changes: {ex["n_changes"]}')


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('CS 6810 PA3 — NLP Adversarial Attacks')
    print(f'Device: {DEVICE}')
    print('=' * 60)

    # Load model and data
    model, tokenizer = load_bert_sst2()
    texts, labels = load_sst2_samples(N_ATTACK)
    print(f'[+] Loaded {len(texts)} SST-2 samples')

    # Load GloVe (adjust path if needed)
    glove_path = './glove.6B.100d.txt'
    word2vec   = load_glove(glove_path)

    # Load SBERT for semantic similarity
    if HAS_SBERT:
        print('[+] Loading SBERT model...')
        sbert = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        sbert = None

    # ── 1. HotFlip Attack ──
    print('\n[1] Running HotFlip (character-level) attack...')
    hotflip_results = hotflip_attack(model, tokenizer, texts, labels)
    if sbert:
        hotflip_results = compute_sbert_similarity(hotflip_results, sbert)

    hf_metrics = compute_metrics(hotflip_results)
    print(f'\nHotFlip Metrics:')
    print(f'  Attack success rate: {hf_metrics["asr"]*100:.1f}%'
          f' ({hf_metrics["n_success"]}/{hf_metrics["n_total"]})')
    if 'avg_edit_dist' in hf_metrics:
        print(f'  Avg char edit distance (successful): {hf_metrics["avg_edit_dist"]:.1f}')
    if 'avg_sbert_sim' in hf_metrics:
        print(f'  Avg SBERT similarity: {hf_metrics["avg_sbert_sim"]:.3f}')

    # ── 2. Word Substitution Attack ──
    print('\n[2] Running Word Substitution (GloVe + POS + beam search) attack...')
    wordsub_results = word_substitution_attack(
        model, tokenizer, texts, labels, word2vec)
    if sbert:
        wordsub_results = compute_sbert_similarity(wordsub_results, sbert)

    ws_metrics = compute_metrics(wordsub_results)
    print(f'\nWord Substitution Metrics:')
    print(f'  Attack success rate: {ws_metrics["asr"]*100:.1f}%'
          f' ({ws_metrics["n_success"]}/{ws_metrics["n_total"]})')
    if 'avg_word_changes' in ws_metrics:
        print(f'  Avg word changes (successful): {ws_metrics["avg_word_changes"]:.1f}')
    if 'avg_sbert_sim' in ws_metrics:
        print(f'  Avg SBERT similarity: {ws_metrics["avg_sbert_sim"]:.3f}')

    # ── 3. Showcase ──
    print_showcase(hotflip_results, wordsub_results, n=20)

    # ── 4. Summary Table ──
    print('\n' + '='*60)
    print('COMPARISON SUMMARY')
    print('='*60)
    print(f'{"Metric":<35} {"HotFlip":>12} {"Word Sub":>12}')
    print('-' * 62)
    print(f'{"Attack Success Rate":<35} '
          f'{hf_metrics["asr"]*100:>11.1f}% '
          f'{ws_metrics["asr"]*100:>11.1f}%')
    if 'avg_sbert_sim' in hf_metrics and 'avg_sbert_sim' in ws_metrics:
        print(f'{"Avg SBERT Semantic Similarity":<35} '
              f'{hf_metrics["avg_sbert_sim"]:>12.3f} '
              f'{ws_metrics["avg_sbert_sim"]:>12.3f}')

    print('\n[+] Done.')


if __name__ == '__main__':
    main()
