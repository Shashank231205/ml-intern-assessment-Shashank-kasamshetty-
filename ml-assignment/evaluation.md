# Evaluation – Trigram Language Model (1-Page Summary)

This document summarizes the design choices made while implementing the **Trigram (N=3) Language Model** for Task 1 of the AI/ML Intern assessment. The goal was to build a clean, efficient, and fully functional model from scratch using Python.

---

## 1. Storing N‑Gram Counts
The model stores trigram statistics using:

```
trigram[(w1, w2)] → Counter(w3)
bigram_totals[(w1, w2)] → total occurrences
```

**Reasons:**
- `defaultdict(Counter)` allows O(1) updates with no initialization overhead.  
- Tuple keys `(w1, w2)` keep indexing clear and efficient.  
- Counters provide built‑in frequency accumulation and sorting.  
- Structure is memory‑efficient for sparse distributions.

This enables quick probability computation for predicting the next word.

---

## 2. Text Cleaning
A custom preprocessing pipeline ensures consistency across Gutenberg texts:

- Convert text to lowercase  
- Strip irregular unicode (`encode/decode`)  
- Replace digits with `<num>`  
- Remove unnecessary symbols while preserving sentence delimiters  
- Reduce multiple spaces to one  
- Sentence splitting via regex `[.!?]`

**Reason:**  
Gutenberg datasets have inconsistent formatting. Clean, normalized text ensures stable tokenization and model behavior.

---

## 3. Unknown Word Handling
Low‑frequency words are replaced with the `<unk>` token.

### Vocabulary:
- Words with frequency > threshold  
- `<s>`, `</s>`, `<unk>`

**Reason:**  
Rare words produce unseen trigram contexts, leading to probability collapse.  
Using `<unk>` stabilizes training and ensures the model can handle novel text.

---

## 4. Padding Strategy
Each sentence is padded as:

```
<s> <s> w1 w2 ... wn </s>
```

**Reasoning:**
- Trigrams require two previous tokens.  
- Double‑start padding allows correct prediction from the first meaningful token.  
- End token ensures natural termination during generation.

---

## 5. Probabilistic Text Generation
Generation starts with:

```
w1 = <s>, w2 = <s>
```

Then loops:
1. Fetch trigram distribution.
2. Convert counts → probabilities.
3. Sample next word using `random.choices`.
4. Slide window `(w1, w2) → (w2, w3)`.
5. Stop if `</s>` is generated.

**Why sampling instead of greedy?**
- Produces more diverse, natural sentences.  
- Avoids repetitive loops common in greedy decoding.

A deterministic **greedy fallback** generator is also included for debugging.

---

## 6. Additional Decisions
### Top‑K Prediction
A helper returns the most probable next words, useful for debugging or autocomplete evaluation.

### Edge‑Case Handling
- Empty input returns an empty string (to satisfy tests).  
- Unseen bigram contexts default to `<unk>`.

### Modular Structure
Cleaning, tokenization, vocabulary building, padding, and training are separated into dedicated private methods for readability and maintainability.

---

## 7. Conclusion
The implemented Trigram Model is:
- Robust to noisy real‑world text  
- Efficient and mathematically correct  
- Fully probabilistic  
- Cleanly structured and extensible  
- Able to pass all provided unit tests  

This design delivers a reliable N‑gram language model built entirely from first principles.
