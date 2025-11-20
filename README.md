# Trigram Language Model (Assessment 1)

This folder contains the implementation for **Task 1** of the AI/ML Intern assessment:  
building a **Trigram (N=3) Language Model** from scratch using Python.

The implementation includes text cleaning, tokenization, unknown-word handling, vocabulary building, trigram counting, and probabilistic text generation.

---

##  Project Structure

```
ml-assignment/
│
├── data/
│   └── example_corpus.txt        # Training corpus
│
├── src/
│   ├── ngram_model.py            # Trigram model implementation
│   ├── generate.py               # Script to train + generate text
│   └── utils.py                  # (Provided in starter repo)
│
├── tests/
│   └── test_ngram.py             # Provided test cases
│
└── evaluation.md                 # Design explanation
```

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

---

##  Running the Trigram Model

Run commands from inside:

```
ml-intern-assessment-Shashank-kasamshetty-/ml-assignment
```

---

### **1. Ensure the training corpus exists**

The model reads the file:

```
data/example_corpus.txt
```

Replace this file if you want to train on another Project Gutenberg book.

---

### **2. Train the model & generate text**

```bash
python src/generate.py
```

This will:

- Load `example_corpus.txt`
- Clean & tokenize text
- Build trigram counts
- Generate a sample sentence

**Example output:**

```
Generated Text:
the rabbit looked quite surprised when alice walked into the garden
```

---

## Running Tests

Tests are located in:

```
tests/test_ngram.py
```

Run them using:

```bash
python -m pytest
```

All tests must pass:

- fit() + generate()
- empty text handling
- short text handling

---

This README describes **Assessment 1 only**.
