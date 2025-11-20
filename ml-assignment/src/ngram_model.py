import re
import random
from collections import defaultdict, Counter
from math import sqrt


class TrigramModel:
    def __init__(self, unk_threshold=1, seed=None):
        self.trigram = defaultdict(Counter)
        self.bigram_totals = Counter()
        self.word_freq = Counter()
        self.vocab = set()

        self.BOS = "<s>"
        self.EOS = "</s>"
        self.UNK = "<unk>"

        self.unk_threshold = unk_threshold
        if seed is not None:
            random.seed(seed)

    def _prepare_text(self, text):
        text = text.lower()
        text = text.encode("ascii", "ignore").decode()
        text = re.sub(r"\d+", "<num>", text)
        text = re.sub(r"[^a-z0-9<>\s,.!?]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]

    def _tokenize(self, sentences):
        return [s.split() for s in sentences]

    def _build_vocab(self, tokenized):
        for sent in tokenized:
            self.word_freq.update(sent)

        for word, freq in self.word_freq.items():
            if freq > self.unk_threshold:
                self.vocab.add(word)

        self.vocab.update({self.BOS, self.EOS, self.UNK})

    def _replace_rare(self, tokenized):
        rep = []
        for sent in tokenized:
            rep.append([w if w in self.vocab else self.UNK for w in sent])
        return rep

    def _pad(self, tokenized):
        out = []
        for sent in tokenized:
            out.append([self.BOS, self.BOS] + sent + [self.EOS])
        return out

    def fit(self, text):
        cleaned = self._prepare_text(text)
        tokenized = self._tokenize(cleaned)
        self._build_vocab(tokenized)
        processed = self._replace_rare(tokenized)
        padded = self._pad(processed)

        for sent in padded:
            for i in range(len(sent) - 2):
                w1, w2, w3 = sent[i], sent[i + 1], sent[i + 2]
                self.trigram[(w1, w2)][w3] += 1
                self.bigram_totals[(w1, w2)] += 1

    def _next_word_distribution(self, w1, w2):
        dist = self.trigram.get((w1, w2))
        if not dist:
            return {}

        total = self.bigram_totals[(w1, w2)]
        return {w: c / total for w, c in dist.items()}

    def generate(self, max_length=50):
        # REQUIRED FIX FOR TESTS
        if not self.trigram or not self.bigram_totals:
            return ""

        w1, w2 = self.BOS, self.BOS
        sequence = []

        for _ in range(max_length):
            if (w1, w2) not in self.trigram:
                break

            dist = self._next_word_distribution(w1, w2)
            if not dist:
                break

            words = list(dist.keys())
            probs = list(dist.values())
            nxt = random.choices(words, probs, k=1)[0]

            if nxt == self.EOS:
                break

            sequence.append(nxt)
            w1, w2 = w2, nxt

        return " ".join(sequence)

    def generate_greedy(self, max_length=50):
        if not self.trigram or not self.bigram_totals:
            return ""

        w1, w2 = self.BOS, self.BOS
        out = []

        for _ in range(max_length):
            if (w1, w2) not in self.trigram:
                break

            dist = self._next_word_distribution(w1, w2)
            if not dist:
                break

            nxt = max(dist.items(), key=lambda x: x[1])[0]
            if nxt == self.EOS:
                break

            out.append(nxt)
            w1, w2 = w2, nxt

        return " ".join(out)

    def predict_topk(self, w1, w2, k=5):
        dist = self._next_word_distribution(w1, w2)
        sorted_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]
