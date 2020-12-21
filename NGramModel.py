import collections
from tqdm import tqdm
import os
from preprocess import clean_sentence
import numpy as np


class NGramModel():
    def __init__(self, train_dir):
        self.queue = collections.deque(maxlen=4)
        self.vocab = set()
        self.unigram = {}
        self.bigram = {}
        self.trigram = {}
        self.quadrigram = {}
        self.total_words = 0

        self.load_model(train_dir)

    def __len__(self):
        return len(self.vocab)

    def load_model(self, train_dir):
        f = open(os.path.join(train_dir), 'r', encoding='utf-8')
        lines = f.readlines()
        pre_vocab = {}

        for line in tqdm(lines):
            self.queue.append('')
            tokens = clean_sentence(line).split()
            for token in tokens:
                self.queue.append(token)

                if token not in pre_vocab:
                    pre_vocab[token] = 1
                else:
                    pre_vocab[token] += 1

                # count self.unigrams
                if token not in self.unigram:
                    self.unigram[token] = 0
                self.unigram[token] += 1

                # use self.queue to count self.bigrams
                if len(self.queue) >= 2:
                    item = tuple(self.queue)[:2]
                    if item not in self.bigram:
                        self.bigram[item] = 0
                    self.bigram[item] += 1

                # use self.queue to count self.trigrams
                if len(self.queue) >= 3:
                    item = tuple(self.queue)[:3]
                    if item not in self.trigram:
                        self.trigram[item] = 0
                    self.trigram[item] += 1

                # use self.queue to count self.quadrigrams
                if len(self.queue) >= 4:
                    item = tuple(self.queue)
                    if item not in self.quadrigram:
                        self.quadrigram[item] = 0
                    self.quadrigram[item] += 1

        self.total_words = len(self.unigram)
        self.unigram[''] = self.total_words
        self.vocab = set(k for k, v in pre_vocab.items() if v > 1)

    def get_prob_unigram(self, word):
        if word not in self.unigram:
            return 0
        return self.unigram[word] / self.total_words

    def get_prob_bigram(self, words):
        if words not in self.bigram:
            return 0
        return self.bigram[words] / self.unigram[words[0]]

    def get_prob_trigram(self, words):
        if words not in self.trigram:
            return 0
        return self.trigram[words] / self.bigram[words[:2]]

    def get_prob_quadrigram(self, words):
        if words not in self.quadrigram:
            return 0
        return self.quadrigram[words] / self.trigram[words[:3]]

    def find_next_word(self, context, limit=10):
        context = clean_sentence(context)
        context = np.concatenate(([''], context.lower().split()))
        first_candidate = ('', 0)
        second_candidate = ('', 0)

        for word in self.vocab:
            if word not in (',', '.', '!', '?'):
                p1 = self.get_prob_unigram((word))
                p2 = self.get_prob_bigram((context[-1], word))
                p3 = self.get_prob_trigram(
                    (context[-2], context[-1], word)) if len(context) >= 3 else 0
                p4 = self.get_prob_quadrigram(
                    (context[-3], context[-2], context[-1], word)) if len(context) >= 4 else 0

                p = 0.02*p1 + 0.2*p2 + 0.3*p3 + 0.4*p4

                if p > first_candidate[1]:
                    second_candidate = first_candidate
                    first_candidate = (word, p)
                elif p > second_candidate[1]:
                    second_candidate = (word, p)

        if first_candidate[0] not in context[-limit:]:
            return first_candidate
        else:
            return second_candidate
