import re
from collections import Counter


class LatexTokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

        self.special_tokens = [
            "<PAD>",
            "<SOS>",
            "<EOS>",
            "<UNK>"
        ]
    def tokenize(self, latex):
            pattern = r'(\\[a-zA-Z]+|\\.|[{}_^]|[0-9]+|[a-zA-Z]|[^\s])'

            tokens = re.findall(pattern, latex)

            return tokens
    def build_vocab(self, latex_list, min_freq = 1):
        counter = Counter()

        for latex in latex_list:
            tokens = self.tokenize(latex)
            counter.update(tokens)

        vocab = self.special_tokens.copy()

        for token, freq in counter.items():
            if freq >=  min_freq:
                vocab.append(token)

        self.token_to_id = {
            token: idx
            for idx, token in enumerate(vocab)
        }

        self.id_to_token = {
            idx: token
            for token, idx in self.token_to_id.items()
        }

    def encode(self, latex):
        tokens = self.tokenize(latex)

        encoded = [
            self.token_to_id.get(token, self.token_to_id["<UNK>"])
            for token in tokens
            ]

        encoded = (
            [self.token_to_id["<SOS>"]]
            + encoded
            + [self.token_to_id["<EOS>"]]
        )

        return encoded

    def decode(self, ids):
        tokens = []

        for idx in ids:
            token = self.id_to_token.get(idx, "<UNK>")

            if token in ["<SOS>", "<PAD>"]:
                continue

            if token == "<EOS>":
                break

            tokens.append(token)

        return ''.join(tokens)
