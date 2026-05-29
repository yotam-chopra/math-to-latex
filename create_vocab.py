import pandas as pd

from tokenizer.latex_tokenizer import LatexTokenizer

labels = pd.read_csv(
    "data/labels/labels.csv"
)

equations = labels["equation"].tolist()

tokenizer = LatexTokenizer()

tokenizer.build_vocab(
    equations
)

tokenizer.save_vocab(
    "vocab/token_vocab.json"
)

print("Saved vocab.")