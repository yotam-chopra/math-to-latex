import torch
import os

from torch.utils.data import DataLoader

from tokenizer.latex_tokenizer import LatexTokenizer

from training.dataset import LatexDataset

from models.model import MathToLatexModel

from training.loss import SequenceLoss

from torch.nn.utils.rnn import pad_sequence

import pandas as pd


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def collate_fn(batch):
    images, tokens = zip(*batch)

    images = torch.stack(images)

    tokens = pad_sequence(
        tokens,
        batch_first=True,
        padding_value=0
    )

    return images, tokens

def train():
    labels = pd.read_csv(
        "../data/labels/labels.csv"
    )

    equations = labels["equation"].tolist()

    tokenizer = LatexTokenizer()

    tokenizer.build_vocab(equations)

    tokenizer.save_vocab(
        "vocab/token_vocab.json"
    )

    dataset = LatexDataset(
        image_dir="../data/rendered",
        labels_file="../data/labels/labels.csv",
        tokenizer=tokenizer
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )

    vocab_size = len(tokenizer.token_to_id)

    model = MathToLatexModel(
        vocab_size=vocab_size
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    loss_fn = SequenceLoss()

    epochs = 50

    for epoch in range(epochs):
        model.train()

        total_loss = 0

        for images, tokens in dataloader:
            images = images.to(DEVICE)

            tokens = tokens.to(DEVICE)

            input_tokens = tokens[:, :-1]

            target_tokens = tokens[:, 1:]

            optimizer.zero_grad()

            logits = model(
                images,
                input_tokens
            )

            loss = loss_fn(
                logits,
                target_tokens
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(dataloader)
        )

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"- Loss: {average_loss:.4f}"
        )
        os.makedirs("../checkpoints", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"checkpoints/model_epoch_{epoch+1}.pth"
        )


if __name__ == "__main__":
    train()