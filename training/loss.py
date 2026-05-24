import torch

import torch.nn as nn


class SequenceLoss(nn.Module):
    def __init__(self, pad_token_id=0):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_token_id
        )

    def forward(self, logits, targets):
        batch_size, seq_length, vocab_size = logits.shape

        logits = logits.reshape(
            batch_size * seq_length,
            vocab_size
        )

        targets = targets.reshape(
            batch_size * seq_length
        )

        loss = self.criterion(
            logits,
            targets
        )

        return loss


if __name__ == "__main__":
    batch_size = 4

    seq_length = 20

    vocab_size = 100

    logits = torch.randn(
        batch_size,
        seq_length,
        vocab_size
    )

    targets = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_length)
    )

    loss_fn = SequenceLoss()

    loss = loss_fn(
        logits,
        targets
    )

    print("Loss:")

    print(loss.item())