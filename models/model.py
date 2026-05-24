import torch

import torch.nn as nn

from models.encoder import ImageEncoder

from models.decoder import LatexDecoder


class MathToLatexModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=256
    ):
        super().__init__()

        self.encoder = ImageEncoder(
            embedding_dim=embedding_dim
        )

        self.decoder = LatexDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )

    def forward(self, images, token_ids):
        encoder_features = self.encoder(images)

        logits = self.decoder(
            encoder_features,
            token_ids
        )

        return logits


if __name__ == "__main__":
    vocab_size = 100

    model = MathToLatexModel(
        vocab_size=vocab_size
    )

    dummy_images = torch.randn(
        4,
        1,
        224,
        224
    )

    dummy_tokens = torch.randint(
        0,
        vocab_size,
        (4, 20)
    )

    output = model(
        dummy_images,
        dummy_tokens
    )

    print("Model output shape:")

    print(output.shape)