import torch

import torch.nn as nn


class LatexDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        num_heads=8,
        num_layers=4,
        max_length=256
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            max_length,
            embedding_dim
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, encoder_features, token_ids):
        batch_size, seq_length = token_ids.shape

        positions = torch.arange(
            seq_length,
            device=token_ids.device
        ).unsqueeze(0)

        token_embeddings = self.token_embedding(token_ids)

        position_embeddings = self.position_embedding(positions)

        decoder_input = (
            token_embeddings
            + position_embeddings
        )

        memory = encoder_features.unsqueeze(1)

        mask = torch.triu(
            torch.ones(
                seq_length,
                seq_length,
                device=token_ids.device
            ),
            diagonal=1
        ).bool()

        output = self.transformer_decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=mask
        )

        logits = self.output_projection(output)

        return logits


if __name__ == "__main__":
    vocab_size = 100

    decoder = LatexDecoder(vocab_size=vocab_size)

    encoder_features = torch.randn(4, 256)

    token_ids = torch.randint(
        0,
        vocab_size,
        (4, 20)
    )

    output = decoder(
        encoder_features,
        token_ids
    )

    print("Decoder output shape:")

    print(output.shape)