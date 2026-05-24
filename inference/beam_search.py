import torch


class BeamSearch:
    def __init__(
        self,
        tokenizer,
        beam_width=3,
        max_length=50
    ):
        self.tokenizer = tokenizer

        self.beam_width = beam_width

        self.max_length = max_length

    def search(
        self,
        decoder,
        encoder_features,
        device
    ):
        start_token = (
            self.tokenizer.token_to_id["<SOS>"]
        )

        end_token = (
            self.tokenizer.token_to_id["<EOS>"]
        )

        beams = [
            ([start_token], 0.0)
        ]

        for _ in range(self.max_length):
            candidates = []

            for tokens, score in beams:
                if tokens[-1] == end_token:
                    candidates.append(
                        (tokens, score)
                    )
                    continue

                token_tensor = torch.tensor(
                    [tokens],
                    dtype=torch.long
                ).to(device)

                with torch.no_grad():
                    logits = decoder(
                        encoder_features,
                        token_tensor
                    )

                log_probs = torch.log_softmax(
                    logits[:, -1, :],
                    dim=-1
                )

                top_probs, top_ids = torch.topk(
                    log_probs,
                    self.beam_width
                )

                for i in range(self.beam_width):
                    next_token = (
                        top_ids[0][i].item()
                    )

                    next_score = (
                        score
                        +
                        top_probs[0][i].item()
                    )

                    next_tokens = (
                        tokens + [next_token]
                    )

                    candidates.append(
                        (
                            next_tokens,
                            next_score
                        )
                    )

            beams = sorted(
                candidates,
                key=lambda x: x[1],
                reverse=True
            )[:self.beam_width]

        best_tokens = beams[0][0]

        return best_tokens