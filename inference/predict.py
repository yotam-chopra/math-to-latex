import torch

from PIL import Image

from torchvision import transforms

from models.model import MathToLatexModel

from tokenizer.latex_tokenizer import LatexTokenizer

from inference.beam_search import BeamSearch

import pandas as pd


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class Predictor:
    def __init__(
        self,
        model,
        tokenizer,
        max_length=50
    ):
        self.model = model

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")

        image_tensor = self.transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(DEVICE)

    def predict(self, image_path):
        self.model.eval()

        image_tensor = self.preprocess_image(
            image_path
        )

        with torch.no_grad():
            encoder_features = self.model.encoder(
                image_tensor
            )

            beam_search = BeamSearch(
                tokenizer=self.tokenizer,
                beam_width=3,
                max_length=self.max_length
            )

            generated_tokens = beam_search.search(
                decoder=self.model.decoder,
                encoder_features=encoder_features,
                device=DEVICE
            )

        latex = self.tokenizer.decode(
            generated_tokens
        )

        return latex


if __name__ == "__main__":
    labels = pd.read_csv(
        "../data/labels/labels.csv"
    )

    equations = labels["equation"].tolist()

    tokenizer = LatexTokenizer()

    tokenizer.build_vocab(equations)

    vocab_size = len(tokenizer.token_to_id)

    model = MathToLatexModel(
        vocab_size=vocab_size
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(
            "../checkpoints/model.pth",
            map_location=DEVICE
        )
    )

    predictor = Predictor(
        model=model,
        tokenizer=tokenizer
    )

    image_path = "../data/rendered/equation_0.png"

    prediction = predictor.predict(
        image_path
    )

    print("Prediction:")

    print(prediction)