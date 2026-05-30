import sys
import torch

from PIL import Image
from torchvision import transforms

from models.model import MathToLatexModel
from tokenizer.latex_tokenizer import LatexTokenizer
from inference.beam_search import BeamSearch


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

    def preprocess_image(
        self,
        image_path
    ):
        image = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = self.transform(
            image
        )

        image_tensor = image_tensor.unsqueeze(
            0
        )

        return image_tensor.to(
            DEVICE
        )

    def predict(
        self,
        image_path
    ):
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


def load_predictor(
    checkpoint_path="training/checkpoints/model_epoch_50.pth"
):
    tokenizer = LatexTokenizer()

    tokenizer.load_vocab(
        "vocab/token_vocab.json"
    )

    vocab_size = len(
        tokenizer.token_to_id
    )

    model = MathToLatexModel(
        vocab_size=vocab_size
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE
        )
    )

    predictor = Predictor(
        model=model,
        tokenizer=tokenizer
    )

    return predictor

def predict_image(
    image_path,
    checkpoint_path="training/checkpoints/model_epoch_50.pth"
):
    predictor = load_predictor(
        checkpoint_path
    )

    return predictor.predict(
        image_path
    )

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python -m inference.predict image.png"
        )
        sys.exit(1)

    image_path = sys.argv[1]

    prediction = predict_image(
        image_path
    )

    print("\nPrediction:\n")
    print(prediction)