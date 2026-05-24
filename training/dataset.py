import os
import pandas as py
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tokenizer.latex_tokenizer import LatexTokenizer

class LatexDataset(Dataset):
    def __init__(
            self,
            image_dir,
            labels_file,
            tokenizer,
            image_size = (224, 224)
    ):
        self.image_dir = image_dir
        self.data = py.read_csv(labels_file)

        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = os.path.join(
            self.image_dir,
            row["filename"]
        )

        equation = row["equation"]

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        token_ids = self.tokenizer.encode(equation)
        token_tensor = torch.tensor(
            token_ids,
            dtype = torch.long
        )

        return image_tensor, token_tensor


if __name__ == "__main__":
    tokenizer = LatexTokenizer()

    labels = py.read_csv("../data/labels/labels.csv")

    equations = labels["equation"].tolist()

    tokenizer.build_vocab(equations)

    dataset = LatexDataset(
        image_dir="../data/rendered",
        labels_file="../data/labels/labels.csv",
        tokenizer=tokenizer
    )

    image, tokens = dataset[0]

    print("Image shape:")
    print(image.shape)

    print("\nToken IDs:")
    print(tokens)

