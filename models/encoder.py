import torch
import torch.nn as nn
from torchvision.models import resnet18

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim = 256):
        super().__init__()

        backbone = resnet18(weights = None)
        backbone.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projection = nn.Linear(
            512,
            embedding_dim
        )

    def forward(self, images):
        features = self.feature_extractor(images)

        pooled = self.pool(features)

        pooled = pooled.flatten(1)

        embeddings = self.projection(pooled)

        return embeddings


if __name__ == "__main__":
    encoder = ImageEncoder()

    dummy_images = torch.randn(4, 1, 224, 224)

    output = encoder(dummy_images)

    print("Encoder output shape:")

    print(output.shape)