import os
import random
import matplotlib
import csv
import random

import matplotlib.pyplot as plt

from synthetic.equation_generator import EquationGenerator


OUTPUT_DIR = "data/rendered"
LABELS_FILE = "data/labels/labels.csv"


class LatexRenderer:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs("data/labels", exist_ok=True)

    def render_equation(self, equation, filename):
        available_fonts = [
            "DejaVu Serif",
            "DejaVu Sans",
            "STIXGeneral"
        ]

        plt.rcParams["font.family"] = (
            random.choice(
                available_fonts
            )
        )
        fig = plt.figure(
            figsize = (
                random.uniform(3, 6),
                random.uniform(1, 2)
            )
        )

        plt.text(
            random.uniform(0.35, 0.65),
            random.uniform(0.35, 0.65),
            f"${equation}$",
            fontsize=random.randint(18, 32),
            ha="center",
            va="center"
        )

        plt.axis("off")

        output_path = os.path.join(OUTPUT_DIR, filename)

        plt.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=random.uniform(
                0.05,
                0.4
            )
        )

        plt.close(fig)

        return output_path


if __name__ == "__main__":
    generator = EquationGenerator(max_depth=2)

    renderer = LatexRenderer()

    with open(LABELS_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["filename", "equation"])

        for i in range(100):
            equation = generator.generate_equation()

            filename = f"equation_{i}.png"

            path = renderer.render_equation(
                equation,
                filename
            )

            writer.writerow([filename, equation])

            print(f"Saved: {path}")
            print(f"Equation: {equation}")