import os
import csv

import matplotlib.pyplot as plt

from equation_generator import EquationGenerator


OUTPUT_DIR = "../data/rendered"
LABELS_FILE = "../data/labels/labels.csv"


class LatexRenderer:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs("../data/labels", exist_ok=True)

    def render_equation(self, equation, filename):
        fig = plt.figure(figsize=(4, 1))

        plt.text(
            0.5,
            0.5,
            f"${equation}$",
            fontsize=20,
            ha="center",
            va="center"
        )

        plt.axis("off")

        output_path = os.path.join(OUTPUT_DIR, filename)

        plt.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0.1
        )

        plt.close(fig)

        return output_path


if __name__ == "__main__":
    generator = EquationGenerator(max_depth=1)

    renderer = LatexRenderer()

    with open(LABELS_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["filename", "equation"])

        for i in range(5000):
            equation = generator.generate_equation()

            filename = f"equation_{i}.png"

            path = renderer.render_equation(
                equation,
                filename
            )

            writer.writerow([filename, equation])

            print(f"Saved: {path}")
            print(f"Equation: {equation}")