import random


VARIABLES = ["x", "y", "a", "b", "c", "z"]

OPERATORS = ["+", "-", "="]


class EquationGenerator:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def random_variable(self):
        return random.choice(VARIABLES)

    def random_operator(self):
        return random.choice(OPERATORS)

    def generate_simple_expression(self):
        left = self.random_variable()

        operator = self.random_operator()

        right = self.random_variable()

        return f"{left} {operator} {right}"

    def generate_power_expression(self, depth):
        base = self.generate_expression(depth + 1)

        exponent = random.randint(2, 5)

        return rf"{{{base}}}^{exponent}"

    def generate_fraction_expression(self, depth):
        numerator = self.generate_expression(depth + 1)

        denominator = self.generate_expression(depth + 1)

        return rf"\frac{{{numerator}}}{{{denominator}}}"

    def generate_square_root_expression(self, depth):
        value = self.generate_expression(depth + 1)

        return rf"\sqrt{{{value}}}"

    def generate_expression(self, depth=0):
        if depth >= self.max_depth:
            return self.random_variable()

        generators = [
            lambda: self.generate_simple_expression(),
            lambda: self.generate_power_expression(depth),
            lambda: self.generate_fraction_expression(depth),
            lambda: self.generate_square_root_expression(depth)
        ]

        generator = random.choice(generators)

        return generator()

    def generate_equation(self):
        return self.generate_expression()

