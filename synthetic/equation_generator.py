import random

VARIABLES = [
    "x",
    "y",
    "z",
    "a",
    "b",
    "c",
    "X_1",
    "X_2",
    "Y_1",
    "Y_2",
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\theta",
    r"\pi",
    "e",
    "n",
    "m"
]

OPERATORS = [
    "+",
    "-",
    "=",
    r"\times"
]

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

        exponent = random.choice([2 ,2, 2, 2, 3])

        return rf"{{{base}}}^{exponent}"

    def generate_fraction_expression(self, depth):
        numerator = self.generate_expression(depth + 1)

        denominator = self.generate_expression(depth + 1)

        return rf"\frac{{{numerator}}}{{{denominator}}}"

    def generate_square_root_expression(self, depth):
        value = self.generate_expression(depth + 1)

        return rf"\sqrt{{{value}}}"

    def generate_function_expression(self):
        function_name = random.choice(
            ["f", "g", "h"]
        )

        var1 = self.random_variable()

        var2 = self.random_variable()

        expression = self.generate_expression(depth=1)

        return (
            rf"{function_name}"
            rf"({var1}, {var2})"
            rf" = "
            rf"{expression}"
        )

    def generate_parenthesized_expression(self):
        expression = self.generate_expression(depth=1)

        return rf"({expression})"

    def generate_trig_expression(self):
        func = random.choice(
            [r"\sin", r"\cos", r"\tan"]
        )

        var = self.random_variable()

        return rf"{func}({var})"

    def generate_log_expression(self):
        var = self.random_variable()

        return rf"\log({var})"

    def generate_sum_expression(self):
        var = random.choice(
            ["x", "y", "a", "b"]
        )

        return (
            rf"\sum_{{i=1}}^{{n}} "
            rf"{var}_i"
        )

    def generate_integral_expression(self):
        var = self.random_variable()

        return (
            rf"\int {var} \, dx"
        )

    def generate_limit_expression(self):
        var = self.random_variable()

        return (
            rf"\lim_{{x \to 0}} "
            rf"{var}"
        )

    def generate_equation_chain(self):
        left = self.generate_expression(depth=1)

        middle = self.generate_expression(depth=1)

        right = self.generate_expression(depth=1)

        return (
            rf"{left} = {middle} = {right}"
        )

    def generate_linear_equation(self):
        x = self.random_variable()
        y = self.random_variable()
        z = self.random_variable()

        return f"{x} + {y} = {z}"

    def generate_slope_equation(self):
        return "y = mx + b"

    def generate_pythagorean(self):
        return "a^2 + b^2 = c^2"

    def generate_quadratic(self):
        return "ax^2 + bx + c = 0"

    def generate_average_formula(self):
        return r"X_1 = \frac{y_1 + y_2}{2}"

    def generate_physics_formula(self):
        return random.choice([
            "F = ma",
            "V = IR",
            "E = mc^2",
            "p = mv"
        ])

    def generate_derivative(self):
        return r"\frac{d}{dx}x^2 = 2x"

    def generate_definite_integral(self):
        return r"\int_0^1 x^2 \, dx"

    def generate_famous_limit(self):
        return r"\lim_{x \to 0}\frac{\sin(x)}{x}"

    def generate_expression(self, depth=0):
        if depth >= self.max_depth:
            return self.random_variable()

        choice = random.random()

        if choice < 0.60:
            generators = [
                lambda: self.generate_linear_equation(),
                lambda: self.generate_slope_equation(),
                lambda: self.generate_pythagorean(),
                lambda: self.generate_quadratic(),
                lambda: self.generate_average_formula(),
                lambda: self.generate_physics_formula()
            ]

        elif choice < 0.90:
            generators = [
                lambda: self.generate_fraction_expression(depth),
                lambda: self.generate_square_root_expression(depth),
                lambda: self.generate_trig_expression(),
                lambda: self.generate_log_expression(),
                lambda: self.generate_power_expression(depth)
            ]

        else:
            generators = [
                lambda: self.generate_sum_expression(),
                lambda: self.generate_integral_expression(),
                lambda: self.generate_limit_expression(),
                lambda: self.generate_function_expression(),
                lambda: self.generate_derivative(),
                lambda: self.generate_definite_integral(),
                lambda: self.generate_famous_limit()
            ]

        generator = random.choice(generators)

        return generator()

    def generate_equation(self):
        return self.generate_expression()

