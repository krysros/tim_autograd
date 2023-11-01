"""
Expressions for Simply Supported Rectangular Plates under Sinusoidal Load
from the book Theory of plates and shells by S. Timoshenko and S. Woinowsky-Krieger
used to verify the results.
"""
from autograd.numpy import cos, pi, sin

from const import a, b, nu, q_0
from core import D

# Maximum deflection, Eq. (124)

w_max = q_0 / (pi**4 * D * (1 / a**2 + 1 / b**2) ** 2)

# Maximum moments, Eq. (125)

M_x_max = (q_0 / (pi**2 * (1 / a**2 + 1 / b**2) ** 2)) * (
    1 / a**2 + nu / b**2
)


M_y_max = (q_0 / (pi**2 * (1 / a**2 + 1 / b**2) ** 2)) * (
    nu / a**2 + 1 / b**2
)

# Shearing forces, Eq. (g)


def _Q_x(x, y):
    return (
        (q_0 / (pi * a * (1 / a**2 + 1 / b**2)))
        * cos((pi * x) / a)
        * sin((pi * y) / b)
    )


def _Q_y(x, y):
    return (
        (q_0 / (pi * b * (1 / a**2 + 1 / b**2)))
        * sin((pi * x) / a)
        * cos((pi * y) / b)
    )
