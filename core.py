from autograd import elementwise_grad as grad
from autograd.numpy import pi, sin

from const import E, a, b, h, nu, q_0

dx, dy = 0, 1


# Flexural rigidity of a plate:
D = (E * h**3) / (12 * (1 - nu**2))


# Deflection of a plate, see Eq. (d) p. 105.
def w(x, y):
    C = q_0 / (pi**4 * D * (1 / a**2 + 1 / b**2) ** 2)
    return C * sin((pi * x) / a) * sin((pi * y) / b)


# Functions based on relations known from theory of thin isotropic plates
# defined as partial derivatives of deflection using automatic differentiation.

# Slopes:


def phi_x(x, y):
    return grad(w, dx)(x, y)


def phi_y(x, y):
    return grad(w, dy)(x, y)


# Bending and twisting moments:


def M_x(x, y):
    return -D * (grad(grad(w, dx), dx)(x, y) + nu * grad(grad(w, dy), dy)(x, y))


def M_y(x, y):
    return -D * (grad(grad(w, dy), dy)(x, y) + nu * grad(grad(w, dx), dx)(x, y))


def M_xy(x, y):
    return D * (1 - nu) * grad(grad(w, dx), dy)(x, y)


# Shearing forces:


def Q_x(x, y):
    return -D * (
        grad(grad(grad(w, dx), dx), dx)(x, y) + grad(grad(grad(w, dx), dy), dy)(x, y)
    )


def Q_y(x, y):
    return -D * (
        grad(grad(grad(w, dx), dx), dy)(x, y) + grad(grad(grad(w, dy), dy), dy)(x, y)
    )


# Generalized shearing forces:


def V_x(x, y):
    return -D * (
        grad(grad(grad(w, dx), dx), dx)(x, y)
        + (2 - nu) * grad(grad(grad(w, dx), dy), dy)(x, y)
    )


def V_y(x, y):
    return -D * (
        grad(grad(grad(w, dy), dy), dy)(x, y)
        + (2 - nu) * grad(grad(grad(w, dx), dx), dy)(x, y)
    )
