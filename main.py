import matplotlib.pyplot as plt
from autograd.numpy import linspace, max, meshgrid, min
from matplotlib import cm

from const import a, b
from core import M_x, M_xy, M_y, Q_x, Q_y, V_x, V_y, phi_x, phi_y, w


def plot_results(X, Y, Z, title):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
    ax.set_zlim(ax.get_zlim()[::-1])
    boundaries = linspace(min(Z), max(Z), N)
    fig.colorbar(surf, shrink=0.75, boundaries=boundaries, pad=0.1)
    plt.title(f"3D plot of ${title}$")
    fname = title.strip("\\").replace("{", "").replace("}", "")
    plt.tight_layout()
    plt.savefig(f"tim_{fname}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    N = 25
    M = int(0.5 * (N - N % 2))

    x = linspace(0, a, N)
    y = linspace(0, b, N)
    X, Y = meshgrid(x, y)

    for i, f in [
        ("w", w),
        ("\\varphi_x", phi_x),
        ("\\varphi_y", phi_y),
        ("M_x", M_x),
        ("M_y", M_y),
        ("M_{xy}", M_xy),
        ("Q_x", Q_x),
        ("Q_y", Q_y),
        ("V_x", V_x),
        ("V_y", V_y),
    ]:
        Z = f(X, Y)
        print(f"max({i}):", max(Z))
        print("Mid 1:", Z[M, 0])
        print("Mid 2:", Z[0, M])
        print(5 * "-")
        plot_results(X, Y, Z, title=i)
