"""Reuse the density figure and add exact velocities on a dense regular grid.

Running this script saves both PNG and PDF in the project's figures folder.
"""
import numpy as np
from plot_contracted_convolution import CENTRES, SIGMAS, density, main


def velocity(x, y, t):
    """Conditional E[Z-X | (1-t)X+tZ=(x,y)], including both endpoints.

    Given mixture component i, the mean is a*mu_i, the variance is
    a²*sigma_i²+t², and Cov(Z-X, aX+tZ)=t-a*sigma_i².
    Posterior component weights are evaluated with a stable softmax.
    """
    a = 1. - t
    variance = a*a * SIGMAS**2 + t*t
    delta_x = x[None, ...] - a * CENTRES[:, 0, None, None]
    delta_y = y[None, ...] - a * CENTRES[:, 1, None, None]
    var = variance[:, None, None]
    log_weights = -np.log(var) - (delta_x**2 + delta_y**2)/(2*var)
    weights = np.exp(log_weights - log_weights.max(axis=0))
    weights /= weights.sum(axis=0)
    slope = ((t - a*SIGMAS**2)/variance)[:, None, None]
    vx = np.sum(weights * (-CENTRES[:, 0, None, None] + slope*delta_x), axis=0)
    vy = np.sum(weights * (-CENTRES[:, 1, None, None] + slope*delta_y), axis=0)
    return vx, vy


def grid_velocity_points(t):
    """Spacing 0.3 near the density and inside the hull of component centres."""
    axis = np.arange(-7.8, 7.81, 0.3)
    gx, gy = np.meshgrid(axis, axis)
    near_density = density(gx, gy, t) >= 0.002
    # The four centres form a diamond: include its interior to bridge modes.
    between_modes = np.abs(gx) + np.abs(gy) <= 3.*(1.-t) + 1e-10
    visible = near_density | between_modes
    return gx[visible, None], gy[visible, None]


if __name__ == '__main__':
    # At t=0, E[Z-X | X=y]=-y; at t=1, E[Z-X | Z=y]=y-E[X]=y.
    gx, gy = np.meshgrid(np.linspace(-5., 5., 11), np.linspace(-5., 5., 11))
    for t, sign in [(0., -1.), (1., 1.)]:
        vx, vy = velocity(gx, gy, t)
        assert np.allclose(vx, sign*gx) and np.allclose(vy, sign*gy)
    main(velocity_field=velocity, output_stem='contracted_convolution_velocity_2d',
         velocity_points=grid_velocity_points)
