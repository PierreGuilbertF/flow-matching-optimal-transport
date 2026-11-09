"""Plot the exact independent interpolation; always export PNG and PDF.

Run with Python, NumPy and Matplotlib from any working directory.
Four components lie on the coordinate axes, with equal mixture weights.
"""
from pathlib import Path
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'flowmatching-matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'figures'
CENTRES = np.array([[3., 0.], [-3., 0.], [0., -3.], [0., 3.]])
SIGMAS = np.array([0.3, 1.0, 0.2, 0.5])
TIMES = np.linspace(0., 1., 8)


def density(x, y, t):
    """Law of (1-t)X+tZ, with independent Z ~ N(0,I)."""
    result = np.zeros_like(x)
    variances = (1. - t)**2 * SIGMAS**2 + t**2
    for mean, variance in zip((1. - t) * CENTRES, variances):
        result += np.exp(-((x - mean[0])**2 + (y - mean[1])**2)
                         / (2. * variance)) / (8. * np.pi * variance)
    return result


def main(velocity_field=None, output_stem="contracted_convolution_2d",
         velocity_points=None):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    axis = np.linspace(-8., 8., 1001)
    x, y = np.meshgrid(axis, axis)
    fields = [density(x, y, float(t)) for t in TIMES]
    # A common nonlinear colour scale keeps both narrow and broad modes visible.
    norm = PowerNorm(gamma=0.35, vmin=0., vmax=np.ceil(10 * max(p.max() for p in fields)) / 10)
    levels = np.array([0.002, 0.01, 0.04, 0.1, 0.3, 1.0, 3.0])
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11,
                         'axes.titlesize': 14, 'axes.labelsize': 12,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.6), sharex=True,
                             sharey=True, layout='constrained')
    for ax, t, p in zip(axes.flat, TIMES, fields):
        image = ax.imshow(p, origin='lower', extent=(-8, 8, -8, 8),
                          cmap='coolwarm', norm=norm, interpolation='nearest',
                          rasterized=True)
        ax.contour(x, y, p, levels=levels[levels < p.max()],
                   colors='#242934', linewidths=0.6, alpha=0.75)
        if velocity_field is not None:
            if velocity_points is None:
                gx, gy = np.meshgrid(np.linspace(-7., 7., 15),
                                     np.linspace(-7., 7., 15))
            else:
                gx, gy = velocity_points(float(t))
            vx, vy = velocity_field(gx, gy, float(t))
            arrows = ax.quiver(gx, gy, vx, vy, color='#171923',
                              angles='xy', scale_units='xy', scale=12.,
                              width=0.0025, headwidth=3.5, pivot='tail')
            if t == 1:
                ax.quiverkey(arrows, 0.80, 0.95, 5., r'$5$',
                             coordinates='axes', labelpos='E',
                             fontproperties={'size': 9})
        ax.set_title(rf'$t={t:.2f}$' + ('  (data)' if t == 0 else
                                      '  (initial)' if t == 1 else ''))
        ax.set_aspect('equal')
        ax.set_xticks([-6, -3, 0, 3, 6])
        ax.set_yticks([-6, -3, 0, 3, 6])
        ax.tick_params(length=3)
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
    for ax in axes[-1]:
        ax.set_xlabel(r'$y_1$')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$y_2$')
    cbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.025,
                        ticks=[0, 0.01, 0.1, 0.3, 0.5, 1])
    cbar.set_label(r'Density $p_t$ (common power scale, exponent $0.35$)')
    cbar.ax.set_yticklabels(['0', '0.01', '0.1', '0.3', '0.5', '1'])
    for extension in ('png', 'pdf'):
        path = OUTPUT / f'{output_stem}.{extension}'
        fig.savefig(path, dpi=240, bbox_inches='tight')
        print(path)
    plt.close(fig)
    # Deterministic checks of the formula and the resolved grid mass.
    assert np.allclose(fields[-1], np.exp(-(x*x+y*y)/2)/(2*np.pi))
    step = axis[1] - axis[0]
    masses = np.array([p.sum()*step**2 for p in fields])
    assert np.all(np.abs(masses - 1) < 0.001), masses
    print('Grid masses:', masses)


if __name__ == '__main__':
    main()
