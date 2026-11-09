"""Light adaptation of the atlas Reynolds transport illustration.

Adapted from fisher-rao-wasserstein-atlas/code/reynolds_transport_probability.py.
Retains its density, material region and nonlinear map, now shown at three
actual times with minimal labels. Always saves PNG and PDF in figures/.
"""
from pathlib import Path
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'flowmatching-matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.colors import Normalize

OUTPUT = Path(__file__).resolve().parents[1]/'figures'
MEAN = np.array([-0.35,0.25])
COV = np.array([[1.20,0.48],[0.48,0.92]])
OMEGA_CENTER = np.array([-0.10,0.10])
OMEGA_AXES = np.array([1.05,0.78])
A, B, LAMBDA = 1.05, 1.10, 0.66

def gaussian_pdf_points(points, mean, cov):
    points = np.asarray(points)
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    diff = points - mean
    quad = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det_cov))
    return norm * np.exp(-0.5 * quad)

def omega_boundary(n=500):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    radial_mod = 1.0 + 0.16 * np.cos(3.0 * t - 0.45)
    x = OMEGA_CENTER[0] + OMEGA_AXES[0] * radial_mod * np.cos(t)
    y = OMEGA_CENTER[1] + OMEGA_AXES[1] * (1.0 + 0.10 * np.sin(2.0 * t + 0.4)) * np.sin(t)
    return np.column_stack([x, y])

def points_in_region(points, boundary):
    path = MplPath(boundary)
    return path.contains_points(points)

def phi(points,t):
    x,y = np.moveaxis(np.asarray(points),-1,0)
    u = x+t*A*np.sin(y)
    v = (1+t*(LAMBDA-1))*y+t*B*np.tanh(u)
    return np.stack([u,v],axis=-1)


def inverse(points,t):
    u,v = np.moveaxis(np.asarray(points),-1,0)
    y = (v-t*B*np.tanh(u))/(1+t*(LAMBDA-1))
    x = u-t*A*np.sin(y)
    return np.stack([x,y],axis=-1)


def main():
    OUTPUT.mkdir(exist_ok=True)
    samples = np.random.default_rng(7).multivariate_normal(MEAN,COV,180)
    boundary = omega_boundary()
    inside = points_in_region(samples,boundary)
    axis = np.linspace(-5,5,801)
    x,y = np.meshgrid(axis,axis)
    grid = np.stack([x,y],axis=-1)
    plt.rcParams.update({'font.family':'serif','font.size':12,'pdf.fonttype':42})
    fig,axes = plt.subplots(1,3,figsize=(11.5,3.8),sharex=True,sharey=True,
                            layout='constrained')
    masses=[]
    for ax,t in zip(axes,[0.,0.5,1.]):
        jac = 1+t*(LAMBDA-1)
        p = gaussian_pdf_points(inverse(grid,t),MEAN,COV)/jac
        transformed = phi(samples,t)
        region = phi(boundary,t)
        ax.imshow(p,origin='lower',extent=(-5,5,-5,5),cmap='Blues',
                  norm=Normalize(0,0.28),alpha=0.6,rasterized=True)
        ax.contour(x,y,p,levels=[0.01,0.04,0.08,0.14,0.2],
                   colors='#4C78A8',linewidths=0.65,alpha=0.65)
        ax.fill(region[:,0],region[:,1],color='#C44E52',alpha=0.12)
        ax.plot(region[:,0],region[:,1],color='#C44E52',lw=1.5)
        ax.scatter(transformed[~inside,0],transformed[~inside,1],
                   s=7,color='#333333',alpha=0.3)
        ax.scatter(transformed[inside,0],transformed[inside,1],
                   s=10,color='#C44E52',alpha=0.85)
        centre = phi(OMEGA_CENTER,t)
        ax.text(centre[0]-0.3,centre[1]+1.1,r'$\Omega$' if t==0 else r'$\Omega_t$',
                color='#A7383D',fontsize=12)
        ax.set_title(rf'$t={t:g}$')
        ax.set(xlim=(-4.5,4.5),ylim=(-3.5,3.5),xticks=[-4,-2,0,2,4],yticks=[-2,0,2])
        ax.set_aspect('equal')
        ax.set_xlabel(r'$y_1$')
        # Independent quadrature on each deformed region checks preservation.
        mask=points_in_region(grid.reshape(-1,2),region).reshape(x.shape)
        masses.append(float(p[mask].sum()*(axis[1]-axis[0])**2))
        assert np.allclose(inverse(phi(samples,t),t),samples)
    axes[0].set_ylabel(r'$y_2$')
    assert max(masses)-min(masses)<0.003, masses
    for ext in ('png','pdf'):
        path=OUTPUT/f'reynolds_transport_probability.{ext}'
        fig.savefig(path,dpi=240,bbox_inches='tight')
        print(path)
    plt.close(fig)
    print('Mass in transported region at each time:',masses)


if __name__=='__main__':
    main()
