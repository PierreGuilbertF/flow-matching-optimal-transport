"""Four-time comparison of FM velocity and the local WO gradient.

The isotropic Gaussian initial law makes the two fields equal analytically.
Compute them separately: conditional Gaussian regression versus mixture score.
Always save PNG and PDF to figures/.
"""
import numpy as np
from plot_contracted_convolution import CENTRES, SIGMAS, OUTPUT, density
from plot_convolution_velocity import velocity
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

TIMES = [0.15, 0.4, 0.65, 0.9]


def wo_gradient(x,y,t):
    variances = ((1-t)**2*SIGMAS**2+t*t)[:,None,None]
    dx = x[None,...]-(1-t)*CENTRES[:,0,None,None]
    dy = y[None,...]-(1-t)*CENTRES[:,1,None,None]
    logw = -np.log(variances)-(dx*dx+dy*dy)/(2*variances)
    weights = np.exp(logw-logw.max(axis=0))
    weights /= weights.sum(axis=0)
    score_x = np.sum(-weights*dx/variances,axis=0)
    score_y = np.sum(-weights*dy/variances,axis=0)
    return -(x+t*score_x)/(1-t), -(y+t*score_y)/(1-t)


def main():
    OUTPUT.mkdir(exist_ok=True)
    axis=np.linspace(-5.5,5.5,601)
    x,y=np.meshgrid(axis,axis)
    gx,gy=np.meshgrid(np.arange(-5.4,5.41,0.3),np.arange(-5.4,5.41,0.3))
    plt.rcParams.update({'font.family':'serif','font.size':11,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,4,figsize=(14.8,7.8),sharex=True,sharey=True,layout='constrained')
    largest_error=0.
    for j,t in enumerate(TIMES):
        p=density(x,y,t)
        mask=(density(gx,gy,t)>=0.002)|(np.abs(gx)+np.abs(gy)<=3*(1-t))
        fm=velocity(gx,gy,t)
        wo=wo_gradient(gx,gy,t)
        largest_error=max(largest_error,max(np.max(np.abs(a-b)) for a,b in zip(fm,wo)))
        for row,field in enumerate([fm,wo]):
            ax=axes[row,j]
            image=ax.imshow(p,origin='lower',extent=(-5.5,5.5,-5.5,5.5),
                            cmap='coolwarm',norm=PowerNorm(0.35,vmin=0,vmax=1),
                            rasterized=True)
            ax.contour(x,y,p,levels=[0.002,0.01,0.04,0.1,0.3],colors='#242934',linewidths=0.5)
            ax.quiver(gx,gy,np.ma.array(field[0],mask=~mask),np.ma.array(field[1],mask=~mask),
                      angles='xy',scale_units='xy',scale=12,color='#171923',width=0.0025)
            name=r'$v_t$' if row==0 else r'$\nabla\phi_t$ (WO)'
            ax.set_title(name+rf', $t={t:.2f}$')
            ax.set(xlim=(-5.5,5.5),ylim=(-5.5,5.5),xticks=[-4,0,4],yticks=[-4,0,4])
            ax.set_aspect('equal')
    for ax in axes[-1]: ax.set_xlabel(r'$y_1$')
    for ax in axes[:,0]: ax.set_ylabel(r'$y_2$')
    fig.colorbar(image,ax=axes,shrink=0.85,pad=0.025,
                 label=r'Density $p_t$ (power scale, exponent 0.35)')
    assert largest_error<1e-10,largest_error
    for ext in ['png','pdf']:
        path=OUTPUT/f'fm_wo_velocity_comparison.{ext}'
        fig.savefig(path,dpi=240,bbox_inches='tight')
        print(path)
    plt.close(fig)
    print('Maximum field difference:',largest_error)


if __name__=='__main__': main()
