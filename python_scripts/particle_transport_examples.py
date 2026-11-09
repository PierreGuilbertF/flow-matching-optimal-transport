"""Four standalone FM/OT particle comparisons; does not edit the manuscript.

Run: python python_scripts/particle_transport_examples.py
Requires NumPy, SciPy and Matplotlib. Saves a separate PNG/PDF and trajectory
archive per example in figures/. All examples use the same 150 initial
N(0,I) samples and particle colours in both rows. Time labels follow the
paper: t=1 is the initial Gaussian, t=0 is the destination.

FM uses independent interpolation (1-s)Z+sX, s=1-t. Gaussian flows are
computed analytically. The separable Laplace flow is solved by inversion
of its exact one-dimensional marginal CDF (the no-flux ODE solution).
The complex GMM FM velocity is integrated by RK4. OT is exact for Gaussian
and separable Laplace examples; the complex GMM uses an optimal assignment
between 2048-point empirical endpoint measures, including the displayed
150 Gaussian samples. That example is a numerical OT approximation.
"""
from pathlib import Path
import os
import tempfile
import json
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'flowmatching-matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr, log_ndtr
from scipy.stats import norm, qmc
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

OUTPUT=Path(__file__).resolve().parents[1]/'figures'
PARTICLES=150
STEPS=301
S=np.linspace(0,1,STEPS)
INITIAL=np.random.default_rng(20261109).normal(size=(PARTICLES,2))
LAPLACE_SCALE=1/np.sqrt(2)  # independent coordinates, each of variance one


def matrix_root(cov):
    values,vectors=np.linalg.eigh(cov)
    return (vectors*np.sqrt(values))@vectors.T


def gaussian_paths(mean,cov):
    fm=np.array([s*mean+INITIAL@matrix_root((1-s)**2*np.eye(2)+s*s*cov).T for s in S])
    destination=mean+INITIAL@matrix_root(cov).T
    ot=(1-S[:,None,None])*INITIAL+S[:,None,None]*destination
    return fm,ot


def laplace_quantile(prob):
    return np.where(prob<0.5,LAPLACE_SCALE*np.log(2*prob),
                    -LAPLACE_SCALE*np.log(2*(1-prob)))


def normal_laplace_cdf(y,s):
    """CDF of (1-s)Z+sX, X Laplace(0,b), Z standard normal.

    The log tail expansion avoids cancellation for very small s.
    """
    y=np.asarray(y)
    if s==0: return ndtr(y)
    if s==1:
        return np.where(y<0,0.5*np.exp(-np.abs(y)/LAPLACE_SCALE),
                        1-0.5*np.exp(-np.abs(y)/LAPLACE_SCALE))
    t=1-s
    beta=s*LAPLACE_SCALE
    terms=[]
    for sign in [1.,-1.]:
        r=sign*y/t-t/beta
        tail=r < -12
        logterm=np.empty_like(y,dtype=float)
        logterm[~tail]=(t*t/(2*beta*beta)-sign*y[~tail]/beta+log_ndtr(r[~tail]))
        z=-r[tail]
        inv=1/(z*z)
        series=1-inv+3*inv**2-15*inv**3+105*inv**4-945*inv**5
        logterm[tail]=-y[tail]**2/(2*t*t)-np.log(z)-0.5*np.log(2*np.pi)+np.log(series)
        terms.append(np.exp(logterm))
    return np.clip(ndtr(y/t)+0.5*(terms[1]-terms[0]),0,1)


def laplace_paths():
    levels=ndtr(INITIAL)
    axis=np.linspace(-14,14,28001)
    fm=[]
    for s in S:
        cdf=normal_laplace_cdf(axis,float(s))
        assert np.min(np.diff(cdf))>-1e-8
        assert np.max(np.abs(cdf+cdf[::-1]-1))<1e-8
        fm.append(np.interp(levels,np.maximum.accumulate(cdf),axis))
    fm=np.array(fm)
    fm[0]=INITIAL
    fm[-1]=laplace_quantile(levels)
    ot=(1-S[:,None,None])*INITIAL+S[:,None,None]*fm[-1]
    return fm,ot


def complex_mixture():
    # Eight anisotropic modes on a bent, asymmetric spiral.
    angles=np.linspace(-0.9*np.pi,0.95*np.pi,8)
    radius=np.linspace(1.6,3.8,8)
    means=np.column_stack((radius*np.cos(angles),0.8*radius*np.sin(angles)))
    covariances=[]
    for i,angle in enumerate(angles):
        rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        scales=np.array([0.18+0.025*i,0.45+0.035*(i%3)])
        covariances.append(rot@np.diag(scales**2)@rot.T)
    return means,np.array(covariances)


def gmm_velocity(points,s,means,covs):
    t=1-s
    variances=s*s*covs+t*t*np.eye(2)
    inverses=np.linalg.inv(variances)
    delta=points[None,:,:]-s*means[:,None,:]
    precision_delta=np.einsum('kij,knj->kni',inverses,delta)
    logweight=-0.5*(np.linalg.slogdet(variances)[1][:,None]
                              +np.sum(delta*precision_delta,axis=2))
    weight=np.exp(logweight-logweight.max(axis=0))
    weight/=weight.sum(axis=0)
    component=means[:,None,:]+np.einsum('kij,knj->kni',s*covs-t*np.eye(2),precision_delta)
    return np.sum(weight[:,:,None]*component,axis=0)


def integrate_gmm(means,covs,substeps):
    points=INITIAL.copy();history=[points.copy()]
    h=(S[1]-S[0])/substeps
    for start in S[:-1]:
        for j in range(substeps):
            s=start+j*h
            k1=gmm_velocity(points,s,means,covs)
            k2=gmm_velocity(points+h*k1/2,s+h/2,means,covs)
            k3=gmm_velocity(points+h*k2/2,s+h/2,means,covs)
            k4=gmm_velocity(points+h*k3,s+h,means,covs)
            points+=h*(k1+2*k2+2*k3+k4)/6
        history.append(points.copy())
    return np.array(history)


def gmm_paths():
    means,covs=complex_mixture()
    fm=integrate_gmm(means,covs,8)
    check=integrate_gmm(means,covs,4)
    error=float(np.linalg.norm(fm-check,axis=2).max())
    assert error<1e-3,error
    # Include exactly INITIAL in the source measure; no nearest-neighbour substitution.
    n=2048
    standard=norm.ppf(qmc.Sobol(2,scramble=True,seed=138).random_base2(11))
    gaussian=np.concatenate((INITIAL,standard[:n-PARTICLES]))
    target=np.concatenate([mean+norm.ppf(qmc.Sobol(2,scramble=True,seed=200+i).random_base2(8))@matrix_root(cov).T
                           for i,(mean,cov) in enumerate(zip(means,covs))])
    cache=OUTPUT/'particle_examples_complex_coupling.npz'
    assigned=None
    if cache.exists():
        with np.load(cache) as saved:
            if np.array_equal(saved['initial_measure'],gaussian) and np.array_equal(saved['target_measure'],target):
                assigned=saved['assigned']
    if assigned is None:
        print('Computing 2048-point complex-mixture OT assignment...',flush=True)
        _,cols=linear_sum_assignment(cdist(gaussian,target,metric='sqeuclidean'))
        assigned=target[cols]
        np.savez_compressed(cache,initial_measure=gaussian,target_measure=target,assigned=assigned)
    ot=(1-S[:,None,None])*INITIAL+S[:,None,None]*assigned[:PARTICLES]
    return fm,ot,{'means':means.tolist(),'covariances':covs.tolist(),
                  'ot_endpoint_samples':n,'rk4_max_resolution_difference':error,
                  'empirical_ot_action':float(np.mean(np.sum((assigned-gaussian)**2,axis=1)))}


def gaussian_density(x,y,mean,cov):
    delta=np.stack((x-mean[0],y-mean[1]),axis=-1)
    quadratic=np.einsum('...i,ij,...j->...',delta,np.linalg.inv(cov),delta)
    return np.exp(-0.5*quadratic)/(2*np.pi*np.sqrt(np.linalg.det(cov)))


def background_density(name,x,y,s,row):
    """Densities of the full laws, never estimated from the displayed 150 points."""
    if name in ['gaussian_to_same_gaussian','gaussian_to_shifted_anisotropic']:
        if name=='gaussian_to_same_gaussian':
            mean=np.zeros(2);cov=np.eye(2)
        else:
            angle=np.pi/4
            rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            mean=np.array([3.,-2.]);cov=rot@np.diag([0.16,2.25])@rot.T
        if row==0:
            covariance=(1-s)**2*np.eye(2)+s*s*cov
        else:
            transform=(1-s)*np.eye(2)+s*matrix_root(cov)
            covariance=transform@transform.T
        return gaussian_density(x,y,s*mean,covariance)
    if name=='gaussian_to_laplace':
        coordinates=[x[0,:],y[:,0]]
        factors=[]
        for axis in coordinates:
            if s==0:
                factor=norm.pdf(axis)
            elif s==1:
                factor=np.exp(-np.abs(axis)/LAPLACE_SCALE)/(2*LAPLACE_SCALE)
            elif row==0:
                h=1e-4
                factor=np.maximum((normal_laplace_cdf(axis+h,s)-normal_laplace_cdf(axis-h,s))/(2*h),0)
            else:
                z=np.linspace(-7.5,7.5,100001)
                destination=laplace_quantile(ndtr(z))
                transformed=(1-s)*z+s*destination
                target_pdf=np.exp(-np.abs(destination)/LAPLACE_SCALE)/(2*LAPLACE_SCALE)
                derivative=(1-s)+s*norm.pdf(z)/target_pdf
                factor=np.interp(axis,transformed,norm.pdf(z)/derivative,left=0,right=0)
            factors.append(factor)
        return factors[1][:,None]*factors[0][None,:]
    means,covs=complex_mixture()
    if row==0 or s==0 or s==1:
        return sum(gaussian_density(x,y,s*mean,(1-s)**2*np.eye(2)+s*s*cov)
                   for mean,cov in zip(means,covs))/len(means)
    with np.load(OUTPUT/'particle_examples_complex_coupling.npz') as cached:
        points=(1-s)*cached['initial_measure']+s*cached['assigned']
    # Smooth the full 2048-particle OT approximation, preserving exact endpoints.
    dx=x[0,1]-x[0,0];dy=y[1,0]-y[0,0]
    xedges=np.r_[x[0,:]-dx/2,x[0,-1]+dx/2]
    yedges=np.r_[y[:,0]-dy/2,y[-1,0]+dy/2]
    mass=np.histogram2d(points[:,1],points[:,0],bins=[yedges,xedges])[0]
    return gaussian_filter(mass,[0.18/dy,0.18/dx],mode='constant')/len(points)/dx/dy


def render(name,title,fm,ot,numerical_ot=False):
    assert np.array_equal(fm[0],ot[0])
    assert np.allclose(fm[0],INITIAL)
    assert np.isfinite(fm).all() and np.isfinite(ot).all()
    colours=plt.get_cmap('turbo')(np.linspace(0.05,0.95,PARTICLES))
    allpoints=np.concatenate((fm.reshape(-1,2),ot.reshape(-1,2)))
    centre=(allpoints.max(axis=0)+allpoints.min(axis=0))/2
    radius=max(np.ptp(allpoints[:,0]),np.ptp(allpoints[:,1]))/2+0.4
    xaxis=np.linspace(centre[0]-radius,centre[0]+radius,501)
    yaxis=np.linspace(centre[1]-radius,centre[1]+radius,501)
    x,y=np.meshgrid(xaxis,yaxis)
    frames=np.linspace(0,STEPS-1,6,dtype=int)
    backgrounds=[[background_density(name,x,y,float(S[i]),row) for i in frames] for row in [0,1]]
    norm_colour=PowerNorm(0.35,vmin=0,vmax=max(p.max() for row in backgrounds for p in row))
    fig,axes=plt.subplots(2,6,figsize=(18.5,6.6),sharex=True,sharey=True,layout='constrained')
    for row,history in enumerate([fm,ot]):
        for j,(ax,i) in enumerate(zip(axes[row],frames)):
            p=backgrounds[row][j]
            image=ax.imshow(p,origin='lower',extent=(xaxis[0],xaxis[-1],yaxis[0],yaxis[-1]),
                            cmap='coolwarm',norm=norm_colour,interpolation='nearest',rasterized=True)
            levels=np.array([0.002,0.01,0.04,0.1,0.2,0.4,0.8])
            ax.contour(x,y,p,levels=levels[levels<p.max()],colors='#242934',linewidths=0.55,alpha=0.75)
            ax.add_collection(LineCollection(np.transpose(history[:i+1],(1,0,2)),
                                             colors=colours,linewidths=0.65,alpha=0.7))
            ax.scatter(history[i,:,0],history[i,:,1],s=8,c=colours,
                       edgecolors='white',linewidths=0.25,zorder=3)
            ax.set(xlim=(centre[0]-radius,centre[0]+radius),ylim=(centre[1]-radius,centre[1]+radius))
            ax.set_aspect('equal');ax.tick_params(labelsize=9)
            for spine in ax.spines.values(): spine.set_color('#aaaaaa')
            if row==0: ax.set_title(rf'$t={1-S[i]:.1f}$',fontsize=13)
    axes[0,0].set_ylabel('Flow Matching\n'+r'$y_2$')
    axes[1,0].set_ylabel(('OT (numerical)' if numerical_ot else 'Optimal transport')+'\n'+r'$y_2$')
    for ax in axes[-1]: ax.set_xlabel(r'$y_1$')
    fig.colorbar(image,ax=axes,shrink=0.85,pad=0.015,
                 label='Density (common power scale, exponent 0.35)')
    fig.suptitle(title,fontsize=15)
    for ext in ['png','pdf']:
        path=OUTPUT/f'particles_{name}.{ext}'
        fig.savefig(path,dpi=240,bbox_inches='tight')
        print(path,flush=True)
    plt.close(fig)
    np.savez_compressed(OUTPUT/f'particles_{name}.npz',s=S,t=1-S,initial=INITIAL,fm=fm,ot=ot)


def main():
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.size':11,'pdf.fonttype':42})
    report={'particles':PARTICLES,'same_initial_points_in_all_examples':True,
            'display':'density colour map, contours, particles and past trajectories; t decreases from 1 to 0',
            'laplace_scale':LAPLACE_SCALE}
    fm,ot=laplace_paths()
    render('gaussian_to_laplace','Standard Gaussian to independent-coordinate Laplace',fm,ot)
    fm,ot,details=gmm_paths()
    report['complex_gmm']=details
    render('gaussian_to_complex_gmm','Standard Gaussian to an eight-component anisotropic Gaussian mixture',fm,ot,True)
    fm,ot=gaussian_paths(np.zeros(2),np.eye(2))
    assert np.allclose(ot,INITIAL[None,...])
    assert np.allclose(fm[STEPS//2],INITIAL/np.sqrt(2))
    render('gaussian_to_same_gaussian','Standard Gaussian to the same standard Gaussian',fm,ot)
    angle=np.pi/4
    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    cov=rot@np.diag([0.16,2.25])@rot.T
    mean=np.array([3.,-2.])
    fm,ot=gaussian_paths(mean,cov)
    report['shifted_gaussian']={'mean':mean.tolist(),'covariance':cov.tolist()}
    render('gaussian_to_shifted_anisotropic','Standard Gaussian to a shifted, rotated anisotropic Gaussian',fm,ot)
    (OUTPUT/'particle_transport_examples_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
