"""Four initial laws -> the same complex GMM: densities and FM particles.

The convolution and probability flux are evaluated by inverse FFT from
analytic Fourier transforms. Small Gaussian mollification (std 0.06) is
applied to BOTH endpoint laws, allowing stable evaluation of compact-support
initial laws. Particles start from the correspondingly mollified initial
law. This is a numerical approximation, not a closed-form continuous flow.
All unmollified initial laws are centred with covariance I. The target is
the eight-component mixture in particle_transport_examples.py.
Always saves PNG/PDF and trajectories/diagnostics to figures/.
"""
import json
import numpy as np
from particle_transport_examples import OUTPUT, complex_mixture, matrix_root
from scipy.special import j1, jv
from scipy.stats import norm
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.collections import LineCollection

GRID=512
HALF=8.
EPS=0.06
TIMES=np.linspace(0,1,151)
PARTICLES=150
DX=2*HALF/GRID
AXIS=np.arange(GRID)*DX-HALF
KX,KY=np.meshgrid(2*np.pi*np.fft.fftfreq(GRID,d=DX),2*np.pi*np.fft.fftfreq(GRID,d=DX))
X,Y=np.meshgrid(AXIS,AXIS)
MEANS,COVS=complex_mixture()
COVS=COVS+EPS**2*np.eye(2)
MEAN=MEANS.mean(axis=0)
LAWS=['gaussian','laplace','disk','square']


def prior_fourier(law,qx,qy):
    if law=='gaussian':
        p=np.exp(-0.5*(qx*qx+qy*qy));dx=-qx*p;dy=-qy*p
    elif law=='laplace':
        p=1/(1+0.5*qx*qx)/(1+0.5*qy*qy)
        dx=-qx/(1+0.5*qx*qx)*p;dy=-qy/(1+0.5*qy*qy)*p
    elif law=='disk':
        r=np.hypot(qx,qy);z=2*r
        p=np.ones_like(r);dx=np.zeros_like(r);dy=np.zeros_like(r)
        nonzero=r>1e-10
        p[nonzero]=2*j1(z[nonzero])/z[nonzero]
        factor=np.zeros_like(r);factor[nonzero]=-2*jv(2,z[nonzero])/r[nonzero]**2
        dx=factor*qx;dy=factor*qy
    else:
        a=np.sqrt(3.)
        def sinc_and_derivative(q):
            z=a*q
            value=np.sinc(z/np.pi)
            small=np.abs(z)<1e-4
            derivative=np.empty_like(z)
            derivative[small]=a*(-z[small]/3+z[small]**3/30)
            derivative[~small]=a*(z[~small]*np.cos(z[~small])-np.sin(z[~small]))/z[~small]**2
            return value,derivative
        px,dpx=sinc_and_derivative(qx);py,dpy=sinc_and_derivative(qy)
        p=px*py;dx=dpx*py;dy=px*dpy
    damping=np.exp(-0.5*EPS**2*(qx*qx+qy*qy))
    return p*damping,(dx-EPS**2*qx*p)*damping,(dy-EPS**2*qy*p)*damping


def target_fourier(qx,qy):
    p=np.zeros_like(qx,dtype=complex);dx=p.copy();dy=p.copy()
    for mean,cov in zip(MEANS,COVS):
        cx=cov[0,0]*qx+cov[0,1]*qy
        cy=cov[1,0]*qx+cov[1,1]*qy
        component=np.exp(-1j*(mean[0]*qx+mean[1]*qy)-0.5*(qx*cx+qy*cy))/len(MEANS)
        p+=component;dx+=(-1j*mean[0]-cx)*component;dy+=(-1j*mean[1]-cy)*component
    return p,dx,dy


def inverse(f):
    return np.fft.fftshift(np.fft.ifft2(f)).real/DX**2


def density_velocity(law,s):
    pz,dzx,dzy=prior_fourier(law,(1-s)*KX,(1-s)*KY)
    px,dxx,dxy=target_fourier(s*KX,s*KY)
    p=inverse(pz*px)
    flux=[inverse(1j*(pz*dxx-dzx*px)),inverse(1j*(pz*dxy-dzy*px))]
    visible=p>1e-8
    velocity=np.array([np.divide(f,p,out=np.zeros_like(f),where=visible) for f in flux])
    return np.maximum(p,0),velocity


def initial_particles(law):
    rng=np.random.default_rng(20261109)
    u=rng.uniform(size=(PARTICLES,2))
    if law=='gaussian': points=norm.ppf(u)
    elif law=='laplace':
        b=1/np.sqrt(2)
        points=np.where(u<0.5,b*np.log(2*u),-b*np.log(2*(1-u)))
    elif law=='disk':
        radius=2*np.sqrt(u[:,0]);angle=2*np.pi*u[:,1]
        points=np.column_stack((radius*np.cos(angle),radius*np.sin(angle)))
    else: points=np.sqrt(3)*(2*u-1)
    return points+EPS*rng.normal(size=points.shape)


def interpolate_velocity(fields,points,s):
    index=np.clip(s,0,1)*(len(TIMES)-1)
    low=min(int(index),len(TIMES)-2);weight=index-low
    coordinates=np.array([(points[:,1]+HALF)/DX,(points[:,0]+HALF)/DX])
    return np.column_stack([(1-weight)*map_coordinates(fields[low,j],coordinates,order=1,mode='nearest')
                            +weight*map_coordinates(fields[low+1,j],coordinates,order=1,mode='nearest')
                            for j in [0,1]])


def integrate(fields,initial,substeps):
    points=initial.copy();history=[points.copy()]
    h=(TIMES[1]-TIMES[0])/substeps
    for start in TIMES[:-1]:
        for j in range(substeps):
            s=start+j*h
            k1=interpolate_velocity(fields,points,s)
            k2=interpolate_velocity(fields,points+h*k1/2,s+h/2)
            k3=interpolate_velocity(fields,points+h*k2/2,s+h/2)
            k4=interpolate_velocity(fields,points+h*k3,s+h)
            points+=h*(k1+2*k2+2*k3+k4)/6
        history.append(points.copy())
    return np.array(history)


def main():
    OUTPUT.mkdir(exist_ok=True)
    snapshots=np.linspace(0,len(TIMES)-1,6,dtype=int)
    backgrounds=[];histories=[];diagnostics={}
    for law in LAWS:
        print(f'Computing {law} convolution and flux...',flush=True)
        fields=np.empty((len(TIMES),2,GRID,GRID),dtype=np.float32)
        panels=[];masses=[]
        for i,s in enumerate(TIMES):
            p,fields[i]=density_velocity(law,float(s))
            masses.append(float(p.sum()*DX**2))
            if i in snapshots: panels.append(p)
        initial=initial_particles(law)
        history=integrate(fields,initial,4)
        check=integrate(fields,initial,2)
        error=float(np.linalg.norm(history-check,axis=2).max())
        assert np.isfinite(history).all()
        assert error<0.02,(law,error)
        assert max(abs(np.asarray(masses)-1))<0.003
        # Check the initial and final mean velocity where density is appreciable.
        for i,expected in [(0,MEAN[None,None,:]-np.stack((X,Y),axis=-1)),
                            (-1,np.stack((X,Y),axis=-1))]:
            p,_=density_velocity(law,float(TIMES[i]))
            mask=p>0.01
            assert np.max(np.abs(np.moveaxis(fields[i],0,-1)[mask]-expected[mask]))<0.02
        diagnostics[law]={'rk4_substep_check':error,'mass_range':[min(masses),max(masses)]}
        backgrounds.append(panels);histories.append(history)
        np.savez_compressed(OUTPUT/f'initial_{law}_to_gmm_particles.npz',s=TIMES,positions=history)
        del fields
        print(f'{law}: trajectories ready; RK4 check {error:.3g}',flush=True)
    colours=plt.get_cmap('turbo')(np.linspace(0.05,0.95,PARTICLES))
    plt.rcParams.update({'font.family':'serif','font.size':11,'pdf.fonttype':42})
    fig,axes=plt.subplots(4,6,figsize=(20,14),sharex=True,sharey=True,layout='constrained')
    colour_norm=PowerNorm(0.35,vmin=0,vmax=max(p.max() for row in backgrounds for p in row))
    labels=['Gaussian','Laplace','Uniform disk','Uniform square']
    for row,law in enumerate(LAWS):
        for col,i in enumerate(snapshots):
            ax=axes[row,col];p=backgrounds[row][col];history=histories[row]
            image=ax.imshow(p,origin='lower',extent=(-HALF,HALF,-HALF,HALF),cmap='coolwarm',
                            norm=colour_norm,interpolation='nearest',rasterized=True)
            levels=np.array([0.002,0.01,0.04,0.1,0.2,0.4])
            ax.contour(X,Y,p,levels=levels[levels<p.max()],colors='#242934',linewidths=0.5,alpha=0.75)
            ax.add_collection(LineCollection(np.transpose(history[:i+1],(1,0,2)),colors=colours,
                                             linewidths=0.55,alpha=0.7))
            ax.scatter(history[i,:,0],history[i,:,1],s=6,c=colours,edgecolors='white',linewidths=0.2,zorder=4)
            if row==0: ax.set_title(rf'$t={1-TIMES[i]:.1f}$',fontsize=13)
            ax.set(xlim=(-5.5,5.5),ylim=(-5.5,5.5),xticks=[-4,0,4],yticks=[-4,0,4]);ax.set_aspect('equal')
        axes[row,0].set_ylabel(labels[row]+'\n'+r'$y_2$',fontsize=13)
    for ax in axes[-1]: ax.set_xlabel(r'$y_1$')
    fig.colorbar(image,ax=axes,shrink=0.8,pad=0.015,label='Density (common power scale, exponent 0.35)')
    for ext in ['png','pdf']:
        path=OUTPUT/f'initial_laws_to_complex_gmm.{ext}'
        fig.savefig(path,dpi=240,bbox_inches='tight');print(path,flush=True)
    plt.close(fig)
    diagnostics.update({'grid':GRID,'time_grid':len(TIMES),'endpoint_mollification_std':EPS,
                        'target_means':MEANS.tolist(),'initial_unmollified_covariance':'I'})
    (OUTPUT/'initial_laws_to_complex_gmm_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')


if __name__=='__main__': main()
