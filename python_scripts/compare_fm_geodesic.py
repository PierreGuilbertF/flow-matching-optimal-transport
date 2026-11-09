"""FM versus an empirical quadratic-OT geodesic at six times.

Equal-weight Sobol samples approximate the endpoints. Solve the exact
finite assignment problem for squared Euclidean cost, then interpolate
assigned endpoints. Smooth particle mass and momentum with the same
Gaussian kernel for plotting; their ratio is the plotted velocity.
This is a numerical approximation to the continuous-density geodesic.
Always save PNG and PDF to figures/, plus coupling and diagnostics.
"""
import json
import numpy as np
from plot_contracted_convolution import CENTRES, SIGMAS, OUTPUT, density
from plot_convolution_velocity import velocity
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import qmc, norm
from scipy.ndimage import gaussian_filter, map_coordinates
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.collections import LineCollection
from animate_generative_flow import integrate

N = 16384
TIMES = np.linspace(1,0,6)
BANDWIDTH = 0.16


def endpoint_samples(n):
    exponent=int(np.log2(n))
    base=norm.ppf(qmc.Sobol(2,scramble=True,seed=47).random_base2(exponent-2))
    source=np.concatenate([mu+sigma*base for mu,sigma in zip(CENTRES,SIGMAS)])
    target=norm.ppf(qmc.Sobol(2,scramble=True,seed=91).random_base2(exponent))
    return source,target


def coupling(n):
    source,target=endpoint_samples(n)
    cache=OUTPUT/f'empirical_ot_coupling_{n}.npz'
    legacy=OUTPUT/'empirical_ot_coupling.npz'
    if not cache.exists() and legacy.exists():
        with np.load(legacy) as old:
            if len(old['source']) == n:
                np.savez_compressed(cache,source=old['source'],target=old['target'])
    if cache.exists():
        with np.load(cache) as saved:
            if np.array_equal(saved['source'],source):
                candidate=saved['target']
                if np.array_equal(np.sort(candidate,axis=0),np.sort(target,axis=0)):
                    action=float(np.mean(np.sum((candidate-source)**2,axis=1)))
                    return source,candidate,action
    print(f'Solving exact quadratic assignment for {n} endpoints...',flush=True)
    cost=cdist(source,target,metric='sqeuclidean')
    rows,cols=linear_sum_assignment(cost)
    target=target[cols]
    action=float(cost[rows,cols].mean())
    np.savez_compressed(cache,source=source,target=target)
    print(f'Optimal empirical action ({n} points): {action:.7f}',flush=True)
    return source,target,action


def smooth_mass_flux(points, displacement, edges):
    step=edges[1]-edges[0]
    kwargs=dict(bins=[edges,edges])
    mass=np.histogram2d(points[:,1],points[:,0],**kwargs)[0]/len(points)/step**2
    flux=[np.histogram2d(points[:,1],points[:,0],weights=displacement[:,j],
                         **kwargs)[0]/len(points)/step**2 for j in [0,1]]
    sigma=BANDWIDTH/step
    return (gaussian_filter(mass,sigma,mode='constant'),
            [gaussian_filter(f,sigma,mode='constant') for f in flux])


def main():
    OUTPUT.mkdir(exist_ok=True)
    _,_,coarse_action=coupling(1024)
    source,target,action=coupling(N)
    displacement=target-source
    np.savez_compressed(OUTPUT/'empirical_ot_coupling.npz',source=source,target=target)

    # Select the SAME 150 Gaussian points for both methods, preserving identities.
    chosen=np.random.default_rng(20261109).choice(N,size=150,replace=False)
    initial=target[chosen].copy()
    integration_times=np.linspace(0,1,301)
    fm_history=integrate(initial,integration_times,8)
    coarse_history=integrate(initial,integration_times,4)
    integration_error=float(np.linalg.norm(fm_history-coarse_history,axis=2).max())
    assert integration_error<1e-3
    # OT particles follow exact constant-velocity segments from the coupling.
    ot_history=(1-integration_times[:,None,None])*initial[None,...]+integration_times[:,None,None]*source[chosen][None,...]
    assert np.array_equal(fm_history[0],ot_history[0])
    np.savez_compressed(OUTPUT/'fm_ot_shared_particles.npz',time=integration_times,
                        initial=initial, coupling_indices=chosen,
                        fm_positions=fm_history,ot_positions=ot_history)
    colours=plt.get_cmap('turbo')(np.linspace(0.05,0.95,len(chosen)))

    edges=np.linspace(-8,8,1281)
    centres=(edges[:-1]+edges[1:])/2
    step=centres[1]-centres[0]
    x,y=np.meshgrid(centres,centres)
    gx,gy=np.meshgrid(np.arange(-5.4,5.41,0.35),np.arange(-5.4,5.41,0.35))
    coordinates=np.array([(gy-centres[0])/step,(gx-centres[0])/step])
    plt.rcParams.update({'font.family':'serif','font.size':11,'pdf.fonttype':42})
    fig,axes=plt.subplots(4,6,figsize=(20,14),sharex=True,sharey=True,layout='constrained')
    masses=[]
    for j,t in enumerate(TIMES):
        fm_density=density(x,y,t)
        geo_density,flux=smooth_mass_flux((1-t)*source+t*target,displacement,edges)
        masses.append(float(geo_density.sum()*step**2))
        local_mass=map_coordinates(geo_density,coordinates,order=1)
        geo_velocity=[-map_coordinates(f,coordinates,order=1)/np.maximum(local_mass,1e-12)
                      for f in flux]
        if j in [0,len(TIMES)-1]:
            geo_density=fm_density.copy()
        fm_velocity=[-f for f in velocity(gx,gy,float(t))]
        frame=int(round((1-t)*(len(integration_times)-1)))
        for row in range(4):
            ax=axes[row,j]
            p=fm_density if row<2 else geo_density
            image=ax.imshow(p,extent=(-8,8,-8,8),origin='lower',cmap='coolwarm',
                            norm=PowerNorm(0.35,vmin=0,vmax=1),rasterized=True)
            ax.contour(x,y,p,levels=[0.002,0.01,0.04,0.1,0.3],colors='#242934',linewidths=0.5)
            if row in [0,2]:
                field=fm_velocity if row==0 else geo_velocity
                visible=(density(gx,gy,t)>=0.002) if row==0 else (local_mass>=0.002)
                visible |= np.abs(gx)+np.abs(gy)<=3*(1-t)+1e-9
                ax.quiver(gx,gy,*[np.ma.array(f,mask=~visible) for f in field],
                          angles='xy',scale_units='xy',scale=12,color='#171923',width=0.0028)
            else:
                history=fm_history if row==1 else ot_history
                trails=LineCollection(np.transpose(history[:frame+1],(1,0,2)),
                                      colors=colours,linewidths=0.55,alpha=0.7)
                ax.add_collection(trails)
                ax.scatter(history[frame,:,0],history[frame,:,1],s=6,c=colours,
                           edgecolors='white',linewidths=0.2,zorder=5)
            if row==0: ax.set_title(rf'$t={t:.1f}$',fontsize=13)
            ax.set(xlim=(-5.5,5.5),ylim=(-5.5,5.5),xticks=[-4,0,4],yticks=[-4,0,4])
            ax.set_aspect('equal')
    for ax in axes[-1]: ax.set_xlabel(r'$y_1$')
    for row,label in enumerate(['FM velocity','FM particles','OT velocity','OT particles']):
        axes[row,0].set_ylabel(label+'\n'+r'$y_2$',fontsize=13)
    fig.colorbar(image,ax=axes,shrink=0.75,pad=0.015,
                 label=r'Density (common power scale, exponent 0.35)')
    for ext in ['png','pdf']:
        path=OUTPUT/f'fm_geodesic_comparison.{ext}'
        fig.savefig(path,dpi=240,bbox_inches='tight')
        print(path,flush=True)
    plt.close(fig)
    rng=np.random.default_rng(9)
    i=rng.integers(0,N,10000);j=rng.integers(0,N,10000)
    monotonicity=np.sum((source[i]-source[j])*(target[i]-target[j]),axis=1)
    assert monotonicity.min()>-1e-8
    assert min(masses)>0.999
    report={'samples':N,'bandwidth':BANDWIDTH,'action_1024':coarse_action,
            'action_fine':action,'relative_action_change':abs(action-coarse_action)/action,
            'displayed_masses':masses,'minimum_sampled_monotonicity':float(monotonicity.min()),
            'shared_particles':len(chosen),'identical_initial_positions':True,
            'fm_rk4_max_resolution_difference':integration_error,
            'direction':'Gaussian t=1 to data t=0; arrows are -v_t',
            'interpretation':'Exact OT between empirical measures; smoothed for visualisation, not an exact continuous geodesic'}
    (OUTPUT/'fm_geodesic_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)


if __name__=='__main__': main()
