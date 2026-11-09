"""Two noise-to-data videos, integrating the exact marginal velocity.

Outputs: two MP4 videos, PNG/PDF final frames, particle trajectories (NPZ),
and a numerical integration check (JSON), all in figures/.
Run with Python, NumPy, Matplotlib and ffmpeg available.
"""
import json
import shutil
import subprocess
import numpy as np
from plot_contracted_convolution import OUTPUT, CENTRES, SIGMAS, density
from plot_convolution_velocity import velocity
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import PowerNorm
from matplotlib.collections import LineCollection

FRAMES = 181
FPS = 24
PARTICLES = 150


def generative_velocity(points, s):
    vx, vy = velocity(points[:, 0, None], points[:, 1, None], 1.-s)
    return -np.column_stack((vx.ravel(), vy.ravel()))


def integrate(initial, times, substeps):
    history = [initial.copy()]
    points = initial.copy()
    for start, end in zip(times[:-1], times[1:]):
        h = (end-start)/substeps
        for j in range(substeps):
            s = start + j*h
            k1 = generative_velocity(points, s)
            k2 = generative_velocity(points+h*k1/2, s+h/2)
            k3 = generative_velocity(points+h*k2/2, s+h/2)
            k4 = generative_velocity(points+h*k3, s+h)
            points += h*(k1+2*k2+2*k3+k4)/6
        history.append(points.copy())
    return np.asarray(history)


def render(mode, times, history):
    stem = 'generative_velocity' if mode == 'velocity' else 'generative_particle_trajectories'
    axis = np.linspace(-6., 6., 401)
    x, y = np.meshgrid(axis, axis)
    levels = [0.002, 0.01, 0.04, 0.1, 0.3]
    fig, ax = plt.subplots(figsize=(8.4, 7.2), layout='constrained')
    image = ax.imshow(density(x, y, 1.), extent=(-6,6,-6,6), origin='lower',
                      cmap='coolwarm', norm=PowerNorm(0.35, vmin=0, vmax=1),
                      interpolation='nearest')
    contours = None
    ax.set(xlim=(-6,6), ylim=(-6,6), xlabel=r'$y_1$', ylabel=r'$y_2$')
    ax.set_aspect('equal')
    title = ax.set_title('')
    fig.colorbar(image, ax=ax, shrink=0.85, label=r'Density $p_{1-s}$ (power scale, exponent 0.35)')
    if mode == 'velocity':
        gx, gy = np.meshgrid(np.arange(-5.7,5.71,0.3), np.arange(-5.7,5.71,0.3))
        arrows = ax.quiver(gx, gy, np.zeros_like(gx), np.zeros_like(gy),
                           color='#171923', angles='xy', scale_units='xy',
                           scale=10, width=0.0025, pivot='tail')
    else:
        # All past segments remain visible; colours identify individual particles.
        colours = plt.get_cmap('turbo')(np.linspace(0.05,0.95,len(history[0])))
        trails = LineCollection([], colors=colours, linewidths=1.0, alpha=0.8, zorder=4)
        ax.add_collection(trails)
        particles = ax.scatter(history[0,:,0], history[0,:,1], s=15,
                               c=colours, edgecolors='white', linewidths=0.35, zorder=5)
    writer = FFMpegWriter(fps=FPS, codec='libx264', bitrate=2600,
                         extra_args=['-pix_fmt','yuv420p','-movflags','+faststart','-threads','2'])
    with writer.saving(fig, str(OUTPUT/f'{stem}.mp4'), dpi=120):
        for i, s in enumerate(times):
            t = 1.-s
            p = density(x,y,t)
            image.set_data(p)
            if contours is not None:
                contours.remove()
            contours = ax.contour(x,y,p,levels=levels,colors='#242934',linewidths=0.65,alpha=0.75)
            title.set_text(rf'Gaussian to Gaussian mixture: $s={s:.2f}$')
            if mode == 'velocity':
                vx, vy = velocity(gx,gy,t)
                visible = (density(gx,gy,t)>=0.002) | (np.abs(gx)+np.abs(gy)<=3*s+1e-10)
                arrows.set_UVC(np.ma.array(-vx,mask=~visible), np.ma.array(-vy,mask=~visible))
            else:
                trails.set_segments(np.transpose(history[:i+1],(1,0,2)))
                particles.set_offsets(history[i])
            writer.grab_frame()
            if i % 60 == 0:
                print(f'{mode}: {i}/{len(times)-1}', flush=True)
        # Brief hold at the final state.
        for _ in range(FPS):
            writer.grab_frame()
    for ext in ('png','pdf'):
        fig.savefig(OUTPUT/f'{stem}_final.{ext}', dpi=200)
    plt.close(fig)
    print(OUTPUT/f'{stem}.mp4', flush=True)


def render_particle_panels(times, history):
    """Eight snapshots; displayed t=1-s decreases from one to zero."""
    axis = np.linspace(-6., 6., 501)
    x, y = np.meshgrid(axis, axis)
    colours = plt.get_cmap('turbo')(np.linspace(0.05, 0.95, len(history[0])))
    fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.6), sharex=True,
                             sharey=True, layout='constrained')
    indices = np.rint(np.linspace(0, len(times)-1, 8)).astype(int)
    for ax, i in zip(axes.flat, indices):
        t = 1.-times[i]
        image = ax.imshow(density(x,y,t), extent=(-6,6,-6,6), origin='lower',
                          cmap='coolwarm', norm=PowerNorm(0.35,vmin=0,vmax=1),
                          interpolation='nearest', rasterized=True)
        ax.contour(x,y,density(x,y,t), levels=[0.002,0.01,0.04,0.1,0.3],
                   colors='#242934', linewidths=0.5, alpha=0.7)
        trails = LineCollection(np.transpose(history[:i+1],(1,0,2)),
                                colors=colours, linewidths=0.55, alpha=0.65)
        ax.add_collection(trails)
        ax.scatter(history[i,:,0], history[i,:,1], s=6, c=colours,
                   edgecolors='white', linewidths=0.2, zorder=5)
        ax.set_title(rf'$t={t:.2f}$')
        ax.set(xlim=(-6,6),ylim=(-6,6),xticks=[-4,0,4],yticks=[-4,0,4])
        ax.set_aspect('equal')
    for ax in axes[-1]:
        ax.set_xlabel(r'$y_1$')
    for ax in axes[:,0]:
        ax.set_ylabel(r'$y_2$')
    fig.colorbar(image, ax=axes, shrink=0.85, pad=0.025,
                 label=r'Density $p_t$ (power scale, exponent 0.35)')
    for ext in ('png','pdf'):
        path = OUTPUT/f'generative_particle_panels.{ext}'
        fig.savefig(path,dpi=240,bbox_inches='tight')
        print(path,flush=True)
    plt.close(fig)


def main(mode="both"):
    candidates = [shutil.which('ffmpeg'), '/opt/homebrew/bin/ffmpeg']
    for binary in candidates:
        if binary and subprocess.run([binary, '-version'], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL).returncode == 0:
            plt.rcParams['animation.ffmpeg_path'] = binary
            break
    else:
        raise RuntimeError('A working ffmpeg is required for MP4 output')
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.size':12,'pdf.fonttype':42})
    times = np.linspace(0,1,FRAMES)
    initial = np.random.default_rng(20261030).normal(size=(PARTICLES,2))
    history = integrate(initial,times,8)
    coarse = integrate(initial,times,4)
    error = float(np.linalg.norm(history-coarse,axis=2).max())
    assert np.isfinite(history).all() and error < 1e-3, error
    np.savez_compressed(OUTPUT/'generative_particle_trajectories.npz',
                        time=times, positions=history, initial=initial)
    report = {'particles':PARTICLES, 'method':'RK4', 'substeps_per_frame':8,
              'max_difference_from_half_resolution':error,
              'density_background':'Exact analytic density, not estimated from the particles',
              'direction':'s=0: initial Gaussian; s=1: data mixture; velocity=-v_(1-s)'}
    (OUTPUT/'generative_integration_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)
    render_particle_panels(times,history)
    if mode in ('both', 'velocity'):
        render('velocity',times,history)
    if mode in ('both', 'particles'):
        render('particles',times,history)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['both', 'velocity', 'particles', 'panels'], default='both')
    main(parser.parse_args().mode)
