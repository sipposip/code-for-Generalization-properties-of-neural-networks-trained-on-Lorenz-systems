
import os

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D    # even though we done use this directly, we need to import to
                                            # enable 3d plotting



plt.rcParams['savefig.bbox'] = 'tight'

plotdir = 'plots_diverse'

os.system(f'mkdir -p {plotdir}')

beta = 8. / 3.
rho = 28
sigma = 10


def lorenz(X, t=0):
    x, y, z = X
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    dXdt = [dxdt, dydt, dzdt]
    return dXdt


def make_lorenz_run(y0, Nsteps, tstep):
    t = np.arange(0, Nsteps) * tstep
    sol = odeint(lorenz, y0, t)
    return sol


Nsteps = 100000
tstep = 0.01
modelrun1 = make_lorenz_run([8, 1, 1], Nsteps, tstep)
# model run has diension (time,variable), where variable is [x,y,z]
# normalize
modelrun1 = (modelrun1 - modelrun1.mean(axis=0)) / modelrun1.std(axis=0)


def lorenz3dplot_scatter(data, title="", cmap=plt.cm.gist_heat_r, **kwargs):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], lw=0.5, cmap=cmap,**kwargs)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_title(title)
    return sc


train_data_full = modelrun1[:int(Nsteps / 2)]
# compute length of tendency vector
tendency = np.sqrt(np.mean((train_data_full[1:] - train_data_full[:-1])**2, axis=1))

sc = lorenz3dplot_scatter(train_data_full[:-1], c=tendency, title='model tendencies')
plt.colorbar(sc)
plt.savefig(f'{plotdir}/lorenz63_model_tendencies.png')