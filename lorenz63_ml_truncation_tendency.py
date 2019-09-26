"""

this script makes lorenz63 simulations with a standard odeint solver, and trains a shallow
neural network on the lorenz63 simulation

the network is evaluated

a network is also trianed on a truncated versoin of the run (everything x>1.5 removed), and evaluated


@author: Sebastian Scher, march 2019
"""
import os
import matplotlib
matplotlib.use('agg')
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import keras
from mpl_toolkits.mplot3d import Axes3D    # even though we done use this directly, we need to import to
                                            # enable 3d plotting

from keras import backend as K
import tensorflow as tf
import keras

# # limit maximum number of CPUs
# config = tf.ConfigProto(intra_op_parallelism_threads=20, inter_op_parallelism_threads=20,
#                         allow_soft_placement=True)
# session = tf.Session(config=config)
# K.set_session(session)



plt.rcParams['savefig.bbox'] = 'tight'

plotdir = 'plots_truncation_tendency'


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


def lorenz3dplot_lines(data, title="", **kwargs):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sc = ax.plot(data[:, 0], data[:, 1], data[:, 2], lw=0.5, **kwargs)
    ax.set_xlabel("X ")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_title(title)
    return sc


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

def compute_3d_density(data):
    extent = (-3, 3.01)
    spacing = 0.3
    bins = np.arange(extent[0], extent[1], spacing)
    hist3d = np.histogramdd(data, bins=[bins, bins, bins], normed=True)

    return hist3d

# %% now train a neural network
n_epoch = 100
lead_time = 1  # in steps
dens_thresh = 0.002

# parameters from tuning
hidden_size = 128
n_hidden=2
lr=3e-05



def train_network(X_train, y_train, selection_strategy, reference_clim = None, n_iter=10):
    """
    traina couple of networsk, and returned the best one
    :param X_train:
    :param y_train:
    :param selection_strategy: one of  'density', 'no_equal_points', 'no_fixpoint','density-full'
    :param reference_clim: reference timeseries used for 'density-full'. igonred for other selection strategies
    :return: trained keras network
    """

    # convert the target (y_train) into tendencies
    y_train = y_train - X_train
    # repeat training n_iter times
    res = []
    for _ in range(n_iter):

        # build and train network
        layers = []
        for _ in range(n_hidden):
            layers.append(keras.layers.Dense(hidden_size, input_shape=(3,), activation='relu'))

        layers.append(keras.layers.Dense(3, activation='linear'))
        network = keras.Sequential(layers)
        print(network.summary())

        optimizer = keras.optimizers.Adam(lr=lr)
        network.compile(optimizer=optimizer, loss='mean_squared_error')

        network.fit(X_train, y_train, epochs=n_epoch, verbose=2, validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                             min_delta=0,
                                                             patience=4,

                                                             verbose=1, mode='auto')]
                    )
        # now make a climate run with the network,  and check whether it does
        # meet the selection criterion in selection_strategy
        clim = make_network_climrun(network, X_train)

        if selection_strategy == 'no_fixpoint':
            # test whether there are adjactants states that are identical down to machine precision
            diffs = np.diff(clim,axis=0)
            # check whether there is a point where diff is zero for all variables
            if not np.any(np.all(diffs==0, axis=1)):
               res = network
               break
            else:
                print('reconstructed attractor collapsed, repeating training')
                continue

        elif selection_strategy == 'no_equal_points':
            # test whether there are points that are identical down to machine precision
            # this we can do by comparing the lenght of clim with the unique values (along axis=0) in clim
            if len(clim) == len(np.unique(clim, axis=0)):
               res = network
               break
            else:
                print('reconstructed attractor collapsed, repeating training')
                continue

        elif selection_strategy == 'density':
            # test whether the difference in 3d density is below a threshold given by dens_thresh
            dens_model, _ = compute_3d_density(X_train)
            dens_network, _ = compute_3d_density(clim)

            rmse_dens = np.sqrt(np.mean((dens_network - dens_model) ** 2))
            res.append([rmse_dens,network])

        elif selection_strategy == 'density-full':
            # test whether the difference in 3d density is below a threshold given by dens_thresh,
            # by using a reference climate (from the full network)
            dens_model, _ = compute_3d_density(reference_clim)
            dens_network, _ = compute_3d_density(clim)

            rmse_dens = np.sqrt(np.mean((dens_network - dens_model) ** 2))
            res.append([rmse_dens, network])



        else:
            raise ValueError(f'selection strategy {selection_strategy} not implemented')


    if selection_strategy in ('density', 'density-full'):
        # select the newtork with best attractor density error
        dens_errors = [e[0] for e in res]
        min_idx = np.argmin(dens_errors)
        network = res[min_idx][1]
        return  network

    elif selection_strategy in ('no_fixpoint', 'no_equal_points'):
        if network == []:
            print('no suitable network found')
            return None

    else:
        raise ValueError(f'selection strategy {selection_strategy} not implemented')



def make_network_climrun(network, test_data):
    # crreate a "climate run" with the network

    N_clim = Nsteps//2
    # as initial state, use a random state from the test data
    y_init_clim = test_data[np.random.randint(len(test_data))]

    network_clim_run = np.zeros((N_clim + 1, 3))
    network_clim_run[0] = y_init_clim
    state = y_init_clim
    state = np.expand_dims(state, axis=0)
    for i in range(N_clim):
        # print(i,N_clim)
        tendency = network.predict(state)
        state = state + tendency
        network_clim_run[i + 1] = state

    return network_clim_run


# split in first and second half
train_data_full = modelrun1[:int(Nsteps / 2)]
test_data_full = modelrun1[int(Nsteps / 2):]

# X and y data are the same, but shifted by lead-time

X_train_full = train_data_full[:-lead_time]
y_train_full = train_data_full[lead_time:]

X_test_full = test_data_full[:-lead_time]
y_test_full = test_data_full[lead_time:]


for selection_strategy in ('density', 'density-full'):

    network_full = train_network(X_train_full, y_train_full, selection_strategy=selection_strategy,
                                 reference_clim = X_train_full   )

    if network_full is None:
        print(f'no network found for selection strategy {selection_strategy}')
        continue
    network_clim_full = make_network_climrun(network_full, test_data_full)

    # plot the climates
    lorenz3dplot_lines(X_train_full, 'full model run', alpha=0.8)
    plt.savefig(f'{plotdir}/model_full_tendency.png')

    lorenz3dplot_lines(network_clim_full, 'network full clim', alpha=0.8)
    plt.savefig(f'{plotdir}/network_clim_full_tendency.png')

    # now with truncated training data,

    # when selecting on density, we can only remove small parts of the attractor, otherwise
    # the selection criterion will never pass
    if selection_strategy == 'density':
        all_lims =  ([[-3,1.5],[-3,3],[-3,3]], )   # removes tip of right wing
    else:
        all_lims = (
                 [[-3, 1.5], [-3, 3], [-3, 3]], # removes tip of right wing
                 [[-3,0],[-3,3],[-3,3]],  # left wing
                 [[0,3],[-3,3],[-3,3]], # right wing
                )


    for lims in all_lims:

        param_string = '_'.join([str(e) for e in (Nsteps, tstep, n_epoch, lead_time, dens_thresh, lims, selection_strategy,
                                                  '_tendency')])

        idcs_train = (X_train_full[:,0] > lims[0][0]) & (X_train_full[:,0] < lims[0][1])  & \
                   (X_train_full[:,1] > lims[1][0]) & (X_train_full[:,1] < lims[1][1])  & \
                   (X_train_full[:,2] > lims[2][0]) & (X_train_full[:,2] < lims[2][1])

        idcs_test = (X_test_full[:,0] > lims[0][0]) & (X_test_full[:,0] < lims[0][1])  & \
                   (X_test_full[:,1] > lims[1][0]) & (X_test_full[:,1] < lims[1][1])  & \
                   (X_test_full[:,2] > lims[2][0]) & (X_test_full[:,2] < lims[2][1])


        X_train_trunc = X_train_full[idcs_train]
        # for y_train_trunc, we have to use exactly the same indices as for X_train_trunc
        y_train_trunc = y_train_full[idcs_train]
        # for y_test_trunc, we have to use exactly the same indices as for X_test_trunc
        X_test_trunc = X_test_full[idcs_test]
        y_test_trunc = y_test_full[idcs_test]

        network_truncated = train_network(X_train_trunc, y_train_trunc, selection_strategy=selection_strategy,
                                          reference_clim = X_train_full)

        if network_full is None:
            print(f'no network found for selection strategy {selection_strategy}')
            continue

        network_clim_trunc = make_network_climrun(network_truncated, X_test_trunc)



        lorenz3dplot_lines(X_train_trunc, 'model truncated', alpha=0.8)
        plt.savefig(f'{plotdir}/model_trunc{param_string}.png')


        lorenz3dplot_lines(network_clim_trunc, 'network trunc clim', alpha=0.8)
        plt.savefig(f'{plotdir}/network_clim_trunc{param_string}.png')

        # compute "weather" prediction errors, for this first make 1 step predicton and then compute
        # the error
        pred_full = X_test_full + network_full.predict(X_test_full)
        pred_trunc = X_test_trunc + network_truncated.predict(X_test_trunc)

        abserr_full = np.abs(pred_full - y_test_full).mean(axis=1)
        abserr_trunc = np.abs(pred_trunc - y_test_trunc).mean(axis=1)

        # now use the trunc net to predict on the whole attractor
        pred_trunc_on_full = X_test_full + network_truncated.predict(X_test_full)
        abserr_trunc_on_full = np.abs(pred_trunc_on_full - y_test_full).mean(axis=1)

        # plot the weather prediction errors
        sc = lorenz3dplot_scatter(X_test_full, c=abserr_full, title='abserror full')
        plt.colorbar(sc)
        plt.savefig(f'{plotdir}/abserror_full{param_string}.png')

        sc = lorenz3dplot_scatter(X_test_full, c=abserr_full, title='abserror full same colorscale',
                                  vmin=abserr_trunc_on_full.min(), vmax=abserr_trunc_on_full.max())
        plt.colorbar(sc)
        plt.savefig(f'{plotdir}/abserror_full_same_colorscale{param_string}.png')


        sc = lorenz3dplot_scatter(X_test_trunc, c=abserr_trunc, title='abserror trunc')
        plt.colorbar(sc)
        plt.savefig(f'{plotdir}/abserror_trunc{param_string}.png')

        sc = lorenz3dplot_scatter(X_test_trunc, c=abserr_trunc, title='abserror trunc same colorscale',
                                  vmin=abserr_full.min(), vmax=abserr_full.max()
                                  )
        plt.colorbar(sc)
        plt.savefig(f'{plotdir}/abserror_trunc_smallcolorbar{param_string}.png')

        # same fixed colorscales as for full-state networks
        if lims == [[-3,0],[-3,3],[-3,3]] or lims == [[0,3],[-3,3],[-3,3]]:
            vmax = 0.7
        else:
            vmax = 0.07

        sc = lorenz3dplot_scatter(X_test_full, c=abserr_trunc_on_full, title='abserror trunc on full',
                                      vmax=vmax, vmin=0)

        plt.colorbar(sc)
        plt.savefig(f'{plotdir}/abserror_trunc_full{param_string}.png')

        sc = lorenz3dplot_scatter(pred_trunc_on_full, c=abserr_trunc_on_full, title='abserror trunc on full')
        plt.colorbar(sc)
        plt.savefig(f'{plotdir}/abserror_trunc_full_predictedpoints{param_string}.png')




        #%% now make a plot where we initialzie a couple of points in the left out chunk
        # for this we need to reverse the selection (we need to have only the data in the left out
        # chunk)
        X_test_leftout = X_test_full[~idcs_test]
        N_init = 30
        init_point_idcs = np.random.choice(np.arange(len(X_test_leftout)), size=N_init)
        init_points = X_test_leftout[init_point_idcs]
        # now make network forecasts started from these points
        N_fc = 200  # steps to forecast
        trajectories = [init_points]
        current = trajectories[0]
        for _ in range(N_fc):
            current = current + network_truncated.predict(current)
            trajectories.append(current)
        trajectories = np.array(trajectories)
        # this has now shape (N_fc+1,N_init,3)
        assert(trajectories.shape==(N_fc+1,N_init,3))

        plt.figure()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for i in range(N_init):
            trajectory = trajectories[:,i,:]
            ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], color='black', alpha=0.6)
            sc=ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c=np.arange(N_fc+1), s=8,alpha=0.6)
        cb = plt.colorbar(sc)
        cb.set_label('time step')
        cb.solids.set_edgecolor("face") # this avoids stripes in the colorbar
        ax.set_xlabel("X ")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plt.savefig(f'{plotdir}/trajectories_initilized_in_leftout_{param_string}.png')