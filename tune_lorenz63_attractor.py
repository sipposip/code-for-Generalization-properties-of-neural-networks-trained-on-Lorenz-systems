
import os
import pickle
import matplotlib
matplotlib.use('agg')
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

from mpl_toolkits.mplot3d import Axes3D



import keras


plt.rcParams['savefig.bbox'] = 'tight'

plotdir = 'plots_tune_lorenz63'
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




def lorenz3dplot_lines(data, title="", **kwargs):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sc = ax.plot(data[:, 0], data[:, 1], data[:, 2], lw=0.5, **kwargs)
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

    return hist3d[0]

# %% now train a neural network


def train_network(X_train, y_train,lr, hidden_size,n_hidden):
    """

    :param X_train:
    :param y_train:
    :param selection_strategy: one of  'density', 'no_equal_points', 'no_fixpoint','density-full'
    :param reference_clim: reference timeseries used for 'density-full'. igonred for other selection strategies
    :return: trained keras network
    """

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



    return network


def make_network_climrun(model, test_data, N_clim):
    # crreate a "climate run" with the network

    # as initial state, use a random state from the test data
    y_init_clim = test_data[np.random.randint(len(test_data))]

    network_clim_run = np.zeros((N_clim + 1, 3))
    network_clim_run[0] = y_init_clim
    state = y_init_clim
    state = np.expand_dims(state, axis=0)
    for i in range(N_clim):
        # print(i,N_clim)
        state = model.predict(state)
        network_clim_run[i + 1] = state

    return network_clim_run


Nsteps = 20000
tstep = 0.01


n_epoch = 100
lead_time = 1  # in steps

x_init = [ -9.42239121, -15.38518668,  18.19349896]

modelrun1 = make_lorenz_run(x_init, Nsteps, tstep)


modelrun1 = (modelrun1 - modelrun1.mean(axis=0)) / modelrun1.std(axis=0)


# split in first and second half
train_data_full = modelrun1[:int(Nsteps / 2)]
test_data_full = modelrun1[int(Nsteps / 2):]

# X and y data are the same, but shifted by lead-time

X_train = train_data_full[:-lead_time]
y_train = train_data_full[lead_time:]

X_test = test_data_full[:-lead_time]
y_test = test_data_full[lead_time:]

dens_true = compute_3d_density(X_train)


tunable_params = dict(
                  lr=[0.00003,0.0001,0.003],
                  n_hidden = [1,2,3,4,6,10],
                  hidden_size = [4,8,16,32,64,128]
        )


param_grid = list(ParameterGrid(tunable_params))
print(f'trying {len(param_grid)} param combinations')
tune_res = []
for i_param,params in enumerate(param_grid):
    # give every param combinatoin 4 changes
    for i in range(4):
        param_string = '_'.join([str(e) for e in (Nsteps, tstep, n_epoch, lead_time)])

        network = train_network(X_train, y_train, **params)

        network_clim = make_network_climrun(network, [X_train[0]], Nsteps//2)
        dens_net  = compute_3d_density(network_clim)

        dens_error = np.mean(np.abs(dens_true - dens_net))

        # also compute short term forecast error
        preds = network.predict(X_test)
        abse_shortterm = np.mean(np.abs(preds-y_test))

        lorenz3dplot_lines(network_clim, title=str(dens_error))
        plt.savefig(f'{plotdir}/nn_attractor_{i_param}_iter{i}.svg')
        res = {'params':params, 'dens_error': dens_error, 'abse_shortterm':abse_shortterm}
        pickle.dump(res, open(f'nn_attractor_{i_param}_iter{i}.pkl','wb'))
        tune_res.append(res)

pickle.dump(tune_res, open('result_tuning_lorenz63.pkl','wb'))